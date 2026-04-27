"""Train the LSTM seq2seq debate chatbot.

Two encoder modes:
    --encoder bilstm  (Phase 2 baseline, optionally with FastText init)
    --encoder xlmr    (Phase 3 hybrid, frozen xlm-roberta-base encoder)

Example:
    uv run python -m src.train --encoder bilstm --epochs 10 --batch-size 32
    uv run python -m src.train --encoder bilstm --fasttext-en path.vec --fasttext-ko path.vec
    uv run python -m src.train --encoder xlmr --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from .data import (
    PAD_ID,
    EOS_ID,
    SOS_ID,
    make_loaders,
    make_loaders_xlmr,
)
from .model.lstm_seq2seq import Seq2Seq


def label_smoothed_nll_loss(logits, targets, label_smoothing: float, ignore_index: int):
    """Cross-entropy with label smoothing, ignoring pad positions."""
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    mask = (targets != ignore_index).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def decode_ids(sp, ids: list[int]) -> str:
    out = []
    for tok in ids:
        if tok == EOS_ID:
            break
        if tok in (PAD_ID, SOS_ID):
            continue
        out.append(int(tok))
    return sp.decode(out)


@torch.no_grad()
def evaluate(model, loader, sp, device, encoder_type, max_decode_len=64, max_batches=None):
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    refs: list[str] = []
    hyps: list[str] = []
    bleu = BLEU(lowercase=True, tokenize="13a")

    for i, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        decoder_input = batch["decoder_input"]
        decoder_target = batch["decoder_target"]

        if encoder_type == "bilstm":
            logits = model(batch["src"], decoder_input, src_lens=batch["src_lens"])
            attn = None
        else:
            logits = model(batch["src"], decoder_input, attention_mask=batch["attention_mask"])
            attn = batch["attention_mask"]

        loss = label_smoothed_nll_loss(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
            label_smoothing=0.0,  # plain NLL for ppl reporting
            ignore_index=PAD_ID,
        )
        n_mask = (decoder_target != PAD_ID).sum().item()
        total_loss += loss.item() * n_mask
        n_tokens += n_mask

        if encoder_type == "bilstm":
            preds = model.greedy_decode(batch["src"], src_lens=batch["src_lens"], max_len=max_decode_len)
        else:
            preds = model.greedy_decode(batch["src"], attention_mask=attn, max_len=max_decode_len)

        for ref_ids, hyp_ids in zip(decoder_target.tolist(), preds.tolist()):
            refs.append(decode_ids(sp, ref_ids))
            hyps.append(decode_ids(sp, hyp_ids))

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    bleu_score = bleu.corpus_score(hyps, [refs]).score if refs else 0.0
    return {"loss": avg_loss, "ppl": ppl, "bleu": bleu_score, "n_eval": len(refs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["bilstm", "xlmr"], default="bilstm")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sp", type=Path, default=Path("data/processed/spm.model"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-src-len", type=int, default=128)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=None,
                        help="Limit eval batches per epoch (for speed)")
    parser.add_argument("--fasttext-en", type=Path, default=None,
                        help="Path to cc.en.300.vec (BiLSTM only)")
    parser.add_argument("--fasttext-ko", type=Path, default=None,
                        help="Path to cc.ko.300.vec (BiLSTM only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.encoder}-{int(time.time())}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {run_dir}")
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    if args.encoder == "bilstm":
        train_loader, valid_loader, test_loader, sp = make_loaders(
            data_dir=args.data_dir, sp_path=args.sp,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
        )
    else:
        train_loader, valid_loader, test_loader, sp, _ = make_loaders_xlmr(
            data_dir=args.data_dir, sp_path=args.sp,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
        )

    vocab_size = sp.get_piece_size()
    print(f"[train] vocab={vocab_size}, batches: train={len(train_loader)} valid={len(valid_loader)}")

    model = Seq2Seq(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        encoder_type=args.encoder,
    ).to(args.device)

    # FastText init (BiLSTM only).
    if args.encoder == "bilstm" and (args.fasttext_en or args.fasttext_ko):
        from .model.embeddings import load_fasttext_for_spm
        vec_paths = [p for p in (args.fasttext_en, args.fasttext_ko) if p and p.exists()]
        if vec_paths:
            init = load_fasttext_for_spm(sp, vec_paths, target_dim=args.embed_dim)
            with torch.no_grad():
                model.encoder.embedding.weight.copy_(init.to(args.device))
                # decoder shares the same embedding when bilstm — already updated.
            print(f"[train] FastText embeddings initialized from {[str(p) for p in vec_paths]}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    best_bleu = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            decoder_input = batch["decoder_input"]
            decoder_target = batch["decoder_target"]

            if args.encoder == "bilstm":
                logits = model(batch["src"], decoder_input, src_lens=batch["src_lens"])
            else:
                logits = model(batch["src"], decoder_input, attention_mask=batch["attention_mask"])

            loss = label_smoothed_nll_loss(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                label_smoothing=args.label_smoothing,
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip
            )
            optimizer.step()
            running += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running/n_batches:.3f}")

        train_loss = running / max(n_batches, 1)
        elapsed = time.time() - t0

        valid_metrics = evaluate(
            model, valid_loader, sp, args.device, args.encoder,
            max_decode_len=args.max_tgt_len, max_batches=args.eval_batches,
        )
        print(
            f"[epoch {epoch}] train_loss={train_loss:.3f} "
            f"valid_loss={valid_metrics['loss']:.3f} ppl={valid_metrics['ppl']:.2f} "
            f"bleu={valid_metrics['bleu']:.2f}  ({elapsed:.0f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **valid_metrics, "elapsed_s": elapsed})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if valid_metrics["bleu"] > best_bleu:
            best_bleu = valid_metrics["bleu"]
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "valid": valid_metrics,
            }
            torch.save(ckpt, run_dir / "best.pt")
            print(f"[epoch {epoch}] saved best (bleu={best_bleu:.2f})")

    # Final test eval.
    print("[test] evaluating on test set with best checkpoint")
    best = torch.load(run_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(
        model, test_loader, sp, args.device, args.encoder,
        max_decode_len=args.max_tgt_len,
    )
    print(f"[test] loss={test_metrics['loss']:.3f} ppl={test_metrics['ppl']:.2f} "
          f"bleu={test_metrics['bleu']:.2f}")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
