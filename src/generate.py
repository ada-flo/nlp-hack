"""Beam-search inference using a trained checkpoint.

Example:
    uv run python -m src.generate \\
        --checkpoint checkpoints/bilstm-XXXX/best.pt \\
        --topic "안락사 허용" \\
        --input "고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch

from .data import EOS_ID, PAD_ID, SOS_ID
from .model.lstm_seq2seq import Seq2Seq


def beam_search(
    model: Seq2Seq,
    src: torch.Tensor,
    *,
    src_lens: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    beam_size: int = 4,
    max_len: int = 64,
    length_penalty: float = 1.0,
    sos_id: int = SOS_ID,
    eos_id: int = EOS_ID,
) -> list[int]:
    """Single-example beam search. src is (1, T)."""
    device = src.device
    model.eval()

    enc_outputs, (h0, c0) = model.encode(src, src_lens, attention_mask)
    src_mask = attention_mask.bool() if attention_mask is not None else (src != PAD_ID)

    n_layers = model.decoder.lstm.num_layers
    h0 = h0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
    c0 = c0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()

    # Each beam: (tokens, log_prob, state, finished).
    beams = [([sos_id], 0.0, (h0, c0), False)]

    for _ in range(max_len):
        if all(f for *_, f in beams):
            break
        candidates = []
        for tokens, score, state, finished in beams:
            if finished:
                candidates.append((tokens, score, state, finished))
                continue
            cur = torch.tensor([tokens[-1]], device=device, dtype=torch.long)
            logits, new_state, _ = model.decoder.forward_step(cur, state, enc_outputs, src_mask)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_lp, top_idx = log_probs.topk(beam_size)
            for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                candidates.append((tokens + [idx], score + lp, new_state, idx == eos_id))
        # length-normalized score for sorting.
        candidates.sort(key=lambda c: c[1] / (max(len(c[0]) - 1, 1) ** length_penalty), reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda c: c[1] / (max(len(c[0]) - 1, 1) ** length_penalty))
    return best[0]


def detok(sp: spm.SentencePieceProcessor, ids: list[int]) -> str:
    out = []
    for tok in ids:
        if tok == EOS_ID:
            break
        if tok in (PAD_ID, SOS_ID):
            continue
        out.append(int(tok))
    return sp.decode(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sp", type=Path, default=Path("data/processed/spm.model"))
    parser.add_argument("--topic", required=True)
    parser.add_argument("--input", dest="input_context", required=True)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.sp))

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt["args"]
    encoder_type = train_args.get("encoder", "bilstm")

    model = Seq2Seq(
        vocab_size=ckpt["vocab_size"],
        embed_dim=train_args.get("embed_dim", 256),
        hidden_dim=train_args.get("hidden_dim", 512),
        enc_layers=train_args.get("enc_layers", 2),
        dec_layers=train_args.get("dec_layers", 1),
        dropout=0.0,
        encoder_type=encoder_type,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    encoder_input = f"{args.topic} <SEP> {args.input_context}"

    if encoder_type == "bilstm":
        ids = sp.encode_as_ids(encoder_input)
        src = torch.tensor([ids], device=args.device, dtype=torch.long)
        src_lens = torch.tensor([len(ids)], device=args.device, dtype=torch.long)
        out_ids = beam_search(model, src, src_lens=src_lens,
                              beam_size=args.beam_size, max_len=args.max_len,
                              length_penalty=args.length_penalty)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        enc = tok(encoder_input, return_tensors="pt").to(args.device)
        out_ids = beam_search(model, enc["input_ids"], attention_mask=enc["attention_mask"],
                              beam_size=args.beam_size, max_len=args.max_len,
                              length_penalty=args.length_penalty)

    output = detok(sp, out_ids)
    print(json.dumps({
        "topic": args.topic,
        "input_context": args.input_context,
        "encoder_input": encoder_input,
        "output": output,
        "encoder": encoder_type,
        "beam_size": args.beam_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
