"""Push a trained checkpoint + tokenizer to a Hugging Face model repo.

Uploads the LSTM seq2seq best.pt, the SentencePiece tokenizer, the training
args/history/test_metrics, and a model card with inference snippet.

Usage:
    HF_TOKEN must be set with WRITE scope.

    uv run python scripts/push_model_to_hf.py \\
        --repo-id ada-flo/nlp-hack-debate-xlmr-lstm \\
        --checkpoint-dir checkpoints/xlmr-1777348083
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_card(
    *,
    repo_id: str,
    args: dict,
    history: list[dict],
    test_metrics: dict,
) -> str:
    encoder = args.get("encoder", "?")
    init_from = args.get("init_from") or "(from scratch)"
    epochs = args.get("epochs", "?")
    batch_size = args.get("batch_size", "?")
    lr = args.get("lr", "?")
    embed_dim = args.get("embed_dim", "?")
    hidden_dim = args.get("hidden_dim", "?")
    enc_layers = args.get("enc_layers", "?")
    dec_layers = args.get("dec_layers", "?")

    history_rows = "\n".join(
        f"| {h['epoch']} | {h['train_loss']:.3f} | {h['loss']:.3f} | {h['ppl']:.1f} | {h['bleu']:.3f} |"
        for h in history
    )

    return f"""\
---
language:
  - en
  - ko
license: cc-by-4.0
library_name: pytorch
tags:
  - seq2seq
  - lstm
  - debate
  - bilingual
pipeline_tag: text-generation
---

# {repo_id}

Bilingual (English + Korean) LSTM seq2seq debate chatbot. The encoder is a
frozen XLM-RoBERTa-base providing contextual hidden states; the decoder is
an LSTM with Bahdanau attention. Trained on debate-shape (topic, PRO, CON)
records plus discourse corpora for fluency.

## Test metrics

- **BLEU**: {test_metrics.get("bleu", 0):.3f}
- **Loss**: {test_metrics.get("loss", 0):.3f}
- **Perplexity**: {test_metrics.get("ppl", 0):.1f}
- **n_eval**: {test_metrics.get("n_eval", 0):,}

## Training history

| Epoch | Train loss | Valid loss | Valid PPL | Valid BLEU |
|---|---|---|---|---|
{history_rows}

## Architecture

- Encoder: **{encoder}** ({"frozen XLM-RoBERTa-base" if encoder == "xlmr" else "from-scratch BiLSTM"})
- Decoder: LSTM with Bahdanau attention
- Embed dim: {embed_dim}, hidden dim: {hidden_dim}
- Encoder layers: {enc_layers}, decoder layers: {dec_layers}
- Tokenizer: SentencePiece, 32k shared bilingual vocab (ships as `spm.model`)

## Training config

- Init from: `{init_from}`
- Epochs: {epochs}, batch size: {batch_size}, lr: {lr}
- Optimizer: Adam, label smoothing: {args.get("label_smoothing", 0.1)}
- Max src/tgt length: {args.get("max_src_len", 128)}/{args.get("max_tgt_len", 128)}

## Files

- `best.pt` — model weights (`state_dict` saved as `model_state` inside the checkpoint dict)
- `spm.model` — SentencePiece tokenizer (32k shared bilingual vocab)
- `args.json` — full training config
- `history.json` — per-epoch validation metrics
- `test_metrics.json` — final held-out test metrics

## Inference

Clone the source repo (https://github.com/ada-flo/nlp-hack), then:

```python
import torch, sentencepiece as spm
from huggingface_hub import hf_hub_download
from src.model.lstm_seq2seq import Seq2Seq

ckpt_path = hf_hub_download("{repo_id}", "best.pt")
sp_path = hf_hub_download("{repo_id}", "spm.model")

sp = spm.SentencePieceProcessor()
sp.Load(sp_path)
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

model = Seq2Seq(
    vocab_size=sp.get_piece_size(),
    embed_dim={embed_dim}, hidden_dim={hidden_dim},
    enc_layers={enc_layers}, dec_layers={dec_layers},
    dropout=0.0, encoder_type="{encoder}",
).cuda().eval()
model.load_state_dict(ckpt["model_state"], strict=False)
```

## Data

Training data is published separately as a HF dataset. See the source repo
(https://github.com/ada-flo/nlp-hack) for the preprocessing pipeline.

## License

CC BY 4.0. Underlying corpora retain their original licenses; consult the
source repo for details before commercial use.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF model repo id (e.g. ada-flo/<name>)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory with best.pt, args.json, history.json, test_metrics.json",
    )
    parser.add_argument(
        "--sp-model",
        type=Path,
        default=Path("data/processed/spm.model"),
        help="SentencePiece tokenizer to ship with the model",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set. Source secure-env/setup.sh first.")

    ckpt_dir: Path = args.checkpoint_dir
    required = ["best.pt", "args.json", "history.json", "test_metrics.json"]
    for fname in required:
        if not (ckpt_dir / fname).exists():
            raise SystemExit(f"missing {ckpt_dir / fname}")
    if not args.sp_model.exists():
        raise SystemExit(f"missing {args.sp_model}")

    train_args = _load_json(ckpt_dir / "args.json")
    history = _load_json(ckpt_dir / "history.json")
    test_metrics = _load_json(ckpt_dir / "test_metrics.json")

    print(f"[push] creating repo {args.repo_id} (private={args.private})")
    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")

    api = HfApi()
    uploads = [
        (ckpt_dir / "best.pt", "best.pt"),
        (args.sp_model, "spm.model"),
        (ckpt_dir / "args.json", "args.json"),
        (ckpt_dir / "history.json", "history.json"),
        (ckpt_dir / "test_metrics.json", "test_metrics.json"),
    ]
    for local, remote in uploads:
        print(f"[push] uploading {local} -> {remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
        )

    if not args.no_card:
        card = _render_card(
            repo_id=args.repo_id,
            args=train_args,
            history=history,
            test_metrics=test_metrics,
        )
        print("[push] uploading README.md (model card)")
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

    print(f"[push] done — https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
