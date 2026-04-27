"""Train a SentencePiece BPE tokenizer on the train split.

Single bilingual model handles EN + KO with shared vocab. Special tokens
match gyehun's plan: <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3, <SEP>=4.

Outputs:
    data/processed/spm.model     # binary SentencePiece model
    data/processed/spm.vocab     # human-readable vocab listing
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm

DEFAULT_TRAIN = Path("data/processed/train.jsonl")
DEFAULT_OUT_PREFIX = Path("data/processed/spm")


def _iter_text(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Train SP on the same string the encoder will see + the target.
            yield r["encoder_input"]
            yield r["decoder_target"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # SP wants a single training file. Stream into a temp file.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
        for line in _iter_text(args.train):
            tmp.write(line.replace("\n", " ") + "\n")
        tmp_path = tmp.name

    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=str(args.out_prefix),
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<PAD>",
            unk_piece="<UNK>",
            bos_piece="<SOS>",
            eos_piece="<EOS>",
            user_defined_symbols=["<SEP>"],
            character_coverage=0.9995,  # cover Hangul, CJK, Latin
            normalization_rule_name="nmt_nfkc",
            split_digits=True,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
        )
        print(f"[vocab] wrote {args.out_prefix}.model and {args.out_prefix}.vocab")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
