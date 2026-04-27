"""Pull the processed dataset from Hugging Face and write JSONL splits.

Usage on the GPU server:

    uv run python scripts/pull_dataset_from_hf.py
    # → data/processed/train.jsonl, valid.jsonl, test.jsonl

This skips the entire preprocess pipeline — the dataset on HF already
contains the merged, split, deduped records ready for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


# HF uses "validation"; our pipeline / training code expects "valid".
SPLIT_RENAME = {"train": "train", "validation": "valid", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="ada-flo/nlp-hack-debate",
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write the JSONL splits",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision/tag/commit to pin",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pull] downloading {args.repo_id}" + (f"@{args.revision}" if args.revision else ""))
    ds = load_dataset(args.repo_id, revision=args.revision)

    for hf_split, local_split in SPLIT_RENAME.items():
        out = args.out_dir / f"{local_split}.jsonl"
        rows = ds[hf_split]
        with out.open("w", encoding="utf-8") as f:
            for record in rows:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[pull] wrote {len(rows):>6} records → {out}")

    print("[pull] done — ready for `uv run python -m src.train`")


if __name__ == "__main__":
    main()
