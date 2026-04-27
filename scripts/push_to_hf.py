"""Push data/processed/{train,valid,test}.jsonl to a Hugging Face dataset repo.

Usage:
    HF_TOKEN must be set with WRITE scope (https://huggingface.co/settings/tokens).

    uv run python scripts/push_to_hf.py --repo-id <user>/<dataset-name>
    uv run python scripts/push_to_hf.py --repo-id <user>/<dataset-name> --private

The script also writes a dataset card (README.md inside the HF repo) with
schema, source breakdown, sample counts, and the synthesis prompt version.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict


PROCESSED_DIR = Path("data/processed")


def _flatten(record: dict) -> dict:
    """Promote universally-useful meta fields to top-level so they're
    filterable in the HF dataset viewer. Source-specific extras stay in `meta`.
    """
    meta = dict(record.get("meta") or {})
    record["is_synthetic"] = bool(meta.pop("is_synthetic", False))
    record["input_stance"] = meta.pop("input_stance", None)
    record["target_stance"] = meta.pop("target_stance", None)
    record["meta"] = meta
    return record


def _load_split(name: str) -> Dataset:
    path = PROCESSED_DIR / f"{name}.jsonl"
    ds = Dataset.from_json(str(path))
    return ds.map(_flatten)


def _split_stats(ds: Dataset) -> dict:
    by_lang = Counter()
    by_source = Counter()
    for row in ds:
        by_lang[row.get("lang", "unk")] += 1
        by_source[row.get("source", "unk")] += 1
    return {"total": len(ds), "by_lang": dict(by_lang), "by_source": dict(by_source)}


def _render_card(repo_id: str, splits: dict[str, dict]) -> str:
    train, valid, test = splits["train"], splits["validation"], splits["test"]

    def table(stats: dict) -> str:
        rows = [f"| {src} | {n:,} |" for src, n in sorted(stats["by_source"].items(), key=lambda x: -x[1])]
        return "\n".join(["| Source | Records |", "|---|---|", *rows])

    return f"""\
---
language:
  - en
  - ko
license: cc-by-4.0
task_categories:
  - text2text-generation
pretty_name: Debate-Themed Dialogue Generation Dataset
size_categories:
  - 10K<n<100K
---

# {repo_id}

Bilingual (English + Korean) training data for an LSTM-based seq2seq debate
chatbot. Each record is a `(topic, input_context, target_output)` triple
plus precomputed `encoder_input` / `decoder_input` / `decoder_target` ready
for seq2seq training.

## Schema

```json
{{
  "id": "ibm_argq_30k_8b4b12caccad",
  "lang": "en",
  "source": "ibm_argq_30k",
  "is_synthetic": false,
  "input_stance": "pro",
  "target_stance": "con",
  "topic": "We should abandon marriage",
  "input_context": "abandoning marriage allows for people to grow as themselves...",
  "target_output": "committment and stability are important in the lives of children...",
  "encoder_input": "We should abandon marriage <SEP> abandoning marriage allows...",
  "decoder_input": "<SOS> committment and stability are important...",
  "decoder_target": "committment and stability are important... <EOS>",
  "meta": {{ "source_record_ids": [], "quality_input_WA": 1.0, "...": "..." }}
}}
```

Top-level fields filterable in the HF dataset viewer:

| Field | Values |
|---|---|
| `lang` | `en`, `ko` |
| `source` | `ibm_argq_30k`, `mc_conversation`, `isotonic_conversation`, `casual_conversation`, `ko_debate_synth`, `korean_petitions` |
| `is_synthetic` | `true`, `false` |
| `input_stance` | `pro`, `con`, `petition_position`, `supportive`, `oppositional`, … |
| `target_stance` | `pro`, `con`, `opposition`, … |

## Splits

| Split | Records | EN | KO |
|---|---|---|---|
| train | {train["total"]:,} | {train["by_lang"].get("en", 0):,} | {train["by_lang"].get("ko", 0):,} |
| validation | {valid["total"]:,} | {valid["by_lang"].get("en", 0):,} | {valid["by_lang"].get("ko", 0):,} |
| test | {test["total"]:,} | {test["by_lang"].get("en", 0):,} | {test["by_lang"].get("ko", 0):,} |

Splits are **topic-level** for debate-shaped sources (motion-grouped records
all land in one split — no leakage). Casual chat and topic-seeded synth use
row-wise split because they share placeholder topics.

## Sources (train split)

{table(train)}

## Source descriptions

- **ibm_argq_30k** — [IBM Argument Quality Ranking 30K](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k). Real human pro/con stance pairs over ~70 motions.
- **mc_conversation** — [mc-ai/conversation_dataset](https://huggingface.co/datasets/mc-ai/conversation_dataset), filtered to `corpus_id=persuasionforgood`. Real persuasion-themed multi-turn dialogue (Persuasion-for-Good corpus).
- **isotonic_conversation** — [Isotonic/human_assistant_conversation](https://huggingface.co/datasets/Isotonic/human_assistant_conversation), filtered to single-turn rows without dialog markers or code-task content.
- **casual_conversation** — [SohamGhadge/casual-conversation](https://huggingface.co/datasets/SohamGhadge/casual-conversation). Casual greeting-style exchanges for conversational fluency.
- **ko_debate_synth** — Topic-seeded debate-pair synthesis (Korean). 98 curated debate motions × 30 LLM-generated PRO/CON pairs each. Uses Qwen3-235B-A22B-Instruct via vLLM at temperature 0.9. Both directions per pair.
- **korean_petitions** — Korean Petitions corpus (청와대 국민청원 2017–2019, via Korpora). Petition title = motion, body (truncated to 280 chars) = `input_context`, vLLM-synthesized counter-argument = `target_output`.

## Synthetic data

Records with `meta.is_synthetic=true` were generated by Qwen3-235B-A22B-Instruct
served via vLLM. Synthesis prompt versions are recorded in
`meta.synthesis_prompt_version`.

| Prompt version | Used by |
|---|---|
| v1 (counterargument) | korean_petitions |
| v1 (debate_pair) | ko_debate_synth |

Prompts: see `src/synth/prompts.py` in the source repository.

## License

CC BY 4.0. Source corpora retain their original licenses; consult each source
link above for redistribution terms before commercial use.

## Repository

Generated by https://github.com/ada-flo/nlp-hack — see that repo for the
full preprocessing pipeline, source adapters, and synth client code.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo id (e.g. username/nlp-hack-debate)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    parser.add_argument(
        "--no-card",
        action="store_true",
        help="Skip writing the dataset card README.md",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set. Source setup.sh or export HF_TOKEN.")

    print(f"[push] loading splits from {PROCESSED_DIR}/")
    ds = DatasetDict(
        {
            "train": _load_split("train"),
            "validation": _load_split("valid"),
            "test": _load_split("test"),
        }
    )
    splits = {name: _split_stats(split) for name, split in ds.items()}
    for name, s in splits.items():
        print(f"  {name}: {s['total']} records, by_lang={s['by_lang']}")

    print(f"[push] uploading splits to {args.repo_id} (private={args.private})")
    ds.push_to_hub(args.repo_id, private=args.private)

    if not args.no_card:
        print(f"[push] writing dataset card to {args.repo_id}")
        from huggingface_hub import HfApi

        card = _render_card(args.repo_id, splits)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    print(f"[push] done — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
