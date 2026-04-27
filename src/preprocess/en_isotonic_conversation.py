"""Isotonic/human_assistant_conversation → seq2seq pairs.

Single-turn human↔assistant Q&A. Use as conversational fluency data with
a uniform `topic = "casual conversation"` per the LSTM control-code design.
"""

from __future__ import annotations

from typing import List

from datasets import load_dataset
from tqdm import tqdm

from ._utils import passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

DATASET = "Isotonic/human_assistant_conversation"
SOURCE = "isotonic_conversation"
TOPIC = "casual conversation"
MAX_RECORDS = 1500

# Multi-turn rows have these markers baked into the `prompt` column. We reject
# them rather than try to clean — splitting reliably is hard, and the dataset
# has plenty of clean single-turn rows.
DIALOG_MARKERS = ("Human:", "Assistant:")
CODE_HINTS = ("def ", "import ", "console.log", "function(", "</html", "```")


def _has_marker(text: str) -> bool:
    return any(m in text for m in DIALOG_MARKERS)


def _looks_like_code(text: str) -> bool:
    return any(h in text for h in CODE_HINTS)


def _clean_prompt(text: str) -> str:
    # Trailing "Output:" marker is part of the dataset's prompt format.
    return (text or "").replace("Output:", "").strip()


def build_records(max_records: int = MAX_RECORDS) -> List[dict]:
    # Stream to bypass a schema mismatch between train and validation parquet
    # files that breaks non-streaming load (validation has `source_text`,
    # train has `text`).
    dataset = load_dataset(DATASET, split="train", streaming=True)
    records: List[dict] = []

    for row in tqdm(dataset, desc=SOURCE, total=max_records):
        prompt = _clean_prompt(row.get("prompt"))
        response = (row.get("response") or "").strip()
        if not (passes_basic_filters(prompt) and passes_basic_filters(response)):
            continue
        if _has_marker(prompt) or _has_marker(response):
            continue
        if _looks_like_code(prompt) or _looks_like_code(response):
            continue
        record = make_seq2seq_record(
            lang="en",
            source=SOURCE,
            topic=TOPIC,
            input_context=prompt,
            target_output=response,
            meta={"is_synthetic": False},
        )
        if record:
            records.append(record)
        if len(records) >= max_records:
            break

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/en_isotonic_conversation.jsonl")
