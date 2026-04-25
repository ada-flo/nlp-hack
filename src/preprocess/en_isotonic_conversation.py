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


def _clean_prompt(text: str) -> str:
    # Trailing "Output:" marker is part of the dataset's prompt format.
    return (text or "").replace("Output:", "").strip()


def build_records(max_records: int = MAX_RECORDS) -> List[dict]:
    dataset = load_dataset(DATASET, split="train")
    records: List[dict] = []

    for row in tqdm(dataset, desc=SOURCE):
        prompt = _clean_prompt(row.get("prompt"))
        response = (row.get("response") or "").strip()
        if not (passes_basic_filters(prompt) and passes_basic_filters(response)):
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
