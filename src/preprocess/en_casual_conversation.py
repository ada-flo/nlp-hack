"""SohamGhadge/casual-conversation → seq2seq pairs.

Single-turn casual question/answer pairs. Uniform topic per the LSTM
control-code design.
"""

from __future__ import annotations

from typing import List

from datasets import load_dataset
from tqdm import tqdm

from ._utils import passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

DATASET = "SohamGhadge/casual-conversation"
SOURCE = "casual_conversation"
TOPIC = "casual conversation"
MAX_RECORDS = 1000


def build_records(max_records: int = MAX_RECORDS) -> List[dict]:
    dataset = load_dataset(DATASET, split="train")
    records: List[dict] = []

    for row in tqdm(dataset, desc=SOURCE):
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        # Casual chat is mostly short greeting-style exchanges — that's the point.
        if not (
            passes_basic_filters(question, min_chars=4, reject_greetings=False)
            and passes_basic_filters(answer, min_chars=4, reject_greetings=False)
        ):
            continue
        record = make_seq2seq_record(
            lang="en",
            source=SOURCE,
            topic=TOPIC,
            input_context=question,
            target_output=answer,
            meta={"is_synthetic": False},
        )
        if record:
            records.append(record)
        if len(records) >= max_records:
            break

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/en_casual_conversation.jsonl")
