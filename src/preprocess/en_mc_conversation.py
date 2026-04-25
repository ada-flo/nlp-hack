"""mc-ai/conversation_dataset → seq2seq pairs.

This is the Persuasion-for-Good corpus repackaged: multi-turn telephone
dialogues with real per-conversation topics ("Discussion on charitable
donations…"). Soft skills include "debate" and "persuasion", so this is
debate-adjacent, not pure casual chat — we keep the real `topic`.
"""

from __future__ import annotations

from typing import List

from datasets import load_dataset
from tqdm import tqdm

from ._utils import adjacent_pairs, passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

DATASET = "mc-ai/conversation_dataset"
SOURCE = "mc_conversation"
MAX_RECORDS = 1500


def _extract_turns(raw_convo: dict) -> List[str]:
    messages = (raw_convo or {}).get("messages") or []
    return [m.get("content", "") for m in messages if isinstance(m, dict)]


def build_records(max_records: int = MAX_RECORDS) -> List[dict]:
    dataset = load_dataset(DATASET, split="train")
    records: List[dict] = []

    for row in tqdm(dataset, desc=SOURCE):
        topic = row.get("topic") or "casual conversation"
        turns = _extract_turns(row.get("raw_convo"))
        for prev, nxt in adjacent_pairs(turns):
            if not (passes_basic_filters(prev) and passes_basic_filters(nxt)):
                continue
            record = make_seq2seq_record(
                lang="en",
                source=SOURCE,
                topic=topic,
                input_context=prev,
                target_output=nxt,
                meta={
                    "is_synthetic": False,
                    "conversation_id": row.get("conversation_id"),
                    "corpus_id": row.get("corpus_id"),
                },
            )
            if record:
                records.append(record)
            if len(records) >= max_records:
                return records

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/en_mc_conversation.jsonl")
