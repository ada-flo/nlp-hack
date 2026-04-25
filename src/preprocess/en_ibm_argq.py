from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .common import make_seq2seq_record, write_jsonl


def stance_name(value: int) -> str:
    if value == 1:
        return "pro"
    if value == -1:
        return "con"
    return "neutral"


def build_records() -> List[dict]:
    dataset = load_dataset(
        "ibm-research/argument_quality_ranking_30k",
        "argument_quality_ranking",
    )

    rows = []
    for split_name, split in dataset.items():
        for row in split:
            rows.append({**row, "split": split_name})

    grouped: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stance = int(row["stance_WA"])
        if stance not in {1, -1}:
            continue
        grouped[row["topic"]][stance].append(row)

    records = []
    for topic, by_stance in tqdm(grouped.items()):
        pros = sorted(by_stance[1], key=lambda x: float(x.get("WA", 0.0)), reverse=True)
        cons = sorted(by_stance[-1], key=lambda x: float(x.get("WA", 0.0)), reverse=True)

        for pro, con in zip(pros, cons):
            for input_row, target_row in [(pro, con), (con, pro)]:
                record = make_seq2seq_record(
                    lang="en",
                    source="ibm_argq_30k",
                    topic=topic,
                    input_context=input_row["argument"],
                    target_output=target_row["argument"],
                    meta={
                        "input_stance": stance_name(int(input_row["stance_WA"])),
                        "target_stance": stance_name(int(target_row["stance_WA"])),
                        "is_synthetic": False,
                        "source_record_ids": [],
                        "quality_input_WA": float(input_row.get("WA", 0.0)),
                        "quality_target_WA": float(target_row.get("WA", 0.0)),
                    },
                )
                if record:
                    records.append(record)

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/en_ibm_argq.jsonl")
