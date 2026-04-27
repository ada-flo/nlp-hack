"""KLUE-NLI → seq2seq pairs (Korean rebuttal-shaped data, no LLM needed).

Per gyehun's revised plan §"NLI contradiction conversion": NLI contradiction
pairs serve as weak rebuttal seeds. We use the direct conversion (premise →
hypothesis where label == contradiction). The pair is not a full debate-style
rebuttal, but it teaches the model "given X, output Y where Y disagrees".

This adapter is a fast path to Korean training data without vLLM.

KLUE-NLI label encoding:
    0: entailment    (skip)
    1: neutral       (skip)
    2: contradiction (use)
"""

from __future__ import annotations

from typing import List

from datasets import load_dataset
from tqdm import tqdm

from ._utils import passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

DATASET_NAME = "klue"
DATASET_CONFIG = "nli"
SOURCE = "klue_nli"
LANG = "ko"
TOPIC = "두 문장의 관점 차이"  # "viewpoint difference between two sentences"
MAX_RECORDS = 5000
CONTRADICTION_LABEL = 2


def build_records(max_records: int = MAX_RECORDS) -> List[dict]:
    records: List[dict] = []
    for split in ("train", "validation"):
        try:
            dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
        except Exception as exc:  # noqa: BLE001
            print(f"[{SOURCE}] failed to load split {split}: {exc!r}")
            continue
        for row in tqdm(dataset, desc=f"{SOURCE}/{split}"):
            if int(row.get("label", -1)) != CONTRADICTION_LABEL:
                continue
            premise = (row.get("premise") or "").strip()
            hypothesis = (row.get("hypothesis") or "").strip()
            # Korean text is shorter per char; relax min slightly.
            if not (
                passes_basic_filters(premise, min_chars=10)
                and passes_basic_filters(hypothesis, min_chars=10)
            ):
                continue
            # Both directions to double the rebuttal-shaped signal.
            for input_text, target_text in [(premise, hypothesis), (hypothesis, premise)]:
                record = make_seq2seq_record(
                    lang=LANG,
                    source=SOURCE,
                    topic=TOPIC,
                    input_context=input_text,
                    target_output=target_text,
                    meta={
                        "is_synthetic": False,
                        "conversion": "nli_contradiction_pair",
                        "split": split,
                    },
                )
                if record:
                    records.append(record)
                    if len(records) >= max_records:
                        return records
    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/ko_klue_nli.jsonl")
