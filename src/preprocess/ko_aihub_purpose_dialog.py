"""AI Hub Purpose-Specific Dialogue (dataSetSn=544) → seq2seq pairs.

Goal-oriented call-center / counter dialogue across shopping / civil-complaint /
education / tourism domains.

Schema:

    {
      "info": [{
        "annotations": {
          "subject": "제품/사용문의",
          "lines": [
            {"text": "A.반갑습니다 ...", "speaker": {"id": "A"}, "speechAct": "인사하기"},
            {"text": "B.문의 좀 ...",   "speaker": {"id": "B"}, "speechAct": "질문하기"},
            ...
          ]
        }
      }]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List

from ._utils import (
    adjacent_pairs,
    merge_speaker_turns,
    passes_basic_filters,
)
from .common import make_seq2seq_record, write_jsonl

SOURCE = "aihub_purpose_dialog"
LANG = "ko"
RAW_DIR = Path("data/raw_manual/aihub/purpose_dialog_544")
OUTPUT = Path("data/interim/ko_aihub_purpose_dialog.jsonl")

# Lines are prefixed with "A." / "B." in the raw text — strip that prefix.
PREFIX_RE = re.compile(r"^\s*[A-Z]\.\s*")


def _walk(raw_dir: Path) -> Iterator[dict]:
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[{SOURCE}] could not read {path}: {exc!r}")


def _extract(data: dict) -> tuple[str, list[tuple[str, str]]]:
    info = data.get("info") or []
    if not info:
        return "", []
    ann = (info[0] or {}).get("annotations") or {}
    topic = ann.get("subject") or "목적 대화"
    lines = ann.get("lines") or []
    pairs: list[tuple[str, str]] = []
    for ln in lines:
        text = PREFIX_RE.sub("", (ln.get("text") or "").strip())
        spk = ((ln.get("speaker") or {}).get("id")) or ""
        if text:
            pairs.append((spk, text))
    return str(topic), pairs


def build_records(raw_dir: Path = RAW_DIR) -> List[dict]:
    if not raw_dir.exists():
        print(f"[{SOURCE}] manual data not found at {raw_dir} — skipping.")
        return []

    records: List[dict] = []
    for data in _walk(raw_dir):
        topic, pairs = _extract(data)
        turns = merge_speaker_turns(pairs)
        for prev, nxt in adjacent_pairs(turns):
            if not (passes_basic_filters(prev) and passes_basic_filters(nxt)):
                continue
            record = make_seq2seq_record(
                lang=LANG,
                source=SOURCE,
                topic=topic,
                input_context=prev,
                target_output=nxt,
                meta={"is_synthetic": False},
            )
            if record:
                records.append(record)
    return records


if __name__ == "__main__":
    write_jsonl(build_records(), OUTPUT)
