"""AI Hub Topic-wise Everyday Text Dialogue → seq2seq pairs.

https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=543

Requires manual download (AI Hub account). Drop the unpacked JSON files at:

    data/raw_manual/aihub/topic_dialogue/

Per gyehun's plan: use everyday-conversation turns as auxiliary fluency
data. Speech-act labels (opinion / disagreement / suggestion / explanation)
make the best filter when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from ._utils import adjacent_pairs, passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

SOURCE = "aihub_topic_dialogue"
LANG = "ko"
RAW_DIR = Path("data/raw_manual/aihub/topic_dialogue")
OUTPUT = Path("data/interim/ko_aihub_topic_dialogue.jsonl")

OPINIONATED_SPEECH_ACTS = ("opinion", "disagreement", "suggestion", "explanation")


def _walk_dialogues(raw_dir: Path) -> Iterator[dict]:
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[{SOURCE}] could not read {path}: {exc!r}")
            continue
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield item


def _is_opinionated(utterance: dict) -> bool:
    speech_act = (utterance.get("speech_act") or utterance.get("act") or "").lower()
    if not speech_act:
        return True  # no labels → keep, downstream filters will catch noise
    return any(tag in speech_act for tag in OPINIONATED_SPEECH_ACTS)


def _extract(item: dict) -> tuple[str, list[str]]:
    header = item.get("header") or {}
    body = item.get("body") or item
    topic = (
        item.get("topic")
        or header.get("topic")
        or header.get("subject")
        or "일상 대화"
    )
    raw_turns = body.get("dialogue") or item.get("dialogue") or item.get("utterances") or []
    turns: list[str] = []
    for u in raw_turns:
        if isinstance(u, dict):
            if not _is_opinionated(u):
                continue
            text = u.get("utterance") or u.get("text") or u.get("content")
        else:
            text = str(u)
        if text:
            turns.append(text.strip())
    return str(topic), turns


def build_records(raw_dir: Path = RAW_DIR) -> List[dict]:
    if not raw_dir.exists():
        print(
            f"[{SOURCE}] manual data not found at {raw_dir} — skipping. "
            f"See https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=543."
        )
        return []

    records: List[dict] = []
    for item in _walk_dialogues(raw_dir):
        topic, turns = _extract(item)
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
