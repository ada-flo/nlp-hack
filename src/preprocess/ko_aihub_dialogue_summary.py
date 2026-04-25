"""AI Hub Korean Dialogue Summarization → seq2seq pairs.

https://aihub.or.kr/aidata/30714

Requires manual download (AI Hub account). Drop the unpacked JSON files at:

    data/raw_manual/aihub/dialogue_summary/

Per gyehun's plan: filter for debate-like / discussion / social-topic
dialogues, then extract adjacent argumentative turns. The summary or topic
field is used as `topic`.

The AI Hub format varies by release; this adapter walks any *.json under
the manual dir and tolerates two common shapes:
    1. {"data": [{"header": {...}, "body": {...}}, ...]}
    2. flat list of dialogue dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from ._utils import adjacent_pairs, passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

SOURCE = "aihub_dialogue_summary"
LANG = "ko"
RAW_DIR = Path("data/raw_manual/aihub/dialogue_summary")
OUTPUT = Path("data/interim/ko_aihub_dialogue_summary.jsonl")

DEBATE_HINTS = ("토론", "논쟁", "사회", "정치", "discussion", "debate")


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


def _looks_debate_like(item: dict) -> bool:
    """Cheap heuristic on header/topic strings."""
    blob = json.dumps(item, ensure_ascii=False).lower()
    return any(hint.lower() in blob for hint in DEBATE_HINTS)


def _extract_turns_and_topic(item: dict) -> tuple[str, list[str]]:
    header = item.get("header") or {}
    body = item.get("body") or item
    topic = (
        item.get("topic")
        or header.get("topic")
        or header.get("subject")
        or item.get("summary")
        or body.get("summary")
        or "한국어 대화"  # generic fallback
    )
    dialogue = body.get("dialogue") or item.get("dialogue") or item.get("utterances") or []
    turns = []
    for u in dialogue:
        if isinstance(u, dict):
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
            f"See https://aihub.or.kr/aidata/30714."
        )
        return []

    records: List[dict] = []
    for item in _walk_dialogues(raw_dir):
        if not _looks_debate_like(item):
            continue
        topic, turns = _extract_turns_and_topic(item)
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
