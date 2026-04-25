"""K-News-Stance → seq2seq pairs.

Korean news stance corpus (https://github.com/ssu-humane/K-News-Stance).
Requires a manual download form — drop the JSON at:

    data/raw_manual/k-news-stance/k-news-stance.json

Per gyehun's plan: use `issue` as topic, pair supportive vs oppositional
segments where both stances exist; otherwise the unmatched side could be
fed into vLLM synth (not implemented here — wire to ko_korean_petitions
synth path if you need that).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ._utils import passes_basic_filters
from .common import make_seq2seq_record, write_jsonl

SOURCE = "k_news_stance"
LANG = "ko"
RAW_PATH = Path("data/raw_manual/k-news-stance/k-news-stance.json")
OUTPUT = Path("data/interim/ko_k_news_stance.jsonl")


def _load_raw(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def build_records(raw_path: Path = RAW_PATH) -> List[dict]:
    if not raw_path.exists():
        print(
            f"[{SOURCE}] manual data not found at {raw_path} — skipping. "
            f"See https://github.com/ssu-humane/K-News-Stance for the download form."
        )
        return []

    rows = _load_raw(raw_path)

    # Group by issue (topic). Stance labels in this corpus are typically
    # "supportive" / "oppositional"; segments may be in `headline`, `lead`,
    # `quotation`, `conclusion`, or per-paragraph fields. Collect any text
    # field that has a stance label.
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        issue = row.get("issue") or row.get("topic")
        if not issue:
            continue
        for segment in row.get("segments") or row.get("paragraphs") or []:
            stance = (segment.get("stance") or "").lower()
            text = (segment.get("text") or "").strip()
            if not stance or not text:
                continue
            if stance.startswith("support"):
                grouped[issue]["pro"].append(text)
            elif stance.startswith("oppos"):
                grouped[issue]["con"].append(text)

    records: List[dict] = []
    for issue, by_stance in grouped.items():
        pros = by_stance.get("pro", [])
        cons = by_stance.get("con", [])
        for pro, con in zip(pros, cons):
            if not (passes_basic_filters(pro) and passes_basic_filters(con)):
                continue
            for input_text, target_text, in_stance, out_stance in [
                (pro, con, "supportive", "oppositional"),
                (con, pro, "oppositional", "supportive"),
            ]:
                record = make_seq2seq_record(
                    lang=LANG,
                    source=SOURCE,
                    topic=issue,
                    input_context=input_text,
                    target_output=target_text,
                    meta={
                        "is_synthetic": False,
                        "input_stance": in_stance,
                        "target_stance": out_stance,
                    },
                )
                if record:
                    records.append(record)
    return records


if __name__ == "__main__":
    write_jsonl(build_records(), OUTPUT)
