"""AI Hub Persona Dialogue (dataSetSn=71302) → seq2seq pairs.

Best-fit auxiliary corpus: real persona profiles + topic + alternating
multi-persona turns. The persona profiles are kept in record meta so a future
adapter can prepend them to the encoder input for `<PERSONA>` conditioning.

Schema:

    {
      "info": {
        "topic": "학교/학업",
        "personas": [
          {"persona_id": 642,  "persona": [{"profile": "..."}, ...]},
          {"persona_id": 1232, "persona": [{"profile": "..."}, ...]}
        ]
      },
      "utterances": [
        {"persona_id": 1232, "text": "..."},
        {"persona_id": 882,  "text": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from ._utils import (
    adjacent_pairs,
    merge_speaker_turns,
    passes_basic_filters,
)
from .common import make_seq2seq_record, write_jsonl

SOURCE = "aihub_persona_dialog"
LANG = "ko"
RAW_DIR = Path("data/raw_manual/aihub/persona_dialog_71302")
OUTPUT = Path("data/interim/ko_aihub_persona_dialog.jsonl")


def _walk(raw_dir: Path) -> Iterator[dict]:
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[{SOURCE}] could not read {path}: {exc!r}")


def _persona_profiles(personas: list[dict]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for p in personas or []:
        pid = p.get("persona_id")
        if pid is None:
            continue
        out[int(pid)] = [
            (item.get("profile") or "").strip()
            for item in (p.get("persona") or [])
            if (item.get("profile") or "").strip()
        ]
    return out


def build_records(raw_dir: Path = RAW_DIR) -> List[dict]:
    if not raw_dir.exists():
        print(f"[{SOURCE}] manual data not found at {raw_dir} — skipping.")
        return []

    records: List[dict] = []
    for data in _walk(raw_dir):
        info = data.get("info") or {}
        topic = info.get("topic") or "페르소나 대화"
        profiles = _persona_profiles(info.get("personas") or [])

        utt_pairs: list[tuple[str, str]] = []
        utt_speakers: list[int] = []
        for u in data.get("utterances") or []:
            text = (u.get("text") or "").strip()
            if not text:
                continue
            pid = u.get("persona_id")
            utt_pairs.append((str(pid), text))
            utt_speakers.append(int(pid) if pid is not None else -1)

        turns = merge_speaker_turns(utt_pairs)
        # Recompute aligned speaker sequence for merged turns.
        merged_speakers: list[int] = []
        last = None
        for spk in utt_speakers:
            if spk != last:
                merged_speakers.append(spk)
                last = spk

        for i, (prev, nxt) in enumerate(adjacent_pairs(turns)):
            if not (passes_basic_filters(prev) and passes_basic_filters(nxt)):
                continue
            target_pid = merged_speakers[i + 1] if i + 1 < len(merged_speakers) else None
            target_profiles = profiles.get(target_pid, []) if target_pid is not None else []
            record = make_seq2seq_record(
                lang=LANG,
                source=SOURCE,
                topic=str(topic),
                input_context=prev,
                target_output=nxt,
                meta={
                    "is_synthetic": False,
                    "target_persona": target_profiles,
                },
            )
            if record:
                records.append(record)
    return records


if __name__ == "__main__":
    write_jsonl(build_records(), OUTPUT)
