"""Merge per-source interim JSONL files into final train/valid/test splits.

Per gyehun's plan §"Step 7. Split by topic" — split by unique topic, not by
row, so the same topic does not leak across train/valid/test.

Reads all `data/interim/*.jsonl`, dedupes on (input_context, target_output),
groups records by `lang`, applies optional per-language sampling caps, and
writes `data/processed/{train,valid,test}.jsonl` with an 80/10/10 ratio.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from ._utils import dedupe_records

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[merge] skipping malformed line in {path}: {exc!r}")


def _write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_all(interim_dir: Path) -> List[dict]:
    if not interim_dir.exists():
        return []
    records: List[dict] = []
    for path in sorted(interim_dir.glob("*.jsonl")):
        loaded = list(_read_jsonl(path))
        records.extend(loaded)
        print(f"[merge] loaded {len(loaded):>6} from {path.name}")
    return records


def _cap_per_language(
    records: List[dict],
    caps: Dict[str, int | None],
    seed: int,
) -> List[dict]:
    """Sample down to `caps[lang]` per language, preserving topic balance."""
    by_lang: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_lang[r.get("lang", "unk")].append(r)

    rng = random.Random(seed)
    out: List[dict] = []
    for lang, items in by_lang.items():
        cap = caps.get(lang)
        if cap is None or len(items) <= cap:
            out.extend(items)
            print(f"[merge] {lang}: {len(items)} records (no cap)")
            continue
        rng.shuffle(items)
        out.extend(items[:cap])
        print(f"[merge] {lang}: capped to {cap} of {len(items)}")
    return out


# Sources that use a single uniform placeholder topic across all records.
# These should be split row-wise — the topic field is a control code, not
# a debate motion, so there's no leakage risk and we want even spread.
ROW_WISE_SOURCES = frozenset(
    {
        "casual_conversation",
        "isotonic_conversation",
        "klue_nli",
    }
)


def _split_key(record: dict) -> tuple:
    """Group records that should never be separated.

    Debate records share a topic = the debate motion, so all records on one
    motion go into a single split (prevents leakage). Sources with uniform
    placeholder topics (casual chat, NLI contradiction pairs) split row-wise
    via the record id to spread them across train/valid/test.
    """
    lang = record.get("lang", "unk")
    source = record.get("source", "")
    if source in ROW_WISE_SOURCES:
        return (lang, f"_rowwise_{source}", record.get("id", ""))
    return (lang, record.get("topic", ""))


def _topic_level_split(
    records: List[dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[List[dict], List[dict], List[dict]]:
    """Split by (lang, topic) so the same topic does not leak across splits.

    Casual-chat records (uniform placeholder topic) are split row-wise so
    every split gets fluency examples.
    """
    by_topic: Dict[tuple, List[dict]] = defaultdict(list)
    for r in records:
        by_topic[_split_key(r)].append(r)

    keys = list(by_topic.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_keys = set(keys[:n_train])
    valid_keys = set(keys[n_train : n_train + n_valid])
    # everything else → test

    train, valid, test = [], [], []
    for key, items in by_topic.items():
        if key in train_keys:
            train.extend(items)
        elif key in valid_keys:
            valid.extend(items)
        else:
            test.extend(items)
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--cap-en", type=int, default=10_000)
    parser.add_argument("--cap-ko", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    records = _load_all(args.interim_dir)
    print(f"[merge] {len(records)} records before dedup")
    records = list(dedupe_records(records))
    print(f"[merge] {len(records)} after dedup")

    records = _cap_per_language(
        records, {"en": args.cap_en, "ko": args.cap_ko}, seed=args.seed
    )
    print(f"[merge] {len(records)} after caps")

    train, valid, test = _topic_level_split(records, DEFAULT_RATIOS, seed=args.seed)
    n_train = _write_jsonl(train, args.out_dir / "train.jsonl")
    n_valid = _write_jsonl(valid, args.out_dir / "valid.jsonl")
    n_test = _write_jsonl(test, args.out_dir / "test.jsonl")

    print(f"[merge] wrote train={n_train} valid={n_valid} test={n_test}")


if __name__ == "__main__":
    main()
