"""Shared filters and helpers for source adapters.

Per gyehun's plan §"Step 5. Filter and deduplicate" — these defaults can be
overridden per source where appropriate.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

# --- length thresholds ---------------------------------------------------

MIN_CHARS = 20
MAX_CHARS = 600

# --- noise patterns ------------------------------------------------------

# Greetings / pure agreement / moderation — too generic to use as input or target.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|"
    r"안녕|고마워|감사합니다|네|아니요|좋아|알겠어|예)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Very short content that is almost certainly low-signal.
def is_too_short(text: str, min_chars: int = MIN_CHARS) -> bool:
    return len(text.strip()) < min_chars


def is_too_long(text: str, max_chars: int = MAX_CHARS) -> bool:
    return len(text.strip()) > max_chars


def is_greeting_only(text: str) -> bool:
    return bool(_GREETING_RE.match(text or ""))


def passes_basic_filters(
    text: str,
    *,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    reject_greetings: bool = True,
) -> bool:
    """Reject empty / too-short / too-long / pure-greeting strings.

    For casual-conversation sources, pass `reject_greetings=False` and a lower
    `min_chars` to keep short greeting-style exchanges that are the whole point
    of those datasets.
    """
    if not text or not text.strip():
        return False
    if is_too_short(text, min_chars) or is_too_long(text, max_chars):
        return False
    if reject_greetings and is_greeting_only(text):
        return False
    return True


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate at sentence boundary if possible, otherwise hard cut."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind("。"))
    if last_punct > max_chars * 0.6:
        return cut[: last_punct + 1].strip()
    return cut.strip()


# --- dialogue extraction -------------------------------------------------

def adjacent_pairs(turns: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (prev, next) for every adjacent pair of turns."""
    for i in range(len(turns) - 1):
        yield turns[i], turns[i + 1]


# --- dedup ---------------------------------------------------------------

def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def dedupe_records(records: Iterable[dict]) -> Iterator[dict]:
    """Drop records whose (input_context, target_output) pair was already seen."""
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (
            normalize_for_dedup(record.get("input_context", "")),
            normalize_for_dedup(record.get("target_output", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        yield record
