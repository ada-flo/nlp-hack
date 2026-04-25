from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SEP = "<SEP>"
SOS = "<SOS>"
EOS = "<EOS>"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text while preserving English and Korean content."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def stable_id(*parts: str, prefix: str = "record") -> str:
    raw = "||".join(parts).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_seq2seq_record(
    *,
    lang: str,
    source: str,
    topic: str,
    input_context: str,
    target_output: str,
    meta: Optional[Dict[str, Any]] = None,
    add_language_tag: bool = False,
) -> Optional[Dict[str, Any]]:
    topic = normalize_text(topic)
    input_context = normalize_text(input_context)
    target_output = normalize_text(target_output)

    if not topic or not input_context or not target_output:
        return None
    if input_context == target_output:
        return None

    lang_tag = f"<{lang.upper()}> " if add_language_tag else ""
    encoder_input = f"{lang_tag}{topic} {SEP} {input_context}"
    decoder_input = f"{SOS} {target_output}"
    decoder_target = f"{target_output} {EOS}"

    return {
        "id": stable_id(lang, source, topic, input_context, target_output, prefix=source),
        "lang": lang,
        "source": source,
        "topic": topic,
        "input_context": input_context,
        "target_output": target_output,
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "meta": meta or {},
    }


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
