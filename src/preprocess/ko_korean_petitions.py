"""Korean Petitions → vLLM-synthesized counter-arguments.

Per gyehun's plan §"LLM-assisted synthesis": petitions provide a topic
(petition title) and a one-sided argument (petition body). We use vLLM
to synthesize the missing rebuttal as `target_output`.

REQUIRES a running vLLM server. Set VLLM_BASE_URL / VLLM_MODEL via .env
or shell. See README "Synthetic data via vLLM".

Records are written chunk-by-chunk so a mid-run crash doesn't cost the
whole batch.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple

# Korpora uses urllib.request, which on macOS Python does not pick up the
# system root CA bundle and fails with CERTIFICATE_VERIFY_FAILED. Point at
# certifi's bundle before importing Korpora.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

from Korpora import Korpora
from tqdm import tqdm

from ..synth.client import make_async_client, resolve_model
from ..synth.counterargument import (
    SynthRequest,
    synthesize_counterargument,
    synthesize_counterarguments_async,
)
from ..synth.prompts import COUNTERARGUMENT_PROMPT_VERSION
from ._utils import truncate
from .common import make_seq2seq_record

SOURCE = "korean_petitions"
LANG = "ko"
OUTPUT = Path("data/interim/ko_korean_petitions.jsonl")

MAX_RECORDS = 10_000
CHUNK_SIZE = 200
CONCURRENCY = 16
PETITION_BODY_MAX_CHARS = 280
MIN_TITLE_CHARS = 6
MIN_BODY_CHARS = 50


def _iter_seeds(max_seeds: int) -> Iterator[Tuple[dict, str, str]]:
    """Yield (meta, topic, input_context) for filtered Korean Petitions."""
    corpus = Korpora.load("korean_petitions")
    yielded = 0
    for item in corpus.train:
        title = (item.title or "").strip()
        body = (item.text or "").strip()
        if len(title) < MIN_TITLE_CHARS or len(body) < MIN_BODY_CHARS:
            continue
        meta = {
            "is_synthetic": True,
            "synthesis_prompt_version": COUNTERARGUMENT_PROMPT_VERSION,
            "input_stance": "petition_position",
            "target_stance": "opposition",
            "category": getattr(item, "category", None),
            "num_agree": getattr(item, "num_agree", None),
            "begin": getattr(item, "begin", None),
            "end": getattr(item, "end", None),
        }
        yield meta, title, truncate(body, PETITION_BODY_MAX_CHARS)
        yielded += 1
        if yielded >= max_seeds:
            return


def _append_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _preflight() -> None:
    """One sync call to fail fast if vLLM isn't reachable."""
    test = SynthRequest(
        topic="환경 보호", input_context="플라스틱 사용을 줄여야 합니다.", language=LANG
    )
    synthesize_counterargument(test, max_tokens=64)


async def _run(max_records: int, chunk_size: int, concurrency: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")  # truncate any prior run

    client = make_async_client()
    model = resolve_model()

    seeds = list(_iter_seeds(max_records))
    print(f"[{SOURCE}] {len(seeds)} petition seeds after filtering")

    written = 0
    failed = 0
    try:
        for start in tqdm(range(0, len(seeds), chunk_size), desc=SOURCE):
            chunk = seeds[start : start + chunk_size]
            requests = [
                SynthRequest(topic=topic, input_context=ctx, language=LANG)
                for _, topic, ctx in chunk
            ]
            results = await synthesize_counterarguments_async(
                requests, client=client, model=model, concurrency=concurrency
            )

            chunk_records: List[dict] = []
            for (meta, topic, ctx), result in zip(chunk, results):
                if not result.ok:
                    failed += 1
                    continue
                record = make_seq2seq_record(
                    lang=LANG,
                    source=SOURCE,
                    topic=topic,
                    input_context=ctx,
                    target_output=result.text,
                    meta=meta,
                )
                if record:
                    chunk_records.append(record)
            _append_jsonl(chunk_records, output)
            written += len(chunk_records)
    finally:
        await client.close()

    print(f"[{SOURCE}] wrote {written} records, {failed} failed → {output}")


def build_and_write(
    *,
    max_records: int = MAX_RECORDS,
    chunk_size: int = CHUNK_SIZE,
    concurrency: int = CONCURRENCY,
    output: Path = OUTPUT,
) -> None:
    print(f"[{SOURCE}] preflight check against vLLM…")
    _preflight()
    asyncio.run(_run(max_records, chunk_size, concurrency, output))


if __name__ == "__main__":
    build_and_write()
