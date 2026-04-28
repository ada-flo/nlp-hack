"""Topic-seeded Korean debate-pair synthesis.

Reads a curated list of Korean debate motions from data/seeds/ko_debate_motions.txt
and asks the LLM to produce a (PRO, CON) argument exchange for each motion.
Each successful exchange is split into two training records (pro→con and
con→pro) so the model learns both directions.

This adapter exists because plain NLI contradiction pairs (KLUE-NLI) don't
match the brief's debate-format training-data shape:

    {"topic": "안락사 허용", "input_context": "<pro arg>", "target_output": "<con arg>"}

REQUIRES a running vLLM server. Set VLLM_BASE_URL / VLLM_MODEL via .env.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import List, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

from ..synth.client import make_async_client, resolve_model
from ..synth.prompts import (
    DEBATE_PAIR_PROMPT_VERSION,
    render_debate_pair_prompt,
)
from .common import make_seq2seq_record

SOURCE = "ko_debate_synth"
LANG = "ko"
SEEDS_PATH = Path("data/seeds/ko_debate_motions.txt")
OUTPUT = Path("data/interim/ko_debate_synth.jsonl")

PAIRS_PER_MOTION = 60  # 60 pairs × 98 motions × 2 directions ≈ 11,760 records
CONCURRENCY = 32
MAX_TOKENS = 240
TEMPERATURE = 0.9  # high for variety across pairs on the same motion

PRO_RE = re.compile(r"^\s*PRO\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CON_RE = re.compile(r"^\s*CON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _load_motions(path: Path) -> List[str]:
    motions: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        motions.append(line)
    return motions


def _parse_pair(text: str) -> Tuple[str, str] | None:
    pro_m = PRO_RE.search(text)
    con_m = CON_RE.search(text)
    if not pro_m or not con_m:
        return None
    pro = pro_m.group(1).strip()
    con = con_m.group(1).strip()
    if not pro or not con or pro == con:
        return None
    return pro, con


async def _generate_one(
    motion: str,
    *,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str] | None:
    prompt = render_debate_pair_prompt(language="Korean", topic=motion)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return _parse_pair(response.choices[0].message.content or "")
        except Exception:
            return None


async def _run() -> None:
    motions = _load_motions(SEEDS_PATH)
    print(f"[{SOURCE}] {len(motions)} motions × {PAIRS_PER_MOTION} pairs each")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("")  # truncate

    client = make_async_client()
    model = resolve_model()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    total_records = 0
    failed = 0

    try:
        # Process motion-by-motion so any single-motion errors stay isolated
        # and the output JSONL grows incrementally.
        for motion in tqdm(motions, desc=SOURCE):
            tasks = [
                _generate_one(motion, client=client, model=model, semaphore=semaphore)
                for _ in range(PAIRS_PER_MOTION)
            ]
            pairs = await asyncio.gather(*tasks)

            chunk_records: List[dict] = []
            for pair in pairs:
                if pair is None:
                    failed += 1
                    continue
                pro, con = pair
                # Two directions per generated pair.
                for input_text, target_text, in_stance, out_stance in [
                    (pro, con, "pro", "con"),
                    (con, pro, "con", "pro"),
                ]:
                    record = make_seq2seq_record(
                        lang=LANG,
                        source=SOURCE,
                        topic=motion,
                        input_context=input_text,
                        target_output=target_text,
                        meta={
                            "is_synthetic": True,
                            "synthesis_prompt_version": DEBATE_PAIR_PROMPT_VERSION,
                            "input_stance": in_stance,
                            "target_stance": out_stance,
                            "conversion": "topic_seeded_pair",
                        },
                    )
                    if record:
                        chunk_records.append(record)

            with OUTPUT.open("a", encoding="utf-8") as f:
                for record in chunk_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += len(chunk_records)
    finally:
        await client.close()

    print(f"[{SOURCE}] wrote {total_records} records, {failed} pair-gens failed")


if __name__ == "__main__":
    asyncio.run(_run())
