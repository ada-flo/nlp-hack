"""AI-Hub topic-seeded debate synthesis.

Uses the AI-Hub corpus as a *topic-domain inspiration source* — we sample
balanced topic categories (e.g. 사회이슈, 교육, 자연/환경, 시사/사회/인문)
and ask Qwen3 to propose a specific debate motion within each domain plus a
PRO/CON exchange. The casual-chat utterances themselves are NOT used: this is
a debate chatbot, not casual conversation, so seed *content* is discarded —
only the topic taxonomy is reused as inspiration for fresh motions.

Each successful generation yields two seq2seq records (pro→con, con→pro).

Usage:
    VLLM_BASE_URL=http://localhost:8001/v1 \\
    VLLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507 \\
    uv run python -m src.preprocess.ko_aihub_seeded_synth --n 3000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from ..synth.client import make_async_client, resolve_model
from ..synth.prompts import (
    DOMAIN_DEBATE_PROMPT_VERSION,
    render_domain_debate_prompt,
)
from .common import make_seq2seq_record

SOURCE = "ko_aihub_seeded_synth"
LANG = "ko"
INTERIM_DIR = Path("data/interim")
OUTPUT = Path("data/interim/ko_aihub_seeded_synth.jsonl")

SEED_FILES = [
    "ko_aihub_topic_dialogue.jsonl",
    "ko_aihub_purpose_dialog.jsonl",
    "ko_aihub_persona_dialog.jsonl",
]

# Pure customer-service / transactional categories — not naturally debatable.
SKIP_TOPICS = {
    "AS문의",
    "절차 문의",
    "등록 문의",
    "등록문의",
    "서류 문의",
    "배송",
    "온오프라인 안내",
    "환불/반품/교환",
    "주문/결제",
    "비용/환불 문의",
    "일정 문의",
    "일정문의",
    "프로그램 문의",
    "프로그램문의",
    "부서안내",
    "민원신고",
    "제품/사용문의",
    "이벤트",
}

DEFAULT_N = 3000
TEMPERATURE = 0.9
MAX_TOKENS = 280
CONCURRENCY = 32

MOTION_RE = re.compile(r"^\s*MOTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PRO_RE = re.compile(r"^\s*PRO\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CON_RE = re.compile(r"^\s*CON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _collect_topics(seed_files: list[str]) -> Counter:
    """Tally topics across seed files (after dropping pure-transactional ones)."""
    topics: Counter = Counter()
    for fname in seed_files:
        path = INTERIM_DIR / fname
        if not path.exists():
            print(f"[{SOURCE}] missing {path}, skipping")
            continue
        for r in _read_jsonl(path):
            t = (r.get("topic") or "").strip()
            if not t or t in SKIP_TOPICS:
                continue
            topics[t] += 1
    return topics


def _sample_domains(n: int, seed: int = 42) -> list[str]:
    """Balanced sample of N topic-strings to drive N synth calls.

    Each topic is repeated proportionally up to a per-topic cap so the
    distribution is flat across the AI-Hub topic taxonomy rather than
    weighted by the (very skewed) record counts.
    """
    topic_counts = _collect_topics(SEED_FILES)
    if not topic_counts:
        return []
    topics = sorted(topic_counts.keys())
    per_topic = max(1, n // len(topics))
    picks: list[str] = []
    for t in topics:
        picks.extend([t] * per_topic)
    rng = random.Random(seed)
    rng.shuffle(picks)
    print(f"[{SOURCE}] {len(topics)} domains × {per_topic} calls each → {len(picks)} target")
    return picks[:n]


def _parse_triple(text: str) -> tuple[str, str, str] | None:
    m_motion = MOTION_RE.search(text)
    m_pro = PRO_RE.search(text)
    m_con = CON_RE.search(text)
    if not (m_motion and m_pro and m_con):
        return None
    motion = m_motion.group(1).strip()
    pro = m_pro.group(1).strip()
    con = m_con.group(1).strip()
    if not motion or not pro or not con or pro == con:
        return None
    return motion, pro, con


async def _generate_one(
    domain: str,
    *,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str, str] | None:
    """Returns (domain, motion, pro, con) or None on failure."""
    prompt = render_domain_debate_prompt(language="Korean", domain=domain)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            parsed = _parse_triple(response.choices[0].message.content or "")
            if parsed is None:
                return None
            motion, pro, con = parsed
            return domain, motion, pro, con
        except Exception:
            return None


async def _run(n: int) -> None:
    domains = _sample_domains(n)
    if not domains:
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("")  # truncate

    client = make_async_client()
    model = resolve_model()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        _generate_one(d, client=client, model=model, semaphore=semaphore)
        for d in domains
    ]

    written = 0
    failed = 0
    try:
        with OUTPUT.open("a", encoding="utf-8") as f:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=SOURCE):
                result = await coro
                if result is None:
                    failed += 1
                    continue
                domain, motion, pro, con = result
                # Two directions per generated debate.
                for input_text, target_text, in_stance, out_stance in [
                    (pro, con, "pro", "con"),
                    (con, pro, "con", "pro"),
                ]:
                    rec = make_seq2seq_record(
                        lang=LANG,
                        source=SOURCE,
                        topic=motion,
                        input_context=input_text,
                        target_output=target_text,
                        meta={
                            "is_synthetic": True,
                            "synthesis_prompt_version": DOMAIN_DEBATE_PROMPT_VERSION,
                            "input_stance": in_stance,
                            "target_stance": out_stance,
                            "conversion": "aihub_domain_seeded_pair",
                            "seed_domain": domain,
                        },
                    )
                    if rec:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
    finally:
        await client.close()

    print(f"[{SOURCE}] wrote {written} records ({written // 2} debates); {failed} synth failures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    asyncio.run(_run(args.n))


if __name__ == "__main__":
    main()
