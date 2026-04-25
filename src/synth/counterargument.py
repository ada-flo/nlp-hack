"""Generate counter-arguments via vLLM (OpenAI-compatible API).

Two entry points:
- synthesize_counterargument: one call, sync. Good for adapter scripts and debugging.
- synthesize_counterarguments_async: many calls with bounded concurrency. Use this
  when running over a whole dataset.

Failures are returned as {"ok": False, "error": ...} rather than raised, so a
single bad row does not abort a long batch run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI, OpenAI

from .client import make_async_client, make_client, resolve_model
from .prompts import (
    COUNTERARGUMENT_PROMPT_VERSION,
    render_counterargument_prompt,
)


@dataclass
class SynthRequest:
    topic: str
    input_context: str
    language: str  # "en" or "ko"


@dataclass
class SynthResult:
    ok: bool
    request: SynthRequest
    text: str | None = None
    error: str | None = None
    prompt_version: str = COUNTERARGUMENT_PROMPT_VERSION


def synthesize_counterargument(
    request: SynthRequest,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> str:
    client = client or make_client()
    model = resolve_model(model)
    prompt = render_counterargument_prompt(
        language=request.language,
        topic=request.topic,
        input_context=request.input_context,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


async def _one_async(
    request: SynthRequest,
    *,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> SynthResult:
    prompt = render_counterargument_prompt(
        language=request.language,
        topic=request.topic,
        input_context=request.input_context,
    )
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return SynthResult(ok=False, request=request, error="empty completion")
            return SynthResult(ok=True, request=request, text=text)
        except Exception as exc:  # noqa: BLE001 — surface upstream as error string
            return SynthResult(ok=False, request=request, error=repr(exc))


async def synthesize_counterarguments_async(
    requests: Sequence[SynthRequest],
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
    concurrency: int = 8,
) -> list[SynthResult]:
    """Run many synth requests concurrently. Returns results in input order."""
    own_client = client is None
    client = client or make_async_client()
    model = resolve_model(model)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        return await asyncio.gather(
            *[
                _one_async(
                    req,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
                for req in requests
            ]
        )
    finally:
        if own_client:
            await client.close()


def _smoke_test() -> None:
    """Manual smoke test: requires a running vLLM server.

    Usage:
        VLLM_BASE_URL=http://localhost:8000/v1 \\
        VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
        uv run python -m src.synth.counterargument
    """
    request = SynthRequest(
        topic="Legalization of Euthanasia",
        input_context="It is inhumane to force someone to continue a life full of suffering.",
        language="en",
    )
    print("Prompt version:", COUNTERARGUMENT_PROMPT_VERSION)
    print("Request:", request)
    print("Response:", synthesize_counterargument(request))


if __name__ == "__main__":
    _smoke_test()
