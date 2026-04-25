"""vLLM client wrapper.

vLLM serves an OpenAI-compatible API. Swap local <-> remote by setting
VLLM_BASE_URL. Everything else (sync code, async code, prompts) stays identical.

Local default:  http://localhost:8000/v1
Remote example: https://my-vllm.example.com/v1
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3


def _resolve_base_url(base_url: str | None) -> str:
    return base_url or os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)


def _resolve_api_key(api_key: str | None) -> str:
    return api_key or os.environ.get("VLLM_API_KEY", DEFAULT_API_KEY)


def resolve_model(model: str | None = None) -> str:
    """Resolve the model id from the argument, env (VLLM_MODEL), or raise."""
    model = model or os.environ.get("VLLM_MODEL")
    if not model:
        raise RuntimeError(
            "No model specified. Pass model=... or set VLLM_MODEL "
            "(e.g. 'Qwen/Qwen2.5-7B-Instruct')."
        )
    return model


def make_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    return OpenAI(
        base_url=_resolve_base_url(base_url),
        api_key=_resolve_api_key(api_key),
        timeout=timeout,
        max_retries=max_retries,
    )


def make_async_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=_resolve_base_url(base_url),
        api_key=_resolve_api_key(api_key),
        timeout=timeout,
        max_retries=max_retries,
    )
