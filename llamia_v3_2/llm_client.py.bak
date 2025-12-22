from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

from .config import DEFAULT_CONFIG, ModelConfig
from .state import Message


def _make_client() -> OpenAI:
    """
    Build an OpenAI / OpenAI-compatible client using DEFAULT_CONFIG.
    - For provider='openai': requires a real API key, uses OpenAI's base URL.
    - For provider='openai_compatible': uses api_base and allows a dummy key.
    """
    cfg = DEFAULT_CONFIG
    base_url = cfg.api_base
    provider = cfg.chat_model.provider
    api_key = os.getenv(cfg.api_key_env, "")

    if provider == "openai":
        if not api_key:
            raise RuntimeError(
                f"Missing API key: env var {cfg.api_key_env} is not set for provider 'openai'."
            )
        # Let the OpenAI client use its default base URL if none specified
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        else:
            return OpenAI(api_key=api_key)

    # provider == "openai_compatible"
    if not base_url:
        raise RuntimeError(
            "provider 'openai_compatible' requires api_base to be set in config.py"
        )

    # Many OpenAI-compatible servers (like Ollama) ignore the key, but the client requires it.
    if not api_key:
        api_key = "dummy"

    return OpenAI(api_key=api_key, base_url=base_url)


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def chat_completion(
    messages: Iterable[Message],
    model_cfg: ModelConfig | None = None,
) -> str:
    """
    Simple wrapper to do a chat completion using the given model config
    (or DEFAULT_CONFIG.chat_model if none provided).
    """
    cfg = model_cfg or DEFAULT_CONFIG.chat_model
    client = get_client()

    # Strip node info before sending to the API
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ]

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=api_messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )

    return resp.choices[0].message.content or ""
