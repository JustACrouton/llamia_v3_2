from __future__ import annotations

"""
LLM Client Abstraction Layer

Key Responsibilities:
- Unified interface for multiple LLM providers
- Response format enforcement
- Error handling and retries
- Token usage tracking

Core Components:
- chat_completion: Main generation function
- Message formatting (system/user/assistant roles)
- Response validation and parsing
"""

import os
import logging
from typing import Iterable

from openai import OpenAI, APIError, APITimeoutError

from .config import DEFAULT_CONFIG, ModelConfig
from .state import Message

logger = logging.getLogger(__name__)

# Cache clients by (provider, base_url, api_key_env) so planner/coder can use Chutes
# while chat/research/critic stay local, etc.
_client_cache: dict[tuple[str, str | None, str], OpenAI] = {}


def _resolve_provider(model_cfg: ModelConfig) -> str:
    # ModelConfig.provider is Literal["openai","openai_compatible"], but keep it robust.
    return getattr(model_cfg, "provider", DEFAULT_CONFIG.chat_model.provider)


def _resolve_base_url(model_cfg: ModelConfig) -> str | None:
    # Prefer per-model override if present, else global config default
    base = getattr(model_cfg, "api_base", None) or DEFAULT_CONFIG.api_base

    # Defensive: avoid accidental /v1/v1
    if base and base.endswith("/v1/v1"):
        base = base[:-3]
    return base


def _resolve_api_key_env(model_cfg: ModelConfig) -> str:
    # Prefer per-model override if present, else global config default
    return getattr(model_cfg, "api_key_env", None) or DEFAULT_CONFIG.api_key_env


def _make_client_for(model_cfg: ModelConfig) -> OpenAI:
    """
    Build an OpenAI / OpenAI-compatible client for a specific model_cfg.
    - provider='openai': requires a real API key, base_url optional.
    - provider='openai_compatible': requires base_url, dummy key allowed if empty.
    """
    provider = _resolve_provider(model_cfg)
    base_url = _resolve_base_url(model_cfg)
    api_key_env = _resolve_api_key_env(model_cfg)
    api_key = os.getenv(api_key_env, "")

    if provider == "openai":
        if not api_key:
            raise RuntimeError(
                f"Missing API key: env var {api_key_env} is not set for provider 'openai'."
            )
        # Let OpenAI client use default base URL if none specified
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    # provider == "openai_compatible"
    if not base_url:
        raise RuntimeError("provider 'openai_compatible' requires api_base to be set (global or per-model).")

    # Many OpenAI-compatible servers (like Ollama) ignore the key, but the OpenAI client requires one.
    if not api_key:
        api_key = "dummy"

    return OpenAI(api_key=api_key, base_url=base_url)


def get_client(model_cfg: ModelConfig) -> OpenAI:
    provider = _resolve_provider(model_cfg)
    base_url = _resolve_base_url(model_cfg)
    api_key_env = _resolve_api_key_env(model_cfg)

    key = (provider, base_url, api_key_env)
    client = _client_cache.get(key)
    if client is None:
        client = _make_client_for(model_cfg)
        _client_cache[key] = client
        logger.info(f"[llm_client] created client provider={provider} base_url={base_url} key_env={api_key_env}")
    return client


def chat_completion(
    messages: Iterable[Message],
    model_cfg: ModelConfig | None = None,
) -> str:
    """
    Chat completion using the given model config (or DEFAULT_CONFIG.chat_model if none).
    IMPORTANT: client is chosen per-model_cfg so roles can route to different backends.
    """
    cfg = model_cfg or DEFAULT_CONFIG.chat_model
    client = get_client(cfg)

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=api_messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
        )
        return resp.choices[0].message.content or ""
    except APITimeoutError as e:
        logger.error(f"API timeout error: {e}")
        return "Error: Request to LLM timed out. Please try again."
    except APIError as e:
        logger.error(f"API error: {e}")
        return f"Error: Failed to get response from LLM. Details: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in chat_completion: {e}")
        return f"Error: Unexpected issue when calling LLM. Details: {str(e)}"
