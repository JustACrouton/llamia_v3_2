from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelProvider = Literal["openai", "openai_compatible"]


@dataclass
class ModelConfig:
    provider: ModelProvider
    model: str
    context_window: int = 32768
    temperature: float = 0.3
    max_output_tokens: int = 2048


@dataclass
class LlamiaConfig:
    # Role models (all optional fall back to chat_model)
    chat_model: ModelConfig
    planner_model: ModelConfig | None = None
    coder_model: ModelConfig | None = None
    research_model: ModelConfig | None = None
    critic_model: ModelConfig | None = None

    # RAG / embeddings (Ollama)
    embed_model: str = "mxbai-embed-large"
    rag_top_k: int = 8

    # Web search (SearXNG recommended)
    web_search_provider: Literal["searxng", "none"] = "searxng"
    searxng_url: str = "http://127.0.0.1:8088"
    web_search_timeout_s: int = 20
    web_search_top_k: int = 5

    # Loop control
    max_loops: int = 3

    # OpenAI / OpenAI-compatible HTTP settings (used by llm_client)
    api_base: str | None = None  # None means "use OpenAI default"
    api_key_env: str = "OPENAI_API_KEY"

    def model_for(self, role: str) -> ModelConfig:
        """Return the ModelConfig for a role, falling back to chat_model."""
        role = role.lower().strip()
        m = None
        if role == "chat":
            m = self.chat_model
        elif role == "planner":
            m = self.planner_model
        elif role == "coder":
            m = self.coder_model
        elif role == "research":
            m = self.research_model
        elif role == "critic":
            m = self.critic_model
        return m or self.chat_model

    def ollama_base_url(self) -> str:
        """
        LlamaIndex's Ollama wrappers want base like http://127.0.0.1:11434
        while OpenAI-compatible wants .../v1.
        """
        base = self.api_base or "http://127.0.0.1:11434/v1"
        return base[:-3] if base.endswith("/v1") else base


DEFAULT_CONFIG = LlamiaConfig(
    chat_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3:32b",
        context_window=32768,
        temperature=0.4,
        max_output_tokens=2048,
    ),
    planner_model=ModelConfig(
        provider="openai_compatible",
        model="qwq:32b",
        context_window=131072,
        temperature=0.2,
        max_output_tokens=2048,
    ),
    coder_model=ModelConfig(
        provider="openai_compatible",
        model="qwen2.5-coder:32b",
        context_window=131072,
        temperature=0.2,
        max_output_tokens=2048,
    ),
    research_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3:32b",
        context_window=32768,
        temperature=0.2,
        max_output_tokens=2048,
    ),
    critic_model=ModelConfig(
        provider="openai_compatible",
        model="qwq:32b",
        context_window=131072,
        temperature=0.2,
        max_output_tokens=1024,
    ),

    embed_model="mxbai-embed-large",
    rag_top_k=8,

    # ✅ your searxng instance
    web_search_provider="searxng",
    searxng_url="http://127.0.0.1:8088",
    web_search_timeout_s=20,
    web_search_top_k=5,

    max_loops=3,
    api_base="http://127.0.0.1:11434/v1",
    api_key_env="OPENAI_API_KEY",
)
