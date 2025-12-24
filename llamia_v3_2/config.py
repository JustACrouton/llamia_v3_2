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

    # Per-role overrides (lets you mix local + API)
    api_base: str | None = None
    api_key_env: str | None = None


@dataclass
class LlamiaConfig:
    chat_model: ModelConfig
    planner_model: ModelConfig | None = None
    coder_model: ModelConfig | None = None
    research_model: ModelConfig | None = None
    critic_model: ModelConfig | None = None

    # RAG / embeddings (local)
    embed_model: str = "mxbai-embed-large"
    rag_top_k: int = 8
    embed_api_base: str = "http://127.0.0.1:11434"  # Ollama base (NOT /v1)

    # Web search
    web_search_provider: Literal["searxng", "none"] = "searxng"
    searxng_url: str = "http://127.0.0.1:8088"
    web_search_timeout_s: int = 20
    web_search_top_k: int = 5

    max_loops: int = 3

    # Global defaults (used only if a role doesn't override)
    api_base: str | None = "http://127.0.0.1:11434/v1"
    api_key_env: str = "OPENAI_API_KEY"

    def model_for(self, role: str) -> ModelConfig:
        role = role.lower().strip()
        if role == "chat":
            return self.chat_model
        if role == "planner":
            return self.planner_model or self.chat_model
        if role == "coder":
            return self.coder_model or self.chat_model
        if role == "research":
            return self.research_model or self.chat_model
        if role == "critic":
            return self.critic_model or self.chat_model
        return self.chat_model

    def api_base_for(self, role: str) -> str | None:
        m = self.model_for(role)
        return m.api_base if m.api_base is not None else self.api_base

    def api_key_env_for(self, role: str) -> str:
        m = self.model_for(role)
        return m.api_key_env if m.api_key_env is not None else self.api_key_env


# -----------------------------
# DEFAULT_CONFIG: planner + coder use Chutes API
# chat + research + critic use local Ollama
# -----------------------------

DEFAULT_CONFIG = LlamiaConfig(
    # LOCAL chat model
    chat_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3:32b",
        context_window=32768,
        temperature=0.4,
        max_output_tokens=2048,
        api_base="http://127.0.0.1:11434/v1",
        api_key_env="OPENAI_API_KEY",  # Ollama typically ignores, but keep non-empty
    ),

    # API planner (Chutes)
#    planner_model=ModelConfig(
#        provider="openai_compatible",
#        model="tngtech/DeepSeek-TNG-R1T2-Chimera",  # swap to whatever you want for planning
#        context_window=131072,
#        temperature=0.2,
#        max_output_tokens=2048,
#        api_base="https://llm.chutes.ai/v1",
#        api_key_env="CHUTES_API_TOKEN",
#    ),

    planner_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3:32b",  # swap to whatever you want for planning
        context_window=32768,
        temperature=0.2,
        max_output_tokens=2048,
        api_base="http://127.0.0.1:11434/v1",
        api_key_env="OPENAI_API_KEY",
    ),

    # API coder (Chutes) - can be different than planner
#    coder_model=ModelConfig(
#        provider="openai_compatible",
#        model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",  # swap to coder-favored model
#        context_window=131072,
#        temperature=0.2,
#        max_output_tokens=2048,
#        api_base="https://llm.chutes.ai/v1",
#        api_key_env="CHUTES_API_TOKEN",
#    ),

    coder_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3-coder:latest",  # swap to whatever you want for planning
        context_window=32768,
        temperature=0.2,
        max_output_tokens=2048,
        api_base="http://127.0.0.1:11434/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    

    # LOCAL research + critic (optional explicit; otherwise they fall back to chat_model)
    research_model=ModelConfig(
        provider="openai_compatible",
        model="qwen3:32b",
        context_window=32768,
        temperature=0.2,
        max_output_tokens=2048,
        api_base="http://127.0.0.1:11434/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    critic_model=ModelConfig(
        provider="openai_compatible",
        model="qwq:32b",
        context_window=131072,
        temperature=0.2,
        max_output_tokens=1024,
        api_base="http://127.0.0.1:11434/v1",
        api_key_env="OPENAI_API_KEY",
    ),

    # embeddings local
    embed_model="mxbai-embed-large",
    rag_top_k=8,
    embed_api_base="http://127.0.0.1:11434",

    # searxng local
    web_search_provider="searxng",
    searxng_url="http://127.0.0.1:8088",
    web_search_timeout_s=20,
    web_search_top_k=5,

    max_loops=3,

    # global defaults (safe to keep local)
    api_base="http://127.0.0.1:11434/v1",
    api_key_env="OPENAI_API_KEY",
)
