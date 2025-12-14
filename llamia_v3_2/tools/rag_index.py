from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from ..config import DEFAULT_CONFIG
from .fs_tools import ROOT_DIR

CHROMA_PATH = ROOT_DIR / ".llamia_chroma"
COLLECTION_NAME = "llamia_repo"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".llamia_chroma",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
}

REQUIRED_EXTS = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".sh", ".bash", ".zsh",
}


def _configure_llamaindex_models() -> None:
    base_url = DEFAULT_CONFIG.ollama_base_url()

    Settings.embed_model = OllamaEmbedding(
        model_name=DEFAULT_CONFIG.embed_model,
        base_url=base_url,
    )

    m = DEFAULT_CONFIG.model_for("research")
    Settings.llm = Ollama(
        model=m.model,
        base_url=base_url,
        request_timeout=120.0,
    )


def _client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _get_collection():
    return _client().get_or_create_collection(COLLECTION_NAME)


def _reset_collection() -> None:
    c = _client()
    try:
        c.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    c.get_or_create_collection(COLLECTION_NAME)


def _collection_count() -> int:
    try:
        return int(_get_collection().count())
    except Exception:
        return 0


def _reader_for_path(root: Path) -> SimpleDirectoryReader:
    excludes: list[str] = []
    for d in EXCLUDE_DIRS:
        excludes.append(f"**/{d}/**")
        excludes.append(f"{d}/**")

    return SimpleDirectoryReader(
        input_dir=str(root),
        recursive=True,
        required_exts=sorted(REQUIRED_EXTS),
        exclude=excludes,
    )


def ingest_repo(force: bool = False) -> int:
    """
    Build (or rebuild) a repo-wide vector index in .llamia_chroma/.

    - force=False: no-op if collection already has vectors
    - force=True: wipe and rebuild
    """
    _configure_llamaindex_models()

    if force:
        _reset_collection()
    else:
        if _collection_count() > 0:
            return 0

    docs = _reader_for_path(ROOT_DIR).load_data()
    if not docs:
        return 0

    collection = _get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    return len(docs)


def query_repo(query: str, top_k: int | None = None) -> str:
    _configure_llamaindex_models()

    k = int(top_k if top_k is not None else DEFAULT_CONFIG.rag_top_k)

    collection = _get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    engine = index.as_query_engine(similarity_top_k=k)
    resp = engine.query(query)
    return str(resp)
