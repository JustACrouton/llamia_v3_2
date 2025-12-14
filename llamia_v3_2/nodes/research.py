from __future__ import annotations

from ..state import LlamiaState
from ..config import DEFAULT_CONFIG
from ..tools.rag_index import ingest_repo, query_repo

NODE_NAME = "research"


def _latest_user_text(state: LlamiaState) -> str:
    for m in reversed(state.messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def research_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    user_text = _latest_user_text(state)
    if not user_text:
        state.log(f"[{NODE_NAME}] no user text; skipping")
        return state

    # Allow explicit reindex
    force = False
    q = (state.research_query or user_text).strip()

    low = q.lower()
    if low.startswith("reindex:"):
        force = True
        q = q.split(":", 1)[1].strip()
    elif low.startswith("research:"):
        q = q.split(":", 1)[1].strip()

    # (Re)ingest repo
    ingested = ingest_repo(force=force)
    state.log(f"[{NODE_NAME}] ingested_docs={ingested} force={force}")

    # Query
    answer = query_repo(query=q, top_k=DEFAULT_CONFIG.rag_top_k)
    state.research_notes = answer

    state.add_message(
        role="system",
        content=f"[research results]\nQuery: {q}\n\n{answer}",
        node=NODE_NAME,
    )

    state.log(f"[{NODE_NAME}] done")
    return state
