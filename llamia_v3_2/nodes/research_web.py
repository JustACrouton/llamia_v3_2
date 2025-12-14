from __future__ import annotations

import httpx

from ..state import LlamiaState
from ..config import DEFAULT_CONFIG

NODE_NAME = "research_web"


def research_web_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if DEFAULT_CONFIG.web_search_provider != "searxng":
        state.add_message("system", "[web_search] disabled in config.", node=NODE_NAME)
        state.log(f"[{NODE_NAME}] disabled")
        # route back safely
        state.next_agent = "planner" if state.mode == "task" else "chat"
        return state

    query = (state.research_query or "").strip()

    # fallback: last user message content
    if not query:
        for m in reversed(state.messages):
            if m.get("role") == "user":
                query = (m.get("content") or "").strip()
                break

    # visible marker in history so you KNOW it ran
    state.add_message(
        "system",
        f"[web_search] provider=searxng url={DEFAULT_CONFIG.searxng_url!r} query={query!r}",
        node=NODE_NAME,
    )

    if not query:
        state.add_message("system", "[web_search] no query provided.", node=NODE_NAME)
        state.next_agent = "planner" if state.mode == "task" else "chat"
        return state

    url = DEFAULT_CONFIG.searxng_url.rstrip("/") + "/search"
    params = {"q": query, "format": "json"}

    try:
        with httpx.Client(timeout=DEFAULT_CONFIG.web_search_timeout_s) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        state.add_message("system", f"[web_search] ERROR: {e!r}", node=NODE_NAME)
        state.log(f"[{NODE_NAME}] error={e!r}")
        # clear trigger so we don’t loop forever
        state.research_query = None
        state.next_agent = "planner" if state.mode == "task" else "chat"
        return state

    results = (data.get("results") or [])[: max(1, int(DEFAULT_CONFIG.web_search_top_k))]

    lines = [f"[web_search results] top_k={len(results)} query={query!r}"]
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip()
        link = (item.get("url") or "").strip()
        snippet = (item.get("content") or "").strip()
        lines.append(f"{i}. {title}\n   {link}\n   {snippet}")

    notes = "\n".join(lines)
    state.research_notes = notes
    state.add_message("system", notes, node=NODE_NAME)

    # ? CRITICAL: clear the trigger so research_web does NOT re-run
    state.research_query = None

    # ? Route back:
    # - if we’re in a task, planner consumes research_notes and continues
    # - if we’re in chat (explicit web:), go to chat
    state.next_agent = "planner" if state.mode == "task" else "chat"

    state.log(f"[{NODE_NAME}] got_results={len(results)} next_agent={state.next_agent}")
    return state

