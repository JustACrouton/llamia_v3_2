from __future__ import annotations

import httpx

from ..state import LlamiaState
from ..config import DEFAULT_CONFIG

NODE_NAME = "research_web"


def _resolve_return_after_web(state: LlamiaState) -> str:
    target = str(getattr(state, "return_after_web", "") or "").strip()
    if target in {"planner", "coder", "chat", "research_web"}:
        return target
    return "planner" if state.mode == "task" else "chat"


def _pop_web_queue(state: LlamiaState) -> str | None:
    queue = getattr(state, "web_queue", None)
    if not isinstance(queue, list) or not queue:
        return None
    while queue:
        q = str(queue.pop(0)).strip()
        if q:
            return q
    return None


def research_web_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    query = (state.research_query or "").strip()
    if not query:
        queued = _pop_web_queue(state)
        if queued:
            query = queued
            state.research_query = query

    if not query:
        state.log(f"[{NODE_NAME}] no research_query; skipping")
        state.next_agent = _resolve_return_after_web(state)
        return state

    if DEFAULT_CONFIG.web_search_provider != "searxng":
        state.add_message("system", "[web_search] disabled in config.", node=NODE_NAME)
        state.research_query = None
        state.next_agent = _resolve_return_after_web(state)
        return state

    # only emit marker if we really have a query
    state.add_message(
        "system",
        f"[web_search] provider=searxng url={DEFAULT_CONFIG.searxng_url!r} query={query!r}",
        node=NODE_NAME,
    )

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
        state.research_query = None
        state.next_agent = _resolve_return_after_web(state)
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

    next_query = _pop_web_queue(state)
    if next_query:
        state.research_query = next_query
        state.next_agent = "research_web"
        state.log(f"[{NODE_NAME}] queued next_query; looping to research_web")
        return state

    state.research_query = None
    state.next_agent = _resolve_return_after_web(state)
    state.log(f"[{NODE_NAME}] got_results={len(results)} next_agent={state.next_agent}")
    return state

