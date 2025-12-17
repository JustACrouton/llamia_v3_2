from __future__ import annotations

import httpx

from ..state import LlamiaState
from ..config import DEFAULT_CONFIG

NODE_NAME = "research_web"


def research_web_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    query = (state.research_query or "").strip()
    if not query:
        state.log(f"[{NODE_NAME}] no research_query; skipping")
        state.next_agent = "planner" if state.mode == "task" else "chat"
        return state

    if DEFAULT_CONFIG.web_search_provider != "searxng":
        state.add_message("system", "[web_search] disabled in config.", node=NODE_NAME)
        state.research_query = None
        state.next_agent = "planner" if state.mode == "task" else "chat"
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

    state.research_query = None
    state.next_agent = "planner" if state.mode == "task" else "chat"
    state.log(f"[{NODE_NAME}] got_results={len(results)} next_agent={state.next_agent}")
    return state


