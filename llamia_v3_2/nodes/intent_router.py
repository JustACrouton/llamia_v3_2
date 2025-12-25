from __future__ import annotations

import time
from typing import Any

from ..state import LlamiaState

NODE_NAME = "intent_router"


def intent_router_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    # Always ensure trace is a list
    try:
        state.trace = list(getattr(state, "trace", []) or [])
    except Exception:
        state.trace = []

    def _trace(event: str, **kw: Any) -> None:
        # Keep trace as structured dicts (graph.py wraps a separate string trace line too)
        state.trace.append(
            {
                "node": NODE_NAME,
                "event": event,
                "turn_id": getattr(state, "turn_id", None),
                "ts": time.time(),
                **kw,
            }
        )

    # If no messages, start in chat
    if not getattr(state, "messages", None):
        state.mode = "chat"
        state.goal = None
        state.intent_kind = "chat"
        state.intent_payload = None
        state.intent_source = "empty"
        state.research_query = None
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.return_after_web = "chat"
        state.return_after_research = "chat"
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] no messages -> chat")
        _trace("route", kind="chat", next_agent="chat")
        return state

    last = state.messages[-1]

    # If last message is not user, this is usually an internal retry cycle.
    if not isinstance(last, dict) or last.get("role") != "user":
        # Contract-violation repair path: keep the task alive.
        if state.mode == "task" and state.goal and (state.fix_instructions or "").strip():
            # If something upstream already selected a retry target, honor it.
            if state.next_agent not in {"planner", "coder", "research", "research_web", "chat"}:
                # Default for repair is coder (it must regenerate artifacts)
                state.next_agent = "coder"
            state.log(f"[{NODE_NAME}] TASK(retry): next_agent={state.next_agent}")
            _trace("route", kind="task_retry", next_agent=state.next_agent)
            return state

        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] last not user -> chat")
        _trace("route", kind="chat", next_agent="chat")
        return state

    intent = getattr(state, "intent_kind", None)
    payload = getattr(state, "intent_payload", None)

    # 0) Explicit web search (highest priority)
    if intent == "research_web":
        q = str(payload or "").strip()
        if not q:
            q = str(last.get("content", "") or "").strip()
        state.mode = "chat"
        state.goal = None
        state.research_query = q
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.return_after_web = "chat"
        state.return_after_research = "chat"
        state.fix_instructions = None
        state.next_agent = "research_web"
        state.log(f"[{NODE_NAME}] WEB: next_agent=research_web query={q!r}")
        _trace("route", kind="web", query=q, next_agent="research_web")
        return state

    # 0.5) Explicit repo research (RAG)
    if intent == "research":
        q = str(payload or "").strip()
        if not q:
            q = str(last.get("content", "") or "").strip()
        state.mode = "chat"
        state.goal = None
        state.research_query = q
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.return_after_web = "chat"
        state.return_after_research = "chat"
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "research"
        state.log(f"[{NODE_NAME}] RESEARCH: next_agent=research query={q!r}")
        _trace("route", kind="research", query=q, next_agent="research")
        return state

    # Clear any old research query when not doing web/research
    state.research_query = None

    # 1) Explicit task
    if intent == "task":
        goal = str(payload or "").strip() or "(unspecified task goal)"
        state.mode = "task"
        state.goal = goal
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.return_after_web = "planner"
        state.return_after_research = "planner"
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "planner"
        state.log(f"[{NODE_NAME}] TASK: mode=task goal={goal!r}")
        _trace("route", kind="task", goal=goal, next_agent="planner")
        return state

    # 3) Default chat
    state.mode = "chat"
    state.goal = None
    state.research_notes = None
    state.web_results = None
    state.web_queue = []
    state.web_search_count = 0
    state.return_after_web = "chat"
    state.return_after_research = "chat"
    state.loop_count = 0
    state.fix_instructions = None
    state.next_agent = "chat"
    state.log(f"[{NODE_NAME}] CHAT: next_agent=chat")
    _trace("route", kind="chat", next_agent="chat")
    return state
