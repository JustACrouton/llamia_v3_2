from __future__ import annotations

import time
from typing import Any

from ..state import LlamiaState

NODE_NAME = "intent_router"


def _strip_repl_prefix(text: str) -> str:
    """
    Users sometimes paste prompts like: 'you> task: ...'
    Strip any leading 'you>' tokens so routing behaves consistently.
    """
    s = (text or "").strip()
    while s.lower().startswith("you>"):
        s = s[4:].lstrip()
    return s


def _extract_task_goal(raw: str) -> str:
    text = _strip_repl_prefix(raw).strip()
    lower = text.lower()
    if lower.startswith("task:"):
        return text[5:].strip() or "(unspecified task goal)"
    if lower.startswith("task "):
        return text[5:].strip() or "(unspecified task goal)"
    return text


def _looks_like_task(raw: str) -> bool:
    lower = _strip_repl_prefix(raw).strip().lower()

    if lower in {"hi", "hey", "hello", "yo", "sup"}:
        return False

    verb_keywords = [
        "write a ",
        "write an ",
        "write the ",
        "write some code",
        "write code",
        "write a script",
        "build a ",
        "build an ",
        "build the ",
        "create a ",
        "create an ",
        "generate code",
        "implement ",
        "make a script",
        "make a program",
        "fix this code",
        "fix the code",
        "refactor this",
    ]

    object_keywords = [
        "script",
        "program",
        "function",
        "module",
        "tool",
        "bot",
        "cli",
        "python script",
        "python program",
    ]

    if any(kw in lower for kw in verb_keywords):
        return True

    if "python" in lower and any(obj in lower for obj in object_keywords):
        return True

    return False


def _looks_like_web_search(text: str) -> bool:
    t = _strip_repl_prefix(text).strip().lower()
    return t.startswith("web:") or t.startswith("search:")


def _extract_web_query(text: str) -> str:
    t = _strip_repl_prefix(text).strip()
    lower = t.lower()
    if lower.startswith("web:"):
        return t.split(":", 1)[1].strip()
    if lower.startswith("search:"):
        return t.split(":", 1)[1].strip()
    return t


def _looks_like_repo_research(text: str) -> bool:
    t = _strip_repl_prefix(text).strip().lower()
    return t.startswith("research:") or t.startswith("reindex:")


def _extract_repo_research_query(text: str) -> str:
    # Keep the prefix for research_node to parse reindex:/research:
    return _strip_repl_prefix(text).strip()


def intent_router_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    try:
        state.trace = list(getattr(state, "trace", []) or [])
    except Exception:
        state.trace = []

    def _trace(event: str, **kw: Any) -> None:
        state.trace.append(
            {
                "node": NODE_NAME,
                "event": event,
                "turn_id": getattr(state, "turn_id", None),
                "ts": time.time(),
                **kw,
            }
        )

    if not state.messages:
        state.mode = "chat"
        state.goal = None
        state.research_query = None
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] no messages -> chat")
        return state

    last = state.messages[-1]
    if last["role"] != "user":
        # If we're retrying a task (e.g., contract violation / repair), keep the task alive.
        if state.mode == "task" and state.goal and (state.fix_instructions or "").strip():
            state.next_agent = "planner"
            state.log(f"[{NODE_NAME}] TASK(retry): next_agent=planner")
            return state

        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] last not user -> chat")
        return state

    text = _strip_repl_prefix(last["content"])
    lower = text.lower().strip()

    # 0) Explicit web search (highest priority)
    if _looks_like_web_search(text):
        q = _extract_web_query(text)
        state.mode = "chat"  # prevent stale task mode
        state.goal = None
        state.research_query = q
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.fix_instructions = None
        state.next_agent = "research_web"
        state.log(f"[{NODE_NAME}] WEB: next_agent=research_web query={q!r}")
        _trace("route", kind="web", query=q, next_agent="research_web")
        return state

    # 0.5) Explicit repo research (RAG)
    if _looks_like_repo_research(text):
        q = _extract_repo_research_query(text)
        state.mode = "chat"  # treat as chat-like query, not a task pipeline
        state.goal = None
        state.research_query = q
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "research"
        state.log(f"[{NODE_NAME}] RESEARCH: next_agent=research query={q!r}")
        _trace("route", kind="research", query=q, next_agent="research")
        return state

    # Clear any old research query when not doing web/research
    state.research_query = None

    # 1) Explicit task
    if lower.startswith("task:") or lower.startswith("task "):
        goal = _extract_task_goal(text)
        state.mode = "task"
        state.goal = goal
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "planner"
        state.log(f"[{NODE_NAME}] TASK: mode=task goal={goal!r}")
        _trace("route", kind="task", goal=goal, next_agent="planner")
        return state

    # 2) Heuristic task
    if _looks_like_task(text):
        state.mode = "task"
        state.goal = text
        state.research_notes = None
        state.web_results = None
        state.web_queue = []
        state.web_search_count = 0
        state.loop_count = 0
        state.fix_instructions = None
        state.next_agent = "planner"
        state.log(f"[{NODE_NAME}] TASK(heur): mode=task goal={text!r}")
        _trace("route", kind="task_heur", goal=text, next_agent="planner")
        return state

    # 3) Default chat
    state.mode = "chat"
    state.goal = None
    state.research_notes = None
    state.web_results = None
    state.web_queue = []
    state.web_search_count = 0
    state.loop_count = 0
    state.fix_instructions = None
    state.next_agent = "chat"
    state.log(f"[{NODE_NAME}] CHAT: next_agent=chat")
    _trace("route", kind="chat", next_agent="chat")
    return state
