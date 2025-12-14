from __future__ import annotations

from ..state import LlamiaState

NODE_NAME = "intent_router"


def _extract_task_goal(raw: str) -> str:
    text = raw.strip()
    lower = text.lower()
    if lower.startswith("task:"):
        return text[5:].strip() or "(unspecified task goal)"
    if lower.startswith("task "):
        return text[5:].strip() or "(unspecified task goal)"
    return text


def _looks_like_task(raw: str) -> bool:
    lower = raw.strip().lower()

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
    t = text.strip().lower()
    return t.startswith("web:") or t.startswith("search:")


def _extract_web_query(text: str) -> str:
    t = text.strip()
    lower = t.lower()
    if lower.startswith("web:"):
        return t.split(":", 1)[1].strip()
    if lower.startswith("search:"):
        return t.split(":", 1)[1].strip()
    return t


def intent_router_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if not state.messages:
        state.mode = "chat"
        state.goal = None
        state.research_query = None
        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] no messages -> chat")
        return state

    last = state.messages[-1]
    if last["role"] != "user":
        state.next_agent = "chat"
        state.log(f"[{NODE_NAME}] last not user -> chat")
        return state

    text = last["content"].strip()
    lower = text.lower()

    # 0) Explicit web search (highest priority)
    if _looks_like_web_search(text):
        q = _extract_web_query(text)
        state.mode = "chat"          # prevent stale task mode
        state.goal = None            # prevent stale goals
        state.research_query = q
        state.next_agent = "research_web"
        state.log(f"[{NODE_NAME}] WEB: next_agent=research_web query={q!r}")
        return state

    # Clear any old web query when not doing web
    state.research_query = None

    # 1) Explicit task
    if lower.startswith("task:") or lower.startswith("task "):
        goal = _extract_task_goal(text)
        state.mode = "task"
        state.goal = goal
        state.next_agent = "planner"
        state.log(f"[{NODE_NAME}] TASK: mode=task goal={goal!r}")
        return state

    # 2) Heuristic task
    if _looks_like_task(text):
        state.mode = "task"
        state.goal = text
        state.next_agent = "planner"
        state.log(f"[{NODE_NAME}] TASK(heur): mode=task goal={text!r}")
        return state

    # 3) Default chat
    state.mode = "chat"
    state.goal = None               # set to None unless you want sticky tasks
    state.next_agent = "chat"
    state.log(f"[{NODE_NAME}] CHAT: next_agent=chat")
    return state
