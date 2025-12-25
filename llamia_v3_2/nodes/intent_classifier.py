from __future__ import annotations

from ..state import LlamiaState

NODE_NAME = "intent_classifier"


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


def intent_classifier_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if not getattr(state, "messages", None):
        state.intent_kind = "chat"
        state.intent_payload = None
        state.intent_source = "empty"
        state.log(f"[{NODE_NAME}] no messages -> intent=chat")
        return state

    last = state.messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        state.intent_kind = None
        state.intent_payload = None
        state.intent_source = "non_user"
        state.log(f"[{NODE_NAME}] last not user -> intent unchanged")
        return state

    text = _strip_repl_prefix(str(last.get("content", "") or ""))
    lower = text.lower().strip()

    if _looks_like_web_search(text):
        state.intent_kind = "research_web"
        state.intent_payload = _extract_web_query(text)
        state.intent_source = "explicit_web"
        state.log(f"[{NODE_NAME}] intent=research_web payload={state.intent_payload!r}")
        return state

    if _looks_like_repo_research(text):
        state.intent_kind = "research"
        state.intent_payload = _extract_repo_research_query(text)
        state.intent_source = "explicit_research"
        state.log(f"[{NODE_NAME}] intent=research payload={state.intent_payload!r}")
        return state

    if lower.startswith("task:") or lower.startswith("task "):
        goal = _extract_task_goal(text)
        state.intent_kind = "task"
        state.intent_payload = goal
        state.intent_source = "explicit_task"
        state.log(f"[{NODE_NAME}] intent=task goal={goal!r}")
        return state

    if _looks_like_task(text):
        state.intent_kind = "task"
        state.intent_payload = text
        state.intent_source = "heuristic_task"
        state.log(f"[{NODE_NAME}] intent=task (heuristic)")
        return state

    state.intent_kind = "chat"
    state.intent_payload = None
    state.intent_source = "default_chat"
    state.log(f"[{NODE_NAME}] intent=chat")
    return state
