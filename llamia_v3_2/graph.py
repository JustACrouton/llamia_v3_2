from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.graph import StateGraph, END

from .state import LlamiaGraphState
from .nodes.intent_router import intent_router_node
from .nodes.chat import chat_node
from .nodes.planner import planner_node
from .nodes.coder import coder_node
from .nodes.executor import executor_node
from .nodes.research import research_node
from .nodes.critic import critic_node
from .nodes.research_web import research_web_node


# -----------------------------
# Trace helpers (trace is list[str] in state.py)
# -----------------------------
def _get_attr(state: Any, name: str, default: Any = None) -> Any:
    if hasattr(state, name):
        return getattr(state, name)
    if isinstance(state, dict):
        return state.get(name, default)
    return default


def _safe_head(s: Any, n: int = 220) -> str:
    txt = str(s or "")
    return txt[:n].replace("\n", "\\n")


def _last_msg_summary(state: Any) -> dict[str, Any]:
    msgs = _get_attr(state, "messages", []) or []
    if not isinstance(msgs, list) or not msgs:
        return {"role": None, "node": None, "len": 0, "head": ""}

    m = msgs[-1] if isinstance(msgs[-1], dict) else None
    if not m:
        return {"role": None, "node": None, "len": 0, "head": ""}

    content = str(m.get("content", "") or "")
    return {
        "role": m.get("role"),
        "node": m.get("node"),
        "len": len(content),
        "head": _safe_head(content, 240),
    }


def _exec_req_summary(state: Any) -> dict[str, Any] | None:
    req = _get_attr(state, "exec_request", None)
    if req is None:
        return None
    try:
        workdir = getattr(req, "workdir", None)
        cmds = list(getattr(req, "commands", []) or [])
    except Exception:
        return None

    shown = [str(c) for c in cmds[:6]]
    return {"workdir": workdir, "commands": shown, "more": max(0, len(cmds) - len(shown))}


def _snapshot(state: Any) -> dict[str, Any]:
    plan = _get_attr(state, "plan", []) or []
    applied = _get_attr(state, "applied_patches", []) or []
    exec_results = _get_attr(state, "exec_results", []) or []

    return {
        "mode": _get_attr(state, "mode", None),
        "next_agent": _get_attr(state, "next_agent", None),
        "goal_head": _safe_head(_get_attr(state, "goal", None), 260),
        "loop_count": _get_attr(state, "loop_count", None),
        "web_search_count": _get_attr(state, "web_search_count", None),
        "research_query": _get_attr(state, "research_query", None),
        "has_fix_instructions": bool(str(_get_attr(state, "fix_instructions", "") or "").strip()),
        "counts": {
            "messages": len(_get_attr(state, "messages", []) or []),
            "plan": len(plan) if isinstance(plan, list) else 0,
            "applied_patches": len(applied) if isinstance(applied, list) else 0,
            "exec_results": len(exec_results) if isinstance(exec_results, list) else 0,
        },
        "last_msg": _last_msg_summary(state),
        "exec_request": _exec_req_summary(state),
    }


def _trace(state: Any, event: dict[str, Any]) -> None:
    # Store as a single line string (easy to grep / parse)
    line = "[trace] " + json.dumps(event, ensure_ascii=False, sort_keys=True)
    try:
        if hasattr(state, "log"):
            state.log(line)
    except Exception:
        pass


def _wrap_step(name: str, fn: Callable[[LlamiaGraphState], LlamiaGraphState]) -> Callable[[LlamiaGraphState], LlamiaGraphState]:
    def _step(state: LlamiaGraphState) -> LlamiaGraphState:
        before = _snapshot(state)
        _trace(state, {"event": "node_enter", "node": name, "snap": before})

        out = fn(state)

        after = _snapshot(out)
        delta = {
            "messages_added": after["counts"]["messages"] - before["counts"]["messages"],
            "plan_delta": after["counts"]["plan"] - before["counts"]["plan"],
            "applied_patches_delta": after["counts"]["applied_patches"] - before["counts"]["applied_patches"],
            "exec_results_delta": after["counts"]["exec_results"] - before["counts"]["exec_results"],
        }
        _trace(out, {"event": "node_exit", "node": name, "snap": after, "delta": delta})
        return out
    return _step


def _wrap_router(name: str, router_fn: Callable[[Any], str]) -> Callable[[Any], str]:
    def _r(state: Any) -> str:
        choice = router_fn(state)
        _trace(state, {"event": "route", "node": name, "choice": choice, "snap": _snapshot(state)})
        return choice
    return _r


# -----------------------------
# Original routing helpers
# -----------------------------
def _get_mode_and_goal(state: Any) -> tuple[str, str | None]:
    if hasattr(state, "mode"):
        return getattr(state, "mode"), getattr(state, "goal", None)
    if isinstance(state, dict):
        return state.get("mode", "chat"), state.get("goal")
    return "chat", None


def _latest_user_text(state: Any) -> str:
    msgs = _get_attr(state, "messages", []) or []
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def _looks_like_research(user_text: str) -> bool:
    t = user_text.lower().strip()
    if t.startswith("research:") or t.startswith("reindex:"):
        return True
    keywords = ["workspace", "repo", "repository", "project files", "codebase", "in this folder"]
    intents = ["what files", "list files", "show files", "summarize files", "what does this do", "explain this project"]
    return any(k in t for k in keywords) and any(i in t for i in intents)


def _route_from_intent(state: Any) -> str:
    nxt = _get_attr(state, "next_agent", None)
    allowed = {"chat", "planner", "research", "research_web"}
    if isinstance(nxt, str) and nxt in allowed:
        return nxt

    user_text = _latest_user_text(state)
    if _looks_like_research(user_text):
        return "research"

    mode, goal = _get_mode_and_goal(state)
    if mode == "task" and goal:
        return "planner"

    return "chat"


def _route_from_planner(state: Any) -> str:
    nxt = _get_attr(state, "next_agent", None)
    if isinstance(nxt, str) and nxt == "research_web":
        return "research_web"
    if isinstance(nxt, str) and nxt == "research":
        return "research"
    return "coder"



def _route_from_research_web(state: Any) -> str:
    mode, goal = _get_mode_and_goal(state)
    fix = _get_attr(state, "fix_instructions", None)

    if mode == "task" and goal:
        return "coder" if (str(fix or "").strip()) else "planner"
    return "chat"


def _route_from_critic(state: Any) -> str:
    nxt = _get_attr(state, "next_agent", None)
    allowed = {"chat", "planner", "coder", "research", "research_web"}
    if isinstance(nxt, str) and nxt in allowed:
        return nxt
    return "chat"

def _route_from_research(state: Any) -> str:
    mode, goal = _get_mode_and_goal(state)
    if mode == "task" and goal:
        return "planner"
    return "chat"

def build_llamia_graph():
    workflow = StateGraph(LlamiaGraphState)

    # Wrap nodes (enter/exit)
    workflow.add_node("intent_router", _wrap_step("intent_router", intent_router_node))
    workflow.add_node("research", _wrap_step("research", research_node))
    workflow.add_node("research_web", _wrap_step("research_web", research_web_node))
    workflow.add_node("planner", _wrap_step("planner", planner_node))
    workflow.add_node("coder", _wrap_step("coder", coder_node))
    workflow.add_node("executor", _wrap_step("executor", executor_node))
    workflow.add_node("critic", _wrap_step("critic", critic_node))
    workflow.add_node("chat", _wrap_step("chat", chat_node))

    workflow.set_entry_point("intent_router")

    # Wrap routers (route decisions)
    workflow.add_conditional_edges(
        "intent_router",
        _wrap_router("intent_router", _route_from_intent),
        {"research_web": "research_web", "research": "research", "planner": "planner", "chat": "chat"},
    )

    workflow.add_conditional_edges(
      "research",
      _wrap_router("research", _route_from_research),
      {"planner": "planner", "chat": "chat"},
)


    workflow.add_conditional_edges(
        "research_web",
        _wrap_router("research_web", _route_from_research_web),
        {"coder": "coder", "planner": "planner", "chat": "chat"},
    )

    workflow.add_conditional_edges(
        "planner",
        _wrap_router("planner", _route_from_planner),
        {"research_web": "research_web", "coder": "coder"},
    )

    workflow.add_edge("coder", "executor")
    workflow.add_edge("executor", "critic")

    workflow.add_conditional_edges(
        "critic",
        _wrap_router("critic", _route_from_critic),
        {"coder": "coder", "planner": "planner", "research": "research", "research_web": "research_web", "chat": "chat"},
    )

    workflow.add_edge("chat", END)
    return workflow.compile()

