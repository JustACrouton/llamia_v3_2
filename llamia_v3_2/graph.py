from __future__ import annotations

from typing import Any

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


def intent_router_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return intent_router_node(state)


def chat_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return chat_node(state)


def planner_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return planner_node(state)


def coder_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return coder_node(state)


def executor_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return executor_node(state)


def research_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return research_node(state)


def research_web_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return research_web_node(state)


def critic_step(state: LlamiaGraphState) -> LlamiaGraphState:
    return critic_node(state)


def _get_mode_and_goal(state: Any) -> tuple[str, str | None]:
    if hasattr(state, "mode"):
        return getattr(state, "mode"), getattr(state, "goal", None)
    if isinstance(state, dict):
        return state.get("mode", "chat"), state.get("goal")
    return "chat", None


def _latest_user_text(state: Any) -> str:
    if hasattr(state, "messages"):
        msgs = getattr(state, "messages", []) or []
    elif isinstance(state, dict):
        msgs = state.get("messages", []) or []
    else:
        msgs = []

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
    nxt = None
    if hasattr(state, "next_agent"):
        nxt = getattr(state, "next_agent", None)
    elif isinstance(state, dict):
        nxt = state.get("next_agent")

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
    """
    Option B: planner may request web search by setting next_agent="research_web".
    Otherwise proceed to coder.
    """
    nxt = None
    if hasattr(state, "next_agent"):
        nxt = getattr(state, "next_agent", None)
    elif isinstance(state, dict):
        nxt = state.get("next_agent")

    if isinstance(nxt, str) and nxt == "research_web":
        return "research_web"
    return "coder"


def _route_from_research_web(state: Any) -> str:
    """
    After web search:
      - task + fix_instructions -> coder (web-first repair)
      - task -> planner
      - chat -> chat
    """
    mode, goal = _get_mode_and_goal(state)

    fix = None
    if hasattr(state, "fix_instructions"):
        fix = getattr(state, "fix_instructions", None)
    elif isinstance(state, dict):
        fix = state.get("fix_instructions")

    if mode == "task" and goal:
        return "coder" if (fix or "").strip() else "planner"
    return "chat"


def _route_from_critic(state: Any) -> str:
    nxt = None
    if hasattr(state, "next_agent"):
        nxt = getattr(state, "next_agent", None)
    elif isinstance(state, dict):
        nxt = state.get("next_agent")

    allowed = {"chat", "planner", "coder", "research", "research_web"}
    if isinstance(nxt, str) and nxt in allowed:
        return nxt

    return "chat"


def build_llamia_graph():
    workflow = StateGraph(LlamiaGraphState)

    workflow.add_node("intent_router", intent_router_step)
    workflow.add_node("research", research_step)
    workflow.add_node("research_web", research_web_step)
    workflow.add_node("planner", planner_step)
    workflow.add_node("coder", coder_step)
    workflow.add_node("executor", executor_step)
    workflow.add_node("critic", critic_step)
    workflow.add_node("chat", chat_step)

    workflow.set_entry_point("intent_router")

    workflow.add_conditional_edges(
        "intent_router",
        _route_from_intent,
        {
            "research_web": "research_web",
            "research": "research",
            "planner": "planner",
            "chat": "chat",
        },
    )

    workflow.add_edge("research", "chat")

    workflow.add_conditional_edges(
        "research_web",
        _route_from_research_web,
        {
            "coder": "coder",
            "planner": "planner",
            "chat": "chat",
        },
    )

    workflow.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "research_web": "research_web",
            "coder": "coder",
        },
    )

    workflow.add_edge("coder", "executor")
    workflow.add_edge("executor", "critic")

    workflow.add_conditional_edges(
        "critic",
        _route_from_critic,
        {
            "coder": "coder",
            "planner": "planner",
            "research": "research",
            "research_web": "research_web",
            "chat": "chat",
        },
    )

    workflow.add_edge("chat", END)
    return workflow.compile()

