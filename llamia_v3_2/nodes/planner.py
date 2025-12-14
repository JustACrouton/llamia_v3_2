from __future__ import annotations

import json

from ..state import LlamiaState, PlanStep
from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion

NODE_NAME = "planner"


PLANNER_SYSTEM_PROMPT = """You are a planning agent for an autonomous developer assistant.

Your job:
- Read the user's HIGH-LEVEL GOAL.
- Produce a small, linear plan of 3-7 steps MAX.
- Each step should be short, clear, and actionable.
- Do NOT write any code here; just describe the steps.

You MUST respond with STRICT JSON ONLY in this format:

{
  "plan": [
    {"id": 1, "description": "First step description"},
    {"id": 2, "description": "Second step description"}
  ]
}
"""


def _needs_web_search(goal: str) -> bool:
    """
    Heuristic: tasks that likely require external factual info / docs.
    Keep it conservative to avoid pointless web hits.
    """
    t = (goal or "").lower().strip()
    if not t:
        return False

    triggers = [
        "look up",
        "lookup",
        "search for",
        "search the web",
        "find documentation",
        "docs",
        "documentation",
        "api",
        "parameter",
        "query parameter",
        "curl",
        "how do i",
        "how to",
        "what is the correct",
        "latest",
        "current",
        "version",
        "release notes",
        "searxng",
    ]
    return any(x in t for x in triggers)


def planner_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if state.mode != "task" or not state.goal:
        state.log(f"[{NODE_NAME}] no goal in task mode; nothing to plan")
        return state

    # Option B: planner can request web search before planning
    if (
        DEFAULT_CONFIG.web_search_provider == "searxng"
        and not (state.research_notes or "").strip()
        and _needs_web_search(state.goal)
    ):
        state.research_query = state.goal.strip()
        state.next_agent = "research_web"
        state.add_message(
            "system",
            f"[planner] requesting web search for goal: {state.research_query!r}",
            node=NODE_NAME,
        )
        state.log(f"[{NODE_NAME}] routed to research_web query={state.research_query!r}")
        return state

    # Build prompt (include research notes if available)
    notes = (state.research_notes or "").strip()
    notes_block = f"\n\nWeb research notes:\n{notes}\n" if notes else ""

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT, "node": NODE_NAME},
        {
            "role": "user",
            "content": f"Goal: {state.goal}{notes_block}",
            "node": NODE_NAME,
        },
    ]

    cfg = DEFAULT_CONFIG.model_for("planner")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")
    raw = chat_completion(messages=messages, model_cfg=cfg)

    plan_steps: list[PlanStep] = []
    try:
        data = json.loads(raw)
        raw_plan = data.get("plan", [])
        if not isinstance(raw_plan, list):
            raise ValueError("plan field is not a list")

        for i, step in enumerate(raw_plan, start=1):
            if not isinstance(step, dict):
                continue
            desc = str(step.get("description", "")).strip()
            if not desc:
                continue
            sid = int(step.get("id", i))
            plan_steps.append(PlanStep(id=sid, description=desc, status="pending"))

    except Exception as e:
        state.log(f"[{NODE_NAME}] ERROR parsing plan JSON: {e!r}")
        plan_steps = [
            PlanStep(id=1, description=f"Attempt to solve goal: {state.goal}", status="pending")
        ]

    state.plan = plan_steps
    state.next_agent = None  # important: don't “stick” routing hints
    state.log(f"[{NODE_NAME}] created {len(plan_steps)} plan steps")
    return state

