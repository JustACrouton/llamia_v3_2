from __future__ import annotations

import json
from typing import Any

from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion
from ..state import LlamiaState, PlanStep

NODE_NAME = "planner"

PLANNER_SYSTEM_PROMPT = """You are a planning agent for an autonomous developer assistant.

Your job:
- Read the user's HIGH-LEVEL GOAL.
- Produce a small, linear plan of 3-7 steps MAX.
- Each step should be short, clear, and actionable.
- Do NOT write any code here; just describe the steps.

STRICT OUTPUT RULES:
- Output MUST be valid JSON only.
- No prose, no markdown fences, no comments.
- Output must start with '{' and end with '}'.

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

    # Local repo-only tasks (patch/diff/test runs) should not trigger web search.
    if any(k in t for k in ["unified diff", "diff --git", ".patch", "git style", "git apply", "git diff"]):
        return False

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


def _try_parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    s = raw.strip()

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Extract likely JSON object
    a = s.find("{")
    b = s.rfind("}")
    if a >= 0 and b > a:
        candidate = s[a : b + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _retry_strict_json(messages: list[dict[str, str]], model_cfg) -> str:
    retry_msgs = list(messages)
    retry_msgs.append(
        {
            "role": "system",
            "content": (
                "STRICT MODE: Your last output was NOT valid JSON.\n"
                "Return STRICT JSON ONLY with a 'plan' list.\n"
                "No prose. No markdown fences.\n"
                "Output must start with '{' and end with '}'."
            ),
            "node": NODE_NAME,
        }
    )
    return chat_completion(messages=retry_msgs, model_cfg=model_cfg)


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

    notes = (state.research_notes or "").strip()
    notes_block = f"\n\nWeb research notes:\n{notes}\n" if notes else ""

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT, "node": NODE_NAME},
        {"role": "user", "content": f"Goal: {state.goal}{notes_block}", "node": NODE_NAME},
    ]

    cfg = DEFAULT_CONFIG.model_for("planner")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")

    raw = chat_completion(messages=messages, model_cfg=cfg)
    state.log(f"[{NODE_NAME}] raw LLM output: {raw!r}")

    data = _try_parse_json_object(raw)
    if data is None:
        raw2 = _retry_strict_json(messages, cfg)
        state.log(f"[{NODE_NAME}] raw LLM output (retry): {raw2!r}")
        data = _try_parse_json_object(raw2)

    plan_steps: list[PlanStep] = []
    try:
        if not isinstance(data, dict):
            raise ValueError("planner output is not a JSON object")

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

        if not plan_steps:
            raise ValueError("empty plan")

    except Exception as e:
        state.log(f"[{NODE_NAME}] ERROR parsing plan JSON: {e!r}")
        plan_steps = [PlanStep(id=1, description=f"Attempt to solve goal: {state.goal}", status="pending")]

    state.plan = plan_steps
    state.next_agent = None  # important: don't “stick” routing hints
    state.log(f"[{NODE_NAME}] created {len(plan_steps)} plan steps")
    return state
