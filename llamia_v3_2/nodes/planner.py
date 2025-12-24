from __future__ import annotations

import json
import re
from typing import Any

from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion
from ..state import LlamiaState, PlanStep

NODE_NAME = "planner"

PLANNER_SYSTEM_PROMPT = """You are an expert planning agent for an autonomous developer assistant. The system runs on Linux. Your role is to decompose complex goals into executable, sequential steps that can be carried out by specialized agents.

PLANNING PRINCIPLES:
- Create 2-8 steps maximum for most tasks
- Each step should be specific, actionable, and testable
- Include verification/checking steps when appropriate
- For technical tasks, specify relevant commands or tools
- Consider dependencies between steps
- If the goal involves multiple domains, plan for appropriate handoffs

OUTPUT FORMAT:
- Return ONLY valid JSON with a "plan" array
- Each step has: "id", "description", "status" (always "pending")

STRICT JSON FORMAT:
{
  "plan": [
    {
      "id": 1,
      "description": "Step description"
    }
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
        "install",
        "setup",
        "configuration",
        "tutorial",
        "example",
        "best practice",
        "security",
        "compatibility",
        "dependency",
        "library",
        "framework",
        "protocol",
        "standard",
    ]
    return any(x in t for x in triggers)


def _try_parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    s = raw.strip()

    # Fast path - direct JSON
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Extract JSON from markdown or mixed content
    json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', s)
    if json_match:
        candidate = json_match.group(0)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Fallback: find content between first { and last }
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
                "ERROR: Your last response was not valid JSON.\n"
                "PRODUCE ONLY JSON with a 'plan' field containing an array of steps.\n"
                "NO TEXT BEFORE OR AFTER JSON.\n"
                "Format: {\"plan\": [{\"id\": 1, \"description\": \"...\"}]}\n"
                "START YOUR RESPONSE WITH { and END WITH }"
            ),
            "node": NODE_NAME,
        }
    )
    return chat_completion(messages=retry_msgs, model_cfg=model_cfg)



def _analyze_goal_complexity(goal: str) -> str:
    """Analyze the goal and return complexity level using an LLM."""
    from llamia_v3_2.llm_client import chat_completion, DEFAULT_CONFIG
    
    prompt = (
        "You are an expert at analyzing task complexity. ",
        "Classify the following task into one of these categories: simple, complex, development.",
        "Simple: tasks that can be completed in 1-2 steps with minimal dependencies.",
        "Complex: tasks requiring multiple steps, coordination, or research.",
        "Development: tasks that involve writing or modifying code, including testing.",
        f"Task: {goal}",
        "Reason step by step, then output only the category (simple, complex, development)."
    )
    
    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model_cfg=DEFAULT_CONFIG.model_for("planner"),
    ).strip().lower()
    
    pass
    if response not in ['simple', 'complex', 'development']:
        return 'complex'
    return response



def _enhance_plan_with_context(plan_steps: list[dict], goal: str, research_notes: str = "") -> list[PlanStep]:
    """Enhance the plan with additional context and validation."""
    enhanced_steps = []
    
    for i, step in enumerate(plan_steps):
        if not isinstance(step, dict):
            continue
            
        desc = str(step.get("description", "")).strip()
        if not desc:
            continue
            
        sid = int(step.get("id", i + 1))
        
        # Create PlanStep with only the original expected fields
        enhanced_steps.append(PlanStep(
            id=sid, 
            description=desc, 
            status="pending"
        ))
    
    return enhanced_steps


def planner_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if state.mode != "task" or not state.goal:
        state.log(f"[{NODE_NAME}] no goal in task mode; nothing to plan")
        return state

    # Check if web search is needed before planning
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

    # Prepare context for planning
    notes = (state.research_notes or "").strip()
    notes_block = f"\n\nWeb research notes:\n{notes}\n" if notes else ""
    
    # Add context about current directory/file context if available
    context_block = ""
    if hasattr(state, 'current_file') and state.current_file:
        context_block += f"\nCurrent file context: {state.current_file}\n"
    if hasattr(state, 'working_dir') and state.working_dir:
        context_block += f"Current working directory: {state.working_dir}\n"

    # Determine planning strategy based on goal complexity
    complexity = _analyze_goal_complexity(state.goal)
    
    # Customize prompt based on complexity
    prompt = PLANNER_SYSTEM_PROMPT
    if complexity == "complex":
        prompt += "\nFor complex tasks: break into smaller, manageable phases with verification between phases."
    elif complexity == "simple":
        prompt += "\nFor simple tasks: focus on direct, efficient execution steps."
    elif complexity == "development":
        prompt += "\nFor development tasks: include testing and verification steps."

    messages = [
        {"role": "system", "content": prompt, "node": NODE_NAME},
        {"role": "user", "content": f"Goal: {state.goal}{context_block}{notes_block}", "node": NODE_NAME},
    ]

    cfg = DEFAULT_CONFIG.model_for("planner")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature} complexity={complexity}")

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

        # Enhance the plan with context and validation
        enhanced_plan = _enhance_plan_with_context(raw_plan, state.goal, notes)
        plan_steps = enhanced_plan

        if not plan_steps:
            raise ValueError("empty plan")

        # Ensure plan steps are properly numbered sequentially
        for i, step in enumerate(plan_steps, start=1):
            step.id = i

    except Exception as e:
        state.log(f"[{NODE_NAME}] ERROR parsing plan JSON: {e!r}")
        # Fallback: create a simple plan
        plan_steps = [
            PlanStep(
                id=1, 
                description=f"Analyze and execute the goal: {state.goal}", 
                status="pending"
            )
        ]

    state.plan = plan_steps
    state.next_agent = None  # Reset routing - let execution flow naturally
    state.log(f"[{NODE_NAME}] created {len(plan_steps)} plan steps: {[step.description for step in plan_steps]}")
    
    # Add plan summary to messages for context
    if plan_steps:
        plan_summary = "\n".join([f"Step {step.id}: {step.description}" for step in plan_steps])
        state.add_message(
            "system",
            f"Generated execution plan:\n{plan_summary}",
            node=NODE_NAME,
        )

    return state