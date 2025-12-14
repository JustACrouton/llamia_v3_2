from __future__ import annotations

import re

from ..state import LlamiaState, ExecResult
from ..config import DEFAULT_CONFIG

NODE_NAME = "critic"


def _latest_user_text(state: LlamiaState) -> str:
    for m in reversed(state.messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _detect_expected_failure(text: str) -> bool:
    t = text.lower()

    fix_markers = [
        "then fix",
        "fix it",
        "fix the",
        "until it succeeds",
        "rerun until",
        "and fix",
        "repair",
    ]
    if any(k in t for k in fix_markers):
        return False

    expected_markers = [
        "should fail",
        "expected to fail",
        "intentionally fail",
        "doesn't exist",
        "does not exist",
        "non-existent",
        "nonexistent",
        "module not found",
        "modulenotfounderror",
        "demonstrate error",
        "trigger an error",
    ]
    return any(k in t for k in expected_markers)


def _last_run_results(state: LlamiaState) -> list[ExecResult]:
    if getattr(state, "last_exec_results", None):
        return state.last_exec_results
    return state.exec_results[-5:] if state.exec_results else []


def _extract_missing_module(stderr: str) -> str | None:
    # Example: ModuleNotFoundError: No module named 'foo'
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
    return m.group(1) if m else None


def _looks_like_needs_web(goal_text: str, stderr: str) -> bool:
    g = goal_text.lower()

    # user explicitly wants lookup/research
    if any(k in g for k in ["look up", "lookup", "search the web", "web search", "find documentation", "docs for", "how do i", "what is the correct"]):
        return True

    # failure types where web help is often useful
    s = stderr.lower()
    if "modulenotfounderror" in s or "no module named" in s:
        return True
    if "command not found" in s or "no such file or directory" in s:
        return True
    if "pip" in s and "error" in s:
        return True

    return False


def _build_web_query(goal_text: str, last: ExecResult) -> str:
    stderr = (last.stderr or "").strip()

    mod = _extract_missing_module(stderr)
    if mod:
        return f"python ModuleNotFoundError No module named {mod} install"

    # fallback: use goal + key stderr snippet
    tail = stderr[-300:] if stderr else ""
    if tail:
        return f"{goal_text.strip()} {tail}".strip()
    return goal_text.strip()


def critic_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    max_loops = DEFAULT_CONFIG.max_loops

    # IMPORTANT: clear stale routing hints unless we set them intentionally
    state.next_agent = None

    if state.loop_count >= max_loops:
        state.next_agent = "chat"
        state.add_message(
            "system",
            f"[critic] loop limit reached ({state.loop_count}/{max_loops}) -> finishing",
            node=NODE_NAME,
        )
        state.log(f"[{NODE_NAME}] done (loop limit)")
        return state

    user_text = _latest_user_text(state)
    goal_text = (state.goal or "") + "\n" + user_text
    state.expected_failure = _detect_expected_failure(goal_text)

    results = _last_run_results(state)
    if not results:
        state.next_agent = "chat"
        state.add_message("system", "[critic] no exec results -> finishing", node=NODE_NAME)
        state.log(f"[{NODE_NAME}] done (no exec results)")
        return state

    last = results[-1]
    failed = last.returncode != 0

    if state.expected_failure and failed:
        state.next_agent = "chat"
        state.add_message("system", "[critic] failure was expected -> finishing", node=NODE_NAME)
        state.log(f"[{NODE_NAME}] done (expected failure)")
        return state

    if failed:
        # Decide whether to web-search before fixing
        stderr = (last.stderr or "").strip()
        tail = stderr[-2000:] if stderr else "(no stderr)"

        needs_web = (
            DEFAULT_CONFIG.web_search_provider == "searxng"
            and _looks_like_needs_web(goal_text, stderr)
        )

        # Avoid spamming web searches
        web_count = int(getattr(state, "web_search_count", 0) or 0)
        max_web = 1  # keep it conservative; bump later if you want
        if needs_web and web_count < max_web:
            setattr(state, "web_search_count", web_count + 1)
            state.loop_count += 1

            q = _build_web_query(goal_text, last)
            state.research_query = q

            # Preserve fix instructions so coder can use web notes after research_web runs
            state.fix_instructions = (
                "Execution failed and may require external info.\n"
                "Use the web_search notes (research_notes) to apply the minimal fix needed so the command passes.\n\n"
                f"Failed command: {last.command}\n"
                f"Return code: {last.returncode}\n\n"
                f"Stderr (tail):\n{tail}\n"
            )

            state.next_agent = "research_web"
            state.add_message(
                "system",
                f"[critic] execution failed -> route to research_web (loop={state.loop_count}, web_count={web_count+1})",
                node=NODE_NAME,
            )
            state.log(f"[{NODE_NAME}] done (route research_web)")
            return state

        # Normal fix path (no web)
        state.loop_count += 1
        state.fix_instructions = (
            "Execution failed. Apply the minimal fix needed so the command passes.\n\n"
            f"Failed command: {last.command}\n"
            f"Return code: {last.returncode}\n\n"
            f"Stderr (tail):\n{tail}\n"
        )
        state.next_agent = "coder"
        state.add_message(
            "system",
            f"[critic] execution failed -> route to coder (loop={state.loop_count})",
            node=NODE_NAME,
        )
        state.log(f"[{NODE_NAME}] done (route coder)")
        return state

    # success
    state.fix_instructions = None
    state.next_agent = "chat"
    state.add_message("system", "[critic] execution ok -> finishing", node=NODE_NAME)
    state.log(f"[{NODE_NAME}] done (success)")
    return state

