from __future__ import annotations

import re
from typing import List

from ..state import LlamiaState, ExecResult


def classify_failure(stderr: str) -> str:
    """Classify the type of failure based on error output."""
    stderr_lower = (stderr or "").lower()

    if "permission denied" in stderr_lower:
        return "permissions"
    if "command not found" in stderr_lower or "no such file or directory" in stderr_lower:
        return "missing_dependency"
    if "transient" in stderr_lower or "timeout" in stderr_lower or "network" in stderr_lower:
        return "transient"
    if "assert" in stderr_lower or "test failed" in stderr_lower or "failed" in stderr_lower:
        return "test_failure"
    if "policy" in stderr_lower or "blocked" in stderr_lower or "disallowed" in stderr_lower:
        return "policy_blocked"
    return "bad_command"


def handle_transient_failure(state: LlamiaState, results: List[ExecResult]) -> LlamiaState:
    state.log("[failure_handler] Handling transient failure with retry")
    state.add_message("system", "Retrying failed command with exponential backoff...", node="failure_handler")

    state.retry_count = getattr(state, "retry_count", 0) + 1
    if state.retry_count > 3:
        state.add_message("system", "Max retry attempts reached. Moving to next step.", node="failure_handler")
        state.retry_count = 0
        state.next_agent = "coder"
    return state


def handle_missing_dependency(state: LlamiaState, results: List[ExecResult]) -> LlamiaState:
    state.log("[failure_handler] Handling missing dependency")

    missing: list[str] = []
    for r in results:
        if r.returncode != 0 and "command not found" in (r.stderr or "").lower():
            cmd0 = (r.command or "").split()[:1]
            if cmd0:
                missing.append(cmd0[0])

    if missing:
        install_cmds: list[str] = []
        for pkg in missing:
            if pkg in ("python", "python3"):
                install_cmds.append(f"sudo apt-get install {pkg}")
            else:
                install_cmds.append(f"pip install {pkg}")

        msg = (
            "Missing dependencies detected: " + ", ".join(missing) + "\n\n"
            "Proposed installation commands:\n"
            + "\n".join("- " + c for c in install_cmds)
        )
        state.add_message("system", msg, node="failure_handler")
    else:
        state.add_message("system", "Missing dependency detected. Please install required packages.", node="failure_handler")

    state.next_agent = "coder"
    return state


def handle_test_failure(state: LlamiaState, results: List[ExecResult]) -> LlamiaState:
    state.log("[failure_handler] Handling test failure")

    details: list[str] = []
    for r in results:
        if r.returncode != 0:
            details.append(
                "Command: " + (r.command or "")
                + "\nReturn code: " + str(r.returncode)
                + "\nError: " + (r.stderr or "")[:500]
            )

    state.add_message(
        "system",
        "Test failures detected. Please analyze and fix:\n\n" + "\n\n".join(details),
        node="failure_handler",
    )
    state.next_agent = "coder"
    return state


def handle_policy_blocked(state: LlamiaState, results: List[ExecResult]) -> LlamiaState:
    state.log("[failure_handler] Handling policy blocked command")

    blocked: list[str] = []
    for r in results:
        stderr_lower = (r.stderr or "").lower()
        if r.returncode != 0 and ("policy" in stderr_lower or "blocked" in stderr_lower or "disallowed" in stderr_lower):
            blocked.append("Blocked command: " + (r.command or "") + "\nReason: " + (r.stderr or ""))

    state.add_message(
        "system",
        "Policy blocked commands detected:\n\n" + "\n\n".join(blocked) + "\n\nPlease adjust commands to comply with policies.",
        node="failure_handler",
    )
    state.next_agent = "coder"
    return state


def handle_bad_command(state: LlamiaState, results: List[ExecResult]) -> LlamiaState:
    state.log("[failure_handler] Handling bad command")

    repaired: list[tuple[str, str]] = []
    for r in results:
        if r.returncode != 0:
            old = (r.command or "")
            new = old.replace("python ", "python3 ")
            new = re.sub(r"\s+", " ", new).strip()
            repaired.append((old, new))
            state.log(f"[failure_handler] Attempting repair: {old} -> {new}")

    if repaired:
        state.add_message(
            "system",
            "Attempting to repair commands:\n" + "\n".join("- " + o + " -> " + n for o, n in repaired),
            node="failure_handler",
        )
        exec_req = getattr(state, "exec_request", None)
        if exec_req is not None:
            exec_req.commands = [n for _, n in repaired]
    else:
        state.add_message(
            "system",
            "Bad command detected. Unable to automatically repair. Please review.",
            node="failure_handler",
        )
        state.next_agent = "coder"

    return state


def handle_failures(state: LlamiaState) -> LlamiaState:
    """Classify and route failures based on state.last_exec_results."""
    failed = [r for r in getattr(state, "last_exec_results", []) if r.returncode != 0]
    if not failed:
        return state

    failure_type = classify_failure(failed[0].stderr)
    state.log("[failure_handler] Classified failure as: " + failure_type)

    if failure_type == "transient":
        return handle_transient_failure(state, failed)
    if failure_type == "missing_dependency":
        return handle_missing_dependency(state, failed)
    if failure_type == "test_failure":
        return handle_test_failure(state, failed)
    if failure_type == "policy_blocked":
        return handle_policy_blocked(state, failed)

    return handle_bad_command(state, failed)

