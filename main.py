from __future__ import annotations

from typing import Any
from collections import Counter

import sys
import signal

from llamia_v3_2.state import (
    LlamiaState,
    PlanStep,
    CodePatch,
    ExecRequest,
    ExecResult,
)
from llamia_v3_2.graph import build_llamia_graph


# -----------------------------
# Invoke safety limits
# -----------------------------
INVOKE_RECURSION_LIMIT = 40        # hard cap on langgraph recursion
INVOKE_TIMEOUT_S = 45              # wall-clock cap per turn (Linux only)


class InvokeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise InvokeTimeout(f"invoke exceeded {INVOKE_TIMEOUT_S}s timeout")


def _set_alarm(seconds: int):
    # SIGALRM works on Linux/macOS. If unavailable, we just skip timeout.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(seconds)))


def _clear_alarm():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


def _make_exec_results(raw_list: Any) -> list[ExecResult]:
    results: list[ExecResult] = []
    if not isinstance(raw_list, list):
        return results

    for r in raw_list:
        if isinstance(r, ExecResult):
            results.append(r)
        elif isinstance(r, dict):
            cmd = str(r.get("command", "")).strip()
            if not cmd:
                continue
            rc = int(r.get("returncode", 0))
            stdout = str(r.get("stdout", ""))
            stderr = str(r.get("stderr", ""))
            results.append(ExecResult(command=cmd, returncode=rc, stdout=stdout, stderr=stderr))
    return results


def _coerce_to_state(raw: Any) -> LlamiaState:
    """
    LangGraph currently returns a plain dict even if we start with a dataclass.
    This helper converts the returned object back into LlamiaState.
    """
    if isinstance(raw, LlamiaState):
        return raw

    if isinstance(raw, dict):
        raw_plan = raw.get("plan") or []
        plan: list[PlanStep] = []
        for i, step in enumerate(raw_plan, start=1):
            if isinstance(step, PlanStep):
                plan.append(step)
            elif isinstance(step, dict):
                desc = str(step.get("description", "")).strip()
                if not desc:
                    continue
                sid = int(step.get("id", i))
                status = step.get("status", "pending")
                plan.append(PlanStep(id=sid, description=desc, status=status))  # type: ignore[arg-type]

        def _make_patches(raw_list: Any) -> list[CodePatch]:
            result: list[CodePatch] = []
            if not isinstance(raw_list, list):
                return result
            for p in raw_list:
                if isinstance(p, CodePatch):
                    result.append(p)
                elif isinstance(p, dict):
                    fp = str(p.get("file_path", "")).strip()
                    if not fp:
                        continue
                    content = str(p.get("content", ""))
                    mode = str(p.get("apply_mode", "overwrite")).lower()
                    if mode not in ("overwrite", "append"):
                        mode = "overwrite"
                    result.append(CodePatch(file_path=fp, content=content, apply_mode=mode))
            return result

        pending_patches = _make_patches(raw.get("pending_patches") or [])
        applied_patches = _make_patches(raw.get("applied_patches") or [])

        raw_exec = raw.get("exec_request")
        exec_req: ExecRequest | None = None
        if isinstance(raw_exec, ExecRequest):
            exec_req = raw_exec
        elif isinstance(raw_exec, dict):
            workdir = str(raw_exec.get("workdir", "workspace")).strip() or "workspace"
            commands = raw_exec.get("commands") or []
            if isinstance(commands, list):
                commands = [str(c).strip() for c in commands if str(c).strip()]
            else:
                commands = []
            if commands:
                exec_req = ExecRequest(workdir=workdir, commands=commands)

        exec_results = _make_exec_results(raw.get("exec_results") or [])
        last_exec_results = _make_exec_results(raw.get("last_exec_results") or [])

        web_queue = raw.get("web_queue") or []
        if not isinstance(web_queue, list):
            web_queue = []
        web_queue = [str(q).strip() for q in web_queue if str(q).strip()]

        return_after_web = str(raw.get("return_after_web", "planner") or "planner").strip() or "planner"

        # NEW: carry turn ids if present
        turn_id = int(raw.get("turn_id", 0) or 0)
        responded_turn_id = int(raw.get("responded_turn_id", -1) or -1)

        return LlamiaState(
            messages=raw.get("messages", []),
            mode=raw.get("mode", "chat"),
            goal=raw.get("goal"),
            plan=plan,
            pending_patches=pending_patches,
            applied_patches=applied_patches,
            exec_request=exec_req,
            exec_results=exec_results,
            last_exec_results=last_exec_results,
            next_agent=raw.get("next_agent"),
            trace=raw.get("trace", []),

            research_query=raw.get("research_query"),
            research_notes=raw.get("research_notes"),
            fix_instructions=raw.get("fix_instructions"),
            loop_count=int(raw.get("loop_count", 0) or 0),
            expected_failure=bool(raw.get("expected_failure", False)),

            web_queue=web_queue,
            web_results=raw.get("web_results"),
            return_after_web=return_after_web,

            # NEW
            turn_id=turn_id,
            responded_turn_id=responded_turn_id,
        )

    state = LlamiaState()
    state.log(f"[main] WARNING: unexpected state type from graph: {type(raw)!r}")
    return state


def run_repl():
    state = LlamiaState()
    app = build_llamia_graph()

    print("Llamia v3.2 (LangGraph + planner + coder + executor + chat). Type 'exit' to quit.\n")
    print("Tips:")
    print("  - Normal message: regular chat mode")
    print("  - 'task: build me X': task mode => planner, coder (writes into workspace/), executor (runs safe commands), then chat.\n")

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # NEW: per-turn id increments (used by chat guard)
        state.turn_id += 1
        state.responded_turn_id = -1

        state.add_message("user", user_input, node=None)

        before_applied_len = len(state.applied_patches)
        before_exec_len = len(state.exec_results)

        try:
            _set_alarm(INVOKE_TIMEOUT_S)
            raw_result = app.invoke(
                state,
                config={"recursion_limit": INVOKE_RECURSION_LIMIT},
            )
        except InvokeTimeout as e:
            # clean stop: no stack trace, don't burn GPU forever
            state.add_message("system", f"[main] {e}", node="main")
            print(f"llamia> [timed out] {e}\n")
            continue
        except KeyboardInterrupt:
            # clean stop for runaway model calls
            print("\nllamia> [interrupted] (Ctrl+C). You can type 'exit' to quit.\n")
            # important: don't keep partially mutated state; just continue loop
            continue
        finally:
            _clear_alarm()

        state = _coerce_to_state(raw_result)

        # Print *new* assistant message (last one)
        if not state.messages or state.messages[-1].get("role") != "assistant":
            print("llamia> [no assistant reply produced]\n")
            continue

        last = state.messages[-1]
        print(f"llamia> {last.get('content','')}\n")
        sys.stdout.flush()

        new_applied = state.applied_patches[before_applied_len:]
        new_exec = state.exec_results[before_exec_len:]

        if state.mode == "task":
            if state.plan:
                print("  [plan]")
                for step in state.plan:
                    print(f"   - ({step.id}) [{step.status}] {step.description}")

            if new_applied:
                counts = Counter((p.file_path, p.apply_mode) for p in new_applied)
                print("  [files]")
                for (fp, mode), n in counts.items():
                    suffix = f" x{n}" if n > 1 else ""
                    print(f"   - {fp} ({mode}){suffix}")

            if state.exec_request and state.exec_request.commands:
                print("  [suggested commands]")
                print(f"   - workdir: {state.exec_request.workdir}")
                for cmd in state.exec_request.commands:
                    print(f"   - {cmd}")

            if new_exec:
                print("  [exec results]")
                for r in new_exec:
                    status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
                    print(f"   - {r.command} -> {status}")

            if state.web_results:
                print("  [web results]")
                print("   " + str(state.web_results).replace("\n", "\n   "))

            print("")
            sys.stdout.flush()


if __name__ == "__main__":
    run_repl()

