from __future__ import annotations

from typing import Any, Dict, List

from llamia_v3_2.state import CodePatch, ExecRequest, ExecResult, LlamiaState, PlanStep

from .logging_utils import tail_lines


def state_snapshot(state: LlamiaState) -> Dict[str, Any]:
    """
    Compact snapshot suitable for JSONL:
      - routing info
      - last N messages (trimmed)
      - trace (if used)
      - last exec results

    This is intentionally “lossy” to keep logs small and safe.
    """
    msgs: list[dict[str, Any]] = []
    for m in (state.messages or [])[-12:]:
        msgs.append(
            {
                "role": m.get("role"),
                "node": m.get("node"),
                "content": tail_lines((m.get("content") or "").strip(), max_chars=2000),
            }
        )

    last_exec: list[dict[str, Any]] = []
    for r in (getattr(state, "last_exec_results", None) or [])[-6:]:
        last_exec.append(
            {
                "command": r.command,
                "returncode": r.returncode,
                "stdout_tail": tail_lines((r.stdout or "").strip(), max_chars=1500),
                "stderr_tail": tail_lines((r.stderr or "").strip(), max_chars=1500),
            }
        )

    return {
        "mode": getattr(state, "mode", None),
        "goal": getattr(state, "goal", None),
        "intent_kind": getattr(state, "intent_kind", None),
        "intent_payload": getattr(state, "intent_payload", None),
        "intent_source": getattr(state, "intent_source", None),
        "next_agent": getattr(state, "next_agent", None),
        "loop_count": getattr(state, "loop_count", None),
        "web_search_count": getattr(state, "web_search_count", None),
        "research_query": getattr(state, "research_query", None),
        "trace": getattr(state, "trace", None),
        "messages_tail": msgs,
        "exec_request": getattr(state, "exec_request", None),
        "last_exec_results_tail": last_exec,
    }


def make_exec_results(raw_list: Any) -> List[ExecResult]:
    """
    Normalize an untyped list of exec results into `list[ExecResult]`.
    """
    results: list[ExecResult] = []
    if not isinstance(raw_list, list):
        return results

    for r in raw_list:
        if isinstance(r, ExecResult):
            results.append(r)
            continue

        if isinstance(r, dict):
            cmd = str(r.get("command", "")).strip()
            if not cmd:
                continue
            rc = int(r.get("returncode", 0))
            stdout = str(r.get("stdout", ""))
            stderr = str(r.get("stderr", ""))
            results.append(ExecResult(command=cmd, returncode=rc, stdout=stdout, stderr=stderr))

    return results


def coerce_to_state(raw: Any) -> LlamiaState:
    """
    LangGraph nodes can accidentally return plain dicts.

    This function ensures we always end up with a real `LlamiaState`,
    while preserving backwards compatibility with earlier shapes.
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
            out: list[CodePatch] = []
            if not isinstance(raw_list, list):
                return out
            for p in raw_list:
                if isinstance(p, CodePatch):
                    out.append(p)
                elif isinstance(p, dict):
                    fp = str(p.get("file_path", "")).strip()
                    if not fp:
                        continue
                    content = str(p.get("content", ""))
                    mode = str(p.get("apply_mode", "overwrite")).lower()
                    if mode not in ("overwrite", "append"):
                        mode = "overwrite"
                    out.append(CodePatch(file_path=fp, content=content, apply_mode=mode))
            return out

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

        exec_results = make_exec_results(raw.get("exec_results") or [])
        last_exec_results = make_exec_results(raw.get("last_exec_results") or [])

        web_queue = raw.get("web_queue") or []
        if not isinstance(web_queue, list):
            web_queue = []
        web_queue = [str(q).strip() for q in web_queue if str(q).strip()]

        return_after_web = str(raw.get("return_after_web", "planner") or "planner").strip() or "planner"
        return_after_research = str(raw.get("return_after_research", "planner") or "planner").strip() or "planner"

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
            return_after_research=return_after_research,
            web_search_count=int(raw.get("web_search_count", 0) or 0),
            intent_kind=raw.get("intent_kind"),
            intent_payload=raw.get("intent_payload"),
            intent_source=raw.get("intent_source"),
            turn_id=turn_id,
            responded_turn_id=responded_turn_id,
        )

    # Worst-case fallback: preserve REPL stability.
    state = LlamiaState()
    state.log(f"[repl] WARNING: unexpected state type from graph: {type(raw)!r}")
    return state
