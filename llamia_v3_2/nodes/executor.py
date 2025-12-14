from __future__ import annotations

from ..state import LlamiaState
from ..tools.exec_tools import run_exec_request

NODE_NAME = "executor"

MAX_STD_TAIL = 1200
MAX_ERR_TAIL = 2000


def _tail(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[-n:]


def executor_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    req = state.exec_request
    if req is None or not req.commands:
        state.log(f"[{NODE_NAME}] no exec_request or commands; nothing to run")
        state.last_exec_results = []
        return state

    results = run_exec_request(req)

    state.last_exec_results = results
    state.exec_results.extend(results)

    lines: list[str] = []
    lines.append(f"[executor] workdir={req.workdir}")
    lines.append("[executor] commands:")

    for r in results:
        status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
        lines.append(f"- {r.command} -> {status}")

        out_tail = _tail((r.stdout or "").strip(), MAX_STD_TAIL).strip()
        err_tail = _tail((r.stderr or "").strip(), MAX_ERR_TAIL).strip()

        if out_tail:
            lines.append("  stdout (tail):")
            for line in out_tail.splitlines():
                lines.append(f"    {line}")

        if err_tail:
            lines.append("  stderr (tail):")
            for line in err_tail.splitlines():
                lines.append(f"    {line}")

    state.add_message("system", "\n".join(lines), node=NODE_NAME)
    state.log(f"[{NODE_NAME}] ran {len(results)} commands")
    return state
