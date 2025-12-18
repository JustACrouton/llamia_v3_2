from __future__ import annotations

from collections import Counter
import sys
import time
from pathlib import Path
from typing import Any, Optional

from llamia_v3_2.graph import build_llamia_graph
from llamia_v3_2.state import LlamiaState

from .config import ReplConfig
from .contract import validate_task_contract
from .input_utils import read_user_input_block
from .logging_utils import append_jsonl, read_if_exists, setup_run_logger
from .paths import RepoPaths
from .repo_utils import dirty_outside_workspace, git_ls_files, git_restore_paths, repo_snapshot_text
from .state_utils import coerce_to_state, state_snapshot
from .timeouts import InvokeTimeout, invoke_timeout


def _reset_turn_fields(state: LlamiaState) -> None:
    """
    Normalize/reset per-turn fields to a known state.
    """
    state.responded_turn_id = -1
    state.web_search_count = 0
    state.web_results = None
    state.research_query = None
    state.research_notes = None
    state.web_queue = []
    state.return_after_web = "planner"
    state.loop_count = 0


def _ensure_turn_fields_exist(state: LlamiaState) -> None:
    """
    Backward compatibility: older LlamiaState versions may not have turn_id fields.
    """
    if not hasattr(state, "turn_id"):
        state.turn_id = 0
    if not hasattr(state, "responded_turn_id"):
        state.responded_turn_id = -1


def run_repl(config: Optional[ReplConfig] = None) -> int:
    """
    Main interactive loop.
    Returns a process exit code (0 for normal exit).
    """
    cfg = config or ReplConfig()

    # __file__ is .../llamia_v3_2/repl/app.py
    # parents[2] is repo root (because: repl -> llamia_v3_2 -> repo_root)
    entry_main = Path(__file__).resolve().parents[2] / "main.py"
    paths = RepoPaths.from_entrypoint(entry_main)

    logger, _text_path, jsonl_path = setup_run_logger(paths)

    state = LlamiaState()
    _ensure_turn_fields_exist(state)

    app = build_llamia_graph()

    print("Llamia v3.2 (LangGraph + planner + coder + executor + chat). Type 'exit' to quit.\n")
    print("Tips:")
    print("  - Normal message: regular chat mode")
    print("  - 'task: build me X': task mode => planner, coder (writes into workspace/), executor (runs safe commands), then chat.\n")

    while True:
        user_input = read_user_input_block()
        if user_input is None:
            print("\nBye.")
            logger.info("[repl] exit at prompt")
            append_jsonl(jsonl_path, {"event": "repl_exit", "reason": "prompt_exit", "ts": time.time()})
            return 0

        if not user_input.strip():
            continue

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            logger.info("[repl] user requested exit")
            append_jsonl(jsonl_path, {"event": "repl_exit", "reason": "user_exit", "ts": time.time()})
            return 0

        # Turn bookkeeping
        state.turn_id += 1
        _reset_turn_fields(state)

        before_applied_len = len(getattr(state, "applied_patches", []))
        before_exec_len = len(getattr(state, "exec_results", []))
        before_msg_len = len(getattr(state, "messages", []))
        before_mode = getattr(state, "mode", None)

        append_jsonl(
            jsonl_path,
            {
                "event": "turn_start",
                "turn_id": state.turn_id,
                "mode_before": before_mode,
                "user_input": user_input,
                "ts": time.time(),
            },
        )

        # Optional repo snapshot injection for task grounding
        if cfg.inject_repo_snapshot and user_input.lstrip().lower().startswith("task:"):
            snap = repo_snapshot_text(paths, max_files=cfg.repo_snapshot_max_files)
            state.add_message("system", f"[repo_snapshot]\n{snap}", node="main")

        state.add_message("user", user_input, node="repl")

        baseline_dirty_outside_ws = dirty_outside_workspace(paths)

        attempt = 0
        raw_result: Any = None

        while True:
            attempt += 1
            t0 = time.time()

            try:
                with invoke_timeout(cfg.invoke_timeout_s):
                    raw_result = app.invoke(state, config={"recursion_limit": cfg.invoke_recursion_limit})
            except InvokeTimeout as e:
                dt = time.time() - t0
                msg = f"invoke exceeded {cfg.invoke_timeout_s}s timeout"
                state.add_message("system", f"[main] {msg}", node="main")
                print(f"llamia> [timed out] {msg}\n")
                logger.warning(f"[turn {state.turn_id}] TIMEOUT after {dt:.2f}s: {msg}")
                append_jsonl(
                    jsonl_path,
                    {
                        "event": "invoke_timeout",
                        "turn_id": state.turn_id,
                        "attempt": attempt,
                        "elapsed_s": dt,
                        "error": str(e),
                        "ts": time.time(),
                    },
                )
                raw_result = None
                break
            except KeyboardInterrupt:
                dt = time.time() - t0
                print("\nllamia> [interrupted] (Ctrl+C). You can type 'exit' to quit.\n")
                logger.warning(f"[turn {state.turn_id}] INTERRUPTED after {dt:.2f}s")
                append_jsonl(
                    jsonl_path,
                    {
                        "elapsed_s": dt,
                        "event": "invoke_interrupt_snapshot",
                        "turn_id": state.turn_id,
                        "attempt": attempt,
                        "snapshot": state_snapshot(state),
                        "improvements_patch": read_if_exists(paths, "workspace/IMPROVEMENTS.patch"),
                        "improvements_md": read_if_exists(paths, "workspace/IMPROVEMENTS.md"),
                        "ts": time.time(),
                    },
                )
                raw_result = None
                break

            dt = time.time() - t0
            state = coerce_to_state(raw_result)

            append_jsonl(
                jsonl_path,
                {
                    "event": "invoke_done",
                    "turn_id": state.turn_id,
                    "attempt": attempt,
                    "elapsed_s": dt,
                    "mode_after": getattr(state, "mode", None),
                    "ts": time.time(),
                },
            )

            # Contract validation for tasks
            if user_input.lstrip().lower().startswith("task:"):
                failures, newly_dirty = validate_task_contract(paths, user_input, baseline_dirty_outside_ws)
                if failures:
                    append_jsonl(
                        jsonl_path,
                        {
                            "failures": failures,
                            "event": "contract_fail_snapshot",
                            "turn_id": state.turn_id,
                            "attempt": attempt,
                            "snapshot": state_snapshot(state),
                            "improvements_patch": read_if_exists(paths, "workspace/IMPROVEMENTS.patch"),
                            "improvements_md": read_if_exists(paths, "workspace/IMPROVEMENTS.md"),
                            "ts": time.time(),
                        },
                    )

                    print("llamia> [contract violation] The task output did not satisfy requirements:")
                    for f in failures:
                        print(f"  - {f}")
                    print("")

                    # Revert any newly dirtied tracked files so retries are safe
                    git_restore_paths(paths, newly_dirty)

                    if attempt >= cfg.max_contract_retries:
                        state.add_message("system", "[main] Contract failed after max retries.", node="main")
                        break

                    # Create a corrective hint to push it back to reality.
                    tracked = git_ls_files(paths)
                    filtered = [
                        p
                        for p in tracked
                        if not p.startswith(("workspace/", ".venv/", ".llamia_chroma/"))
                        and not p.endswith((".bin", ".sqlite3", ".db"))
                    ]
                    tracked_hint = ""
                    if any("Patch does not touch" in f for f in failures) or any("Patch failed" in f for f in failures):
                        sample = filtered[:60]
                        if sample:
                            tracked_hint = "\nAllowed patch targets (git ls-files, filtered):\n- " + "\n- ".join(sample) + "\n"

                    fix_msg = (
                        "[main] CONTRACT VIOLATION.\n"
                        "You MUST fix the failures below, using ONLY workspace/ outputs.\n"
                        "Do NOT claim success until all are satisfied.\n\n"
                        "Failures:\n- " + "\n- ".join(failures) + "\n"
                        + tracked_hint
                        + "\nNow regenerate the required artifacts.\n"
                        "- If a unified diff was requested, it MUST modify existing git-tracked files.\n"
                        "- The patch must apply cleanly to HEAD (git apply --check) and compile (python -m compileall).\n"
                        "- IMPROVEMENTS.md must cite the exact files changed and include code excerpts.\n"
                    )

                    state.fix_instructions = fix_msg
                    _reset_turn_fields(state)
                    state.exec_request = None
                    state.last_exec_results = []
                    state.add_message("system", fix_msg, node="main")
                    continue

            break

        if raw_result is None:
            continue

        after_mode = getattr(state, "mode", None)
        new_applied = state.applied_patches[before_applied_len:] if getattr(state, "applied_patches", None) else []
        new_exec = state.exec_results[before_exec_len:] if getattr(state, "exec_results", None) else []
        new_msgs = state.messages[before_msg_len:] if getattr(state, "messages", None) else []

        if not state.messages or state.messages[-1].get("role") != "assistant":
            print("llamia> [no assistant reply produced]\n")
            append_jsonl(
                jsonl_path,
                {
                    "event": "turn_end",
                    "turn_id": state.turn_id,
                    "mode_after": after_mode,
                    "assistant": None,
                    "new_messages": new_msgs,
                    "new_applied_patches": new_applied,
                    "new_exec_results": new_exec,
                    "ts": time.time(),
                },
            )
            continue

        last = state.messages[-1]
        assistant_text = last.get("content", "")
        print(f"llamia> {assistant_text}\n")
        sys.stdout.flush()

        append_jsonl(
            jsonl_path,
            {
                "event": "turn_end",
                "turn_id": state.turn_id,
                "mode_before": before_mode,
                "mode_after": after_mode,
                "assistant": assistant_text,
                "plan": getattr(state, "plan", []),
                "exec_request": getattr(state, "exec_request", None),
                "new_messages": new_msgs,
                "new_applied_patches": new_applied,
                "new_exec_results": new_exec,
                "web_results": getattr(state, "web_results", None),
                "trace": getattr(state, "trace", None),
                "ts": time.time(),
            },
        )

        # Pretty-print task-mode side channel info (plan/files/exec results).
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

    return 0
