#!/usr/bin/env python3
from __future__ import annotations

from typing import Any
from collections import Counter
from dataclasses import asdict, is_dataclass

import sys
import signal
import select
import json
import time
import logging
import re
import subprocess
from pathlib import Path
from datetime import datetime

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
INVOKE_TIMEOUT_S = 45              # wall-clock cap per turn (Linux/macOS only)

# How many times main.py will auto-retry to satisfy the task contract
MAX_CONTRACT_RETRIES = 2

# Repo snapshot injected into task prompts to reduce hallucinations
INJECT_REPO_SNAPSHOT = True
REPO_SNAPSHOT_MAX_FILES = 250


class InvokeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise InvokeTimeout(f"invoke exceeded {INVOKE_TIMEOUT_S}s timeout")


def _set_alarm(seconds: int):
    # SIGALRM works on Linux/macOS. If unavailable (e.g., Windows), we skip.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(seconds)))


def _clear_alarm():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


# -----------------------------
# Logging
# -----------------------------
def _safe_to_json(obj: Any) -> Any:
    """Convert objects to something JSON serializable."""
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_to_json(x) for x in obj]
    return str(obj)


def _setup_run_logger() -> tuple[logging.Logger, Path, Path]:
    log_dir = Path("workspace") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = log_dir / f"run_{stamp}.log"
    jsonl_path = log_dir / f"run_{stamp}.jsonl"

    logger = logging.getLogger("llamia")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(text_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    # IMPORTANT: do NOT log to stdout or it will corrupt the "you>" prompt.
    # If you want console logs, send to stderr and keep it WARNING+.
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)

    # Print log paths manually (stdout) once at startup
    print(f"[log] text:  {text_path}")
    print(f"[log] jsonl: {jsonl_path}")

    return logger, text_path, jsonl_path


def _append_jsonl(jsonl_path: Path, record: dict[str, Any]) -> None:
    record = _safe_to_json(record)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Git / repo helpers
# -----------------------------
def _run_git(args: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "git not found"


def _git_ls_files() -> list[str]:
    rc, out, _ = _run_git(["ls-files"])
    if rc != 0:
        return []
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return files


def _git_status_porcelain() -> list[str]:
    rc, out, _ = _run_git(["status", "--porcelain"])
    if rc != 0:
        return []
    return [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]



def _porcelain_paths(lines: list[str]) -> set[str]:
    paths: set[str] = set()
    for ln in lines:
        if len(ln) < 4:
            continue
        path = ln[3:].strip()
        # handle renames: "R  old -> new"
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            paths.add(old.strip())
            paths.add(new.strip())
        else:
            paths.add(path)
    return paths


def _dirty_outside_workspace() -> set[str]:
    lines = _git_status_porcelain()
    paths = _porcelain_paths(lines)
    return {p for p in paths if p and not p.startswith("workspace/")}


def _git_restore_paths(paths: set[str]) -> None:
    if not paths:
        return
    plist = sorted(paths)

    # Try modern restore first
    rc, _, _ = _run_git(["restore", "--staged", "--worktree", "--", *plist])
    if rc == 0:
        return

    # Fallback for older git
    _run_git(["reset", "--", *plist])
    _run_git(["checkout", "--", *plist])

def _repo_snapshot_text(max_files: int = REPO_SNAPSHOT_MAX_FILES) -> str:
    files = _git_ls_files()
    if not files:
        # fallback: non-git directory
        root = Path(".")
        files = []
        for p in root.rglob("*"):
            if p.is_file():
                s = str(p)
                files.append(s)
                if len(files) >= max_files * 3:
                    # collect a bit extra before filtering
                    break

    # Filter out noisy/binary/large artifacts so the model doesn't get distracted.
    skip_prefixes = (
        ".venv/",
        "workspace/logs/",
        "workspace/.venv/",
        ".llamia_chroma/",
    )
    skip_exts = (
        ".bin",
        ".sqlite3",
        ".db",
        ".pkl",
        ".pt",
        ".onnx",
    )

    filtered: list[str] = []
    for s in files:
        s = str(s)
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        if "/__pycache__/" in s or s.endswith("/__pycache__"):
            continue
        if any(s.endswith(ext) for ext in skip_exts):
            continue
        filtered.append(s)

    files = sorted(filtered)[:max_files]
    return "Repo files (truncated):\n" + "\n".join(f"- {f}" for f in files)


# -----------------------------
# Task contract checks
# -----------------------------
_WS_PATH_RE = re.compile(r"(workspace/[A-Za-z0-9._\-\/]+)")

def _extract_required_workspace_paths(user_input: str) -> list[str]:
    # Any explicit "workspace/..." is treated as required output
    found = _WS_PATH_RE.findall(user_input)
    # normalize + dedup while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        p = p.strip().rstrip(".")
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _prompt_requests_patch(user_input: str) -> bool:
    s = user_input.lower()
    return ("improvements.patch" in s) or ("unified diff" in s) or ("git style" in s) or ("create workspace/" in s and ".patch" in s)


def _patch_touches_tracked_files(patch_text: str, tracked_files: set[str]) -> bool:
    # Look for: diff --git a/<path> b/<path>
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/").strip()
                b_path = parts[3].removeprefix("b/").strip()
                if a_path in tracked_files or b_path in tracked_files:
                    return True
    return False


def _validate_task_contract(user_input: str, baseline_dirty_outside_ws: set[str] | None = None) -> tuple[list[str], set[str]]:
    failures: list[str] = []

    # 1) Required workspace artifacts
    required = _extract_required_workspace_paths(user_input)
    for rel in required:
        p = Path(rel)
        if not p.exists():
            failures.append(f"Missing required file: {rel}")

    # 2) "Do not modify tracked files" enforcement (baseline-aware)
    newly_dirty: set[str] = set()
    if "do not modify" in user_input.lower() and ("tracked file" in user_input.lower() or "repo code" in user_input.lower()):
        before = baseline_dirty_outside_ws or set()
        after = _dirty_outside_workspace()
        newly_dirty = after - before
        if newly_dirty:
            failures.append("Modified tracked files unexpectedly (new this turn):\n" + "\n".join(sorted(newly_dirty)))

# 3) Patch sanity (if prompt asked for a patch)
    if _prompt_requests_patch(user_input):
        # Find the most likely patch file requested
        patch_paths = [p for p in required if p.lower().endswith(".patch")]
        if patch_paths:
            patch_path = Path(patch_paths[0])
            if patch_path.exists():
                txt = patch_path.read_text(encoding="utf-8", errors="replace")
                tracked = set(_git_ls_files())
                if tracked and not _patch_touches_tracked_files(txt, tracked):
                    failures.append(
                        f"Patch does not touch any existing git-tracked files (likely hallucinated / irrelevant). "
                        f"Regenerate {patch_path.as_posix()} to modify real files from git ls-files."
                    )
            else:
                failures.append(f"Patch file not created: {patch_paths[0]}")

    return failures, newly_dirty


# -----------------------------
# State coercion helpers
# -----------------------------
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

            web_search_count=int(raw.get("web_search_count", 0) or 0),

            turn_id=turn_id,
            responded_turn_id=responded_turn_id,
        )

    state = LlamiaState()
    state.log(f"[main] WARNING: unexpected state type from graph: {type(raw)!r}")
    return state


# -----------------------------
# Paste-aware input
# -----------------------------
def _read_user_input_block() -> str | None:
    """Read one user message; drain any immediately-buffered paste lines."""
    try:
        first = input("you> ")
    except (EOFError, KeyboardInterrupt):
        return None

    if not first.strip():
        return ""

    lines = [first]

    # Drain already-buffered stdin (common when pasting multi-line tasks)
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not r:
            break
        nxt = sys.stdin.readline()
        if not nxt:
            break
        lines.append(nxt.rstrip("\n"))

    return "\n".join(lines).rstrip()


# -----------------------------
# REPL
# -----------------------------
def run_repl():
    logger, _text_path, jsonl_path = _setup_run_logger()

    state = LlamiaState()
    app = build_llamia_graph()

    if not hasattr(state, "turn_id"):
        state.turn_id = 0
    if not hasattr(state, "responded_turn_id"):
        state.responded_turn_id = -1

    print("Llamia v3.2 (LangGraph + planner + coder + executor + chat). Type 'exit' to quit.\n")
    print("Tips:")
    print("  - Normal message: regular chat mode")
    print("  - 'task: build me X': task mode => planner, coder (writes into workspace/), executor (runs safe commands), then chat.\n")

    while True:
        user_input = _read_user_input_block()
        if user_input is None:
            print("\nBye.")
            logger.info("[repl] exit at prompt")
            _append_jsonl(jsonl_path, {"event": "repl_exit", "reason": "prompt_exit", "ts": time.time()})
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            logger.info("[repl] user requested exit")
            _append_jsonl(jsonl_path, {"event": "repl_exit", "reason": "user_exit", "ts": time.time()})
            break

        # Turn bookkeeping
        state.turn_id += 1
        state.responded_turn_id = -1

        before_applied_len = len(getattr(state, "applied_patches", []))
        before_exec_len = len(getattr(state, "exec_results", []))
        before_msg_len = len(getattr(state, "messages", []))
        before_mode = getattr(state, "mode", None)

        _append_jsonl(jsonl_path, {
            "event": "turn_start",
            "turn_id": state.turn_id,
            "mode_before": before_mode,
            "user_input": user_input,
            "ts": time.time(),
        })

        # Inject repo snapshot for tasks (helps reduce hallucinations)
        if INJECT_REPO_SNAPSHOT and user_input.lstrip().lower().startswith("task:"):
            snap = _repo_snapshot_text()
            state.add_message("system", f"[repo_snapshot]\n{snap}", node="main")

        state.add_message("user", user_input, node="repl")

        # Baseline git dirtiness outside workspace (so we only flag NEW changes per turn)
        baseline_dirty_outside_ws = _dirty_outside_workspace()

        # Controlled contract retries
        attempt = 0
        raw_result: Any = None
        while True:
            attempt += 1
            t0 = time.time()
            try:
                _set_alarm(INVOKE_TIMEOUT_S)
                raw_result = app.invoke(
                    state,
                    config={"recursion_limit": INVOKE_RECURSION_LIMIT},
                )
            except InvokeTimeout as e:
                dt = time.time() - t0
                state.add_message("system", f"[main] {e}", node="main")
                print(f"llamia> [timed out] {e}\n")
                logger.warning(f"[turn {state.turn_id}] TIMEOUT after {dt:.2f}s: {e}")
                _append_jsonl(jsonl_path, {
                    "event": "invoke_timeout",
                    "turn_id": state.turn_id,
                    "attempt": attempt,
                    "elapsed_s": dt,
                    "error": str(e),
                    "ts": time.time(),
                })
                raw_result = None
                break
            except KeyboardInterrupt:
                dt = time.time() - t0
                print("\nllamia> [interrupted] (Ctrl+C). You can type 'exit' to quit.\n")
                logger.warning(f"[turn {state.turn_id}] INTERRUPTED after {dt:.2f}s")
                _append_jsonl(jsonl_path, {
                    "event": "invoke_interrupt",
                    "turn_id": state.turn_id,
                    "attempt": attempt,
                    "elapsed_s": dt,
                    "ts": time.time(),
                })
                raw_result = None
                break
            finally:
                _clear_alarm()

            dt = time.time() - t0
            state = _coerce_to_state(raw_result)

            _append_jsonl(jsonl_path, {
                "event": "invoke_done",
                "turn_id": state.turn_id,
                "attempt": attempt,
                "elapsed_s": dt,
                "mode_after": getattr(state, "mode", None),
                "ts": time.time(),
            })

            # If this was a task, enforce the contract
            if user_input.lstrip().lower().startswith("task:"):
                failures, newly_dirty = _validate_task_contract(user_input, baseline_dirty_outside_ws)
                if failures:
                    _append_jsonl(jsonl_path, {
                        "event": "contract_fail",
                        "turn_id": state.turn_id,
                        "attempt": attempt,
                        "failures": failures,
                        "ts": time.time(),
                    })
                    # Print one clear line so you see it's not "SUCCESS"
                    print("llamia> [contract violation] The task output did not satisfy requirements:")
                    for f in failures:
                        print(f"  - {f}")
                    print("")

                    # Revert any newly dirtied tracked files so retries are safe
                    _git_restore_paths(newly_dirty)

                    if attempt >= MAX_CONTRACT_RETRIES:
                        # Stop retrying; user can inspect logs/files
                        state.add_message("system", "[main] Contract failed after max retries.", node="main")
                        break

                    # Add a hard corrective instruction and retry once more


                    # IMPORTANT: ensure the next invoke actually re-enters the task graph.


                    tracked = _git_ls_files()


                    filtered = []


                    for p in tracked:


                        if p.startswith(("workspace/", ".venv/", ".llamia_chroma/")):


                            continue


                        if p.endswith((".bin", ".sqlite3", ".db")):


                            continue


                        filtered.append(p)


                    tracked_hint = ""


                    if any("Patch does not touch" in f for f in failures) or any("patch" in f.lower() for f in failures):


                        sample = filtered[:60]


                        if sample:


                            tracked_hint = "\nAllowed patch targets (git ls-files, filtered):\n- " + "\n- ".join(sample) + "\n"



                    fix_msg = (


                        "[main] CONTRACT VIOLATION.\n"


                        "You MUST fix the failures below, using ONLY workspace/ outputs.\n"


                        "Do NOT claim success until all are satisfied.\n\n"


                        "Failures:\n- " + "\n- ".join(failures) + "\n"


                        + tracked_hint + "\n"


                        "Now regenerate the required artifacts.\n"


                        "- If a unified diff was requested, it MUST modify existing git-tracked files (no invented hello.py).\n"


                        "- Use only files under the repo root (never workspace/ as patch targets).\n"


                    )


                    state.fix_instructions = fix_msg


                    # Reset loop counters so critic doesn't instantly bail.


                    state.loop_count = 0


                    state.web_search_count = 0


                    state.web_queue = []


                    state.web_results = None


                    state.research_query = None


                    state.research_notes = None


                    state.exec_request = None


                    state.last_exec_results = []


                    state.add_message("system", fix_msg, node="main")


                    continue

            # Contract OK (or not a task)
            break

        if raw_result is None:
            # timeout/interrupt path
            continue

        after_mode = getattr(state, "mode", None)
        new_applied = state.applied_patches[before_applied_len:] if getattr(state, "applied_patches", None) else []
        new_exec = state.exec_results[before_exec_len:] if getattr(state, "exec_results", None) else []
        new_msgs = state.messages[before_msg_len:] if getattr(state, "messages", None) else []

        # Print last assistant message if present
        if not state.messages or state.messages[-1].get("role") != "assistant":
            print("llamia> [no assistant reply produced]\n")
            _append_jsonl(jsonl_path, {
                "event": "turn_end",
                "turn_id": state.turn_id,
                "mode_after": after_mode,
                "assistant": None,
                "new_messages": new_msgs,
                "new_applied_patches": new_applied,
                "new_exec_results": new_exec,
                "ts": time.time(),
            })
            continue

        last = state.messages[-1]
        assistant_text = last.get("content", "")
        print(f"llamia> {assistant_text}\n")
        sys.stdout.flush()

        _append_jsonl(jsonl_path, {
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
        })

        # Task-mode summary prints
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
