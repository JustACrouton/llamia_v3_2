#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import logging
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from llamia_v3_2.graph import build_llamia_graph
from llamia_v3_2.state import CodePatch, ExecRequest, ExecResult, LlamiaState, PlanStep

# -----------------------------
# Invoke safety limits
# -----------------------------
INVOKE_RECURSION_LIMIT = 100  # hard cap on langgraph recursion
INVOKE_TIMEOUT_S = 600  # wall-clock cap per turn (Linux/macOS only)

# How many times main.py will auto-retry to satisfy the task contract
MAX_CONTRACT_RETRIES = 10

# Repo snapshot injected into task prompts to reduce hallucinations
INJECT_REPO_SNAPSHOT = True
REPO_SNAPSHOT_MAX_FILES = 250

REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = (REPO_ROOT / "workspace").resolve()

# -----------------------------
# Paths & diffs
# -----------------------------
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_WS_PATH_RE = re.compile(r"(workspace/[A-Za-z0-9._\-\/]+)")


def _abs_repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _extract_required_workspace_paths(user_input: str) -> list[str]:
    """Any explicit 'workspace/...' in the user input is treated as required output."""
    found = _WS_PATH_RE.findall(user_input)
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
    return (
        "improvements.patch" in s
        or "unified diff" in s
        or "git style" in s
        or ("create workspace/" in s and ".patch" in s)
    )


def _patch_touched_files(patch_text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        a = m.group(1).strip()
        if a and a != "/dev/null" and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def _patch_has_substantive_changes(patch_text: str) -> bool:
    """Reject whitespace-only / metadata-only patches."""
    for ln in patch_text.splitlines():
        if ln.startswith(("diff --git ", "index ", "--- ", "+++ ", "@@")):
            continue
        if ln.startswith(("+", "-")):
            body = ln[1:]
            if body.strip():
                return True
    return False


# -----------------------------
# Invoke timeout
# -----------------------------
class InvokeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise InvokeTimeout(f"invoke exceeded {INVOKE_TIMEOUT_S}s timeout")


def _set_alarm(seconds: int) -> None:
    # SIGALRM works on Linux/macOS. If unavailable (e.g., Windows), we skip.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(seconds)))


def _clear_alarm() -> None:
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
    log_dir = WORKSPACE_DIR / "logs"
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
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
def _tail_lines(s: str, max_chars: int = 4000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def _read_if_exists(rel_path: str, max_chars: int = 8000) -> str | None:
    p = _abs_repo_path(rel_path)
    if not p.exists():
        return None
    try:
        return _tail_lines(p.read_text(encoding="utf-8", errors="replace"), max_chars=max_chars)
    except Exception as e:
        return f"[read_error] {e!r}"


def _state_snapshot(state: LlamiaState) -> dict[str, Any]:
    """
    Compact snapshot suitable for JSONL:
    - routing info
    - last N messages (trimmed)
    - trace (if used)
    - last exec results
    """
    msgs = []
    for m in (state.messages or [])[-12:]:
        msgs.append({
            "role": m.get("role"),
            "node": m.get("node"),
            "content": _tail_lines((m.get("content") or "").strip(), max_chars=2000),
        })

    last_exec = []
    for r in (getattr(state, "last_exec_results", None) or [])[-6:]:
        last_exec.append({
            "command": r.command,
            "returncode": r.returncode,
            "stdout_tail": _tail_lines((r.stdout or "").strip(), max_chars=1500),
            "stderr_tail": _tail_lines((r.stderr or "").strip(), max_chars=1500),
        })

    return {
        "mode": getattr(state, "mode", None),
        "goal": getattr(state, "goal", None),
        "next_agent": getattr(state, "next_agent", None),
        "loop_count": getattr(state, "loop_count", None),
        "web_search_count": getattr(state, "web_search_count", None),
        "research_query": getattr(state, "research_query", None),
        "trace": getattr(state, "trace", None),
        "messages_tail": msgs,
        "exec_request": getattr(state, "exec_request", None),
        "last_exec_results_tail": last_exec,
    }


# -----------------------------
# Git / repo helpers
# -----------------------------
def _run_git(args: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
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
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


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
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            paths.add(old.strip())
            paths.add(new.strip())
        else:
            paths.add(path)
    return paths


def _dirty_outside_workspace() -> set[str]:
    paths = _porcelain_paths(_git_status_porcelain())
    return {p for p in paths if p and not p.startswith("workspace/")}


def _git_restore_paths(paths: set[str]) -> None:
    if not paths:
        return
    plist = sorted(paths)

    rc, _, _ = _run_git(["restore", "--staged", "--worktree", "--", *plist])
    if rc == 0:
        return

    _run_git(["reset", "--", *plist])
    _run_git(["checkout", "--", *plist])


def _repo_snapshot_text(max_files: int = REPO_SNAPSHOT_MAX_FILES) -> str:
    files = _git_ls_files()
    if not files:
        # fallback: non-git directory
        files = []
        for p in REPO_ROOT.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(REPO_ROOT)))
                if len(files) >= max_files * 3:
                    break

    skip_prefixes = (
        ".venv/",
        "workspace/logs/",
        "workspace/.venv/",
        ".llamia_chroma/",
    )
    skip_exts = (".bin", ".sqlite3", ".db", ".pkl", ".pt", ".onnx")

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
# Step 2: Patch verification in isolated worktree
# -----------------------------
def _check_patch_in_clean_worktree(patch_abs: Path) -> tuple[bool, str]:
    """
    Verify patch applies cleanly to HEAD, then compileall in an isolated worktree.
    Returns (ok, details).
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    base = Path(tempfile.mkdtemp(prefix="llamia_applycheck_", dir=str(WORKSPACE_DIR)))
    wt_dir = base / "wt"  # must NOT exist before `git worktree add`

    try:
        rc, out, err = _run_git(["worktree", "add", "--detach", str(wt_dir), "HEAD"])
        if rc != 0:
            return False, f"git worktree add failed:\n{err or out}"

        p = subprocess.run(
            ["git", "apply", "--check", str(patch_abs)],
            cwd=str(wt_dir),
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return False, f"git apply --check failed:\n{p.stderr or p.stdout}"

        p2 = subprocess.run(
            ["git", "apply", str(patch_abs)],
            cwd=str(wt_dir),
            capture_output=True,
            text=True,
        )
        if p2.returncode != 0:
            return False, f"git apply failed in worktree:\n{p2.stderr or p2.stdout}"

        p3 = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "."],
            cwd=str(wt_dir),
            capture_output=True,
            text=True,
        )
        if p3.returncode != 0:
            return False, f"compileall failed in worktree:\n{p3.stderr or p3.stdout}"

        return True, "ok"
    finally:
        # best-effort cleanup
        try:
            _run_git(["worktree", "remove", "--force", str(wt_dir)])
        except Exception:
            pass
        shutil.rmtree(base, ignore_errors=True)


# -----------------------------
# Step 3: Grounding requirements for IMPROVEMENTS.md
# -----------------------------
def _check_improvements_md_grounding(md_abs: Path, touched_files: list[str]) -> list[str]:
    fails: list[str] = []
    if not md_abs.exists():
        return [f"Missing required file: {md_abs.relative_to(REPO_ROOT).as_posix()}"]

    md = md_abs.read_text(encoding="utf-8", errors="replace")
    low = md.lower()

    if "root cause" not in low:
        fails.append("IMPROVEMENTS.md must include a Root Cause section.")
    if "verif" not in low:
        fails.append("IMPROVEMENTS.md must include Verification steps.")
    if "```" not in md:
        fails.append("IMPROVEMENTS.md must include at least one fenced code block with an excerpt.")

    # Require it to reference the files it claims to modify
    for fp in touched_files:
        if fp not in md:
            fails.append(f"IMPROVEMENTS.md must mention touched file path: {fp}")

    return fails


# -----------------------------
# Task contract checks
# -----------------------------
def _patch_touches_tracked_files(patch_text: str, tracked_files: set[str]) -> bool:
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/").strip()
                b_path = parts[3].removeprefix("b/").strip()
                if a_path in tracked_files or b_path in tracked_files:
                    return True
    return False


def _validate_task_contract(
    user_input: str,
    baseline_dirty_outside_ws: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    failures: list[str] = []

    required = _extract_required_workspace_paths(user_input)
    for rel in required:
        p = _abs_repo_path(rel)
        if not p.exists():
            failures.append(f"Missing required file: {rel}")

    newly_dirty: set[str] = set()
    if "do not modify" in user_input.lower() and ("tracked file" in user_input.lower() or "repo code" in user_input.lower()):
        before = baseline_dirty_outside_ws or set()
        after = _dirty_outside_workspace()
        newly_dirty = after - before
        if newly_dirty:
            failures.append("Modified tracked files unexpectedly (new this turn):\n" + "\n".join(sorted(newly_dirty)))

    if _prompt_requests_patch(user_input):
        patch_paths = [p for p in required if p.lower().endswith(".patch")]
        if not patch_paths:
            patch_paths = ["workspace/IMPROVEMENTS.patch"]

        patch_rel = patch_paths[0]
        patch_abs = _abs_repo_path(patch_rel)

        if not patch_abs.exists():
            failures.append(f"Patch file not created: {patch_rel}")
        else:
            txt = patch_abs.read_text(encoding="utf-8", errors="replace")
            tracked = set(_git_ls_files())

            if tracked and not _patch_touches_tracked_files(txt, tracked):
                failures.append(
                    "Patch does not touch any existing git-tracked files (likely hallucinated / irrelevant). "
                    f"Regenerate {patch_rel} to modify real files from git ls-files."
                )

            if not _patch_has_substantive_changes(txt):
                failures.append("Patch contains no substantive (+/-) changes (looks whitespace-only or metadata-only).")

            ok, detail = _check_patch_in_clean_worktree(patch_abs)
            if not ok:
                failures.append("Patch failed clean-worktree verification:\n" + detail)

            touched = _patch_touched_files(txt)
            md_paths = [p for p in required if p.lower().endswith(".md")]
            md_rel = md_paths[0] if md_paths else "workspace/IMPROVEMENTS.md"
            md_abs = _abs_repo_path(md_rel)
            failures.extend(_check_improvements_md_grounding(md_abs, touched))

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
def run_repl() -> None:
    logger, _text_path, jsonl_path = _setup_run_logger()

    state = LlamiaState()
    app = build_llamia_graph()

    # backward-compat for older state objects
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
        state.web_search_count = 0
        state.web_results = None
        state.research_query = None
        state.research_notes = None
        state.web_queue = []
        state.return_after_web = "planner"
        state.loop_count = 0

        before_applied_len = len(getattr(state, "applied_patches", []))
        before_exec_len = len(getattr(state, "exec_results", []))
        before_msg_len = len(getattr(state, "messages", []))
        before_mode = getattr(state, "mode", None)

        _append_jsonl(
            jsonl_path,
            {
                "event": "turn_start",
                "turn_id": state.turn_id,
                "mode_before": before_mode,
                "user_input": user_input,
                "ts": time.time(),
            },
        )

        # Inject repo snapshot for tasks (helps reduce hallucinations)
        if INJECT_REPO_SNAPSHOT and user_input.lstrip().lower().startswith("task:"):
            snap = _repo_snapshot_text()
            state.add_message("system", f"[repo_snapshot]\n{snap}", node="main")

        state.add_message("user", user_input, node="repl")

        baseline_dirty_outside_ws = _dirty_outside_workspace()

        attempt = 0
        raw_result: Any = None

        while True:
            attempt += 1
            t0 = time.time()

            try:
                _set_alarm(INVOKE_TIMEOUT_S)
                raw_result = app.invoke(state, config={"recursion_limit": INVOKE_RECURSION_LIMIT})
            except InvokeTimeout as e:
                dt = time.time() - t0
                state.add_message("system", f"[main] {e}", node="main")
                print(f"llamia> [timed out] {e}\n")
                logger.warning(f"[turn {state.turn_id}] TIMEOUT after {dt:.2f}s: {e}")
                _append_jsonl(
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
                _append_jsonl(
                    jsonl_path,
                    {
                        "elapsed_s": dt,
                        "event": "invoke_interrupt_snapshot",
                        "turn_id": state.turn_id,
                        "attempt": attempt,
                        "snapshot": _state_snapshot(state),
                        "improvements_patch": _read_if_exists("workspace/IMPROVEMENTS.patch"),
                        "improvements_md": _read_if_exists("workspace/IMPROVEMENTS.md"),
                        "ts": time.time(),
                    },
                )
                raw_result = None
                break
            finally:
                _clear_alarm()

            dt = time.time() - t0
            state = _coerce_to_state(raw_result)

            _append_jsonl(
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

            if user_input.lstrip().lower().startswith("task:"):
                failures, newly_dirty = _validate_task_contract(user_input, baseline_dirty_outside_ws)
                if failures:
                    _append_jsonl(
                        jsonl_path,
                        {
                            "failures": failures,
                            "event": "contract_fail_snapshot",
                            "turn_id": state.turn_id,
                            "attempt": attempt,
                            "snapshot": _state_snapshot(state),
                            "improvements_patch": _read_if_exists("workspace/IMPROVEMENTS.patch"),
                            "improvements_md": _read_if_exists("workspace/IMPROVEMENTS.md"),
                            "ts": time.time(),
                        },
                    )
                    print("llamia> [contract violation] The task output did not satisfy requirements:")
                    for f in failures:
                        print(f"  - {f}")
                    print("")

                    # Revert any newly dirtied tracked files so retries are safe
                    _git_restore_paths(newly_dirty)

                    if attempt >= MAX_CONTRACT_RETRIES:
                        state.add_message("system", "[main] Contract failed after max retries.", node="main")
                        break

                    # Build a corrective instruction that nudges it back to reality
                    tracked = _git_ls_files()
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

            break

        if raw_result is None:
            continue

        after_mode = getattr(state, "mode", None)
        new_applied = state.applied_patches[before_applied_len:] if getattr(state, "applied_patches", None) else []
        new_exec = state.exec_results[before_exec_len:] if getattr(state, "exec_results", None) else []
        new_msgs = state.messages[before_msg_len:] if getattr(state, "messages", None) else []

        if not state.messages or state.messages[-1].get("role") != "assistant":
            print("llamia> [no assistant reply produced]\n")
            _append_jsonl(
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

        _append_jsonl(
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
