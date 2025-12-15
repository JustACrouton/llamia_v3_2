from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from ..state import ExecRequest, ExecResult
from .fs_tools import ROOT_DIR

# ---- Safety policy (tweak as needed) ----
ALLOWED_BINARIES = {
    "python", "python3",
    "pytest",
    "ruff",
    "mypy",
    "git",
}

# Disallow shell metacharacters (no chaining / redirects).
DISALLOWED_TOKENS = ["&&", "||", "|", ";", ">", "<", "`", "$(", "${"]


def _is_safe_command(cmd: str) -> bool:
    c = cmd.strip()
    if not c:
        return False
    for tok in DISALLOWED_TOKENS:
        if tok in c:
            return False

    try:
        parts = shlex.split(c)
    except Exception:
        return False

    if not parts:
        return False

    exe = parts[0]

    # Extra safety for git: allow only read-only-ish commands + apply --check
    if exe == "git":
        if len(parts) < 2:
            return False
        sub = parts[1]
        allowed_sub = {"status", "diff", "ls-files", "apply"}
        if sub not in allowed_sub:
            return False
        if sub == "apply":
            # Only allow dry-run validation; never apply changes here.
            if "--check" not in parts:
                return False
            # Disallow flags that can write reject files or bypass path safety.
            if any(f in parts for f in ["--reject", "--unsafe-paths"]):
                return False
        return True

    return exe in ALLOWED_BINARIES


def _resolve_workdir(workdir: str) -> Path:
    wd = (ROOT_DIR / workdir).resolve()
    root = ROOT_DIR.resolve()
    if root not in wd.parents and wd != root:
        raise ValueError(f"Unsafe workdir escapes repo: {workdir!r}")
    return wd


def _is_python_fallback(prev_cmd: str, next_cmd: str) -> bool:
    """
    Detect: python X -> python3 X
    Only exact argv match except for python/python3.
    """
    try:
        a = shlex.split(prev_cmd)
        b = shlex.split(next_cmd)
    except Exception:
        return False

    if not a or not b:
        return False
    if a[0] != "python" or b[0] != "python3":
        return False
    return a[1:] == b[1:]


def _normalize_argv(argv: list[str]) -> list[str]:
    """
    Make python invocations use the current venv interpreter reliably.
    """
    if not argv:
        return argv
    if argv[0] in ("python", "python3"):
        argv = [sys.executable] + argv[1:]
    return argv


def run_exec_request(req: ExecRequest) -> list[ExecResult]:
    """
    Run commands sequentially (safer + enables fallback semantics).
    Applies a safety filter and python->python3 fallback skip rule.
    """
    wd = _resolve_workdir(req.workdir)
    results: list[ExecResult] = []

    prev_cmd: str | None = None
    prev_rc: int | None = None

    for cmd in req.commands:
        cmd = str(cmd).strip()
        if not cmd:
            continue

        # Skip python3 fallback if python succeeded for same args
        if prev_cmd is not None and prev_rc == 0 and _is_python_fallback(prev_cmd, cmd):
            continue

        if not _is_safe_command(cmd):
            results.append(
                ExecResult(
                    command=cmd,
                    returncode=126,
                    stdout="",
                    stderr="Blocked by safety filter (disallowed command or shell metacharacters).",
                )
            )
            prev_cmd, prev_rc = cmd, 126
            continue

        try:
            argv = _normalize_argv(shlex.split(cmd))

            proc = subprocess.run(
                argv,
                cwd=str(wd),
                capture_output=True,
                text=True,
                timeout=120,
            )
            res = ExecResult(
                command=cmd,
                returncode=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )
        except subprocess.TimeoutExpired as e:
            res = ExecResult(
                command=cmd,
                returncode=124,
                stdout=(e.stdout or "") if hasattr(e, "stdout") else "",
                stderr="Command timed out.",
            )
        except FileNotFoundError:
            res = ExecResult(
                command=cmd,
                returncode=127,
                stdout="",
                stderr="Executable not found.",
            )
        except Exception as e:
            res = ExecResult(
                command=cmd,
                returncode=1,
                stdout="",
                stderr=f"Executor exception: {e!r}",
            )

        results.append(res)
        prev_cmd, prev_rc = cmd, res.returncode

    return results
