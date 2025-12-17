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
# IMPORTANT: we only block these when they appear as SEPARATE argv tokens after shlex.split(),
# not as substrings inside quoted arguments (e.g. python -c "import os; print(...)").
DISALLOWED_ARG_TOKENS = {"&&", "||", "|", ">", "<", "`"}


def _special_case_git_diff_redirect(cmd: str) -> Path | None:
    """
    Allow exactly:
      git diff --no-color > workspace/IMPROVEMENTS.patch
    without invoking a shell redirect.
    Returns output path if matched.
    """
    normalized = " ".join(cmd.strip().split())
    if normalized == "git diff --no-color > workspace/IMPROVEMENTS.patch":
        return (ROOT_DIR / "workspace" / "IMPROVEMENTS.patch").resolve()
    return None


def _is_safe_command(cmd: str) -> bool:
    c = cmd.strip()
    if not c:
        return False

    try:
        parts = shlex.split(c)
    except Exception:
        return False

    if not parts:
        return False

    # Block shell operators only when they are separate argv tokens.
    # This prevents "python -c ... | cat" etc, while allowing semicolons inside python -c strings.
    if any(p in DISALLOWED_ARG_TOKENS for p in parts):
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
        out_path = _special_case_git_diff_redirect(cmd)
        if out_path is not None:
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    ["git", "diff", "--no-color"],
                    cwd=str(wd),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode == 0:
                    out_path.write_text(proc.stdout or "", encoding="utf-8")
                res = ExecResult(
                    command=cmd,
                    returncode=int(proc.returncode),
                    stdout=f"Wrote git diff output to {out_path.relative_to(ROOT_DIR)}",
                    stderr=proc.stderr or "",
                )
            except Exception as e:
                res = ExecResult(
                    command=cmd,
                    returncode=1,
                    stdout="",
                    stderr=f"Special-case git diff redirect failed: {e!r}",
                )

            results.append(res)
            prev_cmd, prev_rc = cmd, res.returncode
            continue

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
