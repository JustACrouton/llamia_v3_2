from __future__ import annotations

"""
Git helpers for the coder node.

In patch-proposal mode we need to:
- list git-tracked files (to prevent hallucinated diff targets)
- keep the prompt small (filtered list)
"""

from pathlib import Path
from typing import List


def repo_root() -> Path:
    """
    nodes/ -> llamia_v3_2/ -> repo root
    """
    return Path(__file__).resolve().parents[2]


def git_ls_files_all() -> list[str]:
    """
    Return ALL git-tracked file paths (repo-relative).

    Safe failure: returns [] if git is missing or repo isn't a git checkout.
    """
    try:
        import subprocess

        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(repo_root()))
    except Exception:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def git_ls_files_filtered(*, limit: int = 200) -> list[str]:
    """
    Return a filtered subset of git-tracked file paths to keep prompts small.
    """
    try:
        import subprocess

        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(repo_root()))
    except Exception:
        return []

    files = [ln.strip() for ln in out.splitlines() if ln.strip()]

    skip_prefixes = ("workspace/", ".llamia_chroma/", ".venv/")
    skip_exts = (".bin", ".sqlite3", ".db")

    filtered: List[str] = []
    for f in files:
        if any(f.startswith(p) for p in skip_prefixes):
            continue
        if any(f.endswith(ext) for ext in skip_exts):
            continue
        filtered.append(f)
        if len(filtered) >= limit:
            break
    return filtered
