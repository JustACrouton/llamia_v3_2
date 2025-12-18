from __future__ import annotations

"""
Small “glue” helpers for coder node:
- allowlist filtering for commands
- formatting plan text for prompt injection
- required artifact enforcement for patch-proposal mode
"""

from typing import List

from ..state import CodePatch, PlanStep


def filter_safe_commands(cmds: list[str]) -> list[str]:
    """
    Apply an allowlist to command suggestions.

    This protects the executor layer by ensuring commands are:
    - simple
    - reviewable
    - non-shell-tricky (no pipes/redirects/&&)
    """
    allowed_prefixes = (
        "python ",
        "python3 ",
        "python -c ",
        "python3 -c ",
        "python -m ",
        "python3 -m ",
        "pytest",
        "ruff ",
        "mypy ",
        "git ",
    )

    out: list[str] = []
    for c in cmds:
        cc = c.strip()
        if not cc:
            continue
        if cc.startswith(allowed_prefixes):
            out.append(cc)
    return out


def format_plan(plan: List[PlanStep]) -> str:
    """Turn structured plan steps into a human-readable prompt block."""
    lines: list[str] = []
    for step in plan:
        lines.append(f"{step.id}. {step.description} [{step.status}]")
    return "\n".join(lines) if lines else "(no plan steps)"


def safe_pycat_command(rel_path: str) -> str:
    """
    Safe file display using python -c rather than shell tools like cat/head/tail,
    because shell tools are forbidden by policy.

    Notes:
      - Normalize backslashes to forward slashes for portability.
      - Escape single quotes because the python snippet uses single-quoted strings.
    """
    fp = rel_path.replace("\\", "/").replace("'", "\\'")
    return f"python -c \"print(open('{fp}', encoding='utf-8').read())\""


def require_patch_artifacts(patches: list[CodePatch]) -> bool:
    """
    Patch-proposal mode must create BOTH artifacts:
      - IMPROVEMENTS.patch
      - IMPROVEMENTS.md
    """
    needed = {"IMPROVEMENTS.patch", "IMPROVEMENTS.md"}
    have = {p.file_path for p in patches}
    return needed.issubset(have)
