from __future__ import annotations

"""
Patch grounding helpers.

Patch-proposal tasks are error-prone because the model may hallucinate
file contents or diff contexts.

This module:
- extracts likely target file paths from the goal
- reads short numbered excerpts from real repo files
- builds a system prompt block the model can rely on for correct hunks
"""

import re
from pathlib import Path
from typing import List

from .coder_git import repo_root


def is_patch_task(goal: str) -> bool:
    """
    Heuristic: treat the goal as "patch proposal mode" if it asks for a unified diff / .patch.
    """
    g = (goal or "").lower()
    return any(k in g for k in ["unified diff", "diff --git", "git style", ".patch", "improvements.patch"])


_FILE_IN_GOAL_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\\.(?:py|md|toml|yaml|yml|txt))")


def extract_paths_from_goal(goal: str, tracked: set[str]) -> list[str]:
    """
    Extract file paths mentioned in the goal and keep only git-tracked paths.

    This prevents "diff --git a/fake.py b/fake.py" patches.
    """
    hits: list[str] = []
    for m in _FILE_IN_GOAL_RE.finditer(goal or ""):
        p = m.group("path").strip().replace("\\\\", "/")
        if p in tracked and p not in hits:
            hits.append(p)
    return hits


def read_numbered_window(
    relpath: str,
    *,
    anchor_patterns: list[str] | None = None,
    window_before: int = 60,
    window_after: int = 120,
    fallback_max_lines: int = 180,
    max_chars: int = 12000,
) -> str:
    """
    Read a RELEVANT excerpt:
      - If anchor_patterns match a line, return a window around the first match.
      - Else return the first fallback_max_lines lines.

    Always line-number the excerpt so the model can verify what it's seeing.
    IMPORTANT: instruct the model to NOT include line numbers in diff context.
    """
    p = (repo_root() / relpath).resolve()
    try:
        text = p.read_text(encoding="utf-8", errors="replace").replace("\\r\\n", "\\n")
    except Exception as e:
        return f"[could not read {relpath}: {e!r}]"

    lines = text.splitlines()

    start = 0
    end = min(len(lines), fallback_max_lines)

    if anchor_patterns:
        regs = []
        for pat in anchor_patterns:
            try:
                regs.append(re.compile(pat))
            except re.error:
                continue

        hit_idx: int | None = None
        for i, ln in enumerate(lines):
            if any(r.search(ln) for r in regs):
                hit_idx = i
                break

        if hit_idx is not None:
            start = max(0, hit_idx - window_before)
            end = min(len(lines), hit_idx + window_after)

    chunk = lines[start:end]
    numbered = "\\n".join(f"{start + i + 1:04d} {ln}" for i, ln in enumerate(chunk))
    if len(numbered) > max_chars:
        numbered = numbered[:max_chars] + "\\n...[truncated]"
    return numbered


def build_patch_context(goal: str, tracked_files: list[str]) -> str:
    """
    Provide grounded excerpts of relevant files so patch hunks match reality.
    This reduces fake diffs / wrong line contexts.
    """
    tracked_set = set(tracked_files)
    targets = extract_paths_from_goal(goal, tracked_set)

    # Sensible defaults (short list, likely high-impact)
    defaults = [
        "llamia_v3_2/repl/app.py",
        "llamia_v3_2/nodes/critic.py",
        "llamia_v3_2/nodes/intent_router.py",
    ]
    for d in defaults:
        if d in tracked_set and d not in targets:
            targets.append(d)
        if len(targets) >= 4:
            break

    if not targets:
        return ""

    out: List[str] = []
    out.append("[patch_context] Authoritative repo excerpts (use for exact diff hunks).\\n")
    out.append("Do NOT invent code. Patch must match these files exactly.\\n")
    out.append("Line numbers are for reference only; do NOT include them in diff context.\\n")

    for t in targets[:4]:
        out.append(f"\\n--- FILE: {t} (numbered excerpt) ---\\n")
        if t.endswith("repl/app.py"):
            excerpt = read_numbered_window(
                t,
                anchor_patterns=[
                    r"def\\s+run_repl\\b",
                    r"read_user_input_block",
                    r"STRICT|contract|validate",
                ],
                fallback_max_lines=240,
            )
        else:
            excerpt = read_numbered_window(t, anchor_patterns=None, fallback_max_lines=200)

        out.append(excerpt)
        out.append("\\n")

    return "".join(out)
