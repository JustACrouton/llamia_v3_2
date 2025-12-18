from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set, Tuple

from .paths import RepoPaths
from .repo_utils import check_patch_in_clean_worktree, dirty_outside_workspace, git_ls_files


_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_WS_PATH_RE = re.compile(r"(workspace/[A-Za-z0-9._\\-\\/]+)")


def extract_required_workspace_paths(user_input: str) -> List[str]:
    """
    Any explicit 'workspace/...' in the user input is treated as required output.
    """
    found = _WS_PATH_RE.findall(user_input)
    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        p2 = p.strip().rstrip(".")
        if p2 and p2 not in seen:
            out.append(p2)
            seen.add(p2)
    return out


def prompt_requests_patch(user_input: str) -> bool:
    s = user_input.lower()
    return (
        "improvements.patch" in s
        or "unified diff" in s
        or "git style" in s
        or ("create workspace/" in s and ".patch" in s)
    )


def patch_touched_files(patch_text: str) -> List[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _DIFF_FILE_RE.finditer(patch_text):
        a = m.group(1).strip()
        if a and a != "/dev/null" and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def patch_has_substantive_changes(patch_text: str) -> bool:
    for ln in patch_text.splitlines():
        if ln.startswith(("diff --git ", "index ", "--- ", "+++ ", "@@")):
            continue
        if ln.startswith(("+", "-")):
            if ln[1:].strip():
                return True
    return False


def patch_touches_tracked_files(patch_text: str, tracked_files: Set[str]) -> bool:
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/").strip()
                b_path = parts[3].removeprefix("b/").strip()
                if a_path in tracked_files or b_path in tracked_files:
                    return True
    return False


def check_improvements_md_grounding(paths: RepoPaths, md_abs: Path, touched_files: List[str]) -> List[str]:
    fails: list[str] = []
    if not md_abs.exists():
        return [f"Missing required file: {md_abs.relative_to(paths.repo_root).as_posix()}"]

    md = md_abs.read_text(encoding="utf-8", errors="replace")
    low = md.lower()

    if "root cause" not in low:
        fails.append("IMPROVEMENTS.md must include a Root Cause section.")
    if "verif" not in low:
        fails.append("IMPROVEMENTS.md must include Verification steps.")
    if "```" not in md:
        fails.append("IMPROVEMENTS.md must include at least one fenced code block with an excerpt.")

    for fp in touched_files:
        if fp not in md:
            fails.append(f"IMPROVEMENTS.md must mention touched file path: {fp}")

    return fails


def validate_task_contract(
    paths: RepoPaths,
    user_input: str,
    baseline_dirty_outside_ws: Set[str] | None = None,
) -> Tuple[List[str], Set[str]]:
    failures: list[str] = []

    required = extract_required_workspace_paths(user_input)
    for rel in required:
        p = paths.abs_repo_path(rel)
        if not p.exists():
            failures.append(f"Missing required file: {rel}")

    newly_dirty: set[str] = set()
    low = user_input.lower()
    if "do not modify" in low and ("tracked file" in low or "repo code" in low):
        before = baseline_dirty_outside_ws or set()
        after = dirty_outside_workspace(paths)
        newly_dirty = after - before
        if newly_dirty:
            failures.append("Modified tracked files unexpectedly (new this turn):\n" + "\n".join(sorted(newly_dirty)))

    if prompt_requests_patch(user_input):
        patch_paths = [p for p in required if p.lower().endswith(".patch")]
        if not patch_paths:
            patch_paths = ["workspace/IMPROVEMENTS.patch"]

        patch_rel = patch_paths[0]
        patch_abs = paths.abs_repo_path(patch_rel)

        if not patch_abs.exists():
            failures.append(f"Patch file not created: {patch_rel}")
        else:
            txt = patch_abs.read_text(encoding="utf-8", errors="replace")
            tracked = set(git_ls_files(paths))

            if tracked and not patch_touches_tracked_files(txt, tracked):
                failures.append(
                    "Patch does not touch any existing git-tracked files (likely hallucinated / irrelevant). "
                    f"Regenerate {patch_rel} to modify real files from git ls-files."
                )

            if not patch_has_substantive_changes(txt):
                failures.append("Patch contains no substantive (+/-) changes (looks whitespace-only or metadata-only).")

            ok, detail = check_patch_in_clean_worktree(paths, patch_abs)
            if not ok:
                failures.append("Patch failed clean-worktree verification:\n" + detail)

            touched = patch_touched_files(txt)
            md_paths = [p for p in required if p.lower().endswith(".md")]
            md_rel = md_paths[0] if md_paths else "workspace/IMPROVEMENTS.md"
            md_abs = paths.abs_repo_path(md_rel)
            failures.extend(check_improvements_md_grounding(paths, md_abs, touched))

    return failures, newly_dirty
