from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import List, Set, Tuple

from .paths import RepoPaths


def run_git(paths: RepoPaths, args: List[str]) -> Tuple[int, str, str]:
    """
    Run git in repo_root and return (returncode, stdout, stderr).
    If git isn't installed, returns rc=127.
    """
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(paths.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "git not found"


def git_ls_files(paths: RepoPaths) -> List[str]:
    rc, out, _ = run_git(paths, ["ls-files"])
    if rc != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def git_status_porcelain(paths: RepoPaths) -> List[str]:
    rc, out, _ = run_git(paths, ["status", "--porcelain"])
    if rc != 0:
        return []
    return [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]


def porcelain_paths(lines: List[str]) -> Set[str]:
    """
    Extract affected paths from `git status --porcelain` output.

    Handles renames like:
      R  old -> new
    """
    out: set[str] = set()
    for ln in lines:
        if len(ln) < 4:
            continue
        path = ln[3:].strip()
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            out.add(old.strip())
            out.add(new.strip())
        else:
            out.add(path)
    return out


def dirty_outside_workspace(paths: RepoPaths) -> Set[str]:
    """
    Returns modified/added/deleted tracked file paths excluding workspace/.
    """
    changed = porcelain_paths(git_status_porcelain(paths))
    return {p for p in changed if p and not p.startswith("workspace/")}


def git_restore_paths(paths: RepoPaths, files: Set[str]) -> None:
    """
    Best-effort restore of newly dirtied paths after a failed contract attempt.

    Uses:
      - git restore --staged --worktree
      - fallback: git reset + git checkout
    """
    if not files:
        return

    plist = sorted(files)
    rc, _, _ = run_git(paths, ["restore", "--staged", "--worktree", "--", *plist])
    if rc == 0:
        return

    run_git(paths, ["reset", "--", *plist])
    run_git(paths, ["checkout", "--", *plist])


def repo_snapshot_text(paths: RepoPaths, max_files: int) -> str:
    """
    Produce a truncated “tree” of repo files to ground the model.

    Prefers `git ls-files` but falls back to rglob if repo isn't a git checkout.
    """
    files = git_ls_files(paths)
    if not files:
        files = []
        for p in paths.repo_root.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(paths.repo_root)))
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
        s2 = str(s)
        if any(s2.startswith(p) for p in skip_prefixes):
            continue
        if "/__pycache__/" in s2 or s2.endswith("/__pycache__"):
            continue
        if any(s2.endswith(ext) for ext in skip_exts):
            continue
        filtered.append(s2)

    files2 = sorted(filtered)[:max_files]
    return "Repo files (truncated):\n" + "\n".join(f"- {f}" for f in files2)


def check_patch_in_clean_worktree(paths: RepoPaths, patch_abs: Path) -> tuple[bool, str]:
    """
    Verify patch applies cleanly to HEAD, then compileall in an isolated worktree.
    Returns (ok, details).
    """
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)

    base = Path(tempfile.mkdtemp(prefix="llamia_applycheck_", dir=str(paths.workspace_dir)))
    wt_dir = base / "wt"  # must NOT exist before `git worktree add`

    try:
        rc, out, err = run_git(paths, ["worktree", "add", "--detach", str(wt_dir), "HEAD"])
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
        try:
            run_git(paths, ["worktree", "remove", "--force", str(wt_dir)])
        except Exception:
            pass
        shutil.rmtree(base, ignore_errors=True)
