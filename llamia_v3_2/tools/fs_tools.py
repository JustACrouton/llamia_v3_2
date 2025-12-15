from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..state import CodePatch


# Root is the repo root: .../llamia_v3_2/
ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "workspace"


def ensure_workspace() -> Path:
    """
    Ensure the workspace directory exists and return its Path.
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def _normalize_path(file_path: str) -> Path:
    """
    Interpret file_path as relative to the workspace.
    We allow paths like "hello.py" or "subdir/script.py".

    Security:
    - Prevent path traversal (..), absolute paths, and writing outside workspace/.

    If the model includes a leading 'workspace/' in the path, strip it.
    """
    ws = ensure_workspace().resolve()
    cleaned = file_path.lstrip("/")

    if cleaned.startswith("workspace/"):
        cleaned = cleaned[len("workspace/") :]

    # Resolve the final path and ensure it stays within workspace
    target = (ws / cleaned).resolve()
    if ws in target.parents:
        return target
    raise ValueError(f"Refusing to write outside workspace: {file_path!r}")


def apply_patch(patch: CodePatch) -> Path:
    """
    Apply a single patch to the workspace.
    Returns the absolute Path of the file written.
    """
    target = _normalize_path(patch.file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if patch.apply_mode == "append" and target.exists():
        with target.open("a", encoding="utf-8") as f:
            f.write(patch.content)
    else:
        with target.open("w", encoding="utf-8") as f:
            f.write(patch.content)

    return target


def apply_patches(patches: Iterable[CodePatch]) -> list[Path]:
    """
    Apply all patches and return the list of written files.
    """
    written: list[Path] = []
    for patch in patches:
        written.append(apply_patch(patch))
    return written
