from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class RepoPaths:
    """
    Centralizes path derivation so every helper uses the same notion of:
      - repo root (where main.py lives)
      - workspace directory (repo_root/workspace)
    """

    repo_root: Path
    workspace_dir: Path

    @classmethod
    def from_entrypoint(cls, entry_file: Path) -> "RepoPaths":
        """
        Build RepoPaths from the file used as the entrypoint (typically main.py).
        """
        root = entry_file.resolve().parent
        return cls(repo_root=root, workspace_dir=(root / "workspace").resolve())

    def abs_repo_path(self, p: PathLike) -> Path:
        """
        Convert a repo-relative path like 'workspace/IMPROVEMENTS.md' into an absolute Path.
        Absolute inputs are returned unchanged.
        """
        path = Path(p)
        return path if path.is_absolute() else (self.repo_root / path)
