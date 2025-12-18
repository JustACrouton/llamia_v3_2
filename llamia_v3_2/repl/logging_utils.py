from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Tuple

from .paths import RepoPaths


def safe_to_json(obj: Any) -> Any:
    """
    Convert arbitrary objects into something JSON-serializable for JSONL logging.

    We prefer not to crash the REPL due to a logging failure.
    """
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {str(k): safe_to_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_to_json(x) for x in obj]

    return str(obj)


def setup_run_logger(paths: RepoPaths) -> Tuple[logging.Logger, Path, Path]:
    """
    Creates:
      - a human-readable text log
      - a structured JSONL log (machine-friendly)

    IMPORTANT: we do not log to stdout because it would corrupt the interactive prompt.
    """
    log_dir = paths.workspace_dir / "logs"
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

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)

    print(f"[log] text:  {text_path}")
    print(f"[log] jsonl: {jsonl_path}")

    return logger, text_path, jsonl_path


def append_jsonl(jsonl_path: Path, record: dict[str, Any]) -> None:
    """
    Append one JSON object per line (JSONL).

    The record is run through `safe_to_json` to reduce logging-related crashes.
    """
    record2 = safe_to_json(record)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record2, ensure_ascii=False) + "\n")


def tail_lines(s: str, max_chars: int = 4000) -> str:
    """
    Truncate large text to avoid exploding logs and snapshots.
    """
    s2 = s or ""
    if len(s2) <= max_chars:
        return s2
    return s2[:max_chars] + "\n...[truncated]"


def read_if_exists(paths: RepoPaths, rel_path: str, max_chars: int = 8000) -> str | None:
    """
    Best-effort read helper used in snapshots. Returns:
      - None if file does not exist
      - truncated text if it exists
      - an error string if the read fails
    """
    p = paths.abs_repo_path(rel_path)
    if not p.exists():
        return None
    try:
        return tail_lines(p.read_text(encoding="utf-8", errors="replace"), max_chars=max_chars)
    except Exception as e:
        return f"[read_error] {e!r}"
