from __future__ import annotations

import sys
from typing import Optional


def read_user_input_block(prompt: str = "you> ", paste_drain_s: float = 0.02) -> Optional[str]:
    """
    Read one user “turn”, while also being friendly to multi-line pastes.

    Behavior:
      - reads the first line via `input(prompt)`
      - then drains any immediately-buffered lines that arrive within `paste_drain_s`

    Notes:
      - On some platforms (notably Windows), select() on sys.stdin can be unreliable.
        In that case we fall back to only the first line.
    """
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None

    if not first.strip():
        return ""

    lines = [first]

    # Best-effort paste drain using select; if unavailable, just return first line.
    try:
        import select  # local import: not always available/usable

        while True:
            r, _, _ = select.select([sys.stdin], [], [], paste_drain_s)
            if not r:
                break
            nxt = sys.stdin.readline()
            if not nxt:
                break
            lines.append(nxt.rstrip("\n"))
    except Exception:
        pass

    return "\n".join(lines).rstrip()
