from __future__ import annotations

"""
Goal constraint parsing helpers.

These functions interpret user goals like:
- "do not create files"
- "don't run any commands"

They are intentionally conservative: if the goal looks like it forbids something,
we enforce it.
"""

import re


_NO_FILES_RE = re.compile(
    r"(do not|don't)\s+(create|write|modify|edit)\s+(any\s+)?files?"
    r"|no\s+files?"
    r"|without\s+(creating|writing|modifying)\s+files?",
    re.I,
)

_NO_COMMANDS_RE = re.compile(
    r"(do not|don't)\s+(run|execute)\s+(any\s+)?(commands?|cmds?)"
    r"|no\s+(commands?|cmds?)"
    r"|without\s+(running|executing)\s+(commands?|cmds?)",
    re.I,
)


def goal_forbids_files(goal: str) -> bool:
    """True if the goal text forbids creating/modifying files."""
    return bool(_NO_FILES_RE.search(goal or ""))


def goal_forbids_commands(goal: str) -> bool:
    """True if the goal text forbids suggesting/running commands."""
    return bool(_NO_COMMANDS_RE.search(goal or ""))
