#!/usr/bin/env python3
from __future__ import annotations

"""
main.py is intentionally tiny.

All REPL logic lives under `llamia_v3_2.repl` so:
  - the entrypoint stays readable
  - helper logic is testable in isolation
  - “big” utilities (git/contract/logging) don’t clutter the top-level script
"""

import sys

from llamia_v3_2.repl.app import run_repl


if __name__ == "__main__":
    sys.exit(run_repl())
