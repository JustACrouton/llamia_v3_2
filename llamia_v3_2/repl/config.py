from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReplConfig:
    """
    Tunables for REPL behavior.

    Keeping these in a dataclass makes it easy to:
      - override in tests
      - later wire to CLI args/env vars
      - pass around as a single object instead of many globals
    """

    # Hard cap on LangGraph recursion; prevents runaway internal loops.
    invoke_recursion_limit: int = 100

    # Wall-clock cap per turn. Enforced via SIGALRM on Linux/macOS. No-op on Windows.
    invoke_timeout_s: int = 600

    # How many times we auto-retry if a “task:” output violates the contract.
    max_contract_retries: int = 10

    # Inject a truncated repo tree into system prompt for tasks to reduce hallucinations.
    inject_repo_snapshot: bool = True
    repo_snapshot_max_files: int = 250
