from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    node: str | None  # which node produced this (for trace/debug)


@dataclass
class PlanStep:
    id: int
    description: str
    status: Literal["pending", "in_progress", "done", "skipped", "failed"] = "pending"


@dataclass
class CodePatch:
    file_path: str  # e.g. "hello.py" (relative to workspace/)
    content: str  # full file contents
    apply_mode: Literal["overwrite", "append"] = "overwrite"


@dataclass
class ExecRequest:
    workdir: str  # e.g. "workspace"
    commands: list[str]  # e.g. ["python hello.py"]


@dataclass
class ExecResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class LlamiaState:
    """
    Core shared state for Llamia v3.2.

    This state is intentionally "plain" so LangGraph can serialize it to dicts and back.
    """
    messages: list[Message] = field(default_factory=list)


    turn_id: int = 0
    responded_turn_id: int = -1
    
    
    # High-level mode
    mode: Literal["chat", "task"] = "chat"
    goal: str | None = None

    # Task plan
    plan: list[PlanStep] = field(default_factory=list)

    # Code patches
    pending_patches: list[CodePatch] = field(default_factory=list)
    applied_patches: list[CodePatch] = field(default_factory=list)

    # Local research / RAG
    research_query: str | None = None
    research_notes: str | None = None

    # Web search / research_web node
    web_queue: list[str] = field(default_factory=list)   # pending web queries
    web_results: str | None = None                       # last web results summary
    return_after_web: str = "planner"                    # where to go when web finishes


    # Critic web-search throttle (resets per task)
    web_search_count: int = 0

    # Execution request (suggested commands)
    exec_request: ExecRequest | None = None

    # Full history of execution results (across loops)
    exec_results: list[ExecResult] = field(default_factory=list)

    # Latest execution results (from the most recent executor run only)
    last_exec_results: list[ExecResult] = field(default_factory=list)

    # Routing / control
    next_agent: str | None = None
    loop_count: int = 0

    # Critic -> Coder guidance
    fix_instructions: str | None = None

    # If task is intentionally demonstrating a failure (don’t “auto-fix”)
    expected_failure: bool = False

    # Debugging trace
    trace: list[str] = field(default_factory=list)

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        node: str | None = None,
    ) -> None:
        self.messages.append({"role": role, "content": content, "node": node})

    def log(self, text: str) -> None:
        self.trace.append(text)


# Alias used by LangGraph
LlamiaGraphState = LlamiaState
