from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configuration for state management
MAX_MESSAGES = 100
MAX_TRACE_ENTRIES = 1000
MAX_EXEC_RESULTS = 100
MAX_PATCHES = 50

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
        # Limit the number of messages to prevent memory issues
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]
            logger.warning("Message list truncated to prevent memory issues")

    def log(self, text: str) -> None:
        self.trace.append(text)
        # Limit the number of trace entries to prevent memory issues
        if len(self.trace) > MAX_TRACE_ENTRIES:
            self.trace = self.trace[-MAX_TRACE_ENTRIES:]
            logger.warning("Trace list truncated to prevent memory issues")

    def add_exec_result(self, result: ExecResult) -> None:
        self.exec_results.append(result)
        # Limit the number of execution results to prevent memory issues
        if len(self.exec_results) > MAX_EXEC_RESULTS:
            self.exec_results = self.exec_results[-MAX_EXEC_RESULTS:]
            logger.warning("Execution results list truncated to prevent memory issues")

    def add_applied_patch(self, patch: CodePatch) -> None:
        self.applied_patches.append(patch)
        # Limit the number of patches to prevent memory issues
        if len(self.applied_patches) > MAX_PATCHES:
            self.applied_patches = self.applied_patches[-MAX_PATCHES:]
            logger.warning("Applied patches list truncated to prevent memory issues")


# Alias used by LangGraph
LlamiaGraphState = LlamiaState
