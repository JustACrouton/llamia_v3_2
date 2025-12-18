from __future__ import annotations

"""
Coder node (refactored).

This file is intentionally “thin”:
- assembles prompts
- calls the LLM
- validates/coerces output
- applies workspace patches

Most helpers live in:
- coder_prompts.py
- coder_json.py
- coder_constraints.py
- coder_git.py
- coder_patch_context.py
- coder_utils.py
"""

from typing import Any

from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion
from ..state import CodePatch, ExecRequest, LlamiaState
from ..tools.fs_tools import apply_patches

from .coder_constraints import goal_forbids_commands, goal_forbids_files
from .coder_git import git_ls_files_all, git_ls_files_filtered
from .coder_json import retry_strict_json, try_parse_json_object
from .coder_patch_context import build_patch_context, is_patch_task
from .coder_prompts import CODER_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT_PATCH
from .coder_utils import filter_safe_commands, format_plan, require_patch_artifacts, safe_pycat_command

NODE_NAME = "coder"


def _build_recent_context_tail(state: LlamiaState, *, max_messages: int = 8, max_chars: int = 3000) -> str:
    """
    Build a compact “tail” of recent messages to keep the coder grounded.

    We put this in a system message so the model treats it as context,
    not as a user instruction.
    """
    ctx_lines: list[str] = []
    for m in (state.messages or [])[-max_messages:]:
        role = m.get("role") or "?"
        node = m.get("node") or "?"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        ctx_lines.append(f"[{role}:{node}] {content}")

    return "Recent context (tail):\n" + "\n\n".join(ctx_lines) if ctx_lines else ""


def _parse_patches_from_json(data: dict[str, Any]) -> list[CodePatch]:
    """
    Convert the model JSON payload into CodePatch objects.
    Dedupe by file_path (last one wins).
    """
    patches: list[CodePatch] = []
    raw_patches = data.get("patches", [])

    if not isinstance(raw_patches, list):
        return patches

    by_fp: dict[str, CodePatch] = {}
    for idx, p in enumerate(raw_patches, start=1):
        if not isinstance(p, dict):
            continue
        fp = str(p.get("file_path", f"file_{idx}.txt")).strip()
        if not fp:
            continue
        content = str(p.get("content", "")).replace("\r\n", "\n")
        mode = str(p.get("apply_mode", "overwrite")).lower()
        if mode not in ("overwrite", "append"):
            mode = "overwrite"
        by_fp[fp] = CodePatch(file_path=fp, content=content, apply_mode=mode)

    patches = list(by_fp.values())
    return patches


def _parse_exec_from_json(data: dict[str, Any], *, forbid_cmds: bool) -> ExecRequest | None:
    """
    Convert the model JSON "exec" field into an ExecRequest if allowed.
    Applies allowlist + dedupe.
    """
    if forbid_cmds:
        return None

    raw_exec = data.get("exec")
    if not isinstance(raw_exec, dict):
        return None

    workdir = str(raw_exec.get("workdir", "workspace")).strip() or "workspace"
    commands = raw_exec.get("commands") or []

    if isinstance(commands, list):
        commands = [str(c).strip() for c in commands if str(c).strip()]
    else:
        commands = []

    # Dedupe
    seen: set[str] = set()
    deduped: list[str] = []
    for c in commands:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    # Allowlist
    filtered = filter_safe_commands(deduped)
    if not filtered:
        return None

    return ExecRequest(workdir=workdir, commands=filtered)


def coder_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if state.mode != "task" or not state.goal:
        state.log(f"[{NODE_NAME}] not in task mode or missing goal; skipping coding")
        return state

    patch_mode = is_patch_task(state.goal)
    plan_str = format_plan(state.plan)

    # Enforce goal constraints deterministically (non-patch tasks).
    # Patch tasks ALWAYS write workspace artifacts, so "no files" is ignored in patch-proposal mode.
    forbid_files = (not patch_mode) and goal_forbids_files(state.goal)
    forbid_cmds = goal_forbids_commands(state.goal)

    if forbid_files and forbid_cmds:
        state.exec_request = None
        state.fix_instructions = None
        state.next_agent = None
        state.add_message("system", "[coder] Goal forbids files and commands; skipping coder.", node=NODE_NAME)
        state.log(f"[{NODE_NAME}] goal forbids files+commands -> skip")
        return state

    existing_files = sorted({p.file_path for p in state.applied_patches}) if state.applied_patches else []
    existing_files_str = "\n".join(f"- {fp}" for fp in existing_files) if existing_files else "(none yet)"

    repair_block = ""
    if state.fix_instructions:
        repair_block = f"""
FIX INSTRUCTIONS (REPAIR MODE):
{state.fix_instructions}

Existing workspace files you should prefer to edit:
{existing_files_str}
"""

    notes = (state.research_notes or "").strip()
    notes_block = f"\nWeb research notes:\n{notes}\n" if notes else ""

    user_prompt = f"""Goal:
{state.goal}

Plan:
{plan_str}
{repair_block}
{notes_block}
"""

    tracked_all = git_ls_files_all() if patch_mode else []
    tracked_filtered = git_ls_files_filtered(limit=200) if patch_mode else []
    if tracked_filtered:
        user_prompt += "\nRepo tracked files you may reference in a unified diff (filtered):\n" + "\n".join(
            f"- {p}" for p in tracked_filtered
        ) + "\n"

    system_prompt = CODER_SYSTEM_PROMPT_PATCH if patch_mode else CODER_SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt, "node": NODE_NAME}]

    # Patch grounding: provide real excerpts so hunks can match reality
    if patch_mode and tracked_all:
        patch_ctx = build_patch_context(state.goal, tracked_all)
        if patch_ctx:
            messages.append({"role": "system", "content": patch_ctx, "node": NODE_NAME})

    messages.append({"role": "user", "content": user_prompt, "node": NODE_NAME})

    ctx_tail = _build_recent_context_tail(state)
    if ctx_tail:
        messages.append({"role": "system", "content": ctx_tail, "node": NODE_NAME})

    cfg = DEFAULT_CONFIG.model_for("coder")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")

    raw = chat_completion(messages=messages, model_cfg=cfg)
    state.log(f"[{NODE_NAME}] raw LLM output: {raw!r}")

    data = try_parse_json_object(raw)
    if data is None:
        raw2 = retry_strict_json(messages, cfg, node_name=NODE_NAME)
        state.log(f"[{NODE_NAME}] raw LLM output (retry): {raw2!r}")
        data = try_parse_json_object(raw2)

    # If the model cannot produce JSON, force a coder retry (especially important for patch_mode)
    if data is None:
        if patch_mode:
            msg = (
                "[coder] ERROR: model did not return valid JSON for PATCH-PROPOSAL mode.\n"
                "You MUST return STRICT JSON only with patches for IMPROVEMENTS.patch and IMPROVEMENTS.md.\n"
                "No prose, no markdown fences.\n"
            )
            state.add_message("system", msg, node=NODE_NAME)
            state.fix_instructions = msg + "Retry now with strict JSON and use the provided file excerpts for exact diff hunks."
            state.next_agent = "coder"
            return state

        # Non-patch mode: if goal forbids files, do not try to write fallback files.
        if forbid_files:
            state.exec_request = None
            state.fix_instructions = "[coder] Goal forbids files; model output invalid JSON; skipping."
            state.next_agent = None
            state.add_message("system", state.fix_instructions, node=NODE_NAME)
            return state

        # Safe fallback: generate a tiny file in workspace
        patches = [
            CodePatch(
                file_path="generated_script.py",
                content="print('Hello from Llamia coder fallback')\n",
                apply_mode="overwrite",
            )
        ]
        exec_req = ExecRequest(workdir="workspace", commands=["python generated_script.py"])
        written_files = apply_patches(patches)
        state.applied_patches.extend(patches)
        state.exec_request = exec_req
        state.fix_instructions = None
        state.next_agent = None
        state.add_message(
            "system",
            "The coder node created or updated the following files:\n"
            f"- {patches[0].file_path}  (-> {written_files[0]})\n\n"
            "Suggested commands to run:\n"
            f"  (workdir: {exec_req.workdir})\n"
            f"- {exec_req.commands[0]}",
            node=NODE_NAME,
        )
        return state

    # Convert JSON -> patches + exec
    patches = _parse_patches_from_json(data)

    # Enforce "no files" on non-patch tasks
    if forbid_files:
        patches = []

    # Patch tasks MUST create the requested artifacts (fail fast if not)
    if patch_mode and not require_patch_artifacts(patches):
        msg = (
            "[coder] PATCH-PROPOSAL invalid: missing required artifacts.\n"
            "You must create BOTH files:\n"
            "- IMPROVEMENTS.patch\n"
            "- IMPROVEMENTS.md\n"
        )
        state.add_message("system", msg, node=NODE_NAME)
        state.fix_instructions = msg + "Retry with strict JSON only."
        state.next_agent = "coder"
        return state

    exec_req = _parse_exec_from_json(data, forbid_cmds=forbid_cmds)

    # If model suggested only unsafe commands, but patches exist and commands are allowed,
    # generate a safe “display file” command.
    if exec_req is None and (not forbid_cmds) and patches:
        exec_req = ExecRequest(workdir="workspace", commands=[safe_pycat_command(patches[0].file_path)])

    # Apply patches (workspace-only)
    if patches:
        try:
            written_files = apply_patches(patches)
        except Exception as e:
            state.pending_patches = []
            state.exec_request = None
            state.add_message("system", f"[coder] ERROR applying patches: {e!r}", node=NODE_NAME)
            state.log(f"[{NODE_NAME}] ERROR applying patches: {e!r}")
            return state

        state.pending_patches = []
        state.applied_patches.extend(patches)

        summary_lines = ["The coder node created or updated the following files:"]
        for p, path in zip(patches, written_files):
            summary_lines.append(f"- {p.file_path}  (-> {path})")

        if exec_req and exec_req.commands:
            summary_lines.append("")
            summary_lines.append("Suggested commands to run:")
            summary_lines.append(f"  (workdir: {exec_req.workdir})")
            for cmd in exec_req.commands:
                summary_lines.append(f"- {cmd}")

        state.exec_request = exec_req
        state.fix_instructions = None
        state.next_agent = None

        state.add_message("system", "\n".join(summary_lines), node=NODE_NAME)
        state.log(f"[{NODE_NAME}] applied {len(patches)} patches")
        return state

    # No patches; still forward exec if allowed
    state.exec_request = exec_req
    state.fix_instructions = None
    state.next_agent = None
    state.log(f"[{NODE_NAME}] no patches produced")

    if exec_req and exec_req.commands:
        state.add_message(
            "system",
            "The coder node produced no files.\n\n"
            "Suggested commands to run:\n"
            f"  (workdir: {exec_req.workdir})\n"
            + "\n".join(f"- {c}" for c in exec_req.commands),
            node=NODE_NAME,
        )

    return state
