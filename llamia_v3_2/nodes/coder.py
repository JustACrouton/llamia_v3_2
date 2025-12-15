from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..state import LlamiaState, CodePatch, ExecRequest, PlanStep
from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion
from ..tools.fs_tools import apply_patches

NODE_NAME = "coder"


def _is_patch_task(goal: str) -> bool:
    g = (goal or "").lower()
    return any(k in g for k in ["unified diff", "diff --git", "git style", ".patch", "improvements.patch"])


def _git_ls_files_filtered(limit: int = 200) -> list[str]:
    """Return git-tracked file paths (filtered to keep prompts small + relevant)."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(Path(__file__).resolve().parents[2]))
    except Exception:
        return []
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # Drop workspace + big/binary artifacts
    skip_prefixes = ("workspace/", ".llamia_chroma/", ".venv/")
    skip_exts = (".bin", ".sqlite3", ".db")
    filtered: list[str] = []
    for f in files:
        if any(f.startswith(p) for p in skip_prefixes):
            continue
        if any(f.endswith(ext) for ext in skip_exts):
            continue
        filtered.append(f)
        if len(filtered) >= limit:
            break
    return filtered


CODER_SYSTEM_PROMPT = """You are a coding agent for an autonomous developer assistant.

Your job:
- Read the HIGH-LEVEL GOAL and the PLAN steps.
- Decide what files need to be created or updated in the workspace/ directory.
- Generate COMPLETE FILE CONTENTS (not diffs) as patches.
- Suggest commands to run for testing/usage.

Rules:
- All files live under workspace/.
- Use relative paths like "hello.py" (do NOT include "workspace/" in the path).
- Write clean, runnable code.

COMMAND RULES (IMPORTANT):
- Do NOT use: chmod, ./script, shebang execution, &&, ||, pipes, redirects.
- Do NOT use: cat, less, more, head, tail.
- Only suggest commands like:
  - python file.py
  - python3 file.py
  - python -c "..."
  - python -m ...
  - pytest
  - ruff .
  - mypy .

NOTE:
- If you need to show a file's contents, use python -c, e.g.:
  python -c "print(open('file.md', encoding='utf-8').read())"

STAGED EXECUTION (IMPORTANT):
- If the goal says "then fix" / "until it succeeds", do NOT preemptively provide the final fixed state.
- Create the initial attempt + commands that reproduce the failure.
- The critic/coder repair loop will handle fixes in subsequent iterations via FIX INSTRUCTIONS.

REPAIR MODE:
- If FIX INSTRUCTIONS are provided, apply the MINIMAL change needed to make the failing command pass.
- Prefer editing existing files over creating new ones.

You MUST respond with STRICT JSON ONLY in this format:

{
  "patches": [
    {
      "file_path": "hello.py",
      "apply_mode": "overwrite",
      "content": "# full file content here"
    }
  ],
  "exec": {
    "workdir": "workspace",
    "commands": [
      "python hello.py"
    ]
  }
}
"""

CODER_SYSTEM_PROMPT_PATCH = """You are a coding agent for an autonomous developer assistant.

You are in PATCH-PROPOSAL mode.

Your job:
- Propose concrete improvements to the EXISTING repo code as a unified diff.
- You MUST write the requested *.patch and *.md artifacts into workspace/.

Rules:
- You MUST NOT modify tracked repo files directly.
- The .patch content MUST be a unified diff (git style) that targets ONLY existing git-tracked files.
- Do NOT invent new files (e.g., hello.py) unless the goal explicitly requests a new tracked file.
- When writing workspace files, use relative paths WITHOUT the \"workspace/\" prefix.

Patch requirements:
- The patch must include at least one \"diff --git a/<path> b/<path>\" for a real tracked file.
- Prefer small, safe, reviewable changes (logging, contract checks, loop guards, better prompts).

Output STRICT JSON ONLY in this format:
{
  \"patches\": [
    { \"file_path\": \"IMPROVEMENTS.patch\", \"apply_mode\": \"overwrite\", \"content\": \"<unified diff here>\" },
    { \"file_path\": \"IMPROVEMENTS.md\", \"apply_mode\": \"overwrite\", \"content\": \"<explanation + test steps>\" }
  ],
  \"exec\": {
    \"workdir\": \".\",
    \"commands\": [
      \"git status --porcelain\",
      \"git apply --check workspace/IMPROVEMENTS.patch\",
      \"python -m compileall -q .\"
    ]
  }
}
"""



def _filter_safe_commands(cmds: list[str]) -> list[str]:
    allowed_prefixes = (
        "python ",
        "python3 ",
        "python -c ",
        "python3 -c ",
        "python -m ",
        "python3 -m ",
        "pytest",
        "ruff ",
        "mypy ",
    )
    out: list[str] = []
    for c in cmds:
        cc = c.strip()
        if not cc:
            continue
        if cc.startswith(allowed_prefixes):
            out.append(cc)
    return out


def _format_plan(plan: List[PlanStep]) -> str:
    lines = []
    for step in plan:
        lines.append(f"{step.id}. {step.description} [{step.status}]")
    return "\n".join(lines) if lines else "(no plan steps)"


def _safe_pycat_command(rel_path: str) -> str:
    # rel_path is relative to workspace/
    fp = rel_path.replace("\\", "/").replace("'", "\\'")
    return f'python -c "print(open(\'{fp}\', encoding=\'utf-8\').read())"'


def coder_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if state.mode != "task" or not state.goal:
        state.log(f"[{NODE_NAME}] not in task mode or missing goal; skipping coding")
        return state

    plan_str = _format_plan(state.plan)

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

    # PATCH-PROPOSAL mode: provide tracked file list + switch prompts
    is_patch_task = _is_patch_task(state.goal)
    tracked = _git_ls_files_filtered(limit=200) if is_patch_task else []
    if tracked:
        user_prompt += "\nRepo tracked files you may reference in a unified diff (filtered):\n" + "\n".join(f"- {p}" for p in tracked) + "\n"

    system_prompt = CODER_SYSTEM_PROMPT_PATCH if is_patch_task else CODER_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt, "node": NODE_NAME},
        {"role": "user", "content": user_prompt, "node": NODE_NAME},
    ]

    # Include a small tail of conversation context (repo snapshot, contract failures, etc.)
    ctx_lines: list[str] = []
    for m in (state.messages or [])[-8:]:
        role = m.get("role") or "?"
        node = m.get("node") or "?"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        content = content if len(content) <= 3000 else content[:3000] + "\n...[truncated]"
        ctx_lines.append(f"[{role}:{node}] {content}")
    if ctx_lines:
        messages.append({"role": "system", "content": "Recent context (tail):\n" + "\n\n".join(ctx_lines), "node": NODE_NAME})

    cfg = DEFAULT_CONFIG.model_for("coder")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")

    raw = chat_completion(messages=messages, model_cfg=cfg)
    state.log(f"[{NODE_NAME}] raw LLM output: {raw!r}")

    patches: list[CodePatch] = []
    exec_req: ExecRequest | None = None

    try:
        data = json.loads(raw)

        raw_patches = data.get("patches", [])
        if not isinstance(raw_patches, list):
            raise ValueError("patches field is not a list")

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

        raw_exec = data.get("exec")
        if isinstance(raw_exec, dict):
            workdir = str(raw_exec.get("workdir", "workspace")).strip() or "workspace"
            commands = raw_exec.get("commands") or []
            if isinstance(commands, list):
                commands = [str(c).strip() for c in commands if str(c).strip()]
            else:
                commands = []

            # dedupe
            seen = set()
            deduped: list[str] = []
            for c in commands:
                if c not in seen:
                    deduped.append(c)
                    seen.add(c)
            commands = deduped

            # ✅ enforce allowlist (THIS was missing)
            commands = _filter_safe_commands(commands)

            # ✅ if we filtered everything out, inject a safe "show file" command
            if not commands and patches:
                commands = [_safe_pycat_command(patches[0].file_path)]
                workdir = "workspace"

            if commands:
                exec_req = ExecRequest(workdir=workdir, commands=commands)

    except Exception as e:
        state.log(f"[{NODE_NAME}] ERROR parsing coder JSON: {e!r}")
        patches = [
            CodePatch(
                file_path="generated_script.py",
                content="print('Hello from Llamia coder fallback')\n",
                apply_mode="overwrite",
            )
        ]
        exec_req = ExecRequest(workdir="workspace", commands=["python generated_script.py"])

    if patches:
        try:
            written_files = apply_patches(patches)
        except Exception as e:
            # Most likely a path safety issue (eg. attempted write outside workspace)
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
        state.next_agent = None  # don't carry routing hints forward

        state.add_message("system", "\n".join(summary_lines), node=NODE_NAME)
        state.log(f"[{NODE_NAME}] applied {len(patches)} patches")
    else:
        state.log(f"[{NODE_NAME}] no patches produced")

    return state
