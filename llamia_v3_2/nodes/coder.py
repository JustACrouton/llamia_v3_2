from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion
from ..state import CodePatch, ExecRequest, LlamiaState, PlanStep
from ..tools.fs_tools import apply_patches

NODE_NAME = "coder"

# ---------------------------------
# Patch task detection
# ---------------------------------
def _is_patch_task(goal: str) -> bool:
    g = (goal or "").lower()
    return any(k in g for k in ["unified diff", "diff --git", "git style", ".patch", "improvements.patch"])


def _repo_root() -> Path:
    # nodes/ -> llamia_v3_2/ -> repo root
    return Path(__file__).resolve().parents[2]


def _git_ls_files_all() -> list[str]:
    """Return ALL git-tracked file paths."""
    try:
        import subprocess

        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(_repo_root()))
    except Exception:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _git_ls_files_filtered(limit: int = 200) -> list[str]:
    """Return git-tracked file paths (filtered to keep prompts small + relevant)."""
    try:
        import subprocess

        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(_repo_root()))
    except Exception:
        return []

    files = [ln.strip() for ln in out.splitlines() if ln.strip()]

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


# ---------------------------------
# Contract / constraint helpers
# ---------------------------------
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


def _goal_forbids_files(goal: str) -> bool:
    return bool(_NO_FILES_RE.search(goal or ""))


def _goal_forbids_commands(goal: str) -> bool:
    return bool(_NO_COMMANDS_RE.search(goal or ""))


# ---------------------------------
# JSON extraction / retry helpers
# ---------------------------------
def _try_parse_json_object(raw: str) -> dict[str, Any] | None:
    """
    Parse a JSON object even if the model wrapped it in prose.
    We take the first '{' ... last '}' slice and attempt json.loads.
    """
    if not raw:
        return None
    s = raw.strip()

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Fallback: extract likely JSON object substring
    a = s.find("{")
    b = s.rfind("}")
    if a >= 0 and b > a:
        candidate = s[a : b + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _retry_strict_json(messages: list[dict[str, str]], model_cfg) -> str:
    """
    One controlled retry: tells the model its output was invalid.
    """
    retry_msgs = list(messages)
    retry_msgs.append(
        {
            "role": "system",
            "content": (
                "STRICT MODE: Your last output was NOT valid JSON.\n"
                "Return STRICT JSON ONLY that matches the required schema.\n"
                "No prose. No markdown fences. No comments.\n"
                "Output must start with '{' and end with '}'."
            ),
            "node": NODE_NAME,
        }
    )
    return chat_completion(messages=retry_msgs, model_cfg=model_cfg)


# ---------------------------------
# Patch grounding helpers
# ---------------------------------
_FILE_IN_GOAL_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\-]+\.(?:py|md|toml|yaml|yml|txt))")


def _extract_paths_from_goal(goal: str, tracked: set[str]) -> list[str]:
    """Extract file paths mentioned in the goal and keep only git-tracked paths."""
    hits: list[str] = []
    for m in _FILE_IN_GOAL_RE.finditer(goal or ""):
        p = m.group("path").strip()
        if p in tracked and p not in hits:
            hits.append(p)
    return hits


def _read_numbered_window(
    relpath: str,
    *,
    anchor_patterns: list[str] | None = None,
    window_before: int = 60,
    window_after: int = 120,
    fallback_max_lines: int = 180,
    max_chars: int = 12000,
) -> str:
    """
    Read a RELEVANT excerpt:
    - If anchor_patterns match a line, return a window around the first match.
    - Else return the first fallback_max_lines lines.
    Always line-number the excerpt so diffs can match reality.
    """
    p = (_repo_root() / relpath).resolve()
    try:
        text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    except Exception as e:
        return f"[could not read {relpath}: {e!r}]"

    lines = text.splitlines()

    start = 0
    end = min(len(lines), fallback_max_lines)

    if anchor_patterns:
        regs = []
        for pat in anchor_patterns:
            try:
                regs.append(re.compile(pat))
            except re.error:
                continue

        hit_idx: int | None = None
        for i, ln in enumerate(lines):
            if any(r.search(ln) for r in regs):
                hit_idx = i
                break

        if hit_idx is not None:
            start = max(0, hit_idx - window_before)
            end = min(len(lines), hit_idx + window_after)

    chunk = lines[start:end]
    numbered = "\n".join(f"{start + i + 1:04d} {ln}" for i, ln in enumerate(chunk))
    if len(numbered) > max_chars:
        numbered = numbered[:max_chars] + "\n...[truncated]"
    return numbered


def _build_patch_context(goal: str, tracked_files: list[str]) -> str:
    """
    Provide grounded excerpts of relevant files so patch hunks match reality.
    This reduces fake diffs / wrong line contexts.
    """
    tracked_set = set(tracked_files)
    targets = _extract_paths_from_goal(goal, tracked_set)

    # Defaults (keep short + relevant)
    defaults = ["main.py", "llamia_v3_2/nodes/critic.py", "llamia_v3_2/nodes/intent_router.py"]
    for d in defaults:
        if d in tracked_set and d not in targets:
            targets.append(d)
        if len(targets) >= 4:
            break

    if not targets:
        return ""

    out: list[str] = []
    out.append("[patch_context] Authoritative repo excerpts (use for exact diff hunks).\n")
    out.append("Do NOT invent code. Patch must match these files exactly.\n")
    out.append("Line numbers are for reference only; do NOT include them in diff context.\n")

    for t in targets[:4]:
        out.append(f"\n--- FILE: {t} (numbered excerpt) ---\n")

        if t == "main.py":
            excerpt = _read_numbered_window(
                t,
                anchor_patterns=[
                    r"def\s+run_repl\b",
                    r'input\("you>\s*"\)',
                    r'print\("Llamia v3\.2',
                    r"Tips:",
                ],
            )
        else:
            excerpt = _read_numbered_window(t, anchor_patterns=None, fallback_max_lines=200)

        out.append(excerpt)
        out.append("\n")

    return "".join(out)


# ---------------------------------
# Prompts
# ---------------------------------
CODER_SYSTEM_PROMPT = """You are a coding agent for an autonomous developer assistant.

Your job:
- Read the HIGH-LEVEL GOAL and the PLAN steps.
- Create/modify files in the workspace/ directory ONLY when the goal requires it.
- Suggest commands ONLY when the goal requires it.

Rules:
- All files live under workspace/.
- Use relative paths like "hello.py" (do NOT include "workspace/" in the path).

GOAL CONSTRAINTS (MUST OBEY):
- If the goal says NOT to create/modify files, output "patches": [].
- If the goal says NOT to run commands, omit "exec" or set it to null.

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
- Do NOT invent new tracked files unless the goal explicitly requests it.
- When writing workspace files, use relative paths WITHOUT the "workspace/" prefix.

Patch requirements:
- The patch must include at least one "diff --git a/<path> b/<path>" for a real tracked file.
- Prefer small, safe, reviewable changes (logging, contract checks, loop guards, better prompts).
- Do NOT include markdown fences (no ```). The patch file must be raw diff text.
- Avoid fake "index abc..def" lines. Prefer omitting index lines unless you truly know them.

Output STRICT JSON ONLY in this format:
{
  "patches": [
    { "file_path": "IMPROVEMENTS.patch", "apply_mode": "overwrite", "content": "<unified diff here>" },
    { "file_path": "IMPROVEMENTS.md", "apply_mode": "overwrite", "content": "<explanation + test steps>" }
  ],
  "exec": {
    "workdir": ".",
    "commands": [
      "git status --porcelain",
      "git apply --check workspace/IMPROVEMENTS.patch",
      "python -m compileall -q ."
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
        "git ",
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
    fp = rel_path.replace("\\", "/").replace("'", "\\'")
    return f'python -c "print(open(\'{fp}\', encoding=\'utf-8\').read())"'


def _require_patch_artifacts(patches: list[CodePatch]) -> bool:
    needed = {"IMPROVEMENTS.patch", "IMPROVEMENTS.md"}
    have = {p.file_path for p in patches}
    return needed.issubset(have)


def coder_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    if state.mode != "task" or not state.goal:
        state.log(f"[{NODE_NAME}] not in task mode or missing goal; skipping coding")
        return state

    is_patch_task = _is_patch_task(state.goal)
    plan_str = _format_plan(state.plan)

    # Enforce goal constraints deterministically (non-patch tasks).
    # Patch tasks ALWAYS write workspace artifacts, so "no files" is ignored for patch-proposal mode.
    forbid_files = (not is_patch_task) and _goal_forbids_files(state.goal)
    forbid_cmds = _goal_forbids_commands(state.goal)

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

    tracked_all = _git_ls_files_all() if is_patch_task else []
    tracked_filtered = _git_ls_files_filtered(limit=200) if is_patch_task else []
    if tracked_filtered:
        user_prompt += "\nRepo tracked files you may reference in a unified diff (filtered):\n" + "\n".join(
            f"- {p}" for p in tracked_filtered
        ) + "\n"

    system_prompt = CODER_SYSTEM_PROMPT_PATCH if is_patch_task else CODER_SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt, "node": NODE_NAME}]

    # Ground patch tasks with excerpts that include the REPL prompt area (use ALL tracked files)
    if is_patch_task and tracked_all:
        patch_ctx = _build_patch_context(state.goal, tracked_all)
        if patch_ctx:
            messages.append({"role": "system", "content": patch_ctx, "node": NODE_NAME})

    messages.append({"role": "user", "content": user_prompt, "node": NODE_NAME})

    # Recent context tail
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
        messages.append(
            {"role": "system", "content": "Recent context (tail):\n" + "\n\n".join(ctx_lines), "node": NODE_NAME}
        )

    cfg = DEFAULT_CONFIG.model_for("coder")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")

    raw = chat_completion(messages=messages, model_cfg=cfg)
    state.log(f"[{NODE_NAME}] raw LLM output: {raw!r}")

    data = _try_parse_json_object(raw)
    if data is None:
        raw2 = _retry_strict_json(messages, cfg)
        state.log(f"[{NODE_NAME}] raw LLM output (retry): {raw2!r}")
        data = _try_parse_json_object(raw2)

    if data is None:
        if is_patch_task:
            msg = (
                "[coder] ERROR: model did not return valid JSON for PATCH-PROPOSAL mode.\n"
                "You MUST return STRICT JSON only with patches for IMPROVEMENTS.patch and IMPROVEMENTS.md.\n"
                "No prose, no markdown fences.\n"
            )
            state.add_message("system", msg, node=NODE_NAME)
            state.fix_instructions = msg + "Retry now with strict JSON and use the provided file excerpts for exact diff hunks."
            state.next_agent = "coder"
            return state

        state.log(f"[{NODE_NAME}] ERROR parsing coder JSON: invalid JSON")
        # If goal forbids files, do not fall back to writing files
        if forbid_files:
            state.exec_request = None
            state.fix_instructions = "[coder] Goal forbids files; model output invalid JSON; skipping."
            state.next_agent = None
            state.add_message("system", state.fix_instructions, node=NODE_NAME)
            return state

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

    # Build patches + exec request from JSON
    patches: list[CodePatch] = []
    exec_req: ExecRequest | None = None

    raw_patches = data.get("patches", [])
    if isinstance(raw_patches, list):
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

    # Enforce "no files" on non-patch tasks
    if forbid_files:
        patches = []

    # Patch tasks MUST create the requested artifacts (fail fast if not)
    if is_patch_task and not _require_patch_artifacts(patches):
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

    raw_exec = data.get("exec")
    if isinstance(raw_exec, dict) and not forbid_cmds:
        workdir = str(raw_exec.get("workdir", "workspace")).strip() or "workspace"
        commands = raw_exec.get("commands") or []
        if isinstance(commands, list):
            commands = [str(c).strip() for c in commands if str(c).strip()]
        else:
            commands = []

        # Dedupe
        seen = set()
        deduped: list[str] = []
        for c in commands:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        commands = deduped

        # Allowlist
        commands = _filter_safe_commands(commands)

        # If filtered everything, show a file instead (safe) — but only if commands are allowed
        if not commands and patches:
            commands = [_safe_pycat_command(patches[0].file_path)]
            workdir = "workspace"

        if commands:
            exec_req = ExecRequest(workdir=workdir, commands=commands)

    # Enforce "no commands"
    if forbid_cmds:
        exec_req = None

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
    else:
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
