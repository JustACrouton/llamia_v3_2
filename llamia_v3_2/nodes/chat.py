from __future__ import annotations

from ..state import LlamiaState
from ..config import DEFAULT_CONFIG
from ..llm_client import chat_completion

NODE_NAME = "chat"

MAX_STD_TAIL = 1200
MAX_ERR_TAIL = 2000


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


def _strip_repl_prefix(text: str) -> str:
    s = (text or "").strip()
    while s.lower().startswith("you>"):
        s = s[4:].lstrip()
    return s


def _latest_user_text(state: LlamiaState) -> str:
    for m in reversed(state.messages):
        if m.get("role") == "user":
            return _strip_repl_prefix(m.get("content") or "").strip()
    return ""


def _looks_like_web_query(text: str) -> bool:
    t = _strip_repl_prefix(text).strip().lower()
    return t.startswith("web:") or t.startswith("search:")


def _looks_like_repo_research_query(text: str) -> bool:
    t = _strip_repl_prefix(text).strip().lower()
    return t.startswith("research:") or t.startswith("reindex:")


def _last_user_index(state: LlamiaState) -> int:
    for i in range(len(state.messages) - 1, -1, -1):
        if state.messages[i].get("role") == "user":
            return i
    return -1


def _web_ran_this_turn(state: LlamiaState) -> bool:
    ui = _last_user_index(state)
    if ui < 0:
        return False

    for m in state.messages[ui + 1:]:
        if m.get("role") == "system" and m.get("node") == "research_web":
            c = (m.get("content") or "").strip()
            # ONLY count as “ran” if we have actual results
            if c.startswith("[web_search results]"):
                return True
    return False


def _research_ran_this_turn(state: LlamiaState) -> bool:
    ui = _last_user_index(state)
    if ui < 0:
        return False

    for m in state.messages[ui + 1:]:
        if m.get("role") == "system" and m.get("node") == "research":
            c = (m.get("content") or "").strip()
            if c.startswith("[research results]"):
                return True
    return False


def _unique_files_tail(state: LlamiaState, limit: int = 12) -> list[str]:
    seen = set()
    out: list[str] = []
    for p in reversed(state.applied_patches or []):
        fp = getattr(p, "file_path", None)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
        if len(out) >= limit:
            break
    return list(reversed(out))


def _format_exec_summary(state: LlamiaState) -> tuple[str, bool]:
    results = state.last_exec_results or []
    if not results:
        return "No commands were executed.", True

    all_ok = all(r.returncode == 0 for r in results)
    lines: list[str] = []
    for r in results:
        status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
        lines.append(f"- {r.command} -> {status}")

        out_tail = _tail((r.stdout or "").strip(), MAX_STD_TAIL).strip()
        if out_tail:
            lines.append(f"  stdout (tail):\n    {out_tail}")
        err_tail = _tail((r.stderr or "").strip(), MAX_ERR_TAIL).strip()

        if out_tail:
            lines.append("  stdout (tail):")
            for line in out_tail.splitlines():
                lines.append(f"    {line}")
        if err_tail:
            lines.append("  stderr (tail):")
            for line in err_tail.splitlines():
                lines.append(f"    {line}")

    return "\n".join(lines), all_ok


def _task_final_message(state: LlamiaState) -> str:
    goal = (state.goal or "").strip() or "(no goal recorded)"
    files = _unique_files_tail(state)
    exec_summary, all_ok = _format_exec_summary(state)

    lines: list[str] = []
    lines.append(f"Task: {goal}")
    lines.append("")
    if files:
        lines.append("Files updated:")
        for fp in files:
            lines.append(f"- workspace/{fp}")
        lines.append("")

    if _web_ran_this_turn(state):
        lines.append("Web research: yes (research_web ran during this task)")
        lines.append("")

    lines.append("Execution:")
    lines.append(exec_summary)
    lines.append("")
    lines.append("Result: SUCCESS" if all_ok else "Result: FAILED (see stderr above)")
    return "\n".join(lines).strip()


def _web_final_message(state: LlamiaState) -> str:
    notes = (state.research_notes or "").strip()
    if not notes:
        return "No web results were captured."
    lines = notes.splitlines()
    snippet = "\n".join(lines[:40]).strip()
    return f"Here are the web results I fetched:\n\n{snippet}"


def _research_final_message(state: LlamiaState) -> str:
    notes = (state.research_notes or "").strip()
    if not notes:
        return "No repo research results were captured."
    lines = notes.splitlines()
    snippet = "\n".join(lines[:60]).strip()
    return f"Here are the repo research results:\n\n{snippet}"


def _trim_history_for_llm(state: LlamiaState, max_pairs: int = 10) -> list[dict]:
    msgs: list[dict] = []
    for m in state.messages:
        if m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m.get("content", "")})
    return msgs[-(2 * max_pairs):]


CHAT_GUARD_PROMPT = """You are Llamia's chat surface.

Rules:
- Do NOT claim you executed commands or edited files unless system messages from executor/coder indicate that.
- Keep replies concise and actionable.
"""


def chat_node(state: LlamiaState) -> LlamiaState:
    state.log(f"[{NODE_NAME}] starting")

    # HARD GUARD: only one assistant reply per REPL turn
    if state.responded_turn_id == state.turn_id:
        state.log(f"[{NODE_NAME}] already responded for turn_id={state.turn_id}; skipping")
        return state

    user_text = _latest_user_text(state)

    # 1) Task mode: deterministic summary (no LLM call)
    if state.mode == "task" and state.goal:
        reply = _task_final_message(state)
        state.add_message("assistant", reply, node=NODE_NAME)
        state.responded_turn_id = state.turn_id
        state.log(f"[{NODE_NAME}] finished (task summary) reply_len={len(reply)}")
        return state

    # 2) Repo research: deterministic summary (no LLM call)
    if _looks_like_repo_research_query(user_text) or _research_ran_this_turn(state):
        reply = _research_final_message(state)
        state.add_message("assistant", reply, node=NODE_NAME)
        state.responded_turn_id = state.turn_id
        state.log(f"[{NODE_NAME}] finished (research summary) reply_len={len(reply)}")
        return state

    # 3) Web query: deterministic summary (no LLM call)
    if _looks_like_web_query(user_text) or _web_ran_this_turn(state):
        reply = _web_final_message(state)
        state.add_message("assistant", reply, node=NODE_NAME)
        state.responded_turn_id = state.turn_id
        state.log(f"[{NODE_NAME}] finished (web summary) reply_len={len(reply)}")
        return state

    # 4) Normal chat: use model
    has_user = any(m.get("role") == "user" for m in state.messages)
    if not has_user:
        state.log(f"[{NODE_NAME}] no user messages in history; nothing to do")
        return state

    cfg = DEFAULT_CONFIG.model_for("chat")
    state.log(f"[{NODE_NAME}] using model={cfg.model} temp={cfg.temperature}")

    messages = [{"role": "system", "content": CHAT_GUARD_PROMPT, "node": NODE_NAME}]
    for m in _trim_history_for_llm(state):
        messages.append({"role": m["role"], "content": m["content"], "node": NODE_NAME})

    reply = chat_completion(messages=messages, model_cfg=cfg)
    state.add_message("assistant", reply, node=NODE_NAME)
    state.responded_turn_id = state.turn_id
    state.log(f"[{NODE_NAME}] finished reply_len={len(reply)}")
    return state
