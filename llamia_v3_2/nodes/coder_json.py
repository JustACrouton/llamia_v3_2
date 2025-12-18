from __future__ import annotations

"""
JSON extraction + strict retry utilities.

The coder node expects STRICT JSON. Models sometimes:
- wrap JSON in prose
- add markdown fences
- include trailing commentary

We parse defensively and allow exactly ONE strict retry.
"""

import json
from typing import Any

from ..llm_client import chat_completion
from .coder_prompts import STRICT_JSON_RETRY_SYSTEM


def try_parse_json_object(raw: str) -> dict[str, Any] | None:
    """
    Parse a JSON object even if wrapped in prose.

    Strategy:
      1) fast path: raw is exactly {...}
      2) fallback: take first '{' .. last '}' slice
    """
    if not raw:
        return None

    s = raw.strip()

    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

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


def retry_strict_json(messages: list[dict[str, str]], model_cfg, *, node_name: str) -> str:
    """
    One controlled retry: tells the model its output was invalid and demands strict JSON.
    """
    retry_msgs = list(messages)
    retry_msgs.append({"role": "system", "content": STRICT_JSON_RETRY_SYSTEM, "node": node_name})
    return chat_completion(messages=retry_msgs, model_cfg=model_cfg)
