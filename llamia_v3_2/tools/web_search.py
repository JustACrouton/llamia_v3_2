from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class WebResult:
    title: str
    url: str
    content: str = ""
    engine: str | None = None


def searxng_search(
    *,
    base_url: str,
    query: str,
    top_k: int = 5,
    timeout_s: int = 20,
) -> list[WebResult]:
    """
    Call SearXNG JSON API:
      GET {base_url}/search?q=...&format=json

    Returns a small list of WebResult.
    """
    base = base_url.rstrip("/")
    url = f"{base}/search"
    params = {"q": query, "format": "json"}

    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data: Any = resp.json()

    raw_results = data.get("results") or []
    out: list[WebResult] = []

    for r in raw_results:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "").strip()
        link = str(r.get("url") or r.get("link") or "").strip()
        content = str(r.get("content") or r.get("snippet") or "").strip()
        engine = r.get("engine")
        if not link:
            continue
        out.append(WebResult(title=title or link, url=link, content=content, engine=str(engine) if engine else None))
        if len(out) >= top_k:
            break

    return out
