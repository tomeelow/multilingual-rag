"""Thin HTTP client for the RAG API, kept free of Streamlit imports so the
SSE protocol seam stays unit-testable."""

import json
from collections.abc import Iterable, Iterator
from typing import Any

import httpx


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse `data: {json}` lines into event dicts; ignore everything else."""
    for line in lines:
        if line.startswith("data: "):
            yield json.loads(line.removeprefix("data: "))


def stream_chat(
    api_url: str,
    query: str,
    filter_language: str | None,
    cross_lingual: bool,
    use_hyde: bool,
    session_id: str | None = None,
    timeout: float = 120.0,
) -> Iterator[dict[str, Any]]:
    payload = {
        "query": query,
        "filter_language": filter_language,
        "cross_lingual": cross_lingual,
        "use_hyde": use_hyde,
        "session_id": session_id,
    }
    with httpx.stream("POST", f"{api_url}/api/chat", json=payload, timeout=timeout) as r:
        r.raise_for_status()
        yield from iter_sse_events(r.iter_lines())


def get_sources(api_url: str) -> list[dict[str, Any]]:
    return httpx.get(f"{api_url}/api/sources", timeout=10).json()


def get_health(api_url: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{api_url}/api/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None
