"""SQLite response cache for LLM calls.

Every LLM call in the project goes through this layer before touching the
network — re-running evals against a paid API without a cache burns credits
on identical requests. The key hashes the full request (model, messages,
generation params), so any change to the prompt template, retrieved context,
or model invalidates naturally.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import ROOT

_DEFAULT_PATH = ROOT / ".cache" / "llm_cache.sqlite3"


def request_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
