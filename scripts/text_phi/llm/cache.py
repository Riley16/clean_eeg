"""SQLite content-hash cache for LLM responses.

Key is a SHA-256 over `(server_type, model, prompt_hash, input_hash,
context_hash, seed, temperature)` so a rerun with the same inputs and
same seed skips the LLM entirely, and any change to the prompt / input /
context / seed / temperature invalidates the entry.

Cache is PHI-agnostic — it just stores JSON strings keyed by a hash. Any
caller can use it (LLM-redaction is the primary consumer).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_str(text: str) -> str:
    """Public helper for callers that need to hash prompt / input / context
    strings into the components of the cache key."""
    return _sha(text)


def build_cache_key(
    server_type: str,
    model: str,
    prompt_hash: str,
    input_hash: str,
    context_hash: str,
    seed: int,
    temperature: float,
) -> str:
    """Canonical cache-key layout. Changing any component invalidates the
    entry."""
    material = "|".join([
        server_type,
        model,
        prompt_hash,
        input_hash,
        context_hash,
        str(seed),
        f"{float(temperature):.6f}",
    ])
    return _sha(material)


class LLMCache:
    """SQLite-backed cache of raw LLM responses (as JSON strings).

    The cache is a plain key→value store. Composing a key is the caller's
    responsibility via `build_cache_key`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response  TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, response, created_at) "
            "VALUES (?, ?, ?)",
            (key, response, int(time.time())),
        )
        self._conn.commit()

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()
        return cur.rowcount

    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
