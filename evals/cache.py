"""Disk cache for cross-encoder scores, so that re-running the benchmark is cheap.

Latency is never measured through this cache (see ``evals/run.py``).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from rag.retrieval import Reranker


class CachedReranker(Reranker):
    def __init__(self, base: Reranker, cache_path: str | Path):
        self.base = base
        self.model_name = base.model_name
        self.batch_size = base.batch_size
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path))
        self._db.execute("CREATE TABLE IF NOT EXISTS rr (key TEXT PRIMARY KEY, score REAL)")
        self.hits = self.misses = 0

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{query}\x00{text}".encode()).hexdigest()

    def score(self, query: str, texts: list[str]) -> list[float]:
        keys = [self._key(query, t) for t in texts]
        out: list[float | None] = []
        for k in keys:
            row = self._db.execute("SELECT score FROM rr WHERE key = ?", (k,)).fetchone()
            out.append(row[0] if row else None)
        missing = [i for i, v in enumerate(out) if v is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            new = self.base.score(query, [texts[i] for i in missing])
            for i, s in zip(missing, new, strict=True):
                out[i] = s
            self._db.executemany("INSERT OR REPLACE INTO rr VALUES (?, ?)", [(keys[i], out[i]) for i in missing])
            self._db.commit()
        return out  # type: ignore[return-value]
