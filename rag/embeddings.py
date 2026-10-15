"""Embedding backends and an on-disk embedding cache."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings


def _retry(fn, *, attempts: int = 6, base_delay: float = 2.0):
    """Call ``fn`` with exponential backoff on transient API errors (429/5xx)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # provider SDKs raise different exception types
            msg = str(e)
            transient = any(
                code in msg
                for code in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                    "DEADLINE",
                )
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)


class GeminiEmbeddings(Embeddings):
    """Google ``gemini-embedding-001`` via the ``google-genai`` SDK.

    Uses the retrieval task types (``RETRIEVAL_DOCUMENT`` for chunks,
    ``RETRIEVAL_QUERY`` for questions), batches requests, retries transient
    errors and L2-normalises vectors, so L2 distance in FAISS ranks the same
    way as cosine similarity.
    """

    def __init__(
        self, model: str = "gemini-embedding-001", batch_size: int = 100, api_key: str | None = None
    ):
        from google import genai

        self.model = model
        self.batch_size = batch_size
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = _retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model, contents=b, config=types.EmbedContentConfig(task_type=task_type)
                )
            )
            for e in resp.embeddings:
                v = np.asarray(e.values, dtype=np.float32)
                out.append((v / (np.linalg.norm(v) or 1.0)).tolist())
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class FastEmbedEmbeddings(Embeddings):
    """Local, offline embeddings via FastEmbed (ONNX). Zero API key needed."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # FastEmbed returns a generator of numpy arrays
        return [v.tolist() for v in self._model.embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


def make_embeddings(provider: str, model: str) -> Embeddings:
    """Return a LangChain ``Embeddings`` object for ``provider``/``model``."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env or your system environment.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    if provider == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found. Please set it in .env or your system environment.")
        return GeminiEmbeddings(model=model)
    if provider == "fastembed":
        return FastEmbedEmbeddings(model_name=model)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


class CachedEmbeddings(Embeddings):
    """Wrap any ``Embeddings`` with a SQLite cache keyed by (namespace, kind, text).

    Re-running ingestion or evals only pays for texts that were never embedded
    before. ``namespace`` must identify the model (e.g. ``"google/gemini-embedding-001"``)
    so vectors from different models never mix. Queries and documents are
    cached separately because some providers embed them differently.
    """

    def __init__(self, base: Embeddings, cache_path: str | Path, namespace: str):
        self.base = base
        self.namespace = namespace
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{kind}\x00{text}".encode()).hexdigest()

    def _get(self, key: str) -> list[float] | None:
        row = self._db.execute("SELECT vec FROM emb WHERE key = ?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).tolist() if row else None

    def _put_many(self, items: list[tuple[str, list[float]]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )
            self._db.commit()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key("doc", t) for t in texts]
        result: list[list[float] | None] = [self._get(k) for k in keys]
        missing = [i for i, v in enumerate(result) if v is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            new = self.base.embed_documents([texts[i] for i in missing])
            for i, v in zip(missing, new, strict=False):
                result[i] = v
            self._put_many([(keys[i], result[i]) for i in missing])
        return result  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        v = self.base.embed_query(text)
        self._put_many([(key, v)])
        return v
