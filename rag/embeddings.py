"""Embedding backends (all return L2-normalised float32 vectors) and a disk cache.

Normalised vectors + inner product = cosine similarity, which is what the
sentence-transformers paraphrase models are trained for. (The previous code
fed raw, unnormalised FastEmbed vectors into an L2 index.)

``Embedder.max_tokens`` / ``count_tokens`` expose the model's input limit so
the index can split long chunks into windows instead of silently losing
everything after the limit: ``paraphrase-multilingual-MiniLM-L12-v2`` truncates
at 128 tokens, roughly 450-500 characters of Turkish or German text.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a / (np.linalg.norm(a) or 1.0)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def _retry(fn, *, attempts: int = 6, base_delay: float = 2.0):
    """Call ``fn`` with exponential backoff on transient API errors (429/5xx)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # provider SDKs raise different exception types
            msg = str(e)
            transient = any(c in msg for c in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)


class Embedder:
    """Interface. ``name`` must identify provider + model (used as cache namespace)."""

    name: str = "?"
    max_tokens: int | None = None  # None: no practical limit for our chunk sizes

    def count_tokens(self, text: str) -> int:  # pragma: no cover - only needed if max_tokens is set
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    """Local, offline ONNX embeddings via FastEmbed. No API key needed."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        self.name = f"fastembed/{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        tok = self._model.model.tokenizer  # tokenizers.Tokenizer configured by FastEmbed
        trunc = tok.truncation or {}
        self.max_tokens = int(trunc["max_length"]) if trunc.get("max_length") else None
        self._counter = Tokenizer.from_str(tok.to_str())  # independent copy without truncation
        self._counter.no_truncation()
        self._counter.no_padding()

    def count_tokens(self, text: str) -> int:
        return len(self._counter.encode(text).ids)  # includes the special tokens, like the model input

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.stack(list(self._model.embed(list(texts), batch_size=64))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(next(iter(self._model.embed([text]))))


class GeminiEmbedder(Embedder):
    """Google ``gemini-embedding-001`` via ``google-genai`` (retrieval task types, batched)."""

    def __init__(self, model: str = "gemini-embedding-001", batch_size: int = 100):
        from google import genai

        self.name = f"google/{model}"
        self.model = model
        self.batch_size = batch_size
        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = _retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model, contents=b, config=types.EmbedContentConfig(task_type=task_type)
                )
            )
            out.extend(e.values for e in resp.embeddings)
        return _normalise(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 256):
        from openai import OpenAI

        self.name = f"openai/{model}"
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = _retry(lambda b=batch: self._client.embeddings.create(model=self.model, input=b))
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.asarray(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def make_embedder(provider: str, model: str) -> Embedder:
    if provider == "fastembed":
        return FastEmbedEmbedder(model)
    if provider == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env or your environment.")
        return GeminiEmbedder(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Set it in .env or your environment.")
        return OpenAIEmbedder(model)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


class CachedEmbedder(Embedder):
    """SQLite cache keyed by (model name, kind, text). Vectors of different models never mix."""

    def __init__(self, base: Embedder, cache_path: str | Path):
        self.base = base
        self.name = base.name
        self.max_tokens = base.max_tokens
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def count_tokens(self, text: str) -> int:
        return self.base.count_tokens(text)

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha256(f"{self.name}\x00{kind}\x00{text}".encode()).hexdigest()

    def _get(self, key: str) -> np.ndarray | None:
        row = self._db.execute("SELECT vec FROM emb WHERE key = ?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).copy() if row else None

    def _put(self, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )
            self._db.commit()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key("doc", t) for t in texts]
        found = [self._get(k) for k in keys]
        missing = [i for i, v in enumerate(found) if v is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            new = self.base.embed_documents([texts[i] for i in missing])
            for j, i in enumerate(missing):
                found[i] = new[j]
            self._put([(keys[i], found[i]) for i in missing])
        return np.stack(found) if found else np.zeros((0, 0), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        key = self._key("query", text)
        v = self._get(key)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        v = self.base.embed_query(text)
        self._put([(key, v)])
        return v
