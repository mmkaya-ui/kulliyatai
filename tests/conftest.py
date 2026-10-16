"""Shared fixtures: a deterministic fake embedder/reranker so retrieval logic is tested without models."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pytest

from rag.embeddings import Embedder
from rag.retrieval import Reranker

DIM = 2048


class HashEmbedder(Embedder):
    """Bag-of-words hashing embedder: texts sharing words get similar vectors.

    ``max_tokens`` (whitespace tokens + 2 "special" tokens) lets tests exercise
    the windowing code path.
    """

    def __init__(self, max_tokens: int | None = None):
        self.name = f"test/hash-{max_tokens}"
        self.max_tokens = max_tokens
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if self.max_tokens is not None:
            words = words[: self.max_tokens - 2]  # emulate model truncation
        for w in words:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker(Reranker):
    """Scores = number of shared lower-case words (no model)."""

    def __init__(self):
        self.model_name = "test/overlap"
        self.batch_size = 1

    def score(self, query, texts):
        q = set(re.findall(r"\w+", query.lower()))
        return [float(len(q & set(re.findall(r"\w+", t.lower())))) for t in texts]


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


@pytest.fixture
def overlap_reranker():
    return OverlapReranker()


def models_available() -> bool:
    return os.environ.get("RUN_MODEL_TESTS") == "1"
