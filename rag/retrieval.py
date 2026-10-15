"""Retrieval: dense (FAISS), sparse (BM25), Reciprocal Rank Fusion, cross-encoder reranking.

``Retriever.search`` is the single entry point used by the app, ingest checks
and the evals, so the evaluated pipeline is the code that serves users. It
returns :class:`Hit` objects and, if asked, a per-stage latency breakdown.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.embeddings import Embedder
from rag.index import RAGIndex

RRF_K = 60  # standard RRF constant (Cormack et al., SIGIR 2009)
MODES = ("dense", "bm25", "hybrid")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, mode: str = "plain") -> list[str]:
    """Lower-case word tokens for BM25 (TR / EN / DE).

    ``mode``:
      * ``"plain"``   – whole words.
      * ``"prefixN"`` – words truncated to their first N characters: a crude,
        language-independent stemmer for agglutinative Turkish
        ("Meclisinde" -> "mecli"). Can et al. (JASIST 59(3), 2008) found that
        simple word truncation is about as effective as a lemmatizer-based
        stemmer for Turkish IR. Whether it helps on this mixed TR/EN/DE
        corpus is measured in ``evals/``, not assumed.

    ``str.casefold`` maps German ß -> ss; Turkish dotted capital İ is mapped to
    i first (``"İ".lower()`` would give "i̇" with a combining dot).
    """
    tokens = _TOKEN_RE.findall(text.replace("İ", "i").casefold())
    if mode == "plain":
        return tokens
    m = re.fullmatch(r"prefix(\d+)", mode)
    if not m:
        raise ValueError(f"unknown tokenizer mode {mode!r}")
    n = int(m.group(1))
    return [t[:n] for t in tokens]


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion: ``score(d) = sum_i w_i / (k + rank_i(d))``.

    Ranks are 1-based; a document missing from a ranking contributes nothing
    for it. Only ranks matter, which is why RRF can merge unbounded BM25
    scores with cosine similarities without normalisation. Ties keep the
    order of first appearance (deterministic).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # ignore duplicates inside one ranking
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    order = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


class Reranker:
    """Multilingual cross-encoder via FastEmbed (ONNX, CPU)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3-int8", batch_size: int = 16):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts, batch_size=self.batch_size)]


@dataclass
class Hit:
    chunk: Document
    score: float  # dense cosine, BM25 score, RRF score or reranker logit, depending on the last stage
    rerank_score: float | None = None


@dataclass
class SearchTrace:
    timings_ms: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, t0: float) -> None:
        self.timings_ms[stage] = self.timings_ms.get(stage, 0.0) + (time.perf_counter() - t0) * 1000


class Retriever:
    def __init__(
        self,
        index: RAGIndex,
        embedder: Embedder,
        *,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        candidate_k: int = 20,
        reranker: Reranker | None = None,
    ):
        if embedder.name != index.embedder_name:
            raise ValueError(f"index built with {index.embedder_name}, query embedder is {embedder.name}")
        self.index = index
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_k = candidate_k
        self.reranker = reranker
        mode = index.settings.bm25_tokenizer
        self._bm25_mode = mode
        self._bm25 = BM25Okapi([tokenize(self._text(c), mode) or [""] for c in index.chunks])

    def _text(self, chunk: Document) -> str:
        return RAGIndex.index_text(chunk, self.index.settings)

    # --- single retrievers: return [(chunk_index, score)] ------------------

    def _dense(self, query: str, k: int, trace: SearchTrace) -> list[tuple[int, float]]:
        t0 = time.perf_counter()
        qv = self.embedder.embed_query(query)
        trace.add("embed_query", t0)
        t0 = time.perf_counter()
        out = self.index.dense_search(qv, k)
        trace.add("dense_search", t0)
        return out

    def _bm25_search(self, query: str, k: int, trace: SearchTrace) -> list[tuple[int, float]]:
        t0 = time.perf_counter()
        tokens = tokenize(query, self._bm25_mode)
        out: list[tuple[int, float]] = []
        if tokens:
            scores = self._bm25.get_scores(tokens)
            top = np.argsort(-scores, kind="stable")[:k]
            out = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
        trace.add("bm25", t0)
        return out

    def _first_stage(self, query: str, k: int, mode: str, trace: SearchTrace) -> list[tuple[int, float]]:
        if mode == "dense":
            return self._dense(query, k, trace)
        if mode == "bm25":
            return self._bm25_search(query, k, trace)
        n = max(k, self.candidate_k)
        dense = self._dense(query, n, trace)
        sparse = self._bm25_search(query, n, trace)
        t0 = time.perf_counter()
        fused = rrf_fuse(
            [[str(i) for i, _ in dense], [str(i) for i, _ in sparse]], [self.dense_weight, self.bm25_weight]
        )
        trace.add("fusion", t0)
        return [(int(i), s) for i, s in fused[:k]]

    def search(
        self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool = False, trace: SearchTrace | None = None
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        trace = trace if trace is not None else SearchTrace()
        if not rerank:
            return [Hit(self.index.chunks[i], s) for i, s in self._first_stage(query, k, mode, trace)]
        if self.reranker is None:
            raise ValueError("rerank=True but no reranker configured")
        cands = self._first_stage(query, max(k, self.candidate_k), mode, trace)
        t0 = time.perf_counter()
        scores = self.reranker.score(query, [self._text(self.index.chunks[i]) for i, _ in cands])
        trace.add("rerank", t0)
        order = sorted(range(len(cands)), key=lambda j: -scores[j])[:k]
        return [Hit(self.index.chunks[cands[j][0]], scores[j], scores[j]) for j in order]
