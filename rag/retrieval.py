"""Retrieval: dense (FAISS), sparse (BM25), Reciprocal Rank Fusion, optional reranking.

The original app used LangChain's ``EnsembleRetriever`` (weighted RRF over a
FAISS and a BM25 retriever). This module implements the same fusion formula
explicitly so that it is unit-testable, and so the evals can switch between
``dense`` / ``bm25`` / ``hybrid`` / ``hybrid + rerank`` with identical code.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.chunking import strip_context_header

RRF_K = 60  # standard RRF constant (also LangChain EnsembleRetriever's default)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer for BM25 that works for TR / EN / DE.

    ``str.casefold`` handles German ß -> ss. Turkish dotted capital İ is mapped
    to i first (``"İ".lower()`` would otherwise yield "i̇" with a combining dot).
    """
    return _TOKEN_RE.findall(text.replace("İ", "i").casefold())


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion.

    ``score(d) = sum_i  w_i / (k + rank_i(d))`` with 1-based ranks; documents
    missing from a ranking contribute nothing for it. Only ranks matter, not
    raw scores, which is why RRF can merge BM25 scores (unbounded) with
    cosine similarities (bounded) without any normalisation.

    Returns ``(id, score)`` sorted by score descending; ties keep the order of
    first appearance (deterministic).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=False):
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # ignore duplicates inside one ranking
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    order = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def doc_id(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.page_content


class Reranker:
    """Cross-encoder reranker (fastembed / ONNX, runs on CPU). Optional dependency."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3-int8"):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:  # pragma: no cover - depends on optional package
            raise ImportError(
                "Reranking needs the optional 'fastembed' package: pip install fastembed"
            ) from e
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts, batch_size=16)]

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        scores = self.score(query, [d.page_content for d in docs])
        ranked = sorted(zip(docs, scores, strict=False), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_k]]


class HybridRetriever:
    """Dense + BM25 retrieval over the chunks stored in a LangChain vector store.

    BM25 indexes the chunk text **without** the ``[Kaynak: … | Sayfa: …]``
    enrichment header, so file names and the word "Sayfa" don't add noise to
    keyword matching. The dense side embeds the enriched text.
    """

    def __init__(
        self,
        vectorstore,
        documents: list[Document] | None = None,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        candidate_k: int = 20,
        reranker: Reranker | None = None,
    ):
        self.vectorstore = vectorstore
        if documents is None:
            # Extract all documents from FAISS for BM25 indexing
            documents = list(vectorstore.docstore._dict.values())
        if not documents:
            raise ValueError("No documents to index")
        self.documents = documents
        self._by_id = {doc_id(d): d for d in documents}
        self._bm25 = BM25Okapi([tokenize(strip_context_header(d.page_content)) or [""] for d in documents])
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_k = candidate_k
        self.reranker = reranker

    def dense(self, query: str, k: int) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def bm25(self, query: str, k: int) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.documents[i] for i in top if scores[i] > 0]

    def hybrid(self, query: str, k: int) -> list[Document]:
        n = max(k, self.candidate_k)
        dense_ids = [doc_id(d) for d in self.dense(query, n)]
        bm25_ids = [doc_id(d) for d in self.bm25(query, n)]
        fused = rrf_fuse([dense_ids, bm25_ids], [self.dense_weight, self.bm25_weight])
        return [self._by_id[i] for i, _ in fused[:k] if i in self._by_id]

    def retrieve(self, query: str, k: int, mode: str = "hybrid", rerank: bool = False) -> list[Document]:
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        search = getattr(self, mode)
        if not rerank:
            return search(query, k)
        if self.reranker is None:
            raise ValueError("rerank=True but no reranker configured")
        candidates = search(query, max(k, self.candidate_k))
        return self.reranker.rerank(query, candidates, top_k=k)
