"""Retrieval / abstention / extraction metrics and the statistics used to report them.

Relevance is evidence-based (see :mod:`evals.dataset`): a retrieved chunk is
relevant if it contains at least half of one of the item's evidence spans.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document

from evals.dataset import DocText, EvalItem, Span, chunk_span, covers

KS = (1, 3, 5)
MRR_DEPTH = 10


def relevance_vector(chunks: list[Document], item: EvalItem, docs: dict[str, DocText]) -> list[bool]:
    spans = [chunk_span(c, docs) for c in chunks]
    return [any(covers(cs, ev) for ev in item.spans) for cs in spans]


def query_metrics(chunks: list[Document], item: EvalItem, docs: dict[str, DocText]) -> dict[str, float]:
    rel = relevance_vector(chunks, item, docs)
    out = {f"hit@{k}": float(any(rel[:k])) for k in KS}
    first = next((i for i, r in enumerate(rel[:MRR_DEPTH], start=1) if r), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    spans5 = [chunk_span(c, docs) for c in chunks[:5]]
    out["evidence_recall@5"] = sum(any(covers(cs, ev) for cs in spans5) for ev in item.spans) / len(item.spans)
    pages = {(c.metadata.get("source"), c.metadata.get("page")) for c in chunks[:5]}
    out["page_hit@5"] = float(any((item.doc, p) in pages for p in item.pages))
    return out


def sentence_hits_evidence(sent_span: Span, item: EvalItem) -> bool:
    """A sentence supports the answer if it overlaps an evidence span substantially.

    Overlap must reach half of the shorter of the two (a short evidence quote
    inside a long sentence counts, and so does a sentence inside a long quote).
    """
    for ev in item.spans:
        if ev.doc != sent_span.doc:
            continue
        overlap = min(ev.end, sent_span.end) - max(ev.start, sent_span.start)
        if overlap >= 0.5 * min(ev.end - ev.start, sent_span.end - sent_span.start):
            return True
    return False


# --- statistics -------------------------------------------------------------


def mean_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Mean and percentile-bootstrap (1-alpha) CI over queries."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10_000, seed: int = 0):
    """Two-sided paired randomization (sign-flip) test on per-query differences a-b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0 or not np.any(d):
        return {"mean_diff": float(d.mean()) if len(d) else 0.0, "p_value": 1.0, "n": int(len(d))}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    p = (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1)
    return {"mean_diff": float(d.mean()), "p_value": float(p), "n": int(len(d))}


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else math.nan
