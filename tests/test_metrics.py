"""Metric and statistics helpers."""

import math

import numpy as np
import pytest
from langchain_core.documents import Document

from evals.dataset import DocText, EvalItem, Span
from evals.metrics import auroc, mean_ci, paired_permutation_test, query_metrics, sentence_hits_evidence

DOC = DocText("d.pdf", "x" * 1000, [0])


def _chunk(start, n=100):
    return Document(
        page_content="x" * n, metadata={"source": "d.pdf", "page": 1, "start_index": start, "chunk_chars": n}
    )


def test_query_metrics_ranks_and_recall():
    it = EvalItem("i", "test", "answerable", "en", "d.pdf", "q", "a", ["e1", "e2"])
    it.spans = [Span("d.pdf", 500, 540), Span("d.pdf", 900, 940)]
    it.pages = [1]
    ranked = [_chunk(0), _chunk(480), _chunk(100), _chunk(880)]
    m = query_metrics(ranked, it, {"d.pdf": DOC})
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr@10"] == pytest.approx(0.5)
    assert m["evidence_recall@5"] == 1.0  # both evidence spans covered within top 5
    assert query_metrics([_chunk(0)], it, {"d.pdf": DOC})["mrr@10"] == 0.0


def test_sentence_hits_evidence_uses_shorter_length():
    it = EvalItem("i", "test", "answerable", "en", "d.pdf", "q", "a", ["e"])
    it.spans = [Span("d.pdf", 100, 120)]
    assert sentence_hits_evidence(Span("d.pdf", 50, 200), it)  # long sentence containing the quote
    assert sentence_hits_evidence(Span("d.pdf", 105, 115), it)  # short sentence inside the quote
    assert not sentence_hits_evidence(Span("d.pdf", 115, 300), it)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    a, b = mean_ci(v), mean_ci(v)
    assert a == b and a["lo"] <= a["mean"] <= a["hi"] and a["mean"] == pytest.approx(0.7)
    assert math.isnan(mean_ci([])["mean"])


def test_permutation_test():
    rng = np.random.default_rng(1)
    x = rng.random(60)
    assert paired_permutation_test(x + 0.3, x)["p_value"] < 0.001
    assert paired_permutation_test(x, x)["p_value"] == 1.0
    noisy = paired_permutation_test(x, x + rng.normal(0, 0.01, 60))
    assert noisy["p_value"] > 0.05 or abs(noisy["mean_diff"]) < 0.01


def test_auroc():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1], [1]) == 0.5
    assert math.isnan(auroc([], [1]))
