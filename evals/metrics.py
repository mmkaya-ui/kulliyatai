"""Evaluation metric functions: Hit@K, MRR, Citation Correctness, and Refusal Accuracy."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag.chunking import strip_context_header
from rag.prompts import NO_ANSWER, is_refusal, split_claims


def doc_matches(doc: Document, expected_doc: str, expected_pages: list[int]) -> bool:
    """Check if a retrieved Document matches the expected source filename and page."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    if source != expected_doc:
        return False
    if not expected_pages:
        return True
    return page in expected_pages


def hit_at_k(retrieved_docs: list[Document], expected_doc: str, expected_pages: list[int], k: int) -> float:
    """Return 1.0 if any of the top-k retrieved docs matches the target doc & page, else 0.0."""
    candidates = retrieved_docs[:k]
    for doc in candidates:
        if doc_matches(doc, expected_doc, expected_pages):
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_docs: list[Document], expected_doc: str, expected_pages: list[int]) -> float:
    """Compute Reciprocal Rank (1 / rank) for the first relevant retrieved document."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc_matches(doc, expected_doc, expected_pages):
            return 1.0 / rank
    return 0.0


def evaluate_refusal_accuracy(unanswerable_entries: list[dict[str, Any]], answers: list[str]) -> float:
    """Check what fraction of unanswerable questions triggered a proper refusal."""
    if not unanswerable_entries:
        return 1.0
    correct = 0
    for _q, ans in zip(unanswerable_entries, answers, strict=False):
        if is_refusal(ans) or NO_ANSWER in ans or "bulunamadı" in ans.lower() or "not found" in ans.lower():
            correct += 1
    return correct / len(unanswerable_entries)


def evaluate_citation_correctness_heuristic(answer: str, source_docs: list[Document]) -> dict[str, Any]:
    """Evaluate citation accuracy without calling an external LLM.

    Checks:
    1. Are cited indices within range [1..len(source_docs)]?
    2. Does the cited source chunk contain key terms / n-grams from the citing claim?
    """
    claims = split_claims(answer)
    total_citations = 0
    supported_citations = 0
    hallucinated_indices = 0

    doc_texts = {i: strip_context_header(d.page_content).lower() for i, d in enumerate(source_docs, start=1)}

    for claim_text, cite_nums in claims:
        for num in cite_nums:
            total_citations += 1
            if num not in doc_texts:
                hallucinated_indices += 1
                continue

            doc_text = doc_texts[num]
            # Extract content words from claim (length > 3)
            claim_words = [w.lower().strip(".,;:?!'\"()[]") for w in claim_text.split() if len(w) > 3]
            overlap = [w for w in claim_words if w in doc_text]
            # If at least 30% of significant claim words appear in the cited chunk:
            if len(overlap) >= max(1, int(len(claim_words) * 0.25)):
                supported_citations += 1

    accuracy = (supported_citations / total_citations) if total_citations > 0 else 1.0
    return {
        "total_citations": total_citations,
        "supported_citations": supported_citations,
        "hallucinated_indices": hallucinated_indices,
        "accuracy": accuracy,
    }
