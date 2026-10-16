"""Answering on top of retrieval: abstention gate and an offline extractive answerer.

Both work without any LLM or API key:

* **Abstention.** The cross-encoder score of the best passage is used as a
  retrieval-confidence signal. Below a threshold the system answers
  "not found in the sources" instead of guessing. The default threshold is
  calibrated on the dev split (``evals/``) so that about 95 % of answerable
  dev questions pass the gate; its effect on unanswerable questions is then
  measured on the test split.
* **Extractive answers.** The top passages are split into sentences, every
  sentence is scored against the question with the same cross-encoder, and
  the best sentences are returned verbatim with ``[n]`` citations. Every word
  of such an answer is copied from a cited source, so it cannot contain a
  claim that is absent from the cited passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from rag.retrieval import Hit, Reranker

_BOUNDARY_RE = re.compile(r"[.!?…]+[»«\"“”’']*\s+|\n\s*\n")
MIN_SENTENCE_CHARS = 25


@dataclass
class Sentence:
    text: str
    hit_rank: int  # 1-based position of the source passage in the retrieved list
    start: int  # offset inside the passage text
    score: float = 0.0


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Return (offset, sentence) pairs.

    Boundaries are sentence punctuation followed by whitespace, or a blank
    line. Single newlines are *not* boundaries: in PDF text they are usually
    just visual line wraps. Fragments shorter than ``MIN_SENTENCE_CHARS`` are
    merged into the following sentence.
    """
    cuts = sorted({m.end() for m in _BOUNDARY_RE.finditer(text)} | {len(text)})
    out: list[tuple[int, str]] = []
    start = 0
    for end in cuts:
        seg = text[start:end]
        if seg.strip() and (len(seg.strip()) >= MIN_SENTENCE_CHARS or end == len(text)):
            lead = len(seg) - len(seg.lstrip())
            out.append((start + lead, seg.strip()))
            start = end
        elif not seg.strip():
            start = end
    return out


def is_confident(hits: list[Hit], threshold: float | None) -> bool:
    """False when the best reranker score is below ``threshold`` (or nothing was found)."""
    if not hits:
        return False
    if threshold is None or hits[0].rerank_score is None:
        return True
    return hits[0].rerank_score >= threshold


def extractive_answer(
    question: str, hits: list[Hit], reranker: Reranker, *, passages: int = 3, max_sentences: int = 2
) -> list[Sentence]:
    """Best ``max_sentences`` sentences from the first ``passages`` hits, in reading order."""
    cands: list[Sentence] = []
    for rank, h in enumerate(hits[:passages], start=1):
        for off, s in split_sentences(h.chunk.page_content):
            cands.append(Sentence(s, rank, off))
    if not cands:
        return []
    for c, score in zip(cands, reranker.score(question, [c.text for c in cands]), strict=True):
        c.score = score
    best = sorted(cands, key=lambda c: -c.score)[:max_sentences]
    return sorted(best, key=lambda c: (c.hit_rank, c.start))


def format_extractive(sentences: list[Sentence]) -> str:
    """Markdown block quote, one cited sentence per paragraph."""
    return "\n>\n".join(f"> {s.text} [{s.hit_rank}]" for s in sentences)


def source_label(doc: Document) -> str:
    return f"{doc.metadata.get('source', '?')}, s. {doc.metadata.get('page', '?')}"
