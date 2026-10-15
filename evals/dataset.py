"""Evaluation dataset: loading, machine verification and evidence-based relevance.

Gold labels are *not* hand-typed page numbers. Every answerable item carries
one or more ``evidence`` strings copied verbatim from its document. At load
time each string is located in the text that :mod:`rag.loading` extracts from
the PDF (matching is whitespace-insensitive, nothing else), which yields
character spans and therefore pages. Consequences:

* a typo, a wrong document or a quote that is not in the corpus is a hard
  error (``python -m evals.dataset`` and ``tests/test_dataset.py``);
* if the PDFs are re-rendered and page breaks move, labels follow
  automatically;
* relevance can be judged per chunk ("does this chunk contain the evidence?")
  instead of per page.

Unanswerable items list ``absent_terms``: words or phrases that would carry the
answer. The validator checks that none of them occurs anywhere in the corpus
(case-insensitive, whole words).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag.config import ROOT

QUESTIONS_FILE = ROOT / "evals" / "questions.jsonl"
LANGS = {"tr", "en", "de"}

# A chunk counts as relevant if it contains at least this share of one evidence span.
MIN_EVIDENCE_COVERAGE = 0.5


@dataclass(frozen=True)
class Span:
    doc: str
    start: int  # offsets into DocText.text
    end: int


@dataclass
class DocText:
    """Concatenated page texts of one PDF, with page offsets."""

    name: str
    text: str
    page_starts: list[int]  # page_starts[i] = offset of page i+1

    def page_of(self, offset: int) -> int:
        lo, hi = 0, len(self.page_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.page_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    @classmethod
    def from_pages(cls, name: str, pages: list[Document]) -> DocText:
        pages = sorted(pages, key=lambda d: d.metadata["page"])
        n_pages = max(d.metadata.get("total_pages", d.metadata["page"]) for d in pages)
        by_page = {d.metadata["page"]: d.page_content for d in pages}
        parts, starts, pos = [], [], 0
        for p in range(1, n_pages + 1):
            starts.append(pos)
            t = by_page.get(p, "")
            parts.append(t)
            pos += len(t) + 1  # "\n" joiner
        return cls(name, "\n".join(parts), starts)


@dataclass
class EvalItem:
    id: str
    split: str
    type: str
    lang: str
    doc: str | None
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    absent_terms: list[str] = field(default_factory=list)
    why_unanswerable: str | None = None
    spans: list[Span] = field(default_factory=list)  # filled by resolve()
    pages: list[int] = field(default_factory=list)  # filled by resolve()

    @property
    def answerable(self) -> bool:
        return self.type == "answerable"

    @property
    def doc_lang(self) -> str | None:
        return self.doc[:2] if self.doc else None

    @property
    def cross_lingual(self) -> bool:
        return self.answerable and self.lang != self.doc_lang


class DatasetError(ValueError):
    pass


def load_items(path: Path = QUESTIONS_FILE) -> list[EvalItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(EvalItem(**json.loads(line)))
            except TypeError as e:
                raise DatasetError(f"{path.name}:{n}: {e}") from e
    return items


def _pattern(evidence: str) -> re.Pattern:
    return re.compile(r"\s+".join(re.escape(tok) for tok in evidence.split()))


def corpus_texts(pages: list[Document]) -> dict[str, DocText]:
    by_doc: dict[str, list[Document]] = {}
    for d in pages:
        by_doc.setdefault(d.metadata["source"], []).append(d)
    return {name: DocText.from_pages(name, ps) for name, ps in by_doc.items()}


def resolve(items: list[EvalItem], docs: dict[str, DocText]) -> list[str]:
    """Locate evidence, check absent terms; fill spans/pages. Return a list of errors."""
    errors: list[str] = []
    seen: set[str] = set()
    for it in items:
        where = f"[{it.id}]"
        if it.id in seen:
            errors.append(f"{where} duplicate id")
        seen.add(it.id)
        if it.split not in {"dev", "test"}:
            errors.append(f"{where} split must be dev/test")
        if it.lang not in LANGS:
            errors.append(f"{where} lang must be one of {sorted(LANGS)}")
        if it.type == "answerable":
            if it.doc not in docs:
                errors.append(f"{where} unknown doc {it.doc!r}")
                continue
            if not it.evidence:
                errors.append(f"{where} answerable item without evidence")
            it.spans, pages = [], set()
            for ev in it.evidence:
                matches = list(_pattern(ev).finditer(docs[it.doc].text))
                if len(matches) != 1:
                    errors.append(f"{where} evidence found {len(matches)}x (need exactly 1): {ev[:70]!r}")
                    continue
                s = Span(it.doc, matches[0].start(), matches[0].end())
                it.spans.append(s)
                pages.update(range(docs[it.doc].page_of(s.start), docs[it.doc].page_of(s.end - 1) + 1))
            it.pages = sorted(pages)
        elif it.type == "unanswerable":
            if it.evidence:
                errors.append(f"{where} unanswerable item must not have evidence")
            if not it.absent_terms:
                errors.append(f"{where} unanswerable item needs absent_terms")
            for term in it.absent_terms:
                rx = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, term.split())) + r"(?!\w)", re.I)
                hits = [name for name, d in docs.items() if rx.search(d.text)]
                if hits:
                    errors.append(f"{where} absent term {term!r} occurs in {hits}")
        else:
            errors.append(f"{where} type must be answerable/unanswerable")
    return errors


def load_verified(pages: list[Document], path: Path = QUESTIONS_FILE) -> list[EvalItem]:
    """Load + resolve, raising :class:`DatasetError` listing every problem."""
    items = load_items(path)
    errors = resolve(items, corpus_texts(pages))
    if errors:
        raise DatasetError("questions.jsonl failed verification:\n  " + "\n  ".join(errors))
    return items


# --- relevance -------------------------------------------------------------


def chunk_span(chunk: Document, docs: dict[str, DocText]) -> Span | None:
    """Character span of a chunk inside its document (chunks never cross pages)."""
    m = chunk.metadata
    d = docs.get(m.get("source"))
    if d is None or "page" not in m or "start_index" not in m:
        return None
    start = d.page_starts[m["page"] - 1] + m["start_index"]
    return Span(d.name, start, start + int(m.get("chunk_chars", len(chunk.page_content))))


def covers(chunk: Span | None, evidence: Span) -> bool:
    if chunk is None or chunk.doc != evidence.doc:
        return False
    overlap = min(chunk.end, evidence.end) - max(chunk.start, evidence.start)
    return overlap >= MIN_EVIDENCE_COVERAGE * (evidence.end - evidence.start)


def relevant(chunk: Document, item: EvalItem, docs: dict[str, DocText]) -> bool:
    cs = chunk_span(chunk, docs)
    return any(covers(cs, s) for s in item.spans)


def main() -> int:
    from rag.loading import load_pdfs_with_pymupdf

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pages = load_pdfs_with_pymupdf(ROOT / "sample_docs", log=lambda *_: None)
    items = load_items()
    errors = resolve(items, corpus_texts(pages))
    for e in errors:
        print("ERROR", e)
    ans = [i for i in items if i.answerable]
    print(
        f"{len(items)} items: {len(ans)} answerable ({sum(i.cross_lingual for i in ans)} cross-lingual), "
        f"{len(items) - len(ans)} unanswerable; dev={sum(i.split == 'dev' for i in items)}, "
        f"test={sum(i.split == 'test' for i in items)}; {len(errors)} errors"
    )
    if "--pages" in sys.argv:
        for i in ans:
            print(f"  {i.id:5} {i.doc:34} pages={i.pages}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
