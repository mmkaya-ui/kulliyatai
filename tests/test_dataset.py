"""The evaluation set is machine-verified against the PDFs (no models needed)."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from evals.dataset import (
    DatasetError,
    DocText,
    EvalItem,
    Span,
    chunk_span,
    corpus_texts,
    covers,
    load_items,
    load_verified,
    resolve,
)
from rag.chunking import split_documents
from rag.loading import load_pdfs_with_pymupdf

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pages():
    return load_pdfs_with_pymupdf(ROOT / "sample_docs", log=lambda *_: None)


def test_questions_file_verifies_against_corpus(pages):
    items = load_verified(pages)  # raises DatasetError listing every problem
    ans = [i for i in items if i.answerable]
    un = [i for i in items if not i.answerable]
    assert len(items) == len({i.id for i in items})
    assert ans and un
    assert all(i.pages for i in ans)
    assert {i.split for i in items} == {"dev", "test"}
    # every document and every question language is represented in both splits
    for split in ("dev", "test"):
        assert {i.doc for i in ans if i.split == split} == {p.metadata["source"] for p in pages}
        assert {i.lang for i in ans if i.split == split} == {"tr", "en", "de"}
    assert any(i.cross_lingual for i in ans)


def _item(**kw):
    base = dict(
        id="x",
        split="dev",
        type="answerable",
        lang="en",
        doc="en_us_constitution.pdf",
        question="q",
        answer="a",
        evidence=["shall have the sole Power of Impeachment"],
    )
    base.update(kw)
    return EvalItem(**base)


def test_bad_items_are_rejected(pages):
    docs = corpus_texts(pages)
    errors = resolve([_item(evidence=["This sentence is not in the Constitution."])], docs)
    assert any("found 0x" in e for e in errors)
    errors = resolve([_item(evidence=["the United States"])], docs)  # ambiguous quote
    assert any("need exactly 1" in e for e in errors)
    errors = resolve([_item(doc="missing.pdf")], docs)
    assert any("unknown doc" in e for e in errors)
    errors = resolve([_item(type="unanswerable", evidence=[], absent_terms=["Senate"])], docs)
    assert any("absent term 'Senate'" in e for e in errors)
    assert resolve([_item()], docs) == []


def test_evidence_matching_is_whitespace_insensitive(pages):
    docs = corpus_texts(pages)
    it = _item(evidence=["shall   have the\nsole Power of Impeachment"])
    assert resolve([it], docs) == [] and it.pages


def test_load_items_reports_line_numbers(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "a", "unknown_field": 1}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="q.jsonl:1"):
        load_items(p)


def test_chunk_spans_align_with_document_text(pages):
    docs = corpus_texts(pages)
    for c in split_documents(pages, 700, 100):
        s = chunk_span(c, docs)
        assert docs[s.doc].text[s.start : s.end] == c.page_content


def test_page_of_and_coverage():
    d = DocText("d.pdf", "aaaa\nbbbb", [0, 5])
    assert d.page_of(0) == 1 and d.page_of(4) == 1 and d.page_of(5) == 2
    ev = Span("d.pdf", 10, 20)
    assert covers(Span("d.pdf", 0, 15), ev)  # exactly 50 %
    assert not covers(Span("d.pdf", 0, 14), ev)
    assert not covers(Span("other.pdf", 0, 100), ev)
    doc = Document(page_content="x" * 5, metadata={"source": "d.pdf", "page": 2, "start_index": 1, "chunk_chars": 3})
    assert chunk_span(doc, {"d.pdf": d}) == Span("d.pdf", 6, 9)
