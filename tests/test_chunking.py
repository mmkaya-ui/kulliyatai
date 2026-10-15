"""Chunking, contextual headers and embedding windows."""

import pytest
from langchain_core.documents import Document

from rag.chunking import context_header, split_documents, strip_context_header, token_windows, with_header


def test_split_documents_no_empty_chunks_and_metadata():
    docs = [
        Document(page_content="First non-empty page.\n\nParagraph two.", metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="   \n\n\t  ", metadata={"source": "doc1.pdf", "page": 2}),
        Document(page_content="Third page with meaningful text.", metadata={"source": "doc1.pdf", "page": 3}),
    ]
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    assert chunks
    for c in chunks:
        assert c.page_content.strip()
        m = c.metadata
        assert m["chunk_id"] == f"doc1.pdf:p{m['page']}:{m['start_index']}"
        assert m["chunk_chars"] == len(c.page_content)


def test_start_index_points_into_page_text():
    text = "Alpha beta gamma. " * 40
    page = Document(page_content=text, metadata={"source": "a.pdf", "page": 1})
    for c in split_documents([page], chunk_size=120, chunk_overlap=20):
        s = c.metadata["start_index"]
        assert text[s : s + c.metadata["chunk_chars"]] == c.page_content


def test_exact_overlap_on_unbreakable_text():
    text = "abcdefghijklmnopqrstuvwxyz0123456789" * 5
    chunks = split_documents([Document(page_content=text, metadata={"source": "s.pdf", "page": 1})], 50, 15)
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.page_content[-15:] == b.page_content[:15]
        assert b.metadata["start_index"] == a.metadata["start_index"] + 35


def test_split_documents_invalid_params():
    docs = [Document(page_content="Hello world", metadata={"source": "doc.pdf", "page": 1})]
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_documents(docs, chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents(docs, chunk_size=100, chunk_overlap=100)


def test_context_header_roundtrip():
    chunk = Document(
        page_content="Madde 1 – Türkiye Devleti bir Cumhuriyettir.", metadata={"source": "k.pdf", "page": 4}
    )
    assert context_header(chunk.metadata) == "[Kaynak: k.pdf | Sayfa: 4]"
    assert context_header({}) == "[Kaynak: Bilinmeyen | Sayfa: ?]"
    enriched = with_header(chunk)
    assert enriched.startswith("[Kaynak: k.pdf | Sayfa: 4]\n")
    assert strip_context_header(enriched) == chunk.page_content
    assert strip_context_header(chunk.page_content) == chunk.page_content  # no header: unchanged


def count_words(text: str) -> int:
    return len(text.split())


def test_token_windows_cover_text_and_respect_limit():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    windows = token_windows(text, count_words, max_tokens=20)
    assert all(count_words(w) <= 20 for w in windows)
    covered = {tok for w in windows for tok in w.split()}
    assert covered == set(words)  # nothing is lost
    # 50 % stride: every window starts 10 words after the previous one
    starts = [words.index(w.split()[0]) for w in windows]
    assert all(b - a == 10 for a, b in zip(starts, starts[1:], strict=False))


def test_token_windows_short_text_is_one_window():
    assert token_windows("kısa bir metin", count_words, max_tokens=10) == ["kısa bir metin"]
    assert token_windows("   ", count_words, max_tokens=10) == []
