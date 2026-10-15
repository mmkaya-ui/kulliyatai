"""Unit tests for document chunking, overlap correctness, and contextual enrichment."""

import pytest
from langchain_core.documents import Document

from rag.chunking import (
    context_header,
    enrich_chunks_with_context,
    split_documents,
    strip_context_header,
)


def test_split_documents_no_empty_chunks():
    docs = [
        Document(
            page_content="First non-empty page.\n\nParagraph two.", metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(page_content="   \n\n\t  ", metadata={"source": "doc1.pdf", "page": 2}),  # whitespace only
        Document(page_content="Third page with meaningful text.", metadata={"source": "doc1.pdf", "page": 3}),
    ]
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.page_content.strip(), "Chunk content must not be empty or whitespace"
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["source"] == "doc1.pdf"


def test_split_documents_overlap_correct():
    # 1. Test continuous string where overlap is exact character-level
    continuous_text = "abcdefghijklmnopqrstuvwxyz0123456789" * 5
    docs = [Document(page_content=continuous_text, metadata={"source": "sample.pdf", "page": 1})]

    chunk_size = 50
    chunk_overlap = 15
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) >= 3
    for i in range(len(chunks) - 1):
        # Character overlap between adjacent chunks must equal chunk_overlap
        curr_suffix = chunks[i].page_content[-chunk_overlap:]
        next_prefix = chunks[i + 1].page_content[:chunk_overlap]
        assert curr_suffix == next_prefix, f"Exact overlap failed between chunk {i} and {i + 1}"
        assert chunks[i + 1].metadata["start_index"] == chunks[i].metadata["start_index"] + (
            chunk_size - chunk_overlap
        )

    # 2. Test natural text where start_index of next chunk is before end of current chunk
    natural_text = (
        "First complete sentence here. Second complete sentence here. Third complete sentence here."
    )
    nat_docs = [Document(page_content=natural_text, metadata={"source": "natural.pdf", "page": 1})]
    nat_chunks = split_documents(nat_docs, chunk_size=40, chunk_overlap=10)
    for i in range(len(nat_chunks) - 1):
        curr_end = nat_chunks[i].metadata["start_index"] + len(nat_chunks[i].page_content)
        next_start = nat_chunks[i + 1].metadata["start_index"]
        assert next_start <= curr_end, "Adjacent chunks must overlap or touch in source text"


def test_split_documents_invalid_params():
    docs = [Document(page_content="Hello world", metadata={"source": "doc.pdf", "page": 1})]

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_documents(docs, chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and smaller than chunk_size"):
        split_documents(docs, chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and smaller than chunk_size"):
        split_documents(docs, chunk_size=100, chunk_overlap=150)


def test_enrich_chunks_with_context_and_strip():
    chunks = [
        Document(
            page_content="Madde 1 - Türkiye Devleti bir Cumhuriyettir.",
            metadata={"source": "anayasalar.pdf", "page": 1, "start_index": 0},
        )
    ]
    enriched = enrich_chunks_with_context(chunks)

    assert len(enriched) == 1
    assert enriched[0].page_content.startswith("[Kaynak: anayasalar.pdf | Sayfa: 1]\n")
    assert "Türkiye Devleti bir Cumhuriyettir" in enriched[0].page_content
    assert enriched[0].metadata["source"] == "anayasalar.pdf"

    # Test reverse stripping
    stripped = strip_context_header(enriched[0].page_content)
    assert stripped == "Madde 1 - Türkiye Devleti bir Cumhuriyettir."


def test_context_header_formatting():
    hdr = context_header({"source": "kanun.pdf", "page": 4})
    assert hdr == "[Kaynak: kanun.pdf | Sayfa: 4]"

    hdr_missing = context_header({})
    assert hdr_missing == "[Kaynak: Bilinmeyen | Sayfa: ?]"
