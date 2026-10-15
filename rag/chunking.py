"""Chunking, contextual headers and embedding windows."""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Respect paragraph boundaries first, then lines, sentences, words, characters.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_HEADER_RE = re.compile(r"^\[Kaynak: [^\]\n]*\]\n")


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """Split page documents into overlapping character chunks.

    - Sizes are in characters (``length_function=len``), not tokens.
    - Pages are split independently, so a chunk never spans two pages; that
      keeps page citations exact.
    - Whitespace-only chunks are dropped.
    - Each chunk gets ``start_index`` (offset in its page), ``chunk_chars``
      and a stable ``chunk_id`` = ``"<source>:p<page>:<start_index>"``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=SEPARATORS,
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    for c in chunks:
        m = c.metadata
        m["chunk_chars"] = len(c.page_content)
        m["chunk_id"] = f"{m.get('source', '?')}:p{m.get('page', '?')}:{m.get('start_index', 0)}"
    return chunks


def context_header(metadata: dict) -> str:
    source = metadata.get("source", "Bilinmeyen")
    page = metadata.get("page", "?")
    return f"[Kaynak: {source} | Sayfa: {page}]"


def with_header(chunk: Document) -> str:
    """Chunk text prefixed with its ``[Kaynak: … | Sayfa: …]`` header."""
    return f"{context_header(chunk.metadata)}\n{chunk.page_content}"


def strip_context_header(text: str) -> str:
    """Remove a leading ``[Kaynak: … | Sayfa: …]`` header line, if present."""
    return _HEADER_RE.sub("", text, count=1)


def token_windows(
    text: str, count_tokens: Callable[[str], int], max_tokens: int, stride_ratio: float = 0.5
) -> list[str]:
    """Cut ``text`` into word-aligned windows of at most ``max_tokens`` tokens.

    Consecutive windows start ``stride_ratio`` of a window apart, so every
    sentence is fully contained in at least one window unless it is longer
    than a window. Used to embed chunks that exceed the embedding model's
    input limit (see :mod:`rag.index`): the chunk's dense score is the max over
    its windows ("MaxP", Dai & Callan, SIGIR 2019).
    """
    words = text.split()
    if not words:
        return []
    if count_tokens(text) <= max_tokens:
        return [" ".join(words)]
    windows, i = [], 0
    while i < len(words):
        lo, hi = i + 1, len(words)
        while lo < hi:  # largest end index whose window fits
            mid = (lo + hi + 1) // 2
            if count_tokens(" ".join(words[i:mid])) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        windows.append(" ".join(words[i:lo]))
        if lo >= len(words):
            break
        i += max(1, int((lo - i) * stride_ratio))
    return windows
