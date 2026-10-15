"""Chunking and contextual enrichment."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Respect paragraph boundaries first, then lines, sentences, words, characters.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_HEADER_RE = re.compile(r"^\[Kaynak: [^\]\n]*\]\n")


def split_documents(
    documents: list[Document], chunk_size: int = 1500, chunk_overlap: int = 300
) -> list[Document]:
    """Split page documents into overlapping character chunks.

    - Sizes are in characters (``length_function=len``), not tokens.
    - Pages are split independently, so a chunk never spans two pages; that
      keeps page citations exact.
    - Whitespace-only chunks are dropped.
    - Each chunk gets ``start_index`` (offset in its page) and a stable
      ``chunk_id`` = ``"<source>:p<page>:<start_index>"``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    # --- Smart Chunking (larger chunks, more overlap) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Larger chunks = more context per retrieval
        chunk_overlap=chunk_overlap,  # More overlap = better continuity
        length_function=len,
        separators=SEPARATORS,  # Respect paragraph boundaries
        add_start_index=True,
    )
    chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]
    for c in chunks:
        m = c.metadata
        m["chunk_id"] = f"{m.get('source', '?')}:p{m.get('page', '?')}:{m.get('start_index', 0)}"
    return chunks


def context_header(metadata: dict) -> str:
    source = metadata.get("source", "Bilinmeyen")
    page = metadata.get("page", "?")
    return f"[Kaynak: {source} | Sayfa: {page}]"


# --- UPGRADE 2: Contextual Chunk Enrichment ---
# Prepends source metadata INTO each chunk before embedding
# This makes retrieval smarter — the AI knows WHERE info came from
def enrich_chunks_with_context(chunks: list[Document]) -> list[Document]:
    """Add source context directly into chunk text for better retrieval."""
    enriched = []
    for chunk in chunks:
        # Prepend context header to the actual text
        enriched_content = f"{context_header(chunk.metadata)}\n{chunk.page_content}"

        enriched.append(
            Document(
                page_content=enriched_content,
                metadata=dict(chunk.metadata),  # Keep original metadata too
            )
        )

    return enriched


def strip_context_header(text: str) -> str:
    """Inverse of :func:`enrich_chunks_with_context` for a single chunk text."""
    return _HEADER_RE.sub("", text, count=1)
