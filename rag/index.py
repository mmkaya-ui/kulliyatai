"""Persistent hybrid index: chunk store + dense vectors (FAISS, inner product) + BM25.

On-disk layout of an index directory (no pickle anywhere, so loading an index
cannot execute code):

* ``chunks.jsonl``  – one JSON object per chunk: ``text`` + ``metadata``
* ``vectors.npy``   – float32 matrix, one L2-normalised row per *dense unit*
* ``owner.npy``     – int32, ``owner[i]`` = chunk index of dense unit ``i``
* ``meta.json``     – build settings (embedder name, chunking, windows, …)

A dense unit is either a whole chunk or, with ``dense_windows=True``, one
window of a chunk that is longer than the embedding model's input limit.
A chunk's dense score is the maximum over its units (MaxP).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from rag.chunking import context_header, split_documents, token_windows
from rag.embeddings import Embedder

FORMAT_VERSION = 2


@dataclass(frozen=True)
class IndexSettings:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    contextual_header: bool = True  # prefix "[Kaynak: file | Sayfa: n]" to indexed text
    dense_windows: bool = True  # split chunks longer than the embedder's limit into windows
    bm25_tokenizer: str = "prefix5"  # see rag.retrieval.tokenize


class RAGIndex:
    def __init__(
        self,
        chunks: list[Document],
        vectors: np.ndarray,
        owner: np.ndarray,
        settings: IndexSettings,
        embedder_name: str,
        stats: dict | None = None,
    ):
        if len(vectors) != len(owner):
            raise ValueError("vectors and owner must have the same length")
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.owner = np.asarray(owner, dtype=np.int32)
        self.settings = settings
        self.embedder_name = embedder_name
        self.stats = stats or {}
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)

    # --- building ---------------------------------------------------------

    @staticmethod
    def index_text(chunk: Document, settings: IndexSettings) -> str:
        if settings.contextual_header:
            return f"{context_header(chunk.metadata)}\n{chunk.page_content}"
        return chunk.page_content

    @classmethod
    def build(cls, pages: list[Document], embedder: Embedder, settings: IndexSettings) -> RAGIndex:
        chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("no text to index")
        units, owner = [], []
        truncated = 0
        for ci, ch in enumerate(chunks):
            header = f"{context_header(ch.metadata)}\n" if settings.contextual_header else ""
            body = ch.page_content
            limit = embedder.max_tokens
            if limit is not None and embedder.count_tokens(header + body) > limit:
                if settings.dense_windows:
                    # header tokens minus the 2 special tokens counted twice, plus a 2-token safety margin
                    budget = limit - (embedder.count_tokens(header) - 2 if header else 0) - 2
                    for w in token_windows(body, embedder.count_tokens, budget):
                        units.append(header + w)
                        owner.append(ci)
                    continue
                truncated += 1  # embedded anyway; the model only sees the first `limit` tokens
            units.append(header + body)
            owner.append(ci)
        vectors = embedder.embed_documents(units)
        stats = {
            "n_chunks": len(chunks),
            "n_dense_units": len(units),
            "n_chunks_truncated_by_embedder": truncated,
            "n_pages": len(pages),
        }
        return cls(chunks, vectors, np.asarray(owner), settings, embedder.name, stats)

    # --- dense search -----------------------------------------------------

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k chunks by max-over-units cosine similarity: [(chunk_index, score)]."""
        k = min(k, len(self.chunks))
        n_units = len(self.owner)
        want = min(n_units, max(4 * k, 64))
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        while True:
            scores, ids = self._faiss.search(q, want)
            out, seen = [], set()
            for s, u in zip(scores[0], ids[0], strict=True):
                if u < 0:
                    continue
                c = int(self.owner[u])
                if c not in seen:  # units come sorted by score: first hit is the chunk's max
                    seen.add(c)
                    out.append((c, float(s)))
                    if len(out) == k:
                        return out
            if want >= n_units:
                return out
            want = min(n_units, want * 2)

    # --- persistence ------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write to a temp dir next to ``directory``, then swap it in (atomic-ish)."""
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".index_tmp_", dir=directory.parent))
        try:
            with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
            np.save(tmp / "vectors.npy", self.vectors)
            np.save(tmp / "owner.npy", self.owner)
            meta = {
                "format_version": FORMAT_VERSION,
                "embedder": self.embedder_name,
                "dim": int(self.vectors.shape[1]),
                "settings": asdict(self.settings),
                "stats": self.stats,
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            if directory.exists():
                shutil.rmtree(directory)
            tmp.rename(directory)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def read_meta(directory: str | Path) -> dict:
        p = Path(directory) / "meta.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — build the index with `python ingest.py`")
        meta = json.loads(p.read_text(encoding="utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"index at {directory} has format {meta.get('format_version')}, expected {FORMAT_VERSION}; "
                "rebuild it with `python ingest.py`"
            )
        return meta

    @classmethod
    def load(cls, directory: str | Path, embedder_name: str | None = None) -> RAGIndex:
        directory = Path(directory)
        meta = cls.read_meta(directory)
        if embedder_name is not None and embedder_name != meta["embedder"]:
            raise ValueError(f"index was built with {meta['embedder']}, not {embedder_name}")
        chunks = []
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunks.append(Document(page_content=row["text"], metadata=row["metadata"]))
        return cls(
            chunks,
            np.load(directory / "vectors.npy"),
            np.load(directory / "owner.npy"),
            IndexSettings(**meta["settings"]),
            meta["embedder"],
            meta.get("stats"),
        )
