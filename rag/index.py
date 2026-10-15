"""Building and loading the FAISS index (shared by ingest.py, app.py and evals)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

META_FILE = "meta.json"


def build_faiss(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    ids = [c.metadata.get("chunk_id") for c in chunks]
    return FAISS.from_documents(chunks, embeddings, ids=ids if all(ids) else None)


def save_index_atomically(db: FAISS, db_dir: Path, meta: dict) -> None:
    """Write the index to a temp dir first, then swap it in.

    The original ingest deleted ``vector_db/`` *before* embedding, so a failed
    API call left you with no index at all.
    """
    db_dir = Path(db_dir)
    db_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".vector_db_tmp_", dir=db_dir.parent))
    try:
        db.save_local(str(tmp))
        (tmp / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if db_dir.exists():
            shutil.rmtree(db_dir)
        tmp.rename(db_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_meta(db_dir: Path) -> dict:
    """Return the build metadata, or {} for indexes built before meta.json existed."""
    p = Path(db_dir) / META_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_faiss(db_dir: Path, embeddings: Embeddings) -> FAISS:
    # The docstore is a pickle: only load indexes you built yourself.
    return FAISS.load_local(str(db_dir), embeddings, allow_dangerous_deserialization=True)
