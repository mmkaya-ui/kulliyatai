"""Central configuration, read from environment variables (and ``.env``).

Every value that used to be hard-coded in app.py / ingest.py lives here so it
can be overridden without editing code. Relative paths are resolved against
the repository root (not the current working directory).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default embedding model per provider. The provider/model used to build an
# index is written to ``<db_dir>/meta.json`` and re-used at query time, so the
# app can never load an index with a mismatching embedding model.
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-large",  # 3072 dims
    "google": "gemini-embedding-001",  # 3072 dims
    "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384 dims, local ONNX
}


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_dir: Path
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    dense_weight: float
    bm25_weight: float
    candidate_k: int
    rerank_model: str
    answer_language: str

    @classmethod
    def from_env(cls) -> Settings:
        default_p = "openai" if os.environ.get("OPENAI_API_KEY") else ("fastembed")
        provider = os.environ.get("EMBEDDING_PROVIDER", default_p).strip().lower()
        if provider not in DEFAULT_EMBEDDING_MODELS:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {sorted(DEFAULT_EMBEDDING_MODELS)}, got {provider!r}"
            )
        return cls(
            data_dir=_path(os.environ.get("DATA_DIR", "sample_docs")),
            db_dir=_path(os.environ.get("DB_DIR", "vector_db")),
            embedding_provider=provider,
            embedding_model=os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[provider],
            chunk_size=int(os.environ.get("CHUNK_SIZE", 1500)),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 300)),
            # 60% vector (meaning) + 40% keyword (exact match) — original weights
            dense_weight=float(os.environ.get("DENSE_WEIGHT", 0.6)),
            bm25_weight=float(os.environ.get("BM25_WEIGHT", 0.4)),
            candidate_k=int(os.environ.get("CANDIDATE_K", 20)),
            # Apache-2.0, multilingual cross-encoder (int8 ONNX, ~0.6 GB, CPU).
            rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8"),
            # "tr" keeps the original behaviour (always answer in Turkish);
            # "auto" answers in the language of the question.
            answer_language=os.environ.get("ANSWER_LANGUAGE", "tr"),
        )


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(ROOT / ".env")
