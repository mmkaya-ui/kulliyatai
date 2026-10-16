"""Central configuration, read from environment variables (and ``.env``).

Retrieval defaults are the settings selected on the **dev** split by
``python -m evals.run`` (see ``evals/RESULTS.md``); ``tests/test_docs_consistency.py``
fails if they disagree with ``evals/results/results.json``. Relative paths
are resolved against the repository root, not the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag.index import IndexSettings

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # local ONNX, 384-d
    "openai": "text-embedding-3-large",
    "google": "gemini-embedding-001",
}

# Selected on the dev split (evals/RESULTS.md, "Selected settings").
DEFAULT_INDEX = IndexSettings(
    chunk_size=1000,
    chunk_overlap=150,
    contextual_header=True,
    dense_windows=True,
    bm25_tokenizer="prefix5",
)
DEFAULT_CANDIDATE_K = 30
# Reranker score below which the app answers "not found" (5th percentile of the
# best-passage score over answerable dev questions).
DEFAULT_ABSTAIN_THRESHOLD = -2.292

# Chat models. Model IDs as listed by the providers on 2026-10-15; override via env.
# None of them is exercised by the tests or the benchmark (no API keys are used there).
DEFAULT_LLMS = {
    "gemini_fast": "gemini-3.8-flash",
    "gemini_pro": "gemini-3.1-pro-preview",
    "openai_fast": "gpt-4o-mini",
    "openai_pro": "gpt-4o",
    "openrouter_claude": "anthropic/claude-sonnet-5.5",
}


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_dir: Path
    embedding_provider: str
    embedding_model: str
    index: IndexSettings
    dense_weight: float
    bm25_weight: float
    candidate_k: int
    rerank_model: str
    abstain_threshold: float | None
    answer_language: str
    llms: dict

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        provider = env("EMBEDDING_PROVIDER", "fastembed").strip().lower()
        if provider not in DEFAULT_EMBEDDING_MODELS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {sorted(DEFAULT_EMBEDDING_MODELS)}, got {provider!r}")
        d = DEFAULT_INDEX
        index = IndexSettings(
            chunk_size=int(env("CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(env("CHUNK_OVERLAP", d.chunk_overlap)),
            contextual_header=_bool(env("CONTEXTUAL_HEADER", str(d.contextual_header))),
            dense_windows=_bool(env("DENSE_WINDOWS", str(d.dense_windows))),
            bm25_tokenizer=env("BM25_TOKENIZER", d.bm25_tokenizer),
        )
        thr = env("ABSTAIN_THRESHOLD", str(DEFAULT_ABSTAIN_THRESHOLD)).strip()
        return cls(
            data_dir=_path(env("DATA_DIR", "sample_docs")),
            db_dir=_path(env("DB_DIR", "vector_db")),
            embedding_provider=provider,
            embedding_model=env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[provider],
            index=index,
            dense_weight=float(env("DENSE_WEIGHT", 1.0)),
            bm25_weight=float(env("BM25_WEIGHT", 1.0)),
            candidate_k=int(env("CANDIDATE_K", DEFAULT_CANDIDATE_K)),
            rerank_model=env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8"),
            abstain_threshold=None if thr.lower() in {"", "none", "off"} else float(thr),
            # "tr" keeps the original behaviour (always answer in Turkish);
            # "auto" answers in the language of the question.
            answer_language=env("ANSWER_LANGUAGE", "tr"),
            llms={k: env(f"LLM_{k.upper()}", v) for k, v in DEFAULT_LLMS.items()},
        )


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(ROOT / ".env")
