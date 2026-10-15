"""End-to-end with the real local models (opt-in: RUN_MODEL_TESTS=1, models are downloaded on first use)."""

from pathlib import Path

import pytest
from conftest import models_available

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not models_available(), reason="set RUN_MODEL_TESTS=1 to run tests that load ONNX models"),
]

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    import ingest

    db = tmp_path_factory.mktemp("db") / "vector_db"
    assert ingest.create_index(["--db-dir", str(db), "--no-cache"]) == 0
    return db


def test_ingest_and_retrieve(built_index):
    from rag.config import Settings
    from rag.embeddings import make_embedder
    from rag.index import RAGIndex
    from rag.retrieval import Reranker, Retriever

    s = Settings.from_env()
    idx = RAGIndex.load(built_index)
    assert idx.settings == s.index and idx.stats["n_chunks_truncated_by_embedder"] == 0
    r = Retriever(idx, make_embedder("fastembed", s.embedding_model), candidate_k=s.candidate_k, reranker=Reranker())
    hits = r.search("Who presides when the President of the United States is tried?", k=3, rerank=True)
    assert hits[0].chunk.metadata["source"] == "en_us_constitution.pdf"
    assert "Chief Justice shall preside" in hits[0].chunk.page_content


def test_streamlit_app_offline_answer_and_refusal(built_index, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DB_DIR", str(built_index))
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "STREAMLIT_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("1924 Teşkîlât-ı Esâsîye Kanunu'na göre mebus seçme yaşı kaçtır?").run()
    assert not at.exception
    reply = at.chat_message[-1].markdown[0].value
    assert "Yirmi iki yaşını bitiren" in reply and "[1]" in reply and "tr_teskilati_esasiye_1924.pdf" in reply

    # Clearly out-of-corpus question: refused by the abstention gate. (In-domain "near miss"
    # questions are often *not* refused; see the abstention numbers in evals/RESULTS.md.)
    at.chat_input[0].set_value("What is the boiling point of mercury?").run()
    assert "bulunamadı" in at.chat_message[-1].markdown[0].value
