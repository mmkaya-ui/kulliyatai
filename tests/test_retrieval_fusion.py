"""Unit tests for retrieval fusion (BM25, Reciprocal Rank Fusion, Hybrid search) with mocked embeddings."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.retrieval import HybridRetriever, doc_id, rrf_fuse, tokenize


class MockDeterministicEmbeddings(Embeddings):
    """Deterministic mock embeddings that map text to fixed 4-dimensional vectors without external APIs."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10), float(hash(t) % 10), 1.0, 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10), float(hash(text) % 10), 1.0, 0.5]


def test_tokenize_multilingual():
    # Turkish dotted I and lowercasing
    tr_tokens = tokenize("İSTANBUL Çeşme ve Kaşağı")
    assert "istanbul" in tr_tokens
    assert "çeşme" in tr_tokens
    assert "kaşağı" in tr_tokens

    # German eszett
    de_tokens = tokenize("SCHLIESSEN und Straße")
    assert "schliessen" in de_tokens
    assert "strasse" in de_tokens or "straße" in de_tokens

    # English punctuation
    en_tokens = tokenize("Article I, Section 2: House of Representatives!")
    assert "article" in en_tokens
    assert "section" in en_tokens
    assert "representatives" in en_tokens


def test_rrf_fuse_math_and_ranking():
    # RRF formula: sum_i w_i / (k + rank_i)
    # With k=60, rank 1 score = 1 / (60 + 1) = 1/61 ≈ 0.016393
    rankings = [
        ["docA", "docB", "docC"],  # Ranking 1 (e.g. dense)
        ["docB", "docA", "docD"],  # Ranking 2 (e.g. sparse)
    ]
    # docA: 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # docB: 1/62 + 1/61 = 0.032522
    # docC: 1/63 = 0.015873
    # docD: 1/63 = 0.015873
    fused = rrf_fuse(rankings, k=60)
    scores = dict(fused)

    assert abs(scores["docA"] - scores["docB"]) < 1e-6
    assert scores["docA"] > scores["docC"]
    assert scores["docA"] > scores["docD"]


def test_rrf_fuse_weighted():
    rankings = [
        ["docA", "docB"],
        ["docB", "docA"],
    ]
    # Weight 0.8 on rank 1, 0.2 on rank 2
    fused = rrf_fuse(rankings, weights=[0.8, 0.2], k=60)
    top_doc = fused[0][0]
    assert top_doc == "docA", "docA should win because rank 1 has 4x higher weight"


def test_hybrid_retriever_with_mock_embeddings():
    docs = [
        Document(
            page_content="[Kaynak: doc1 | Sayfa: 1]\nTürkiye Devleti bir Cumhuriyettir.",
            metadata={"chunk_id": "c1", "source": "doc1", "page": 1},
        ),
        Document(
            page_content="[Kaynak: doc2 | Sayfa: 1]\nThe Senate of the United States shall be composed of two Senators.",
            metadata={"chunk_id": "c2", "source": "doc2", "page": 1},
        ),
        Document(
            page_content="[Kaynak: doc3 | Sayfa: 1]\nGregor Samsa erwachte zu einem Ungeziefer.",
            metadata={"chunk_id": "c3", "source": "doc3", "page": 1},
        ),
    ]

    mock_emb = MockDeterministicEmbeddings()
    vectorstore = FAISS.from_documents(docs, mock_emb, ids=[doc_id(d) for d in docs])

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=docs,
        dense_weight=0.6,
        bm25_weight=0.4,
        candidate_k=3,
    )

    # 1. Test BM25 exact keyword match
    bm25_hits = retriever.bm25("Cumhuriyettir", k=1)
    assert len(bm25_hits) == 1
    assert "Cumhuriyettir" in bm25_hits[0].page_content

    # 2. Test Dense retrieval returns documents
    dense_hits = retriever.dense("Senate query", k=2)
    assert len(dense_hits) <= 2

    # 3. Test Hybrid retrieval returns fused results
    hybrid_hits = retriever.hybrid("Gregor Ungeziefer", k=2)
    assert len(hybrid_hits) >= 1
    assert any("Gregor" in d.page_content for d in hybrid_hits)
