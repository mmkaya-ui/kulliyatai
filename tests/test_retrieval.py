"""Index + retrieval logic with a deterministic fake embedder and reranker (no models)."""

import json

import numpy as np
import pytest
from conftest import HashEmbedder, OverlapReranker
from langchain_core.documents import Document

from rag.index import IndexSettings, RAGIndex
from rag.retrieval import Retriever, rrf_fuse, tokenize

PAGES = [
    Document(page_content="Madde 1 – Türkiye Devleti bir Cumhuriyettir.", metadata={"source": "tr.pdf", "page": 1}),
    Document(
        page_content="The Senate of the United States shall be composed of two Senators from each State.",
        metadata={"source": "en.pdf", "page": 1},
    ),
    Document(
        page_content="Als Gregor Samsa eines Morgens erwachte, fand er sich zu einem Ungeziefer verwandelt.",
        metadata={"source": "de.pdf", "page": 1},
    ),
]
PLAIN = IndexSettings(
    chunk_size=500, chunk_overlap=50, contextual_header=False, dense_windows=False, bm25_tokenizer="plain"
)


def test_tokenize_multilingual_and_prefix():
    assert tokenize("İSTANBUL Çeşme ve Kaşağı") == ["istanbul", "çeşme", "ve", "kaşağı"]
    assert "strasse" in tokenize("Straße")  # casefold: ß -> ss
    assert tokenize("Meclisinde Meclise", "prefix5") == ["mecli", "mecli"]
    assert tokenize("ab", "prefix5") == ["ab"]
    with pytest.raises(ValueError):
        tokenize("x", "stem")


def test_rrf_fuse_math_ties_and_weights():
    fused = dict(rrf_fuse([["A", "B", "C"], ["B", "A", "D"]], k=60))
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["A"] == pytest.approx(fused["B"])
    assert fused["C"] == pytest.approx(1 / 63)
    assert rrf_fuse([["A", "B"], ["B", "A"]], weights=[0.8, 0.2])[0][0] == "A"
    assert rrf_fuse([["A", "A", "B"]])[0] == ("A", pytest.approx(1 / 61))  # duplicates ignored
    with pytest.raises(ValueError):
        rrf_fuse([["A"]], weights=[1, 2])


def test_bm25_dense_hybrid_and_rerank(hash_embedder, overlap_reranker):
    idx = RAGIndex.build(PAGES, hash_embedder, PLAIN)
    r = Retriever(idx, hash_embedder, candidate_k=3, reranker=overlap_reranker)
    assert r.search("Cumhuriyettir", k=1, mode="bm25")[0].chunk.metadata["source"] == "tr.pdf"
    assert r.search("Gregor Samsa Ungeziefer", k=1, mode="dense")[0].chunk.metadata["source"] == "de.pdf"
    hyb = r.search("Senators each State", k=3, mode="hybrid")
    assert hyb[0].chunk.metadata["source"] == "en.pdf"
    rr = r.search("two Senators from each State", k=2, mode="hybrid", rerank=True)
    assert rr[0].chunk.metadata["source"] == "en.pdf" and rr[0].rerank_score == 5.0
    with pytest.raises(ValueError):
        r.search("x", mode="sparse")


def test_retriever_refuses_mismatched_embedder(hash_embedder):
    idx = RAGIndex.build(PAGES, hash_embedder, PLAIN)
    with pytest.raises(ValueError, match="index built with"):
        Retriever(idx, HashEmbedder(max_tokens=10))


def test_dense_windows_recover_text_beyond_model_limit():
    """A fact at the end of a long chunk is invisible to a truncating embedder unless windows are used."""
    filler = " ".join(f"filler{i}" for i in range(60))
    page = Document(page_content=f"{filler} The secret word is zebra.", metadata={"source": "long.pdf", "page": 1})
    other = Document(page_content="Zebra crossing rules for pedestrians.", metadata={"source": "o.pdf", "page": 1})
    emb = HashEmbedder(max_tokens=20)
    single = RAGIndex.build([page, other], emb, IndexSettings(2000, 50, False, False, "plain"))
    windows = RAGIndex.build([page, other], emb, IndexSettings(2000, 50, False, True, "plain"))
    assert single.stats["n_chunks"] == 2  # the long page stays one chunk
    assert single.stats["n_chunks_truncated_by_embedder"] == 1
    assert windows.stats["n_dense_units"] > windows.stats["n_chunks"]
    q = emb.embed_query("secret word zebra")
    assert single.chunks[single.dense_search(q, 1)[0][0]].metadata["source"] == "o.pdf"
    assert windows.chunks[windows.dense_search(q, 1)[0][0]].metadata["source"] == "long.pdf"


def test_dense_search_is_max_over_windows_and_unique():
    vecs = np.array([[1, 0], [0.6, 0.8], [0, 1], [0.8, 0.6]], dtype=np.float32)
    owner = np.array([0, 0, 1, 2])
    chunks = [Document(page_content=str(i), metadata={}) for i in range(3)]
    idx = RAGIndex(chunks, vecs, owner, PLAIN, "test")
    res = idx.dense_search(np.array([1.0, 0.0]), k=3)
    assert [c for c, _ in res] == [0, 2, 1]
    assert res[0][1] == pytest.approx(1.0)  # chunk 0 scored by its best unit


def test_save_load_roundtrip_without_pickle(tmp_path, hash_embedder):
    idx = RAGIndex.build(PAGES, hash_embedder, IndexSettings(500, 50, True, True, "prefix5"))
    idx.save(tmp_path / "db")
    files = sorted(p.name for p in (tmp_path / "db").iterdir())
    assert files == ["chunks.jsonl", "meta.json", "owner.npy", "vectors.npy"]
    meta = json.loads((tmp_path / "db" / "meta.json").read_text())
    assert meta["embedder"] == hash_embedder.name and meta["settings"]["bm25_tokenizer"] == "prefix5"
    loaded = RAGIndex.load(tmp_path / "db", embedder_name=hash_embedder.name)
    assert [c.page_content for c in loaded.chunks] == [c.page_content for c in idx.chunks]
    assert np.allclose(loaded.vectors, idx.vectors)
    with pytest.raises(ValueError, match="built with"):
        RAGIndex.load(tmp_path / "db", embedder_name="other/model")
    q = hash_embedder.embed_query("Cumhuriyettir")
    assert loaded.dense_search(q, 2) == idx.dense_search(q, 2)


def test_overlap_reranker_fixture_sanity():
    assert OverlapReranker().score("a b", ["a", "a b", "c"]) == [1.0, 2.0, 0.0]
