"""Offline extractive answers, abstention gate and the optional LLM evaluation (stub LLM)."""

from conftest import HashEmbedder, OverlapReranker
from langchain_core.documents import Document

from evals.dataset import EvalItem, corpus_texts, resolve
from evals.generation import evaluate
from rag.answer import extractive_answer, format_extractive, is_confident, split_sentences
from rag.index import IndexSettings, RAGIndex
from rag.prompts import NO_ANSWER
from rag.retrieval import Hit, Retriever


def test_split_sentences_offsets_and_line_wraps():
    text = "Madde 1 – Türkiye Devleti bir Cumhuriyettir.\nMadde 2 – Kısa. Daha uzun bir cümle burada\nyer alıyor! Son"
    out = split_sentences(text)
    assert [s for _, s in out] == [
        "Madde 1 – Türkiye Devleti bir Cumhuriyettir.",
        "Madde 2 – Kısa. Daha uzun bir cümle burada\nyer alıyor!",  # short fragment merged, wrap kept
        "Son",
    ]
    assert all(text[o : o + len(s)] == s for o, s in out)


def _hits(*texts):
    return [Hit(Document(page_content=t, metadata={"source": "s.pdf", "page": 1}), 0.0) for t in texts]


def test_extractive_answer_picks_best_sentences_with_citations():
    hits = _hits(
        "The House shall chuse their Speaker. The Senate shall have the sole Power to try all Impeachments.",
        "Each Senator shall have one Vote, and the Senators are chosen for six Years.",
    )
    sents = extractive_answer("How many years are Senators chosen for?", hits, OverlapReranker(), max_sentences=1)
    assert len(sents) == 1 and sents[0].hit_rank == 2 and "six Years" in sents[0].text
    assert format_extractive(sents).startswith("> Each Senator") and sents[0].text in hits[1].chunk.page_content
    assert extractive_answer("x", [], OverlapReranker()) == []


def test_is_confident():
    h = _hits("a")
    assert is_confident(h, None)  # no reranker score -> no gate
    h[0].rerank_score = 0.5
    assert is_confident(h, 0.0) and not is_confident(h, 1.0)
    assert not is_confident([], 0.0)


def test_generation_eval_with_stub_llm():
    pages = [
        Document(
            page_content="Madde 2 – Türkiye Devletinin makarrı Ankara şehridir.",
            metadata={"source": "tr_x.pdf", "page": 1, "total_pages": 1},
        ),
        Document(
            page_content="The Senate shall have the sole Power to try all Impeachments.",
            metadata={"source": "en_x.pdf", "page": 1, "total_pages": 1},
        ),
    ]
    docs = corpus_texts(pages)
    items = [
        EvalItem(
            "a",
            "test",
            "answerable",
            "tr",
            "tr_x.pdf",
            "Başkent neresidir? Ankara",
            "Ankara",
            ["makarrı Ankara şehridir"],
        ),
        EvalItem("u", "test", "unanswerable", "en", None, "Who is the king?", NO_ANSWER, absent_terms=["king"]),
    ]
    assert resolve(items, docs) == []
    emb = HashEmbedder()
    retriever = Retriever(
        RAGIndex.build(pages, emb, IndexSettings(500, 50, False, False, "plain")), emb, reranker=OverlapReranker()
    )

    def llm(prompt: str) -> str:
        if "Who is the king?" in prompt:
            return NO_ANSWER
        n = 1 if prompt.index("Ankara") < prompt.index("Senate") else 2
        return f"Ankara [{n}]. Ayrıca bkz. [9]."

    judge_calls = []

    def judge(prompt: str) -> str:
        judge_calls.append(prompt)
        return '{"supports_claim": true, "explanation": "ok"}'

    res = evaluate(llm, retriever, items, docs, k=2, judge=judge)
    assert res["refusal_accuracy"]["mean"] == 1.0
    assert res["answer_rate"]["mean"] == 1.0
    assert res["invalid_citation_rate"]["mean"] == 1.0  # [9] does not exist
    assert res["cites_evidence_rate"]["mean"] == 1.0
    assert res["judge_support_rate"]["mean"] == 1.0 and judge_calls
