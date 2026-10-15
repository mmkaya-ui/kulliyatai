"""Unit tests for prompt construction, citation extraction, and refusal detection."""

from langchain_core.documents import Document

from rag.prompts import (
    NO_ANSWER,
    build_prompt,
    format_docs,
    is_refusal,
    parse_citations,
    split_claims,
)


def test_build_prompt_structure_and_language():
    docs = [
        Document(
            page_content="[Kaynak: kanun.pdf | Sayfa: 1]\nTürkiye Devleti bir Cumhuriyettir.",
            metadata={"source": "kanun.pdf", "page": 1},
        ),
        Document(
            page_content="[Kaynak: kanun.pdf | Sayfa: 2]\nBaşkenti Ankara şehridir.",
            metadata={"source": "kanun.pdf", "page": 2},
        ),
    ]

    # Test Turkish instruction
    prompt_tr = build_prompt("Devletin şekli nedir?", docs, language="tr")
    assert "Cevabını Türkçe ver." in prompt_tr
    assert "[1] (kanun.pdf, Sayfa 1)" in prompt_tr
    assert "[2] (kanun.pdf, Sayfa 2)" in prompt_tr
    assert NO_ANSWER in prompt_tr
    assert "Devletin şekli nedir?" in prompt_tr

    # Test English instruction
    prompt_en = build_prompt("What is the state form?", docs, language="en")
    assert "Cevabını İngilizce ver." in prompt_en


def test_parse_citations_patterns():
    # Single citation
    assert parse_citations("Türkiye Devleti bir cumhuriyettir [1].") == [1]

    # Multiple adjoining citations [1][2]
    assert parse_citations("Başkenti Ankara şehridir [1][2].") == [1, 2]

    # Comma separated [1, 3]
    assert parse_citations("Meclis görevleri şunlardır [1, 3].") == [1, 3]

    # Boundary filtering: if only 2 documents exist, drop hallucinated citation [99]
    assert parse_citations("Claim with invalid citation [1, 99].", n_docs=2) == [1]


def test_split_claims_and_citations():
    text = (
        "Türkiye Devleti bir Cumhuriyettir [1]. "
        "Resmî dili Türkçe ve başkenti Ankara şehridir [1, 2]. "
        "Bu genel bir iddiadır."
    )
    claims = split_claims(text)
    assert len(claims) == 3

    assert claims[0][1] == [1]
    assert claims[1][1] == [1, 2]
    assert claims[2][1] == []  # no citation


def test_is_refusal_detection():
    assert is_refusal(f"Bilgi yetersiz: {NO_ANSWER}")
    assert is_refusal(NO_ANSWER)
    assert not is_refusal("Türkiye Devleti bir Cumhuriyettir [1].")


def test_format_docs_strips_internal_header():
    docs = [
        Document(
            page_content="[Kaynak: file.pdf | Sayfa: 5]\nActual text without header noise.",
            metadata={"source": "file.pdf", "page": 5},
        )
    ]
    formatted = format_docs(docs)
    assert "[1] (file.pdf, Sayfa 5)" in formatted
    assert "Actual text without header noise." in formatted
    # Internal metadata bracket should not be duplicated
    assert "[Kaynak: file.pdf | Sayfa: 5]\nActual" not in formatted
