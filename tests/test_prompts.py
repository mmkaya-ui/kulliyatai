"""Prompt construction, citation extraction and refusal detection."""

import pytest
from langchain_core.documents import Document

from rag.prompts import (
    NO_ANSWER,
    build_prompt,
    format_docs,
    invalid_citations,
    is_refusal,
    parse_citations,
    split_claims,
)

DOCS = [
    Document(
        page_content="[Kaynak: kanun.pdf | Sayfa: 1]\nTürkiye Devleti bir Cumhuriyettir.",
        metadata={"source": "kanun.pdf", "page": 1},
    ),
    Document(page_content="Başkenti Ankara şehridir.", metadata={"source": "kanun.pdf", "page": 2}),
]


def test_build_prompt_structure_and_language():
    p = build_prompt("Devletin şekli nedir?", DOCS, language="tr")
    assert "Cevabını Türkçe ver." in p
    assert "[1] (kanun.pdf, Sayfa 1)" in p and "[2] (kanun.pdf, Sayfa 2)" in p
    assert NO_ANSWER in p and "Devletin şekli nedir?" in p
    assert "Cevabını İngilizce ver." in build_prompt("What?", DOCS, language="en")
    with pytest.raises(ValueError):
        build_prompt("x", DOCS, language="fr")


def test_format_docs_strips_internal_header():
    out = format_docs(DOCS)
    assert "Türkiye Devleti bir Cumhuriyettir." in out
    assert "[Kaynak: kanun.pdf | Sayfa: 1]\nTürkiye" not in out


def test_parse_citations_patterns():
    assert parse_citations("Cumhuriyettir [1].") == [1]
    assert parse_citations("Ankara [1][2].") == [1, 2]
    assert parse_citations("Görevler [1, 3].") == [1, 3]
    assert parse_citations("Invalid [1, 99].", n_docs=2) == [1]
    assert invalid_citations("Invalid [1, 99] and [7].", n_docs=2) == [7, 99]
    assert invalid_citations("Fine [2].", n_docs=2) == []


def test_split_claims_and_citations():
    claims = split_claims("Cumhuriyettir [1]. Ankara şehridir [1, 2]. Bu genel bir iddiadır.")
    assert [c for _, c in claims] == [[1], [1, 2], []]


def test_is_refusal_detection():
    assert is_refusal(f"Bilgi yetersiz: {NO_ANSWER}")
    assert not is_refusal("Türkiye Devleti bir Cumhuriyettir [1].")
