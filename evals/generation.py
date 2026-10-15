"""Optional LLM answer evaluation (needs an API key; never run by the tests or CI).

    python -m evals.generation --provider google --model gemini-3.8-flash --split test

Measures, for the retrieval pipeline selected in ``config.py``:

* ``refusal_accuracy`` – unanswerable questions answered with ``NO_ANSWER``;
* ``answer_rate`` – answerable questions *not* refused;
* ``invalid_citation_rate`` – answers citing a passage number that does not exist;
* ``cites_evidence_rate`` – answered questions where at least one cited passage
  contains the gold evidence (machine-checked, no judge);
* ``judge_support_rate`` (with ``--judge``) – share of (sentence, citation)
  pairs that the LLM judge (``judge_prompts.CITATION_VERIFIER_PROMPT``) says
  are supported. This is an LLM opinion, reported as such.

Results go to ``evals/results/generation_<provider>_<model>.json``. Nothing in
this file has been run for the published results (see RESULTS.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable

from evals.dataset import EvalItem, corpus_texts, load_verified, relevant
from evals.judge_prompts import CITATION_VERIFIER_PROMPT
from evals.metrics import mean_ci
from rag.chunking import strip_context_header
from rag.prompts import build_prompt, invalid_citations, is_refusal, parse_citations, split_claims
from rag.retrieval import Retriever

LLM = Callable[[str], str]


def _judge_supports(judge: LLM, claim: str, n: int, passage: str) -> bool | None:
    raw = judge(CITATION_VERIFIER_PROMPT.format(claim=claim, citation_number=n, passage=passage))
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return bool(json.loads(m.group(0))["supports_claim"]) if m else None
    except (json.JSONDecodeError, KeyError):
        return None


def evaluate(
    llm: LLM,
    retriever: Retriever,
    items: list[EvalItem],
    docs,
    *,
    k: int = 4,
    rerank: bool = True,
    language: str = "auto",
    judge: LLM | None = None,
) -> dict:
    refused_un, answered, invalid, cites_ev, judged = [], [], [], [], []
    samples = []
    for it in items:
        hits = retriever.search(it.question, k=k, mode="hybrid", rerank=rerank)
        ans = llm(build_prompt(it.question, [h.chunk for h in hits], language))
        refused = is_refusal(ans)
        samples.append({"id": it.id, "answer": ans, "refused": refused})
        if not it.answerable:
            refused_un.append(float(refused))
            continue
        answered.append(float(not refused))
        if refused:
            continue
        invalid.append(float(bool(invalid_citations(ans, len(hits)))))
        cited = parse_citations(ans, len(hits))
        cites_ev.append(float(any(relevant(hits[n - 1].chunk, it, docs) for n in cited)))
        if judge is not None:
            for claim, nums in split_claims(ans):
                for n in nums:
                    if 1 <= n <= len(hits):
                        v = _judge_supports(judge, claim, n, strip_context_header(hits[n - 1].chunk.page_content))
                        if v is not None:
                            judged.append(float(v))
    out = {
        "refusal_accuracy": mean_ci(refused_un),
        "answer_rate": mean_ci(answered),
        "invalid_citation_rate": mean_ci(invalid),
        "cites_evidence_rate": mean_ci(cites_ev),
        "samples": samples,
    }
    if judge is not None:
        out["judge_support_rate"] = mean_ci(judged)
    return out


def _make_llm(provider: str, model: str) -> LLM:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=2)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, temperature=0)
    else:
        raise SystemExit(f"unknown provider {provider!r}")
    chain = PromptTemplate.from_template("{p}") | chat | StrOutputParser()
    return lambda prompt: chain.invoke({"p": prompt})


def main(argv=None) -> int:
    from rag.config import ROOT, Settings, load_dotenv_if_available
    from rag.embeddings import make_embedder
    from rag.index import RAGIndex
    from rag.loading import load_pdfs_with_pymupdf
    from rag.retrieval import Reranker

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=["google", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--judge", action="store_true", help="also run the LLM citation judge (same model)")
    args = ap.parse_args(argv)
    load_dotenv_if_available()
    s = Settings.from_env()
    pages = load_pdfs_with_pymupdf(ROOT / "sample_docs", log=lambda *_: None)
    items = [i for i in load_verified(pages) if i.split == args.split]
    embedder = make_embedder("fastembed", s.embedding_model)
    retriever = Retriever(
        RAGIndex.build(pages, embedder, s.index), embedder, candidate_k=s.candidate_k, reranker=Reranker(s.rerank_model)
    )
    llm = _make_llm(args.provider, args.model)
    res = evaluate(llm, retriever, items, corpus_texts(pages), judge=llm if args.judge else None)
    res["config"] = {"provider": args.provider, "model": args.model, "split": args.split, "judge": args.judge}
    out = ROOT / "evals" / "results" / f"generation_{args.provider}_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
