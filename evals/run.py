"""Reproducible retrieval benchmark: dev-split selection, test-split reporting.

    python -m evals.run              # full run: ~20 min on 2 CPU cores with a warm evals/.cache; much longer on first run
    python -m evals.run --no-rerank  # first-stage retrieval only (minutes)

Protocol
--------
1. ``evals/questions.jsonl`` is verified against the PDFs (evidence quotes must
   occur exactly once; absent terms must not occur). Any error aborts the run.
2. **Dev split only:** a grid of index settings (chunk size x contextual header
   x dense windows x BM25 tokenizer) is evaluated for each retrieval mode and
   the best setting per mode is selected by MRR@10 (ties: Hit@5, then grid
   order). The reranker's candidate count is selected the same way, and the
   abstention threshold is set to the 5th percentile of the best-passage
   reranker score over answerable dev questions.
3. **Test split only:** the selected systems, one-factor ablations and the
   pre-v2 settings are reported with 95 % bootstrap CIs and paired
   randomization tests. Nothing on the test split is used for selection.
4. Latency is measured in a separate pass with the disk caches bypassed, after a
   warm-up, per stage (median and p95), on the machine recorded in the output.

Outputs: ``evals/results/results.json`` (everything, machine-readable) and
``evals/RESULTS.md`` + the README benchmark block, both rendered from it by
:mod:`evals.report`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from evals.cache import CachedReranker
from evals.dataset import QUESTIONS_FILE, EvalItem, Span, chunk_span, corpus_texts, load_verified
from evals.metrics import (
    auroc,
    mean_ci,
    paired_permutation_test,
    percentile,
    query_metrics,
    sentence_hits_evidence,
)
from rag.answer import extractive_answer
from rag.config import ROOT, load_dotenv_if_available
from rag.embeddings import CachedEmbedder, FastEmbedEmbedder
from rag.index import IndexSettings, RAGIndex
from rag.loading import load_pdfs_with_pymupdf
from rag.retrieval import Reranker, Retriever, SearchTrace

RESULTS_DIR = ROOT / "evals" / "results"
CACHE_DIR = ROOT / "evals" / ".cache"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"
DEPTH = 10  # retrieved list length used for metrics
CANDIDATE_KS = (10, 20, 30)
COVERAGE_TARGET = 0.95  # abstention: share of answerable dev questions that must pass the gate
METRICS = ("hit@1", "hit@3", "hit@5", "mrr@10", "evidence_recall@5")

CHUNKINGS = ((500, 75), (1000, 150), (1500, 225))
GRID = [
    IndexSettings(cs, ov, hdr, win, tok)
    for cs, ov in CHUNKINGS
    for hdr in (False, True)
    for win in (False, True)
    for tok in ("plain", "prefix5")
]
# Settings of the pipeline before this benchmark existed (README v0.2): 1500/300 chunks,
# header, single vector per chunk (silently truncated), plain BM25, RRF weights 0.6/0.4.
LEGACY = IndexSettings(1500, 300, True, False, "plain")
LEGACY_WEIGHTS = (0.6, 0.4)


def log(*a):
    print(*a, flush=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def environment() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    pkgs = {}
    for name in ("fastembed", "onnxruntime", "faiss-cpu", "rank-bm25", "PyMuPDF", "numpy", "langchain-text-splitters"):
        try:
            pkgs[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    except OSError:
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "git_commit": commit,
        "git_dirty": dirty,
        "questions_sha256": sha256_file(QUESTIONS_FILE),
        "corpus_manifest_sha256": sha256_file(ROOT / "sample_docs" / "corpus.json"),
    }


def settings_label(s: IndexSettings) -> str:
    return (
        f"chunk {s.chunk_size}/{s.chunk_overlap}, header {'on' if s.contextual_header else 'off'}, "
        f"dense {'windows' if s.dense_windows else 'single'}, bm25 {s.bm25_tokenizer}"
    )


class Bench:
    def __init__(self, pages, items: list[EvalItem], use_rerank: bool):
        self.pages = pages
        self.items = items
        self.docs = corpus_texts(pages)
        self.embedder = CachedEmbedder(FastEmbedEmbedder(EMBED_MODEL), CACHE_DIR / "embeddings.sqlite")
        self.reranker = CachedReranker(Reranker(RERANK_MODEL), CACHE_DIR / "rerank.sqlite") if use_rerank else None
        self._indexes: dict[IndexSettings, RAGIndex] = {}

    def index(self, s: IndexSettings) -> RAGIndex:
        if s not in self._indexes:
            self._indexes[s] = RAGIndex.build(self.pages, self.embedder, s)
        return self._indexes[s]

    def retriever(self, s: IndexSettings, weights=(1.0, 1.0), candidate_k: int = 20) -> Retriever:
        return Retriever(
            self.index(s),
            self.embedder,
            dense_weight=weights[0],
            bm25_weight=weights[1],
            candidate_k=candidate_k,
            reranker=self.reranker,
        )

    def run(self, r: Retriever, items: list[EvalItem], mode: str, rerank: bool = False) -> dict:
        per_query, hits_by_id = {}, {}
        for it in items:
            hits = r.search(it.question, k=DEPTH, mode=mode, rerank=rerank)
            hits_by_id[it.id] = hits
            if it.answerable:
                per_query[it.id] = query_metrics([h.chunk for h in hits], it, self.docs)
        summary = {m: float(np.mean([q[m] for q in per_query.values()])) for m in METRICS}
        return {"per_query": per_query, "summary": summary, "hits": hits_by_id}


def select(results: dict) -> object:
    """Key with the best (mrr@10, hit@5); earlier keys win ties."""
    best, best_key = None, None
    for key, res in results.items():
        score = (round(res["summary"]["mrr@10"], 6), round(res["summary"]["hit@5"], 6))
        if best is None or score > best:
            best, best_key = score, key
    return best_key


def with_ci(res: dict) -> dict:
    return {m: mean_ci([q[m] for q in res["per_query"].values()]) for m in METRICS}


def compare(a: dict, b: dict, metric: str = "mrr@10") -> dict:
    ids = sorted(set(a["per_query"]) & set(b["per_query"]))
    return paired_permutation_test([a["per_query"][i][metric] for i in ids], [b["per_query"][i][metric] for i in ids])


def latency_pass(
    index: RAGIndex, embedder, items: list[EvalItem], mode: str, rerank: bool, candidate_k: int, reranker=None
):
    """Embedder and reranker with the disk caches bypassed: warm-up on 3 queries, then time every query."""
    r = Retriever(index, embedder, candidate_k=candidate_k, reranker=reranker)
    for it in items[:3]:
        r.search(it.question, k=DEPTH, mode=mode, rerank=rerank)
    stages: dict[str, list[float]] = {}
    totals = []
    for it in items:
        tr = SearchTrace()
        t0 = time.perf_counter()
        r.search(it.question, k=DEPTH, mode=mode, rerank=rerank, trace=tr)
        totals.append((time.perf_counter() - t0) * 1000)
        for k, v in tr.timings_ms.items():
            stages.setdefault(k, []).append(v)
    summ = lambda xs: {"median_ms": percentile(xs, 50), "p95_ms": percentile(xs, 95), "n": len(xs)}  # noqa: E731
    return {"total": summ(totals), "stages": {k: summ(v) for k, v in stages.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-rerank", action="store_true", help="skip everything that needs the cross-encoder")
    ap.add_argument("--no-latency", action="store_true", help="skip the uncached latency pass")
    args = ap.parse_args(argv)
    load_dotenv_if_available()
    use_rerank = not args.no_rerank
    t_start = time.time()

    pages = load_pdfs_with_pymupdf(ROOT / "sample_docs", log=lambda *_: None)
    items = load_verified(pages)
    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]
    dev_ans = [i for i in dev if i.answerable]
    test_ans = [i for i in test if i.answerable]
    log(
        f"dataset verified: {len(items)} items, dev {len(dev)} ({len(dev_ans)} answerable), test {len(test)} ({len(test_ans)} answerable)"
    )

    b = Bench(pages, items, use_rerank)
    out: dict = {
        "environment": environment(),
        "protocol": {
            "selection_split": "dev",
            "report_split": "test",
            "selection_metric": "mrr@10 (ties: hit@5)",
            "depth": DEPTH,
            "rrf_weights": [1.0, 1.0],
            "rrf_k": 60,
            "bootstrap_resamples": 10_000,
            "permutation_resamples": 10_000,
            "embedder": f"fastembed/{EMBED_MODEL}",
            "reranker": RERANK_MODEL,
            "relevance": "chunk contains >= 50% of an evidence span",
            "n_items": {
                "dev": len(dev),
                "dev_answerable": len(dev_ans),
                "test": len(test),
                "test_answerable": len(test_ans),
            },
        },
    }

    # ---------------- 1. dev grid --------------------------------------------------------------
    log(f"[dev] grid: {len(GRID)} index settings x 3 modes")
    grid: dict[str, dict] = {"dense": {}, "bm25": {}, "hybrid": {}}
    for s in GRID:
        for mode in grid:
            grid[mode][s] = b.run(b.retriever(s), dev_ans, mode)
        log(f"  {settings_label(s)}: " + ", ".join(f"{m}={grid[m][s]['summary']['mrr@10']:.3f}" for m in grid))
    chosen = {mode: select(grid[mode]) for mode in grid}
    out["dev_grid"] = [{"settings": asdict(s), "mode": mode, **grid[mode][s]["summary"]} for mode in grid for s in GRID]
    out["selected"] = {mode: asdict(s) for mode, s in chosen.items()}
    log("[dev] selected: " + "; ".join(f"{m}: {settings_label(s)}" for m, s in chosen.items()))

    best = chosen["hybrid"]
    chosen_k = None
    if use_rerank:
        rr_dev = {}
        for k in CANDIDATE_KS:
            rr_dev[k] = b.run(b.retriever(best, candidate_k=k), dev_ans, "hybrid", rerank=True)
            log(f"  [dev] rerank candidate_k={k}: mrr@10={rr_dev[k]['summary']['mrr@10']:.3f}")
        chosen_k = select(rr_dev)
        out["dev_rerank"] = [{"candidate_k": k, **rr_dev[k]["summary"]} for k in CANDIDATE_KS]
        out["selected"]["rerank_candidate_k"] = chosen_k

    # ---------------- 2. test: main systems ------------------------------------------------------
    log("[test] main systems")
    systems = {
        "BM25": (chosen["bm25"], "bm25", False, (1.0, 1.0), 20),
        "Dense": (chosen["dense"], "dense", False, (1.0, 1.0), 20),
        "Hybrid (RRF)": (best, "hybrid", False, (1.0, 1.0), 20),
        "Pre-v2 settings (hybrid)": (LEGACY, "hybrid", False, LEGACY_WEIGHTS, 20),
    }
    if use_rerank:
        systems["Hybrid + rerank"] = (best, "hybrid", True, (1.0, 1.0), chosen_k)
    test_res = {}
    for name, (s, mode, rr, w, ck) in systems.items():
        test_res[name] = b.run(b.retriever(s, w, ck), test, mode, rerank=rr)
        log(f"  {name}: " + ", ".join(f"{m}={v:.3f}" for m, v in test_res[name]["summary"].items()))
    out["test_systems"] = [
        {
            "name": n,
            "settings": asdict(systems[n][0]),
            "mode": systems[n][1],
            "rerank": systems[n][2],
            "rrf_weights": list(systems[n][3]),
            "candidate_k": systems[n][4] if systems[n][2] else None,
            "metrics": with_ci(test_res[n]),
        }
        for n in systems
    ]
    pairs = [("Hybrid (RRF)", "BM25"), ("Hybrid (RRF)", "Dense"), ("Hybrid (RRF)", "Pre-v2 settings (hybrid)")]
    if use_rerank:
        pairs = [("Hybrid + rerank", "Hybrid (RRF)"), ("Hybrid + rerank", "BM25"), *pairs]
    out["test_significance"] = [
        {"a": a, "b": c, "metric": m, **compare(test_res[a], test_res[c], m)}
        for a, c in pairs
        for m in ("mrr@10", "hit@1")
    ]

    # breakdown by language of document / cross-lingual
    def group(it: EvalItem) -> str:
        return "cross-lingual" if it.cross_lingual else f"{it.doc_lang} docs, same-language questions"

    groups: dict[str, list[str]] = {}
    for it in test_ans:
        groups.setdefault(group(it), []).append(it.id)
    out["test_breakdown"] = [
        {
            "system": n,
            "group": g,
            "n": len(ids),
            "hit@5": float(np.mean([test_res[n]["per_query"][i]["hit@5"] for i in ids])),
            "mrr@10": float(np.mean([test_res[n]["per_query"][i]["mrr@10"] for i in ids])),
        }
        for n in test_res
        for g, ids in sorted(groups.items())
    ]

    # ---------------- 3. test: one-factor ablations around the selected hybrid --------------------
    log("[test] ablations")
    ablations = {
        ("dense windows -> single vector" if best.dense_windows else "single vector -> dense windows"): replace(
            best, dense_windows=not best.dense_windows
        ),
        "contextual header toggled": replace(best, contextual_header=not best.contextual_header),
        "bm25 tokenizer toggled": replace(
            best, bm25_tokenizer="plain" if best.bm25_tokenizer != "plain" else "prefix5"
        ),
    }
    for cs, ov in CHUNKINGS:
        if cs != best.chunk_size:
            ablations[f"chunk size {cs}"] = replace(best, chunk_size=cs, chunk_overlap=ov)
    out["test_ablations"] = []
    for name, s in ablations.items():
        res = b.run(b.retriever(s), test, "hybrid")
        out["test_ablations"].append(
            {
                "change": name,
                "settings": asdict(s),
                "metrics": with_ci(res),
                "vs_selected": compare(res, test_res["Hybrid (RRF)"]),
            }
        )
        log(f"  {name}: mrr@10={res['summary']['mrr@10']:.3f}")

    # index stats for the selected settings
    out["index_stats"] = {settings_label(s): b.index(s).stats for s in {best, LEGACY, *chosen.values()}}

    # ---------------- 4. abstention + extractive answers (reranker) ----------------------------------
    if use_rerank:
        log("[dev/test] abstention + extractive answers")
        r_best = b.retriever(best, candidate_k=chosen_k)
        top = {}
        for it in dev + test:
            hits = r_best.search(it.question, k=DEPTH, mode="hybrid", rerank=True)
            top[it.id] = hits
        conf = {i: (h[0].rerank_score if h else -1e9) for i, h in top.items()}
        thr = percentile([conf[i.id] for i in dev_ans], 100 * (1 - COVERAGE_TARGET))
        test_un = [i for i in test if not i.answerable]
        dense_top = {
            it.id: (test_res["Dense"]["hits"][it.id][0].score if test_res["Dense"]["hits"][it.id] else -1)
            for it in test
        }
        bm25_top = {
            it.id: (test_res["BM25"]["hits"][it.id][0].score if test_res["BM25"]["hits"][it.id] else 0) for it in test
        }
        out["abstention"] = {
            "signal": "reranker score of the best passage",
            "threshold": thr,
            "threshold_rule": f"{100 * (1 - COVERAGE_TARGET):.0f}th percentile over answerable dev questions",
            "test_answerable_kept": mean_ci([float(conf[i.id] >= thr) for i in test_ans]),
            "test_unanswerable_refused": mean_ci([float(conf[i.id] < thr) for i in test_un]),
            "test_auroc": {
                "reranker_top_score": auroc([conf[i.id] for i in test_ans], [conf[i.id] for i in test_un]),
                "dense_top_cosine": auroc([dense_top[i.id] for i in test_ans], [dense_top[i.id] for i in test_un]),
                "bm25_top_score": auroc([bm25_top[i.id] for i in test_ans], [bm25_top[i.id] for i in test_un]),
            },
            "n_test_unanswerable": len(test_un),
        }
        log(
            f"  threshold={thr:.3f} kept={out['abstention']['test_answerable_kept']['mean']:.3f} refused={out['abstention']['test_unanswerable_refused']['mean']:.3f}"
        )

        ext1, ext2, e2e = [], [], []
        for it in test:
            hits = top[it.id]
            passed = conf[it.id] >= thr
            if not it.answerable:
                e2e.append(float(not passed))
                continue
            sents = extractive_answer(it.question, hits, b.reranker, passages=3, max_sentences=2)
            by_score = sorted(sents, key=lambda s: -s.score)

            def sent_span(s, hits=hits) -> Span | None:
                cs = chunk_span(hits[s.hit_rank - 1].chunk, b.docs)
                return Span(cs.doc, cs.start + s.start, cs.start + s.start + len(s.text)) if cs else None

            h1 = bool(by_score) and sentence_hits_evidence(sent_span(by_score[0]), it)
            h2 = any(sentence_hits_evidence(sent_span(s), it) for s in sents)
            ext1.append(float(h1))
            ext2.append(float(h2))
            e2e.append(float(passed and h2))
        out["extractive"] = {
            "definition": "sentence counts if it overlaps an evidence span by >= 50% of the shorter of the two",
            "top1_sentence_supports": mean_ci(ext1),
            "any_of_2_sentences_supports": mean_ci(ext2),
            "end_to_end_test_accuracy": mean_ci(e2e),
            "end_to_end_definition": "answerable: passes the gate AND one of the 2 returned sentences is evidence; "
            "unanswerable: refused by the gate",
        }
        log(f"  extractive top1={np.mean(ext1):.3f} any2={np.mean(ext2):.3f} e2e={np.mean(e2e):.3f}")

    # ---------------- 5. latency (fresh models, no caches) ----------------------------------------
    if not args.no_latency:
        log("[test] latency pass (uncached)")
        raw_embedder = b.embedder.base  # same model instance, cache wrapper bypassed (avoids a 2nd copy in RAM)
        lat = {
            "BM25": latency_pass(b.index(chosen["bm25"]), raw_embedder, test, "bm25", False, 20),
            "Dense": latency_pass(b.index(chosen["dense"]), raw_embedder, test, "dense", False, 20),
            "Hybrid (RRF)": latency_pass(b.index(best), raw_embedder, test, "hybrid", False, 20),
        }
        if use_rerank:
            raw = b.reranker.base  # the model behind the cache: every call is computed
            lat["Hybrid + rerank"] = latency_pass(b.index(best), raw_embedder, test, "hybrid", True, chosen_k, raw)
            subset = test_ans[:16]
            curve = []
            for k in CANDIDATE_KS:
                res = b.run(b.retriever(best, candidate_k=k), test, "hybrid", rerank=True)
                curve.append(
                    {
                        "candidate_k": k,
                        "metrics": with_ci(res),
                        "latency": latency_pass(b.index(best), raw_embedder, subset, "hybrid", True, k, raw),
                    }
                )
                log(
                    f"  candidate_k={k}: mrr@10={res['summary']['mrr@10']:.3f} median={curve[-1]['latency']['total']['median_ms']:.0f} ms"
                )
            out["rerank_tradeoff"] = curve
        out["latency"] = lat
        for n, v in lat.items():
            log(f"  {n}: median {v['total']['median_ms']:.1f} ms, p95 {v['total']['p95_ms']:.1f} ms")

    out["generation"] = {
        "status": "not_measured",
        "reason": "no LLM was called in this run; see evals/generation.py to evaluate an LLM",
    }
    out["runtime_s"] = round(time.time() - t_start, 1)
    out["cache"] = {"embeddings": {"hits": b.embedder.hits, "misses": b.embedder.misses}}
    if b.reranker:
        out["cache"]["rerank"] = {"hits": b.reranker.hits, "misses": b.reranker.misses}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / ("results.json" if use_rerank else "results_no_rerank.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote {path.relative_to(ROOT)} in {out['runtime_s']} s")
    if use_rerank:
        from evals import report

        report.main([])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
