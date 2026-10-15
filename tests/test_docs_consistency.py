"""Documentation and defaults must agree with the measured results."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from evals import report
from rag.config import DEFAULT_ABSTAIN_THRESHOLD, DEFAULT_CANDIDATE_K, DEFAULT_INDEX

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "evals" / "results" / "results.json").read_text(encoding="utf-8"))


def test_results_md_and_readme_block_are_generated_from_results_json():
    assert report.main(["--check"]) == 0, "run `python -m evals.report` after `python -m evals.run`"


def test_config_defaults_are_the_dev_selected_settings():
    assert asdict(DEFAULT_INDEX) == RESULTS["selected"]["hybrid"]
    assert DEFAULT_CANDIDATE_K == RESULTS["selected"]["rerank_candidate_k"]
    assert abs(DEFAULT_ABSTAIN_THRESHOLD - RESULTS["abstention"]["threshold"]) < 5e-4


def test_results_were_produced_from_the_committed_data():
    env = RESULTS["environment"]
    import hashlib

    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()  # noqa: E731
    assert env["questions_sha256"] == sha("evals/questions.jsonl")
    assert env["corpus_manifest_sha256"] == sha("sample_docs/corpus.json")


def test_no_llm_quality_numbers_are_claimed():
    assert RESULTS["generation"]["status"] == "not_measured"
    # The README's "Corrections" section quotes the old, withdrawn claims on purpose.
    readme = (ROOT / "README.md").read_text(encoding="utf-8").split("## Corrections to earlier versions")[0]
    text = readme + (ROOT / "evals" / "RESULTS.md").read_text(encoding="utf-8")
    assert not re.search(r"(refusal|citation)[^\n]{0,40}100(\.0)?\s?%", text, re.I)


def test_relative_links_in_markdown_exist():
    for md in [
        ROOT / "README.md",
        ROOT / "evals" / "RESULTS.md",
        ROOT / "sample_docs" / "SOURCES.md",
        ROOT / "docs" / "git-history.md",
    ]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
            if "://" in target:
                continue
            assert (md.parent / target).exists(), f"{md.name}: broken link {target}"
