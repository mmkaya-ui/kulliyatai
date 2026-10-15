"""The bundled corpus is byte-for-byte reproducible from the committed sources."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "sample_docs" / "corpus.json").read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_module():
    spec = importlib.util.spec_from_file_location("build_sample_docs", ROOT / "scripts" / "build_sample_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_hashes_match_files():
    assert _sha(ROOT / MANIFEST["font"]["file"]) == MANIFEST["font"]["sha256"]
    for d in MANIFEST["documents"]:
        stem = d["file"][:-4]
        assert _sha(ROOT / "sample_docs" / d["file"]) == d["pdf_sha256"], d["file"]
        assert _sha(ROOT / "sample_docs" / "source" / f"{stem}.txt") == d["source_sha256"], stem
        assert d["language"] == stem[:2]


def test_rebuild_is_byte_identical():
    assert _build_module().main(["--check"]) == 0


def test_pdf_text_equals_source_text():
    """Every word of the source text is in the PDF text layer, in order (plus title and page footers)."""
    for d in MANIFEST["documents"]:
        stem = d["file"][:-4]
        src = (ROOT / "sample_docs" / "source" / f"{stem}.txt").read_text(encoding="utf-8").split()
        words = []
        with fitz.open(ROOT / "sample_docs" / d["file"]) as pdf:
            assert pdf.page_count == d["pages"]
            for page in pdf:
                for line in page.get_text().splitlines():
                    if not re.fullmatch(r"- \d+ -", line.strip()):
                        words += line.split()
        assert words == src, stem
