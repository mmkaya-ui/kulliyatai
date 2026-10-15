"""Render the public-domain sample corpus (``sample_docs/*.pdf``) offline.

Inputs are committed to the repo, so the build needs no network and gives the
same bytes on every OS:

* ``sample_docs/source/<name>.txt`` — first line is the title, paragraphs are
  separated by blank lines, single newlines are kept as line breaks.
* ``scripts/fonts/DejaVuSans.ttf`` — bundled Unicode font (Bitstream Vera
  license, see ``scripts/fonts/LICENSE-DejaVu.txt``).

Every page gets a ``- N -`` footer so that page citations can be checked by
eye. ``sample_docs/corpus.json`` records language, provenance and the SHA-256
of every source text and PDF; ``tests/test_corpus.py`` fails if they drift.

Usage::

    python scripts/build_sample_docs.py            # rebuild PDFs + corpus.json
    python scripts/build_sample_docs.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "sample_docs"
SOURCE_DIR = DOCS_DIR / "source"
MANIFEST = DOCS_DIR / "corpus.json"
FONT = ROOT / "scripts" / "fonts" / "DejaVuSans.ttf"

# Provenance of each text. ``derived_from`` describes how the committed text
# was obtained; see scripts/legacy/recover_sources_from_v1_pdfs.py.
CORPUS = {
    "tr_teskilati_esasiye_1924": {
        "language": "tr",
        "work": "Teşkîlât-ı Esâsîye Kanunu (Constitution of Turkey, 1924), amended text",
        "upstream": "https://tr.wikisource.org/wiki/Te%C5%9Fk%C3%AEl%C3%A2t-%C4%B1_Es%C3%A2s%C3%AEye_Kanunu_(1924)",
        "public_domain_basis": "Officially published legislation; Turkish copyright law (FSEK) makes "
        "its reproduction and distribution free ('Resmen yayımlanan veya ilân olunan mevzuat ve "
        "kararların çoğaltılması, yayılması ... serbesttir').",
    },
    "tr_omer_seyfettin_hikayeler": {
        "language": "tr",
        "work": "Ömer Seyfettin, 'Kaşağı' and 'Başını Vermeyen Şehit'",
        "upstream": "https://tr.wikisource.org/wiki/Ka%C5%9Fa%C4%9F%C4%B1",
        "public_domain_basis": "Author died 1920; public domain (life + 70 years).",
    },
    "en_us_constitution": {
        "language": "en",
        "work": "The Constitution of the United States (1787 text, no amendments)",
        "upstream": "https://www.gutenberg.org/ebooks/5",
        "public_domain_basis": "Public-domain U.S. founding document.",
    },
    "en_scandal_in_bohemia": {
        "language": "en",
        "work": "Arthur Conan Doyle, 'A Scandal in Bohemia' (1891)",
        "upstream": "https://www.gutenberg.org/ebooks/1661",
        "public_domain_basis": "Author died 1930; public domain (life + 70 years).",
    },
    "de_kafka_die_verwandlung": {
        "language": "de",
        "work": "Franz Kafka, 'Die Verwandlung' (1915)",
        "upstream": "https://www.gutenberg.org/ebooks/22367",
        "public_domain_basis": "Author died 1924; public domain (life + 70 years).",
    },
}
DERIVED_FROM = (
    "Recovered losslessly (word-for-word) from the v1 PDFs of git commit 5770964 by "
    "scripts/legacy/recover_sources_from_v1_pdfs.py; v1 itself was downloaded from 'upstream'. "
    "HTML entities and wiki ':' indentation were removed from the Seyfettin text."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("body", size=8)
        self.cell(0, 8, f"- {self.page_no()} -", align="C")


def render_pdf(title: str, body: str, out: Path) -> int:
    pdf = _PDF(format="A4")
    pdf.set_creation_date(datetime(2000, 1, 1, tzinfo=UTC))  # deterministic bytes
    pdf.set_title(title)
    pdf.set_producer("kulliyatai scripts/build_sample_docs.py")
    pdf.add_font("body", fname=str(FONT))
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    pdf.set_font("body", size=15)
    pdf.multi_cell(0, 8, title)
    pdf.ln(4)
    pdf.set_font("body", size=10.5)
    for para in body.split("\n\n"):
        pdf.multi_cell(0, 5.5, para)
        pdf.ln(2.5)
    pdf.output(str(out))
    return pdf.page_no()


def read_source(name: str) -> tuple[str, str]:
    text = (SOURCE_DIR / f"{name}.txt").read_text(encoding="utf-8").strip("\n")
    title, _, body = text.partition("\n\n")
    return title.strip(), body.strip()


def build(out_dir: Path) -> dict:
    manifest = {"font": {"file": "scripts/fonts/DejaVuSans.ttf", "sha256": sha256(FONT)}, "documents": []}
    for name, info in CORPUS.items():
        title, body = read_source(name)
        pdf_path = out_dir / f"{name}.pdf"
        pages = render_pdf(title, body, pdf_path)
        manifest["documents"].append(
            {
                "file": f"{name}.pdf",
                "title": title,
                **info,
                "derived_from": DERIVED_FROM,
                "pages": pages,
                "chars": len(body),
                "source_sha256": sha256(SOURCE_DIR / f"{name}.txt"),
                "pdf_sha256": sha256(pdf_path),
            }
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild in a temp dir and compare, write nothing")
    args = ap.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build(Path(tmp))
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ok = manifest == committed
        for d in manifest["documents"]:
            on_disk = sha256(DOCS_DIR / d["file"]) if (DOCS_DIR / d["file"]).exists() else None
            same = on_disk == d["pdf_sha256"]
            ok &= same
            print(f"{d['file']}: {'OK' if same else 'MISMATCH'} ({d['pages']} pages)")
        print("corpus.json:", "OK" if manifest == committed else "MISMATCH")
        return 0 if ok else 1

    manifest = build(DOCS_DIR)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for d in manifest["documents"]:
        print(f"{d['file']}: {d['pages']} pages, {d['chars']:,} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
