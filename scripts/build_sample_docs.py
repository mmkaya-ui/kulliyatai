"""Build the public-domain sample corpus in ``sample_docs/``.

Downloads 5 public-domain texts (Turkish, English, German), strips
boilerplate, and renders each to a PDF with page-number footers so that
retrieval citations ("page N") can be checked by eye.

Usage:
    python scripts/build_sample_docs.py          # (re)build all PDFs

The generated PDFs are committed to the repo, so you only need this script
to verify provenance or to regenerate them. See sample_docs/SOURCES.md.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "sample_docs"
UA = {"User-Agent": "kulliyatai-sample-builder/1.0 (educational RAG demo)"}

# Unicode TTF fonts we can use (Turkish ı/ş/ğ and German ß/ä need Unicode).
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def gutenberg(book_id: int) -> str:
    """Return a Project Gutenberg plain-text book without the PG header/footer.

    Removing the header/footer (and therefore the "Project Gutenberg"
    trademark) is what PG asks for when redistributing the public-domain text.
    """
    txt = _get(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
    txt = txt.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", txt)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", txt)
    if not start or not end:
        raise RuntimeError(f"Could not find PG markers in book {book_id}")
    return txt[start.end() : end.start()].strip()


def _strip_templates(text: str) -> str:
    """Remove (possibly nested) ``{{...}}`` wiki templates."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{{", i):
            depth, i = depth + 1, i + 2
        elif text.startswith("}}", i) and depth:
            depth, i = depth - 1, i + 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def clean_wikitext(wt: str) -> str:
    wt = _strip_templates(wt.replace("\r\n", "\n"))
    wt = re.split(r"\n==\s*(Dipnotlar|Kaynakça|Notlar)\s*==", wt)[0]
    wt = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", "", wt)  # html tags
    wt = re.sub(r"\[\[(?:Kategori|Category|Dosya|File):[^\]]*\]\]", "", wt)
    wt = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", wt)  # [[a|b]] -> b
    wt = re.sub(r"'{2,}", "", wt)  # bold / italics
    wt = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", wt, flags=re.M)  # headings
    wt = re.sub(r"[ \t]+\n", "\n", wt)
    return re.sub(r"\n{3,}", "\n\n", wt).strip()


def wikisource_tr(title: str) -> str:
    """Return the cleaned plain text of a Turkish Wikisource page."""
    q = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
    )
    data = json.loads(_get(f"https://tr.wikisource.org/w/api.php?{q}"))
    if "parse" not in data:
        raise RuntimeError(f"Wikisource page {title!r} not found: {data.get('error')}")
    return clean_wikitext(data["parse"]["wikitext"])


def between(text: str, start: str, end: str | None, occurrence: int = 1) -> str:
    """Slice ``text`` from the n-th occurrence of ``start`` up to ``end``."""
    idx = -1
    for _ in range(occurrence):
        idx = text.index(start, idx + 1)
    stop = text.index(end, idx + len(start)) if end else len(text)
    return text[idx:stop].strip()


def unwrap(text: str) -> str:
    """Join hard-wrapped lines inside paragraphs (Gutenberg wraps at ~70 cols)."""
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p).strip() for p in paras if p.strip())


def find_font() -> str:
    env = os.environ.get("SAMPLE_DOCS_FONT")
    for path in [env, *FONT_CANDIDATES]:
        if path and Path(path).exists():
            return path
    sys.exit("No Unicode TTF font found. Install fonts-dejavu-core or set SAMPLE_DOCS_FONT.")


class _PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("body", size=8)
        self.cell(0, 8, f"- {self.page_no()} -", align="C")


def render_pdf(title: str, body: str, out: Path, font: str) -> int:
    pdf = _PDF(format="A4")
    pdf.set_creation_date(datetime(2000, 1, 1, tzinfo=UTC))  # deterministic output
    pdf.set_title(title)
    pdf.add_font("body", fname=font)
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


def build() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    font = find_font()

    # Ömer Seyfettin (1884-1920): public domain worldwide.
    stories = ["Kaşağı", "Başını Vermeyen Şehit"]
    seyfettin = "\n\n".join(f"{t}\n\n{wikisource_tr(t)}" for t in stories)

    # Kafka "Die Verwandlung" (1915): Kafka died 1924, public domain worldwide.
    # Drop the producer credits / 1917 title page; start at chapter "I.".
    kafka_all = gutenberg(22367)
    kafka = kafka_all[kafka_all.index("\nI.\n") :].strip()

    # Doyle, "A Scandal in Bohemia" (1891): Doyle died 1930, public domain worldwide.
    holmes_all = gutenberg(1661)
    holmes = between(holmes_all, "I. A SCANDAL IN BOHEMIA", "II. THE RED-HEADED LEAGUE")

    # U.S. Constitution, original 1787 text (this edition has no amendments).
    # U.S. federal government work: public domain. Drop the PG editors' notes.
    const_all = gutenberg(5)
    constitution = const_all[const_all.index("We the people") :].strip()

    docs = {
        # Official legal text: excluded from copyright under Turkish law
        # (FSEK; Wikisource tags it {{FSEK-32}}). Final 1960 amended version.
        "tr_teskilati_esasiye_1924.pdf": (
            "Teşkîlât-ı Esâsîye Kanunu (1924)",
            wikisource_tr("Teşkîlât-ı Esâsîye Kanunu (1924)"),
        ),
        "tr_omer_seyfettin_hikayeler.pdf": ("Ömer Seyfettin — Hikâyeler", seyfettin),
        "en_us_constitution.pdf": ("The Constitution of the United States (1787)", constitution),
        "en_scandal_in_bohemia.pdf": ("A Scandal in Bohemia — Arthur Conan Doyle", holmes),
        "de_kafka_die_verwandlung.pdf": ("Die Verwandlung — Franz Kafka", kafka),
    }

    for name, (title, raw) in docs.items():
        body = unwrap(raw) if name.startswith(("en_", "de_")) else raw.replace("\r\n", "\n")
        body = re.sub(r"\n{3,}", "\n\n", body)
        body = re.sub(r"[\U00010000-\U0010FFFF]", "", body)  # decorative ornaments
        pages = render_pdf(title, body, OUT_DIR / name, font)
        print(f"{name}: {pages} pages, {len(body):,} chars")


if __name__ == "__main__":
    build()
