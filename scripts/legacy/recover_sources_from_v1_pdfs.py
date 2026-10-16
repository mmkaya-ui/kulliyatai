"""One-time migration: recover the plain-text sources of the v1 sample PDFs.

Why this exists
---------------
The first version of ``sample_docs/`` (git commit ``5770964``) was rendered
straight from texts downloaded at build time, and only the PDFs were committed.
Rebuilding them needed network access to Project Gutenberg / Wikisource and
the Windows Arial font, so the corpus could not be rebuilt reproducibly. The
Turkish story text also still contained raw HTML entities (``&mdash;``,
``&acirc;``) and wiki indentation colons.

This script reads the v1 PDFs, rebuilds the paragraph structure from the page
layout and writes ``sample_docs/source/<name>.txt``. Those text files are now
the committed, canonical source; ``scripts/build_sample_docs.py`` renders the
PDFs from them offline with a bundled font.

How paragraphs are recovered (fpdf2 layout of v1)
-------------------------------------------------
* Lines inside a paragraph are 5.5 mm apart; paragraphs get 2.5 mm extra, so a
  baseline gap above ``PARA_GAP_PT`` starts a new paragraph.
* v1 used justified text: every wrapped line ends at the right margin. A line
  that ends before it, inside a paragraph, was an explicit line break.
* At a page break, the paragraph continues when the page's last line is a
  full-width (wrapped) line.

Usage (needs the v1 PDFs, e.g. ``git show 5770964:sample_docs/<file>``)::

    python scripts/legacy/recover_sources_from_v1_pdfs.py <v1_pdf_dir> <out_dir>
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import fitz

PARA_GAP_PT = 19.0  # line pitch is 15.6 pt, paragraph pitch 22.7 pt
FULL_WIDTH_X1 = 530.0  # justified lines end at x≈535.8 on A4 with 20 mm margins
FOOTER_RE = re.compile(r"^- \d+ -$")


def _lines(page: fitz.Page) -> list[tuple[float, float, float, str]]:
    """Return (y0, x1, size, text) for every visual line, top to bottom.

    PyMuPDF can split one widely-justified line into several "lines"; pieces
    sharing a baseline are merged back left-to-right.
    """
    pieces = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"])
            if text.strip():
                x0, y0, x1, _ = line["bbox"]
                pieces.append((round(y0, 1), x0, x1, line["spans"][0]["size"], text))
    rows: dict[float, list] = {}
    for y0, x0, x1, size, text in pieces:
        key = next((k for k in rows if abs(k - y0) < 1.0), y0)
        rows.setdefault(key, []).append((x0, x1, size, text))
    out = []
    for y0, parts in rows.items():
        parts.sort()
        text = re.sub(r" {2,}", " ", " ".join(p[3] for p in parts)).strip()  # justified spacing
        if text and not FOOTER_RE.match(text):
            out.append((y0, max(p[1] for p in parts), parts[0][2], text))
    return sorted(out, key=lambda t: t[0])


def recover(pdf_path: Path) -> tuple[str, str]:
    """Return (title, body) where paragraphs are separated by blank lines."""
    doc = fitz.open(pdf_path)
    title_parts: list[str] = []
    paras: list[str] = []
    cur = ""
    prev_y = prev_x1 = None
    for page_index, page in enumerate(doc):
        lines = _lines(page)
        for i, (y, x1, size, text) in enumerate(lines):
            if page_index == 0 and size > 12:  # title, rendered at 15 pt
                title_parts.append(text)
                continue
            new_para = cur == "" or (i > 0 and prev_y is not None and y - prev_y > PARA_GAP_PT)
            if i == 0 and cur and prev_x1 is not None and prev_x1 < FULL_WIDTH_X1:
                new_para = True  # previous page ended its paragraph on a short line
            if new_para:
                if cur:
                    paras.append(cur)
                cur = text
            elif prev_x1 is not None and prev_x1 >= FULL_WIDTH_X1 and not text.startswith(":"):
                cur += " " + text  # soft wrap (wiki dialogue lines start with ':')
            else:
                cur += "\n" + text  # explicit line break inside the paragraph
            prev_y, prev_x1 = y, x1
        prev_y = None  # gaps are only meaningful within one page
    if cur:
        paras.append(cur)
    return " ".join(title_parts), "\n\n".join(paras)


def clean_wiki_residue(text: str) -> str:
    """Decode HTML entities and drop MediaWiki ':' indentation left in v1."""
    text = html.unescape(text)
    text = re.sub(r"(?m)^:\s*", "", text)
    return text


def main(src_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(Path(src_dir).glob("*.pdf")):
        title, body = recover(pdf)
        if pdf.name.startswith("tr_omer_seyfettin"):
            body = clean_wiki_residue(body)
            # v1 printed the first story title twice (section title + wiki heading)
            body = body.replace("Kaşağı\n\nKaşağı\n", "Kaşağı\n\n", 1)
        (out / f"{pdf.stem}.txt").write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        print(f"{pdf.name}: {len(body):,} chars, {body.count(chr(10) * 2) + 1} paragraphs")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
