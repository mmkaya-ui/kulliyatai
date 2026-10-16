"""Layout-aware PDF loading with PyMuPDF.

Per page, :func:`extract_pages` does four things that plain ``page.get_text()``
does not:

1. **Reading order for multi-column pages.** Text blocks are classified as
   left column, right column or full width. Between two full-width blocks
   (titles, wide paragraphs), the whole left column is emitted before the
   right column, instead of interleaving lines by height.
2. **Running header/footer removal.** Lines in the top or bottom margin band
   whose text (with digits masked) repeats on at least half of the pages, and
   bare page numbers there, are dropped. Without this every chunk of the
   bundled corpus would contain ``- N -``.
3. **De-hyphenation.** ``exam-\\nple`` becomes ``example`` when the next line
   starts with a lower-case letter (``Vice-\\nPresident`` is kept).
4. **OCR fallback.** Pages without a text layer but with images are OCR'd
   with Tesseract through PyMuPDF (``ocr="auto"``), if Tesseract and the
   requested language data are installed. Otherwise the page is reported as
   ``needs_ocr`` instead of silently disappearing.

Each page Document carries ``metadata["extraction"]`` (``text`` / ``ocr`` /
``needs_ocr``) and ``metadata["columns"]`` so ingestion can report what it did.
Page numbers are 1-based, as printed in a PDF viewer.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

MARGIN_BAND = 0.08  # top/bottom 8 % of the page height count as header/footer zone
MIN_PAGES_FOR_REPEAT = 3
_PAGE_NUMBER_RE = re.compile(r"^[\W_]*(?:page|sayfa|seite)?\s*\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?[\W_]*$", re.I)
_DIGITS_RE = re.compile(r"\d+")
_HYPHEN_RE = re.compile(r"(\w)-\n(?=[a-zäöüßçğıöşâîû])")


@dataclass
class PageText:
    page: int  # 1-based
    text: str
    extraction: str  # "text" | "ocr" | "needs_ocr" | "empty"
    columns: int = 1
    removed_lines: list[str] = field(default_factory=list)


@dataclass
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[str]


def _blocks(page: fitz.Page, textpage=None) -> list[_Block]:
    out = []
    for b in page.get_text("dict", textpage=textpage)["blocks"]:
        if b.get("type", 0) != 0:
            continue
        lines = []
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"])
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            if text:
                lines.append(text)
        if lines:
            x0, y0, x1, y1 = b["bbox"]
            out.append(_Block(x0, y0, x1, y1, lines))
    return out


def _reading_order(blocks: list[_Block], page_width: float) -> tuple[list[_Block], int]:
    """Order blocks for 1- or 2-column layouts; return (ordered blocks, n_columns)."""
    if not blocks:
        return [], 1
    mid = page_width / 2
    tol = page_width * 0.02
    left = [b for b in blocks if b.x1 <= mid + tol]
    right = [b for b in blocks if b.x0 >= mid - tol]
    two_columns = len(left) >= 2 and len(right) >= 2
    if not two_columns:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0)), 1

    def kind(b: _Block) -> str:
        if b.x1 <= mid + tol:
            return "L"
        if b.x0 >= mid - tol:
            return "R"
        return "F"  # spans the gutter

    ordered: list[_Block] = []
    segment: list[_Block] = []

    def flush() -> None:
        ordered.extend(sorted((b for b in segment if kind(b) == "L"), key=lambda b: b.y0))
        ordered.extend(sorted((b for b in segment if kind(b) == "R"), key=lambda b: b.y0))
        segment.clear()

    for b in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        if kind(b) == "F":
            flush()
            ordered.append(b)
        else:
            segment.append(b)
    flush()
    return ordered, 2


def _mask(line: str) -> str:
    return _DIGITS_RE.sub("#", line.strip().lower())


def _margin_lines(blocks: list[_Block], height: float) -> list[str]:
    top, bottom = height * MARGIN_BAND, height * (1 - MARGIN_BAND)
    return [ln for b in blocks if b.y1 <= top or b.y0 >= bottom for ln in b.lines]


def dehyphenate(text: str) -> str:
    return _HYPHEN_RE.sub(r"\1", text)


def _ocr_textpage(page: fitz.Page, language: str):
    """Return an OCR TextPage, or None if Tesseract/language data is missing."""
    try:
        return page.get_textpage_ocr(language=language, dpi=300, full=True)
    except Exception:  # RuntimeError when tessdata or the language is not installed
        return None


def ocr_available(language: str = "eng") -> bool:
    """True when PyMuPDF can find Tesseract data for every ``+``-joined language."""
    try:
        tessdata = fitz.get_tessdata()
    except Exception:
        return False
    return all((Path(tessdata) / f"{lang}.traineddata").exists() for lang in language.split("+"))


def extract_pages(
    path: str | Path,
    *,
    ocr: str = "auto",
    ocr_language: str | None = None,
    remove_headers: bool = True,
) -> list[PageText]:
    """Extract every page of one PDF. ``ocr`` is ``"auto"`` (image-only pages) or ``"off"``."""
    ocr_language = ocr_language or os.environ.get("OCR_LANGUAGES", "tur+eng+deu")
    with fitz.open(path) as doc:
        raw: list[tuple[list[_Block], float, float, str]] = []
        for page in doc:
            blocks = _blocks(page)
            mode = "text"
            if not blocks and page.get_images():
                tp = _ocr_textpage(page, ocr_language) if ocr == "auto" else None
                if tp is not None:
                    blocks, mode = _blocks(page, textpage=tp), "ocr"
                else:
                    mode = "needs_ocr"
            raw.append((blocks, page.rect.width, page.rect.height, mode))

    repeated: set[str] = set()
    if remove_headers and len(raw) >= MIN_PAGES_FOR_REPEAT:
        counts = Counter(m for b, _, h, _ in raw for m in {_mask(ln) for ln in _margin_lines(b, h)})
        threshold = max(MIN_PAGES_FOR_REPEAT, len(raw) // 2)
        repeated = {m for m, c in counts.items() if c >= threshold}

    pages: list[PageText] = []
    for i, (blocks, width, height, mode) in enumerate(raw, start=1):
        removed: list[str] = []
        if remove_headers:
            margin = set(_margin_lines(blocks, height))
            kept_blocks = []
            for b in blocks:
                lines = [
                    ln for ln in b.lines if not (ln in margin and (_mask(ln) in repeated or _PAGE_NUMBER_RE.match(ln)))
                ]
                removed += [ln for ln in b.lines if ln not in lines]
                if lines:
                    kept_blocks.append(_Block(b.x0, b.y0, b.x1, b.y1, lines))
            blocks = kept_blocks
        ordered, columns = _reading_order(blocks, width)
        text = dehyphenate("\n".join("\n".join(b.lines) for b in ordered))
        if not text.strip() and mode == "text":
            mode = "empty"
        pages.append(PageText(i, text, mode, columns, removed))
    return pages


def load_pdfs_with_pymupdf(data_path: str | Path, log=print, **extract_kwargs) -> list[Document]:
    """Load every PDF in ``data_path``: one Document per page that has text.

    ``metadata``: ``source`` (file name), ``page`` (1-based), ``total_pages``,
    ``extraction`` and ``columns``. Pages that need OCR but could not get it are
    logged, not returned.
    """
    data_path = Path(data_path)
    documents: list[Document] = []
    pdf_files = sorted(p for p in data_path.iterdir() if p.suffix.lower() == ".pdf")
    if not pdf_files:
        log("No PDF files found!")
        return documents

    log(f"Found {len(pdf_files)} PDF files. Loading with PyMuPDF (layout-aware)...")
    for filepath in pdf_files:
        try:
            pages = extract_pages(filepath, **extract_kwargs)
        except Exception as e:  # keep going: one broken PDF must not stop ingestion
            log(f"  FAIL: {filepath.name}: {e}")
            continue
        stats = Counter(p.extraction for p in pages)
        for p in pages:
            if p.text.strip():
                documents.append(
                    Document(
                        page_content=p.text,
                        metadata={
                            "source": filepath.name,
                            "page": p.page,
                            "total_pages": len(pages),
                            "extraction": p.extraction,
                            "columns": p.columns,
                        },
                    )
                )
        detail = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        removed = sum(len(p.removed_lines) for p in pages)
        log(f"  OK: {filepath.name} ({len(pages)} pages: {detail}; header/footer lines removed: {removed})")
        if stats.get("needs_ocr"):
            log(
                f"  WARN: {stats['needs_ocr']} image-only page(s) in {filepath.name} were skipped: "
                "install Tesseract + language data (OCR_LANGUAGES) to index them."
            )
    return documents
