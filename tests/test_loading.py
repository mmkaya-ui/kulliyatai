"""Layout-aware PDF extraction on synthetic PDFs (built in-test with PyMuPDF)."""

from pathlib import Path

import fitz
import pytest

from rag.loading import dehyphenate, extract_pages, load_pdfs_with_pymupdf, ocr_available

A4 = fitz.paper_rect("a4")


def _save(doc: fitz.Document, tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    doc.save(path)
    return path


def test_two_column_reading_order(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_textbox(fitz.Rect(50, 50, 545, 90), "FULL WIDTH TITLE", fontsize=14)
    left = " ".join(f"leftword{i}" for i in range(20))
    right = " ".join(f"rightword{i}" for i in range(20))
    # Two side-by-side columns; each holds several paragraphs (blocks).
    for k in range(3):
        y = 110 + k * 120
        assert page.insert_textbox(fitz.Rect(50, y, 285, y + 110), f"L{k} {left}", fontsize=9) >= 0
        assert page.insert_textbox(fitz.Rect(310, y, 545, y + 110), f"R{k} {right}", fontsize=9) >= 0
    path = _save(doc, tmp_path, "two_col.pdf")

    (p,) = extract_pages(path)
    assert p.columns == 2
    order = [p.text.index(tag) for tag in ("FULL WIDTH TITLE", "L0 ", "L1 ", "L2 ", "R0 ", "R1 ", "R2 ")]
    assert order == sorted(order), "left column must be read completely before the right column"


def test_single_column_is_top_to_bottom(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    for k in range(4):
        page.insert_text((72, 100 + 40 * k), f"line {k}")
    (p,) = extract_pages(_save(doc, tmp_path, "one_col.pdf"))
    assert p.columns == 1
    assert [ln for ln in p.text.splitlines()] == ["line 0", "line 1", "line 2", "line 3"]


def test_running_header_and_page_numbers_removed(tmp_path):
    doc = fitz.open()
    for n in range(1, 6):
        page = doc.new_page(width=A4.width, height=A4.height)
        page.insert_text((72, 30), "Annual Report 2024 — Confidential")  # running header
        page.insert_text((72, 200), f"Body text of page {n} mentions the year 2024.")
        page.insert_text((290, A4.height - 25), f"- {n} -")  # footer page number
    pages = extract_pages(_save(doc, tmp_path, "headers.pdf"))
    for n, p in enumerate(pages, start=1):
        assert "Annual Report" not in p.text
        assert f"- {n} -" not in p.text
        assert f"Body text of page {n} mentions the year 2024." in p.text  # body lines survive
        assert len(p.removed_lines) == 2


def test_headers_kept_for_short_documents(tmp_path):
    doc = fitz.open()
    for _ in range(2):  # below MIN_PAGES_FOR_REPEAT: nothing is "repeated" yet
        page = doc.new_page()
        page.insert_text((72, 30), "Running title")
        page.insert_text((72, 200), "Body")
    pages = extract_pages(_save(doc, tmp_path, "short.pdf"))
    assert all("Running title" in p.text for p in pages)


def test_dehyphenate():
    assert dehyphenate("exam-\nple text") == "example text"
    assert dehyphenate("Vice-\nPresident") == "Vice-\nPresident"  # capitalised: real hyphen
    assert dehyphenate("büyük-\nlüğü") == "büyüklüğü"


def test_image_only_page_without_ocr_is_reported(tmp_path):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 50), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    doc = fitz.open()
    doc.new_page().insert_image(fitz.Rect(72, 72, 272, 122), pixmap=pix)
    (p,) = extract_pages(_save(doc, tmp_path, "scan.pdf"), ocr="off")
    assert p.extraction == "needs_ocr" and p.text == ""


@pytest.mark.skipif(not ocr_available("eng"), reason="Tesseract English data not installed")
def test_ocr_fallback_reads_scanned_page(tmp_path):
    # Render real text to an image, then build a PDF page that contains only that image.
    src = fitz.open()
    src.new_page(width=400, height=120).insert_text((20, 60), "Hybrid retrieval works offline", fontsize=20)
    pix = src[0].get_pixmap(dpi=200)
    scan = fitz.open()
    scan.new_page(width=400, height=120).insert_image(fitz.Rect(0, 0, 400, 120), pixmap=pix)
    path = _save(scan, tmp_path, "scan.pdf")

    (p,) = extract_pages(path, ocr="auto", ocr_language="eng")
    assert p.extraction == "ocr"
    assert "Hybrid retrieval works offline" in " ".join(p.text.split())


def test_load_pdfs_metadata_on_sample_corpus():
    root = Path(__file__).resolve().parent.parent
    docs = load_pdfs_with_pymupdf(root / "sample_docs", log=lambda *_: None)
    assert {d.metadata["source"] for d in docs} == {
        "tr_teskilati_esasiye_1924.pdf",
        "tr_omer_seyfettin_hikayeler.pdf",
        "en_us_constitution.pdf",
        "en_scandal_in_bohemia.pdf",
        "de_kafka_die_verwandlung.pdf",
    }
    assert all(d.metadata["page"] >= 1 and d.metadata["extraction"] == "text" for d in docs)
    # The "- N -" footers of the sample PDFs must not leak into page text.
    assert not any(
        line.strip().startswith("- ") and line.strip().endswith(" -")
        for d in docs
        for line in d.page_content.splitlines()
    )


@pytest.mark.skipif(not ocr_available("tur"), reason="Tesseract Turkish data (tur.traineddata) not installed")
def test_ocr_fallback_turkish(tmp_path):
    font = Path(__file__).resolve().parent.parent / "scripts" / "fonts" / "DejaVuSans.ttf"
    src = fitz.open()
    page = src.new_page(width=460, height=120)
    page.insert_text((20, 60), "Türkiye Devleti bir Cumhuriyettir", fontsize=20, fontname="dv", fontfile=str(font))
    pix = page.get_pixmap(dpi=200)
    scan = fitz.open()
    scan.new_page(width=460, height=120).insert_image(fitz.Rect(0, 0, 460, 120), pixmap=pix)
    (p,) = extract_pages(_save(scan, tmp_path, "scan_tr.pdf"), ocr="auto", ocr_language="tur")
    assert p.extraction == "ocr"
    assert "Türkiye Devleti bir Cumhuriyettir" in " ".join(p.text.split())
