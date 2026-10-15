"""PDF loading with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF — superior PDF text extraction
from langchain_core.documents import Document


# --- UPGRADE 1: PyMuPDF Loader (better than PyPDF) ---
# Extracts text with superior layout preservation, handles columns, tables, Turkish chars
def load_pdfs_with_pymupdf(data_path: str | Path, log=print) -> list[Document]:
    """Load PDFs using PyMuPDF for better text extraction quality.

    Returns one Document per non-empty page. ``metadata["page"]`` is the
    **1-based** page number as printed in a PDF viewer (PyMuPDF indexes from 0;
    the original code stored the 0-based index and displayed it as the page,
    which made every citation one page too low).
    """
    data_path = Path(data_path)
    documents: list[Document] = []
    pdf_files = sorted(p for p in data_path.iterdir() if p.suffix.lower() == ".pdf")

    if not pdf_files:
        log("No PDF files found!")
        return documents

    log(f"Found {len(pdf_files)} PDF files. Loading with PyMuPDF...")

    for filepath in pdf_files:
        try:
            with fitz.open(filepath) as doc:
                num_pages = doc.page_count
                for page_index in range(num_pages):
                    # Extract text with layout preservation
                    text = doc[page_index].get_text("text")

                    if text.strip():  # Skip empty pages
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filepath.name,
                                    "page": page_index + 1,
                                    "total_pages": num_pages,
                                },
                            )
                        )
            log(f"  OK: {filepath.name} ({num_pages} pages)")
        except Exception as e:  # keep going: one broken PDF must not stop ingestion
            log(f"  FAIL: {filepath.name}: Error - {e}")

    return documents
