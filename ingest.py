"""Build the hybrid index from a folder of PDFs.

    python ingest.py                         # .env / defaults (sample_docs -> vector_db)
    python ingest.py --data-dir my_pdfs --db-dir vector_db_private
    python ingest.py --chunk-size 1500 --chunk-overlap 225

Settings come from environment variables (see .env.example); CLI flags win.
Image-only (scanned) pages are OCR'd when Tesseract and the OCR_LANGUAGES
data are installed; otherwise they are listed as skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from rag.config import ROOT, Settings, load_dotenv_if_available
from rag.embeddings import CachedEmbedder, make_embedder
from rag.index import RAGIndex
from rag.loading import load_pdfs_with_pymupdf

EMBED_CACHE = ROOT / "evals" / ".cache" / "embeddings.sqlite"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="folder with PDFs (default: $DATA_DIR or sample_docs)")
    p.add_argument("--db-dir", help="output index folder (default: $DB_DIR or vector_db)")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--no-ocr", action="store_true", help="never OCR image-only pages")
    p.add_argument("--no-cache", action="store_true", help="do not use the on-disk embedding cache")
    return p.parse_args(argv)


def create_index(argv=None) -> int:
    load_dotenv_if_available()
    args = parse_args(argv)
    s = Settings.from_env()
    data_dir = (ROOT / args.data_dir) if args.data_dir else s.data_dir
    db_dir = (ROOT / args.db_dir) if args.db_dir else s.db_dir
    settings = s.index
    if args.chunk_size:
        settings = replace(settings, chunk_size=args.chunk_size)
    if args.chunk_overlap is not None:
        settings = replace(settings, chunk_overlap=args.chunk_overlap)

    embedder = make_embedder(s.embedding_provider, s.embedding_model)  # fails fast without a key
    if not args.no_cache:
        embedder = CachedEmbedder(embedder, EMBED_CACHE)

    print(f"\n[LOAD] {data_dir}")
    pages = load_pdfs_with_pymupdf(data_dir, ocr="off" if args.no_ocr else "auto")
    if not pages:
        print("[ERROR] No text found.")
        return 1
    print(f"[INFO] {len(pages)} pages with text.")

    print(f"\n[INDEX] {embedder.name}; {settings}")
    t0 = time.time()
    index = RAGIndex.build(pages, embedder, settings)
    index.save(db_dir)
    st = index.stats
    print(f"\n[SUCCESS] {db_dir} in {time.time() - t0:.1f}s")
    print(f"   {st['n_chunks']} chunks, {st['n_dense_units']} dense vectors ({index.vectors.shape[1]}-d)")
    if st["n_chunks_truncated_by_embedder"]:
        print(
            f"   WARNING: {st['n_chunks_truncated_by_embedder']} chunks exceed the embedder's input limit "
            "and were truncated (enable DENSE_WINDOWS to avoid this)"
        )
    if isinstance(embedder, CachedEmbedder):
        print(f"   embedding cache: {embedder.hits} hits, {embedder.misses} new")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Turkish characters on Windows consoles
    sys.exit(create_index())
