"""Build the vector index from a folder of PDFs.

    python ingest.py                         # uses .env / defaults (sample_docs -> vector_db)
    python ingest.py --data-dir my_pdfs      # your own private PDFs
    python ingest.py --chunk-size 1000 --chunk-overlap 200

Settings come from environment variables (see .env.example); CLI flags win.
"""

import argparse
import sys
import time

from rag.chunking import enrich_chunks_with_context, split_documents
from rag.config import ROOT, Settings, load_dotenv_if_available
from rag.embeddings import CachedEmbeddings, make_embeddings
from rag.index import build_faiss, save_index_atomically
from rag.loading import load_pdfs_with_pymupdf

EMBED_CACHE = ROOT / "evals" / ".cache" / "embeddings.sqlite"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="folder with PDFs (default: $DATA_DIR or sample_docs)")
    p.add_argument("--db-dir", help="output index folder (default: $DB_DIR or vector_db)")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--no-cache", action="store_true", help="do not use the on-disk embedding cache")
    return p.parse_args(argv)


def create_vector_db(argv=None):
    # 1. SETUP
    load_dotenv_if_available()
    args = parse_args(argv)
    s = Settings.from_env()
    data_dir = ROOT / args.data_dir if args.data_dir else s.data_dir
    db_dir = ROOT / args.db_dir if args.db_dir else s.db_dir
    chunk_size = args.chunk_size or s.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else s.chunk_overlap

    embeddings = make_embeddings(s.embedding_provider, s.embedding_model)  # fails fast without a key
    if not args.no_cache:
        embeddings = CachedEmbeddings(embeddings, EMBED_CACHE, f"{s.embedding_provider}/{s.embedding_model}")

    # --- Load PDFs with PyMuPDF ---
    print(f"\n[LOAD] Loading PDFs from '{data_dir}'...")
    documents = load_pdfs_with_pymupdf(data_dir)

    if not documents:
        print("[ERROR] No documents loaded!")
        return 1

    print(f"\n[INFO] Loaded {len(documents)} pages total.")

    print(f"\n[CHUNK] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap} chars)...")
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"   Created {len(chunks)} chunks.")

    # --- Contextual Enrichment ---
    print("\n[ENRICH] Enriching chunks with source context...")
    enriched_chunks = enrich_chunks_with_context(chunks)
    print(f"   Enriched {len(enriched_chunks)} chunks.")

    # --- UPGRADE 3: Embedding model (configurable, recorded in meta.json) ---
    print(f"\n[EMBED] Generating embeddings with {s.embedding_provider}/{s.embedding_model}...")
    t0 = time.time()
    db = build_faiss(enriched_chunks, embeddings)
    dim = db.index.d

    # Build into a temp folder and swap, so a failure never deletes the old index
    meta = {
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.embedding_model,
        "dim": dim,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_chunks": len(enriched_chunks),
        "n_pages": len(documents),
        "data_dir": str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir),
    }
    save_index_atomically(db, db_dir, meta)

    print(f"\n[SUCCESS] Database saved to '{db_dir}' in {time.time() - t0:.1f}s.")
    print(f"   {len(enriched_chunks)} enriched chunks")
    print(f"   {dim}-dim embeddings ({s.embedding_provider}/{s.embedding_model})")
    print("   Parsed with PyMuPDF")
    print("   Context-enriched for smarter retrieval")
    if isinstance(embeddings, CachedEmbeddings):
        print(f"   Embedding cache: {embeddings.hits} hits, {embeddings.misses} new")
    return 0


if __name__ == "__main__":
    # Fix Windows console encoding for Turkish characters
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(create_vector_db())
