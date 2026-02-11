import os
import sys
import shutil

# Fix Windows console encoding for Turkish characters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import fitz  # PyMuPDF — superior PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. SETUP
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env or your system environment.")

DATA_PATH = "my_pdfs"
DB_PATH = "vector_db"

# --- UPGRADE 1: PyMuPDF Loader (better than PyPDF) ---
# Extracts text with superior layout preservation, handles columns, tables, Turkish chars
def load_pdfs_with_pymupdf(data_path):
    """Load PDFs using PyMuPDF for better text extraction quality."""
    documents = []
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found!")
        return documents
    
    print(f"Found {len(pdf_files)} PDF files. Loading with PyMuPDF...")
    
    for filename in sorted(pdf_files):
        filepath = os.path.join(data_path, filename)
        try:
            doc = fitz.open(filepath)
            num_pages = doc.page_count
            for page_num in range(num_pages):
                page = doc[page_num]
                # Extract text with layout preservation
                text = page.get_text("text")
                
                if text.strip():  # Skip empty pages
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "total_pages": num_pages
                        }
                    ))
            doc.close()
            print(f"  OK: {filename} ({num_pages} pages)")
        except Exception as e:
            print(f"  FAIL: {filename}: Error - {e}")
    
    return documents


# --- UPGRADE 2: Contextual Chunk Enrichment ---
# Prepends source metadata INTO each chunk before embedding
# This makes retrieval smarter — the AI knows WHERE info came from
def enrich_chunks_with_context(chunks):
    """Add source context directly into chunk text for better retrieval."""
    enriched = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "Bilinmeyen")
        page = chunk.metadata.get("page", "?")
        
        # Prepend context header to the actual text
        enriched_content = f"[Kaynak: {source} | Sayfa: {page}]\n{chunk.page_content}"
        
        enriched.append(Document(
            page_content=enriched_content,
            metadata=chunk.metadata  # Keep original metadata too
        ))
    
    return enriched


def create_vector_db():
    # Clear old database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"[CLEARED] Old database at {DB_PATH}")

    # --- Load PDFs with PyMuPDF ---
    print(f"\n[LOAD] Loading PDFs from '{DATA_PATH}'...")
    documents = load_pdfs_with_pymupdf(DATA_PATH)
    
    if not documents:
        print("[ERROR] No documents loaded!")
        return

    print(f"\n[INFO] Loaded {len(documents)} pages total.")

    # --- Smart Chunking (larger chunks, more overlap) ---
    print("\n[CHUNK] Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,       # Larger chunks = more context per retrieval
        chunk_overlap=300,     # More overlap = better continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Respect paragraph boundaries
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    # --- Contextual Enrichment ---
    print("\n[ENRICH] Enriching chunks with source context...")
    enriched_chunks = enrich_chunks_with_context(chunks)
    print(f"   Enriched {len(enriched_chunks)} chunks.")

    # --- UPGRADE 3: Best Embedding Model ---
    print("\n[EMBED] Generating embeddings with text-embedding-3-large (3072 dims)...")
    print("   (This is the highest quality embedding model available)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create FAISS index
    db = FAISS.from_documents(enriched_chunks, embeddings)
    
    # Save locally
    db.save_local(DB_PATH)
    
    print(f"\n[SUCCESS] Ultra-premium database saved to '{DB_PATH}'.")
    print(f"   {len(enriched_chunks)} enriched chunks")
    print(f"   3072-dim embeddings (text-embedding-3-large)")
    print(f"   Parsed with PyMuPDF")
    print(f"   Context-enriched for smarter retrieval")


if __name__ == "__main__":
    create_vector_db()