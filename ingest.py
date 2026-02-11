import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # CHANGED

# 1. SETUP
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env if present

# Ensure OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env or your system environment.")

DATA_PATH = "my_pdfs"
DB_PATH = "vector_db"

def create_vector_db():
    # Clear old database if exists
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Cleared old database at {DB_PATH}")

    print(f"Loading PDFs from '{DATA_PATH}'...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found!")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. CHUNK
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} text chunks.")

    # 3. EMBED & STORE (FAISS VERSION)
    print("Generating embeddings... (This costs a tiny amount of OpenAI credit)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create the FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    
    # Save it locally
    db.save_local(DB_PATH)
    
    print(f"Success! Database saved to '{DB_PATH}'.")

if __name__ == "__main__":
    create_vector_db()