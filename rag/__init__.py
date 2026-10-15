"""Kulliyat AI — reusable RAG core (loading, chunking, embeddings, retrieval, prompts).

``app.py`` (Streamlit UI), ``ingest.py`` (index builder) and ``evals/`` all import
from here, so the evaluated pipeline is the same code that serves users.
"""
