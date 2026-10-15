# Kulliyat AI — Streamlit app with the sample corpus pre-indexed and both ONNX models baked in.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

# curl: health check; tesseract + tur/deu/eng data: OCR fallback for scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl tesseract-ocr tesseract-ocr-eng tesseract-ocr-tur tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Index the bundled corpus and download the reranker at build time (needs network
# during `docker build`), so the running container needs no downloads.
RUN python ingest.py --no-cache \
    && python -c "from rag.retrieval import Reranker; Reranker()"

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py"]
