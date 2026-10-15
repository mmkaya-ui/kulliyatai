# Dockerfile for Kulliyat AI
FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install system dependencies (fonts for Unicode rendering, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and sample data
COPY . .

# Ingest sample docs at build time so the container is ready immediately
RUN python ingest.py --data-dir sample_docs --db-dir vector_db

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
