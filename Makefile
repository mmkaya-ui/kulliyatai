.PHONY: help install ingest run eval test lint clean docker-build docker-run

PYTHON ?= python
STREAMLIT ?= streamlit
PYTEST ?= pytest
RUFF ?= ruff

help:
	@echo "Kulliyat AI — Available Commands:"
	@echo "  make install      Install Python dependencies from requirements.txt"
	@echo "  make ingest       Ingest PDFs into vector_db (reads DATA_DIR or sample_docs/)"
	@echo "  make run          Launch the Streamlit web app"
	@echo "  make eval         Run retrieval & answer evaluation benchmarks"
	@echo "  make test         Run pytest unit tests"
	@echo "  make lint         Run ruff linter and formatter checks"
	@echo "  make docker-build Build Docker container image"
	@echo "  make docker-run   Run Docker container on port 8501"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) ingest.py

run:
	$(STREAMLIT) run app.py

eval:
	$(PYTHON) -m evals.run_evals

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check .
	$(RUFF) format --check .

clean:
	rm -rf __pycache__ rag/__pycache__ tests/__pycache__ evals/__pycache__
	rm -rf .pytest_cache .ruff_cache

docker-build:
	docker build -t kulliyatai:latest .

docker-run:
	docker run -p 8501:8501 --env-file .env kulliyatai:latest
