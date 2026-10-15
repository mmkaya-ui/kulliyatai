.PHONY: help install install-dev ingest run test test-models lint format verify-data corpus eval eval-quick report clean docker-build docker-run

PYTHON ?= python

help:
	@echo "Kulliyat AI — commands:"
	@echo "  make install       runtime dependencies (requirements.txt)"
	@echo "  make install-dev   runtime + test/lint tools, exact versions from requirements.lock"
	@echo "  make ingest        build the index from sample_docs/ (or DATA_DIR) into vector_db/ (or DB_DIR)"
	@echo "  make run           start the Streamlit app"
	@echo "  make test          unit tests (no models, no network)"
	@echo "  make test-models   + end-to-end tests that load the local FastEmbed models"
	@echo "  make lint          ruff check + format check"
	@echo "  make verify-data   verify evals/questions.jsonl against the PDFs"
	@echo "  make corpus        verify sample_docs/*.pdf rebuild byte-identically"
	@echo "  make eval          full benchmark (~20 min on 2 cores with warm caches; longer on first run)"
	@echo "  make eval-quick    benchmark without the cross-encoder (minutes; writes results_no_rerank.json)"
	@echo "  make report        re-render RESULTS.md and the README block from results.json"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.lock

ingest:
	$(PYTHON) ingest.py

run:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest

test-models:
	RUN_MODEL_TESTS=1 $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

verify-data:
	$(PYTHON) -m evals.dataset

corpus:
	$(PYTHON) scripts/build_sample_docs.py --check

eval:
	$(PYTHON) -m evals.run

eval-quick:
	$(PYTHON) -m evals.run --no-rerank --no-latency

report:
	$(PYTHON) -m evals.report

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache

docker-build:
	docker build -t kulliyatai:latest .

docker-run:
	docker run -p 8501:8501 --env-file .env kulliyatai:latest
