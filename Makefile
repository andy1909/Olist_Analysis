# =============================================================================
# Olist E-Commerce Analysis — Makefile
# =============================================================================
PYTHON = .venv/bin/python
PIP = .venv/bin/pip

.PHONY: help install run forecast nlp diagnostics test clean lint

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

run: ## Run the full ETL & analytics pipeline
	$(PYTHON) main.py

forecast: ## Run demand forecasting models (univariate + LSTM)
	$(PYTHON) scripts/run_forecasting.py

nlp: ## Run NLP sentiment analysis on reviews
	$(PYTHON) scripts/run_nlp.py

diagnostics: ## Run time series diagnostic checks
	$(PYTHON) scripts/run_diagnostics.py

test: ## Run automated unit tests
	$(PYTHON) -m pytest tests/ -v

clean: ## Remove generated outputs (processed data, figures, logs)
	rm -rf data/processed/*.csv data/processed/*.xlsx
	rm -rf reports/figures/*.png
	rm -rf logs/*.log
	rm -rf models/*.model
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint: ## Run basic code quality checks
	$(PYTHON) -m py_compile main.py
	$(PYTHON) -m py_compile src/data/ingestion.py
	$(PYTHON) -m py_compile src/data/processing.py
	$(PYTHON) -m py_compile src/models/analytics.py
	$(PYTHON) -m py_compile src/models/nlp.py
