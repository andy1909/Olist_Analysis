# =============================================================================
# Olist E-Commerce Analysis — Makefile
# =============================================================================
PYTHON   = .venv/bin/python
PIP      = .venv/bin/pip
TEST_DIR = tests

.PHONY: help venv install run forecast nlp diagnostics \
        test test-unit test-integration clean lint

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────

venv: ## Create virtual environment (.venv)
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: ## Install/upgrade Python dependencies into existing .venv
	$(PIP) install -r requirements.txt

# ── Pipeline ──────────────────────────────────────────────────────────────────

run: ## Run the full ETL & analytics pipeline (always run first)
	$(PYTHON) main.py

forecast: ## Run demand forecasting models — requires main.py to have run
	$(PYTHON) scripts/run_forecasting.py

nlp: ## Run NLP sentiment analysis on reviews — requires main.py to have run
	$(PYTHON) scripts/run_nlp.py

diagnostics: ## Run time-series diagnostic checks — requires main.py to have run
	$(PYTHON) scripts/run_diagnostics.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Run all 150 unit + integration tests
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -v

test-unit: ## Run unit tests only (fast, no file I/O to real data)
	$(PYTHON) -m unittest discover -s $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only (end-to-end pipeline checks)
	$(PYTHON) -m unittest discover -s $(TEST_DIR)/integration -v

# ── Code Quality ──────────────────────────────────────────────────────────────

lint: ## Syntax-check all source files (py_compile)
	$(PYTHON) -m py_compile main.py
	$(PYTHON) -m py_compile src/data/ingestion.py
	$(PYTHON) -m py_compile src/data/processing.py
	$(PYTHON) -m py_compile src/models/analytics.py
	$(PYTHON) -m py_compile src/models/nlp.py
	$(PYTHON) -m py_compile src/utils/visualization.py
	$(PYTHON) -m py_compile src/utils/logger.py
	$(PYTHON) -m py_compile src/utils/exception.py
	$(PYTHON) -m py_compile src/utils/utils.py
	@echo "✅ Syntax OK — all source files compiled without errors."

# ── Clean ─────────────────────────────────────────────────────────────────────

clean: ## Remove all generated outputs (processed data, figures, logs, models)
	rm -rf data/processed/*.csv data/processed/*.xlsx
	rm -rf reports/figures/*.png
	rm -rf logs/*.log
	rm -rf models/*.model
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaned all generated outputs."
