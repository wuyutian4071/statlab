.PHONY: help install lint format-check format typecheck test cov check reproduce clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Sync all dependencies (incl. dev group) into a local venv
	uv sync --all-groups

lint:  ## Run ruff lint checks
	uv run ruff check src tests

format-check:  ## Check formatting without modifying files (what CI actually runs)
	uv run ruff format --check src tests

format:  ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run the test suite
	uv run pytest

cov:  ## Run tests with coverage report
	uv run pytest --cov=statlab --cov-report=term-missing

check: lint format-check typecheck test  ## Run the full CI gate locally (must match ci.yml exactly)

reproduce:  ## Regenerate all synthetic data and results from scratch (deterministic)
	uv run statlab gen-synth --out data/synthetic/panel.parquet --seed 7
	uv run statlab ingest --source synthetic --out data/bars --n 1000 --seed 7
	@echo "M2: synthetic panel + partitioned bar dataset regenerated. Later milestones extend this."

clean:  ## Remove caches and generated artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml
	rm -rf data results reports/*.html mlruns
	find . -type d -name __pycache__ -exec rm -rf {} +
