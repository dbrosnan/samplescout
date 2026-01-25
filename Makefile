.PHONY: setup setup-dev test lint format check clean demo benchmark help

PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

# Default audio file for benchmarks
AUDIO ?= samples/input/demo.wav

help: ## Show this help message
	@echo "SampleScout - AI Audio Sampling System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo ""
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

setup-dev: ## Install with dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install
	@echo ""
	@echo "Dev setup complete! Activate with: source $(VENV)/bin/activate"

test: ## Run tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run linters
	$(FLAKE8) src/ tests/
	$(MYPY) src/

format: ## Format code with black and isort
	$(ISORT) src/ tests/
	$(BLACK) src/ tests/

check: format lint test ## Run all checks (format, lint, test)

demo: ## Run interactive demo
	$(VENV)/bin/python run_demo.py

benchmark: ## Run benchmarks (use AUDIO=path/to/file.wav)
	$(VENV)/bin/python -m samplescout benchmark $(AUDIO)

classify: ## Classify audio (use AUDIO=path/to/file.wav)
	$(VENV)/bin/python -m samplescout classify $(AUDIO)

separate: ## Separate audio (use AUDIO=path/to/file.wav)
	$(VENV)/bin/python -m samplescout separate $(AUDIO) --output samples/output/

clean: ## Remove build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Remove venv and all generated files
	rm -rf $(VENV)/
	rm -rf samples/output/*

docker-build: ## Build Docker image
	docker build -t samplescout:latest .

docker-run: ## Run in Docker container
	docker run -it --rm -v $(PWD)/samples:/app/samples samplescout:latest
