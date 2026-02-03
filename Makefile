.PHONY: install install-dev lint format type-check test test-cov clean run dashboard pre-commit security help

# Default target
.DEFAULT_GOAL := help

install:
	uv sync

install-dev:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run black --check src tests

format:
	uv run ruff check --fix src tests
	uv run black src tests

type-check:
	uv run mypy src

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

coverage: test-cov
	@echo "Coverage report generated in htmlcov/index.html"

pre-commit:
	uv run pre-commit run --all-files

security:
	uv run bandit -r src -ll

# Cross-platform clean (works on Windows and Unix)
clean:
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ rd /s /q __pycache__
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.ruff_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (htmlcov) do @if exist "%%d" rd /s /q "%%d"
	@del /s /q *.pyc 2>nul || echo.
	@del /s /q *.pyo 2>nul || echo.
	@del /s /q *.pyd 2>nul || echo.
	@del /s /q .coverage 2>nul || echo.
else
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
endif

check: format lint type-check test

run:
	uv run python -m src.main

run-compare:
	uv run python -m src.main --mode compare

run-comprehensive:
	uv run python -m src.main --mode comprehensive

dashboard:
	uv run streamlit run src/streamlit_dashboard.py

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install all dependencies (including dev)"
	@echo ""
	@echo "  lint             Run linters (ruff, black --check)"
	@echo "  format           Format code (ruff --fix, black)"
	@echo "  type-check       Run mypy type checker"
	@echo "  pre-commit       Run all pre-commit hooks"
	@echo "  security         Run bandit security scanner"
	@echo ""
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  coverage         Run tests with coverage (alias for test-cov)"
	@echo ""
	@echo "  run              Run single forecast (default mode)"
	@echo "  run-compare      Run model comparison"
	@echo "  run-comprehensive  Run comprehensive analysis"
	@echo "  dashboard        Start Streamlit dashboard"
	@echo ""
	@echo "  clean            Remove cache files and build artifacts"
	@echo "  check            Run format, lint, type-check, and test"
	@echo ""
