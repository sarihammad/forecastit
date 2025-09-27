.PHONY: help install lint format typecheck test cov train serve-api serve-app docker-build up down download-data generate-synth clean

# Default target
help: ## Show this help message
	@echo "ForecastIt - Intelligent Demand & Sales Forecasting System"
	@echo "========================================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install dependencies with Poetry
	poetry install

install-dev: ## Install with development dependencies
	poetry install --with dev

install-dl: ## Install with deep learning extras
	poetry install --extras dl

# Code quality
lint: ## Run linting with ruff
	poetry run ruff check forecastit/ tests/

format: ## Format code with black and isort
	poetry run black forecastit/ tests/
	poetry run isort forecastit/ tests/

typecheck: ## Run type checking with mypy
	poetry run mypy --strict forecastit/

# Testing
test: ## Run tests with pytest
	poetry run pytest --maxfail=1 -q

test-fast: ## Run fast tests only (skip slow tests)
	poetry run pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests
	poetry run pytest tests/ -v -m integration

cov: ## Run tests with coverage
	poetry run pytest --cov=forecastit --cov-report=term-missing

# Data and synthetic generation
generate-synth: ## Generate synthetic dataset
	poetry run python -c "from forecastit.data.make_synthetic import generate_synthetic_data; generate_synthetic_data()"

download-data: ## Download external datasets (optional)
	@echo "External data download not implemented - using synthetic data instead"
	@echo "Run 'make generate-synth' to create synthetic dataset"

# Training and modeling
train: ## Run training pipeline with Prefect
	poetry run python -c "from forecastit.flows.training_flow import run_training_flow; run_training_flow()"

train-local: ## Run training without Prefect (direct execution)
	poetry run python scripts/train_local.py

# API and App serving
serve-api: ## Start FastAPI server
	poetry run uvicorn forecastit.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-app: ## Start Streamlit app
	poetry run streamlit run forecastit/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# MLflow UI
mlflow-ui: ## Start MLflow tracking UI
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Docker
docker-build: ## Build Docker images
	docker-compose build

up: ## Start all services with docker-compose
	docker-compose up --build

down: ## Stop all services
	docker-compose down

up-detached: ## Start services in background
	docker-compose up --build -d

logs: ## View logs from docker-compose
	docker-compose logs -f

# Development utilities
clean: ## Clean up generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

pre-commit-install: ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	poetry run pre-commit run --all-files

# Full pipeline
pipeline: generate-synth train ## Run full pipeline: generate data and train models

validate: lint typecheck test cov ## Run all validation checks

# Quick start
quickstart: install generate-synth train serve-api ## Quick start: install, generate data, train, and serve API

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Monitoring
monitor: ## Start monitoring dashboard (placeholder)
	@echo "Monitoring dashboard not yet implemented"

# Production deployment
deploy-staging: ## Deploy to staging environment (placeholder)
	@echo "Staging deployment not yet implemented"

deploy-prod: ## Deploy to production environment (placeholder)
	@echo "Production deployment not yet implemented"
