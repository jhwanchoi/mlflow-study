.PHONY: help install setup start stop logs clean train evaluate test lint format

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)MLflow Vision Training System$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies using Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	poetry install
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

setup: ## Setup environment (create .env from template)
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Creating .env file...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN).env file created! Please update values if needed.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

##@ Infrastructure

start: ## Start MLflow infrastructure (Docker Compose)
	@echo "$(BLUE)Starting MLflow infrastructure...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Infrastructure started!$(NC)"
	@echo ""
	@echo "$(BLUE)Services:$(NC)"
	@echo "  - MLflow UI: http://localhost:5000"
	@echo "  - MinIO Console: http://localhost:9001 (minio/minio123)"
	@echo ""

stop: ## Stop MLflow infrastructure
	@echo "$(BLUE)Stopping MLflow infrastructure...$(NC)"
	docker-compose down
	@echo "$(GREEN)Infrastructure stopped!$(NC)"

restart: stop start ## Restart MLflow infrastructure

logs: ## Show logs from MLflow infrastructure
	docker-compose logs -f

status: ## Check infrastructure status
	@echo "$(BLUE)Infrastructure Status:$(NC)"
	@docker-compose ps

clean-infra: ## Clean infrastructure (remove volumes)
	@echo "$(RED)Warning: This will remove all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "$(GREEN)Infrastructure cleaned!$(NC)"; \
	fi

##@ Terraform

tf-init: ## Initialize Terraform
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	cd terraform/local && terraform init
	@echo "$(GREEN)Terraform initialized!$(NC)"

tf-plan: ## Plan Terraform infrastructure
	@echo "$(BLUE)Planning Terraform infrastructure...$(NC)"
	cd terraform/local && terraform plan

tf-apply: ## Apply Terraform infrastructure
	@echo "$(BLUE)Applying Terraform infrastructure...$(NC)"
	cd terraform/local && terraform apply
	@echo "$(GREEN)Infrastructure deployed!$(NC)"

tf-destroy: ## Destroy Terraform infrastructure
	@echo "$(RED)Warning: This will destroy all infrastructure!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd terraform/local && terraform destroy; \
		echo "$(GREEN)Infrastructure destroyed!$(NC)"; \
	fi

##@ Training

train: ## Train model locally
	@echo "$(BLUE)Starting training...$(NC)"
	python -m src.training.train

train-docker: ## Train model in Docker container
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t mlflow-vision-training:latest .
	@echo "$(BLUE)Starting training in container...$(NC)"
	docker run --rm \
		--network mlflow-study_mlflow-network \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		--env-file .env \
		mlflow-vision-training:latest

evaluate: ## Evaluate model (requires RUN_ID environment variable)
	@if [ -z "$(RUN_ID)" ]; then \
		echo "$(RED)Error: RUN_ID not set$(NC)"; \
		echo "Usage: make evaluate RUN_ID=<mlflow_run_id>"; \
		exit 1; \
	fi
	@echo "$(BLUE)Evaluating model from run: $(RUN_ID)$(NC)"
	python -m src.training.evaluate $(RUN_ID)

##@ Development

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run black src/ tests/
	poetry run isort src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

lint: ## Lint code with flake8 and mypy
	@echo "$(BLUE)Linting code...$(NC)"
	poetry run flake8 src/ tests/
	poetry run mypy src/
	@echo "$(GREEN)Linting complete!$(NC)"

test: ## Run all tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ -v --cov=src --cov-report=term --cov-report=html
	@echo "$(GREEN)Tests complete! Open htmlcov/index.html for coverage report$(NC)"

test-fast: ## Run fast tests only (skip slow tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	poetry run pytest tests/ -v -m "not slow" --cov=src --cov-report=term
	@echo "$(GREEN)Fast tests complete!$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	poetry run pytest-watch tests/ -- -v

test-coverage: ## Generate detailed coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated!$(NC)"
	@echo "Open htmlcov/index.html to view"

jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	poetry run jupyter notebook notebooks/

##@ Cleanup

clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/
	@echo "$(GREEN)Cleaned temporary files!$(NC)"

clean-data: ## Clean downloaded datasets
	@echo "$(RED)Warning: This will remove all downloaded data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/; \
		echo "$(GREEN)Data cleaned!$(NC)"; \
	fi

clean-checkpoints: ## Clean model checkpoints
	@echo "$(RED)Warning: This will remove all checkpoints!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/; \
		echo "$(GREEN)Checkpoints cleaned!$(NC)"; \
	fi

clean-all: clean clean-data clean-checkpoints clean-infra ## Clean everything

##@ Utilities

shell: ## Open Python shell with environment loaded
	@echo "$(BLUE)Opening Python shell...$(NC)"
	poetry run python

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in Docker container...$(NC)"
	docker run --rm -it \
		--network mlflow-study_mlflow-network \
		-v $(PWD):/app \
		--env-file .env \
		mlflow-vision-training:latest bash

mlflow-ui: ## Open MLflow UI in browser
	@echo "$(BLUE)Opening MLflow UI...$(NC)"
	open http://localhost:5000 || xdg-open http://localhost:5000 || echo "Please open http://localhost:5000 in your browser"

minio-ui: ## Open MinIO console in browser
	@echo "$(BLUE)Opening MinIO console...$(NC)"
	open http://localhost:9001 || xdg-open http://localhost:9001 || echo "Please open http://localhost:9001 in your browser"
