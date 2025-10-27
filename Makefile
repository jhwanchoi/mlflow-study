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
	poetry run python -m src.training.train

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
	poetry run python -m src.training.evaluate $(RUN_ID)

##@ Hyperparameter Tuning (Ray Tune)

tune: ## Run hyperparameter tuning with Ray Tune (default: 10 trials)
	@echo "$(BLUE)Starting Ray Tune hyperparameter optimization...$(NC)"
	@TRIALS=$${TRIALS:-10}; \
	CONCURRENT=$${CONCURRENT:-2}; \
	echo "Running $$TRIALS trials with max $$CONCURRENT concurrent trials"; \
	poetry run python -m src.tuning.ray_tune

tune-quick: ## Quick tuning test (5 trials, 2 concurrent)
	@echo "$(BLUE)Starting quick Ray Tune test...$(NC)"
	TRIALS=5 CONCURRENT=2 poetry run python -c "\
from src.tuning import tune_model; \
tune_model(num_samples=5, max_concurrent_trials=2)"

tune-extensive: ## Extensive tuning (50 trials, 4 concurrent)
	@echo "$(BLUE)Starting extensive Ray Tune optimization...$(NC)"
	TRIALS=50 CONCURRENT=4 poetry run python -c "\
from src.tuning import tune_model; \
tune_model(num_samples=50, max_concurrent_trials=4)"

tune-results: ## Show Ray Tune results summary
	@echo "$(BLUE)Ray Tune Results:$(NC)"
	@if [ -d ray_results ]; then \
		echo ""; \
		echo "$(YELLOW)Recent tuning experiments:$(NC)"; \
		ls -lt ray_results/ | head -10; \
		echo ""; \
		echo "$(YELLOW)To view detailed results, check:$(NC)"; \
		echo "  - MLflow UI: http://localhost:5001"; \
		echo "  - Ray results directory: ./ray_results/"; \
	else \
		echo "$(YELLOW)No tuning results found. Run 'make tune' first.$(NC)"; \
	fi

tune-clean: ## Clean Ray Tune results
	@echo "$(RED)Warning: This will remove all Ray Tune results!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ray_results/; \
		echo "$(GREEN)Ray Tune results cleaned!$(NC)"; \
	fi

tune-test: ## Run Ray Tune integration test (2 trials, 3 epochs)
	@echo "$(BLUE)Running Ray Tune integration test...$(NC)"
	poetry run python tests/test_ray_tune.py

test-mlflow-artifacts: ## Test MLflow artifact storage (models, files)
	@echo "$(BLUE)Testing MLflow artifact storage...$(NC)"
	poetry run python tests/test_mlflow_artifacts.py

##@ Development

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	poetry run pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	poetry run pre-commit run --all-files

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

test-docker: ## Run tests in Docker container (isolated environment)
	@echo "$(BLUE)Building test Docker image...$(NC)"
	docker build -t mlflow-vision-training:test --target development .
	@echo "$(BLUE)Running tests in container...$(NC)"
	docker run --rm \
		-v $(PWD)/htmlcov:/app/htmlcov \
		mlflow-vision-training:test \
		poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests complete! Coverage report: htmlcov/index.html$(NC)"

test-docker-fast: ## Run fast tests in Docker container
	@echo "$(BLUE)Building test Docker image...$(NC)"
	docker build -t mlflow-vision-training:test --target development .
	@echo "$(BLUE)Running fast tests in container...$(NC)"
	docker run --rm \
		mlflow-vision-training:test \
		poetry run pytest tests/ -v -m "not slow" --cov=src --cov-report=term
	@echo "$(GREEN)Fast tests complete!$(NC)"

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

clean-ray: ## Clean Ray Tune results
	@echo "$(BLUE)Cleaning Ray Tune results...$(NC)"
	rm -rf ray_results/
	@echo "$(GREEN)Ray Tune results cleaned!$(NC)"

clean-all: clean clean-data clean-checkpoints clean-ray clean-infra ## Clean everything

##@ Remote (EKS)

remote-mlflow-ui: ## Open remote MLflow UI in browser
	@echo "$(BLUE)Opening remote MLflow UI...$(NC)"
	@MLFLOW_URI=$$(grep MLFLOW_TRACKING_URI .env | cut -d'=' -f2); \
	if [ -z "$$MLFLOW_URI" ] || [ "$$MLFLOW_URI" = "http://localhost:5001" ]; then \
		echo "$(YELLOW)Remote MLflow URI not configured in .env$(NC)"; \
		echo "Set MLFLOW_TRACKING_URI=https://mlflow.company.com"; \
	else \
		echo "Opening: $$MLFLOW_URI"; \
		open $$MLFLOW_URI || xdg-open $$MLFLOW_URI || echo "Please open $$MLFLOW_URI in your browser"; \
	fi

test-remote-connection: ## Test connection to remote MLflow server
	@echo "$(BLUE)Testing remote MLflow connection...$(NC)"
	@poetry run python -c "\
import os; \
from dotenv import load_dotenv; \
import mlflow; \
load_dotenv(); \
print('MLflow URI:', os.getenv('MLFLOW_TRACKING_URI')); \
print('Username:', os.getenv('MLFLOW_TRACKING_USERNAME')); \
mlflow.set_experiment('connection-test'); \
with mlflow.start_run(): \
    mlflow.log_param('test', 'makefile'); \
print('$(GREEN)✅ Connection successful!$(NC)'); \
"

switch-to-remote: ## Switch from local to remote MLflow
	@echo "$(BLUE)Switching to remote MLflow...$(NC)"
	@if [ ! -f .env.local.backup ]; then \
		cp .env .env.local.backup; \
		echo "$(GREEN)Backed up local .env to .env.local.backup$(NC)"; \
	fi
	@echo "$(YELLOW)Please update .env with remote settings:$(NC)"
	@echo "  MLFLOW_TRACKING_URI=https://mlflow.company.com"
	@echo "  MLFLOW_TRACKING_USERNAME=your_username"
	@echo "  MLFLOW_TRACKING_PASSWORD=your_password"
	@echo "  AWS_REGION=us-west-2"

switch-to-local: ## Switch from remote to local MLflow
	@echo "$(BLUE)Switching to local MLflow...$(NC)"
	@if [ -f .env.local.backup ]; then \
		cp .env.local.backup .env; \
		echo "$(GREEN)Restored local .env from backup$(NC)"; \
	else \
		echo "$(YELLOW)No backup found, using .env.example$(NC)"; \
		cp .env.example .env; \
	fi

##@ Deployment (EKS)

deploy-eks: ## Deploy EKS infrastructure (Phase 5)
	@echo "$(BLUE)Deploying EKS infrastructure...$(NC)"
	@if [ -f scripts/setup/02-deploy-eks.sh ]; then \
		./scripts/setup/02-deploy-eks.sh; \
	else \
		echo "$(RED)Error: scripts/setup/02-deploy-eks.sh not found$(NC)"; \
		echo "Please create deployment scripts first"; \
	fi

deploy-mlflow: ## Deploy MLflow server on EKS
	@echo "$(BLUE)Deploying MLflow server...$(NC)"
	@if [ -f scripts/setup/03-deploy-mlflow.sh ]; then \
		./scripts/setup/03-deploy-mlflow.sh; \
	else \
		echo "$(RED)Error: scripts/setup/03-deploy-mlflow.sh not found$(NC)"; \
	fi

verify-eks: ## Verify EKS deployment
	@echo "$(BLUE)Verifying EKS deployment...$(NC)"
	@if [ -f scripts/setup/06-verify-all.sh ]; then \
		./scripts/setup/06-verify-all.sh; \
	else \
		kubectl get nodes; \
		kubectl get pods -n ml-platform; \
		kubectl get svc -n ml-platform; \
	fi

##@ Model Serving (BentoML)

serve: ## Start BentoML model serving (requires MODEL_RUN_ID or MODEL_NAME)
	@echo "$(BLUE)Starting BentoML model serving...$(NC)"
	@if [ -z "$(MODEL_RUN_ID)" ] && [ -z "$(MODEL_NAME)" ]; then \
		echo "$(RED)Error: Either MODEL_RUN_ID or MODEL_NAME must be set$(NC)"; \
		echo "Usage:"; \
		echo "  make serve MODEL_RUN_ID=<mlflow_run_id>"; \
		echo "  make serve MODEL_NAME=<model_name> MODEL_VERSION=<version>"; \
		exit 1; \
	fi
	docker-compose up -d bentoml
	@echo "$(GREEN)BentoML server started!$(NC)"
	@echo ""
	@echo "$(BLUE)API Endpoints:$(NC)"
	@echo "  - Predict Image: POST http://localhost:3000/predict_image"
	@echo "  - Predict Batch: POST http://localhost:3000/predict_batch"
	@echo "  - Model Info: GET http://localhost:3000/get_model_info"
	@echo "  - Health: GET http://localhost:3000/health"
	@echo ""

serve-stop: ## Stop BentoML model serving
	@echo "$(BLUE)Stopping BentoML server...$(NC)"
	docker-compose stop bentoml
	@echo "$(GREEN)BentoML server stopped!$(NC)"

serve-logs: ## Show BentoML server logs
	docker-compose logs -f bentoml

serve-build: ## Build BentoML service (bento build)
	@echo "$(BLUE)Building BentoML service...$(NC)"
	cd src/serving && bentoml build
	@echo "$(GREEN)BentoML service built!$(NC)"

serve-list: ## List built BentoML services
	@echo "$(BLUE)Available BentoML services:$(NC)"
	bentoml list

serve-containerize: ## Containerize BentoML service (requires BENTO_TAG)
	@if [ -z "$(BENTO_TAG)" ]; then \
		echo "$(RED)Error: BENTO_TAG not set$(NC)"; \
		echo "Usage: make serve-containerize BENTO_TAG=<tag>"; \
		exit 1; \
	fi
	@echo "$(BLUE)Containerizing BentoML service...$(NC)"
	bentoml containerize $(BENTO_TAG)
	@echo "$(GREEN)BentoML service containerized!$(NC)"

serve-test: ## Test BentoML API endpoints
	@echo "$(BLUE)Testing BentoML API...$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Health check:$(NC)"
	curl -s -X POST http://localhost:3000/health -H "Content-Type: application/json" -d '{}' | python3 -m json.tool || true
	@echo ""
	@echo "$(YELLOW)2. Model info:$(NC)"
	curl -s -X POST http://localhost:3000/get_model_info -H "Content-Type: application/json" -d '{}' | python3 -m json.tool || true
	@echo ""
	@echo "$(GREEN)API test complete!$(NC)"
	@echo "$(YELLOW)Tip: Use 'make serve-test-predict' to test image prediction$(NC)"

serve-test-predict: ## Test image prediction (requires test images)
	@echo "$(BLUE)Testing image prediction...$(NC)"
	@if [ ! -f test_cat.png ]; then \
		echo "$(YELLOW)No test images found. Creating test images...$(NC)"; \
		poetry run python create_test_image.py; \
	fi
	@echo ""
	@echo "$(YELLOW)Testing predictions on sample images:$(NC)"
	@echo ""
	@for img in test_cat.png test_dog.png test_airplane.png test_ship.png; do \
		if [ -f $$img ]; then \
			echo "$(BLUE)Testing: $$img$(NC)"; \
			curl -s -X POST http://localhost:3000/predict_image -F "image=@$$img" | python3 -m json.tool | grep -E "(predicted_class|confidence)" | head -2; \
			echo ""; \
		fi; \
	done
	@echo "$(GREEN)Prediction test complete!$(NC)"

serve-test-all: serve-test serve-test-predict ## Run all BentoML tests (health, model info, predictions)

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

mlflow-ui: ## Open MLflow UI in browser (local)
	@echo "$(BLUE)Opening local MLflow UI...$(NC)"
	open http://localhost:5001 || xdg-open http://localhost:5001 || echo "Please open http://localhost:5001 in your browser"

minio-ui: ## Open MinIO console in browser (local)
	@echo "$(BLUE)Opening MinIO console...$(NC)"
	open http://localhost:9001 || xdg-open http://localhost:9001 || echo "Please open http://localhost:9001 in your browser"

bentoml-ui: ## Open BentoML API documentation in browser
	@echo "$(BLUE)Opening BentoML API docs...$(NC)"
	open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Please open http://localhost:3000 in your browser"
