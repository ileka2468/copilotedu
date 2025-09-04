.PHONY: help build up down logs clean test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

up-build: ## Build and start all services
	docker-compose up --build -d

down: ## Stop all services
	docker-compose down

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show logs from API gateway
	docker-compose logs -f api-gateway

logs-frontend: ## Show logs from frontend
	docker-compose logs -f frontend

logs-grading: ## Show logs from grading service
	docker-compose logs -f grading-service

logs-processor: ## Show logs from processor service
	docker-compose logs -f content-processor

clean: ## Remove all containers, networks, and volumes
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-all: ## Remove everything including images
	docker-compose down -v --remove-orphans
	docker system prune -af

restart: ## Restart all services
	docker-compose restart

restart-api: ## Restart API gateway
	docker-compose restart api-gateway

restart-frontend: ## Restart frontend
	docker-compose restart frontend

db-shell: ## Connect to PostgreSQL database
	docker-compose exec postgres psql -U teachercopilot -d teachercopilot

redis-shell: ## Connect to Redis
	docker-compose exec redis redis-cli

setup: ## Initial setup - copy env file and build
	cp .env.example .env
	@echo "Please edit .env file with your API keys before running 'make up-build'"

dev-frontend: ## Run frontend in development mode
	cd frontend && npm install && npm run dev

dev-backend: ## Run backend in development mode
	cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

test: ## Run tests
	@echo "Running tests..."
	# Add test commands here when implemented

status: ## Show status of all services
	docker-compose ps