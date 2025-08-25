# Meta MCP Makefile

.PHONY: help dev test run docker-build migrate clean install lint format type-check

# Default target
help:
	@echo "Meta MCP Development Commands:"
	@echo "  dev           Start development environment"
	@echo "  test          Run test suite"
	@echo "  run           Start the server"
	@echo "  docker-build  Build Docker image"
	@echo "  migrate       Run database migrations"
	@echo "  install       Install dependencies"
	@echo "  lint          Run code linting"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  clean         Clean up temporary files"

# Development environment
dev: install migrate
	docker-compose up postgres -d
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Run the server
run:
	python -m app.main

# Build Docker image
docker-build:
	docker build -t meta-mcp:latest .

# Run database migrations
migrate:
	alembic upgrade head

# Linting
lint:
	ruff check app/ tests/

# Format code
format:
	black app/ tests/
	ruff check --fix app/ tests/

# Type checking
type-check:
	mypy app/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Full stack development
dev-full:
	docker-compose --profile full up -d

# Stop all services
stop:
	docker-compose down

# Reset development database
reset-db:
	docker-compose down
	docker volume rm minerva_postgres_data
	docker-compose up postgres -d
	sleep 5
	$(MAKE) migrate

# Production build
prod-build:
	docker build -t meta-mcp:production --target production .

# Run database in development
db:
	docker-compose up postgres -d