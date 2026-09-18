# ============================================
# Makefile — 南航智慧出行管家
# 常用命令一键执行
# ============================================

.PHONY: install test lint format run api docker-up docker-build clean

# ── 环境 ──

install:
	pip install -r requirements.txt
	pip install pytest ruff pre-commit mypy
	pre-commit install

# ── 测试 ──

test:
	python -m pytest tests/ -v --tb=short

test-coverage:
	python -m pytest tests/ --cov=agent --cov=eval --cov-report=term-missing

# ── 代码质量 ──

lint:
	ruff check agent/ eval/ app.py api.py logger.py

format:
	ruff check --fix agent/ eval/ app.py api.py logger.py
	black --line-length=120 agent/ eval/ app.py api.py logger.py

precommit:
	pre-commit run --all-files

# ── 运行 ──

run:
	python app.py

api:
	uvicorn api:app --reload --port 8000

# ── Docker ──

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ── 清理 ──

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "logs" -exec rm -rf {} + 2>/dev/null || true

all: install format lint test
