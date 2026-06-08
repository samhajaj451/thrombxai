.PHONY: help install dev lint format type test smoke docker clean

help:
	@echo "install  install runtime package"
	@echo "dev      install with dev extras"
	@echo "lint     ruff + isort check"
	@echo "format   black + isort write"
	@echo "type     mypy --strict"
	@echo "test     pytest"
	@echo "smoke    train two steps on the smoke config"
	@echo "docker   build the container image"

install:
	pip install .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	isort --check-only .
	black --check .

format:
	isort .
	black .

type:
	mypy --strict thrombxai

test:
	pytest -q

smoke:
	python -m thrombxai.floor.weave experiment=_smoke

docker:
	docker build -t thrombxai:latest .

clean:
	rm -rf outputs artifacts checkpoints .mypy_cache .ruff_cache .pytest_cache
