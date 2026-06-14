.PHONY: train loop predict dashboard test test-fast lint help

help:
	@echo "Comandos disponibles:"
	@echo "  make train          - Entrena modelo canónico"
	@echo "  make loop           - Loop engineering con 20 trials"
	@echo "  make predict        - Predice un partido (usar ARGS=...)"
	@echo "  make dashboard      - Muestra dashboard de entrenamientos"
	@echo "  make test           - Corre tests y gates"
	@echo "  make test-fast      - Solo tests rápidos"
	@echo "  make lint           - Corre ruff"

train:
	python scripts/train.py

loop:
	python scripts/train.py --loop --trials 20

predict:
	python scripts/predict.py $(ARGS)

dashboard:
	python scripts/training_dashboard.py

test:
	python scripts/run-tests.py --gate

test-fast:
	python scripts/run-tests.py --fast

lint:
	python -m ruff check scripts/ predictors/ backtest/ tests/
