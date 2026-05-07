# CANDLE/FCIS Makefile

.PHONY: help install test demo clean run

help:
	@echo "CANDLE/FCIS - Available Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make demo       - Run quick demo"
	@echo "  make run        - Run full pipeline"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make notebook   - Start Jupyter notebook"

install:
	pip install -r requirements.txt
	@echo "Dependencies installed!"

demo:
	python demo.py

run:
	python run.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/raw/* data/processed/* results/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned generated files"

notebook:
	jupyter notebook notebooks/

lint:
	flake8 src/ --max-line-length=100

format:
	black src/ --line-length=100
