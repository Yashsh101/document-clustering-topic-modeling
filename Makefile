.PHONY: help install dev lint test train evaluate streamlit clean docker-build docker-run

help:
	@echo "Document Clustering & Topic Modeling - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint            Run linters (ruff, black)"
	@echo "  make format          Format code with black"
	@echo "  make test            Run tests with pytest"
	@echo ""
	@echo "Pipeline:"
	@echo "  make download-data   Download sample datasets"
	@echo "  make train           Train the pipeline"
	@echo "  make evaluate        Generate evaluation report"
	@echo "  make streamlit       Run Streamlit app"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           Remove artifacts and cache"

install:
	pip install -r requirements.txt
	python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

install-dev:
	pip install -r requirements.txt
	python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

lint:
	ruff check . --exclude "__pycache__"
	black --check . --exclude "__pycache__"

format:
	black . --exclude "__pycache__"

test:
	pytest tests/ -v --cov=src --cov-report=html

download-data:
	python scripts/download_data.py

train:
	python scripts/train.py --data-dir data/sample --n-clusters 5 --n-topics 5

train-tuned:
	python scripts/train.py --data-dir data/sample --n-clusters 5 --n-topics 5 --max-features 500

evaluate:
	python scripts/evaluate.py --artifact-dir artifacts --output artifacts/reports/evaluation_report.txt

streamlit:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t doc-clustering:latest .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts doc-clustering:latest

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/ build/ *.egg-info/
