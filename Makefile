# Makefile for Dry Eye Assessment Project
# Windows: Use `nmake` or install `make` via chocolatey/cygwin

.PHONY: help install install-all install-backend install-frontend setup-gemini check-gemini \
        run-backend run-frontend standardize olap clean

help:
	@echo "=========================================="
	@echo "  Dry Eye Assessment - Make Commands"
	@echo "=========================================="
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install          - Install all dependencies (backend + frontend)"
	@echo "  make install-backend  - Install backend dependencies only"
	@echo "  make install-frontend - Install frontend dependencies only"
	@echo ""
	@echo "🤖 Gemini AI Setup:"
	@echo "  make setup-gemini     - Interactive Gemini API key setup"
	@echo "  make check-gemini     - Check Gemini configuration"
	@echo ""
	@echo "🚀 Run Services:"
	@echo "  make run-backend      - Start FastAPI backend"
	@echo "  make run-frontend     - Start React frontend"
	@echo ""
	@echo "📊 Data Processing:"
	@echo "  make standardize      - Run data standardization pipeline"
	@echo "  make olap            - Build OLAP aggregates"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean           - Remove cache and temp files"
	@echo ""
	@echo "💡 Quick Start:"
	@echo "  1. make install       (install dependencies)"
	@echo "  2. make setup-gemini  (configure AI - optional)"
	@echo "  3. make run-backend   (start backend)"
	@echo "  4. make run-frontend  (start frontend in new terminal)"
	@echo ""

install: install-backend install-frontend
	@echo "✅ All dependencies installed successfully!"

install-all: install

standardize:
	python scripts/standardize.py \
		--input DryEyeDisease/Dry_Eye_Dataset.csv \
		--output data/standardized/clean_assessments.parquet \
		--report data/standardized/data_quality_report.json

olap:
	python -m backend.scripts.olap_build \
		--input data/standardized/clean_assessments.parquet \
		--outdir analytics/duckdb/agg

install-backend:
	@echo "📦 Installing backend dependencies..."
	@echo "   Method 1 (Recommended): uv sync"
	@echo "   Method 2 (Alternative): pip install -r requirements.txt"
	@echo ""
	@command -v uv >/dev/null 2>&1 && uv sync || pip install -r requirements.txt
	@echo "✅ Backend dependencies installed!"

install-frontend:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ Frontend dependencies installed!"

setup-gemini:
	@echo "🤖 Setting up Gemini AI..."
	@command -v python >/dev/null 2>&1 && python setup_gemini.bat || echo "Python not found!"

check-gemini:
	@echo "🔍 Checking Gemini configuration..."
	python check_gemini.py

check-gemini-api:
	@echo "🔍 Testing Gemini API connection..."
	python check_gemini.py --test-api

list-models:
	@echo "📋 Listing available Gemini models..."
	python list_models.py

run-backend:
	python backend/run.py

run-frontend:
	cd frontend && npm run dev

# Windows alternative (use task runner)
win-standardize:
	python scripts/standardize.py --input DryEyeDisease/Dry_Eye_Dataset.csv --output data/standardized/clean_assessments.parquet --report data/standardized/data_quality_report.json

win-olap:
	python -m backend.scripts.olap_build --input data/standardized/clean_assessments.parquet --outdir analytics/duckdb/agg

win-backend:
	python backend/run.py

win-frontend:
	cd frontend && npm run dev

# Cleanup
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name ".venv" -prune -o -type d -name "node_modules" -prune -o -type f -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed!"

# Show versions
versions:
	@echo "📌 Installed versions:"
	@echo "Python: $$(python --version 2>&1)"
	@echo "Node: $$(node --version 2>&1)"
	@echo "npm: $$(npm --version 2>&1)"
	@command -v uv >/dev/null 2>&1 && echo "UV: $$(uv --version 2>&1)" || echo "UV: Not installed"
	@python -c "import google.genai; print(f'google-genai: {google.genai.__version__ if hasattr(google.genai, \"__version__\") else \"Unknown\"}')" 2>/dev/null || echo "google-genai: Not installed"

