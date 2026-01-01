#!/bin/bash
# ============================================
# Setup All Dependencies - Dry Eye Assessment
# ============================================

echo ""
echo "================================================"
echo "  Dry Eye Assessment - Complete Setup"
echo "================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "[STEP 1/5] Checking Python..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python not found! Please install Python 3.10+ first."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
$PYTHON_CMD --version
echo ""

# Check Node.js
echo "[STEP 2/5] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Node.js not found! Please install Node.js 18+ first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

node --version
npm --version
echo ""

# Install Backend Dependencies
echo "[STEP 3/5] Installing Backend Dependencies..."
echo "------------------------------------------------"

if command -v uv &> /dev/null; then
    echo "Using UV (faster)..."
    uv sync
else
    echo "UV not found, using pip instead..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to install backend dependencies!"
    exit 1
fi
echo -e "${GREEN}[SUCCESS]${NC} Backend dependencies installed!"
echo ""

# Install Frontend Dependencies
echo "[STEP 4/5] Installing Frontend Dependencies..."
echo "------------------------------------------------"
cd frontend
npm install
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to install frontend dependencies!"
    exit 1
fi
cd ..
echo -e "${GREEN}[SUCCESS]${NC} Frontend dependencies installed!"
echo ""

# Setup Gemini AI (Optional)
echo "[STEP 5/5] Gemini AI Setup (Optional)..."
echo "------------------------------------------------"
read -p "Do you want to setup Gemini AI now? (y/n): " setup_gemini
if [[ "$setup_gemini" =~ ^[Yy]$ ]]; then
    echo ""
    chmod +x setup_gemini.sh
    ./setup_gemini.sh
else
    echo "Skipping Gemini AI setup. You can run './setup_gemini.sh' later."
fi
echo ""

# Summary
echo "================================================"
echo "  SETUP COMPLETE!"
echo "================================================"
echo ""
echo "[INSTALLED]"
echo "  - Backend dependencies (FastAPI, ML libraries, etc.)"
echo "  - Frontend dependencies (React, Vite, etc.)"
if [[ "$setup_gemini" =~ ^[Yy]$ ]]; then
    echo "  - Gemini AI configured"
else
    echo "  - Gemini AI: Not configured (optional)"
fi
echo ""
echo "[NEXT STEPS]"
echo "  1. Start backend:  uv run python backend/run.py"
echo "     or:             python backend/run.py"
echo ""
echo "  2. Start frontend (in new terminal):"
echo "     cd frontend"
echo "     npm run dev"
echo ""
echo "  3. Open browser: http://localhost:5173"
echo ""
echo "[OPTIONAL]"
if [[ ! "$setup_gemini" =~ ^[Yy]$ ]]; then
    echo "  - Setup Gemini AI: ./setup_gemini.sh"
fi
echo "  - Check Gemini:    python check_gemini.py"
echo "  - View help:       make help"
echo ""
echo "================================================"
