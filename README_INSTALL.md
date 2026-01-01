# Installation Guide - Complete

Hướng dẫn đầy đủ về cài đặt hệ thống Dry Eye Assessment.

## 📋 Yêu cầu hệ thống

### Bắt buộc

- **Python**: 3.10 - 3.13
- **Node.js**: 18+
- **npm**: 8+

### Khuyến nghị

- **UV**: Package manager nhanh cho Python
- **Make**: Để sử dụng Makefile commands
- **Git**: Để clone repository

## 🚀 Cài đặt nhanh (Recommended)

### Windows

```cmd
REM Chạy script tự động
setup_all.bat

REM Script sẽ:
REM 1. Check Python & Node.js
REM 2. Cài backend dependencies (UV hoặc pip)
REM 3. Cài frontend dependencies (npm)
REM 4. Setup Gemini AI (optional)
```

### Linux/Mac

```bash
# Chạy script tự động
chmod +x setup_all.sh
./setup_all.sh

# Script sẽ:
# 1. Check Python & Node.js
# 2. Cài backend dependencies (UV hoặc pip)
# 3. Cài frontend dependencies (npm)
# 4. Setup Gemini AI (optional)
```

## 📦 Các phương pháp cài đặt

### Method 1: Sử dụng Make (Recommended for developers)

```bash
# Cài đặt tất cả
make install

# Hoặc từng phần
make install-backend   # Backend only
make install-frontend  # Frontend only

# Setup Gemini
make setup-gemini

# Check installation
make versions
make check-gemini
```

### Method 2: Sử dụng UV (Fastest)

```bash
# Backend
uv sync

# Frontend
cd frontend && npm install

# Gemini (optional)
python setup_gemini.py
```

### Method 3: Sử dụng pip (Traditional)

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install

# Gemini (optional)
setup_gemini.bat  # Windows
./setup_gemini.sh # Linux/Mac
```

## 🔧 Chi tiết cài đặt từng component

### Backend

**Dependencies gồm:**
- FastAPI (Web framework)
- google-genai (Gemini AI - NEW!)
- ML libraries (XGBoost, LightGBM, CatBoost)
- Data processing (Pandas, Polars, DuckDB)

**Install:**

```bash
# Option 1: UV (10x faster)
uv sync

# Option 2: pip
pip install -r requirements.txt

# Option 3: From pyproject.toml
pip install -e .
```

**Verify:**

```bash
python -c "import fastapi; print(fastapi.__version__)"
python -c "from google import genai; print('Gemini OK')"
```

### Frontend

**Dependencies gồm:**
- React 18
- TypeScript
- Vite
- React Router

**Install:**

```bash
cd frontend
npm install

# Hoặc sử dụng yarn
yarn install
```

**Verify:**

```bash
cd frontend
npm list react
npm list typescript
```

### Gemini AI (Optional nhưng khuyến nghị)

**Setup:**

```bash
# Interactive setup
setup_gemini.bat      # Windows
./setup_gemini.sh     # Linux/Mac

# Hoặc thủ công tạo .env
echo "GEMINI_API_KEY=your_api_key" > .env
```

**Verify:**

```bash
python check_gemini.py
python check_gemini.py --test-api
```

**Get API Key:**
1. Visit: https://makersuite.google.com/app/apikey
2. Login with Google account
3. Create/Get API Key
4. Copy key (starts with `AIzaSy...`)

## ✅ Kiểm tra cài đặt

### Quick Check

```bash
# Check Python
python --version

# Check Node
node --version
npm --version

# Check UV (optional)
uv --version

# Check all versions
make versions
```

### Detailed Check

```bash
# Backend packages
pip list | grep -E "fastapi|google-genai|xgboost"

# Frontend packages
cd frontend && npm list --depth=0

# Gemini configuration
python check_gemini.py --test-api
```

### Run Tests

```bash
# Backend tests
python -m pytest

# Frontend tests
cd frontend && npm test
```

## 🚀 Chạy hệ thống

### Start Backend

```bash
# Method 1: UV
uv run python backend/run.py

# Method 2: Direct
python backend/run.py

# Method 3: Make
make run-backend
```

Backend runs at: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/healthz

### Start Frontend (Terminal mới)

```bash
# Method 1: npm
cd frontend && npm run dev

# Method 2: Make
make run-frontend
```

Frontend runs at: http://localhost:5173

## 🐛 Troubleshooting

### Python/pip issues

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Clear pip cache
pip cache purge

# Reinstall
pip install --force-reinstall -r requirements.txt
```

### Node/npm issues

```bash
# Clear npm cache
npm cache clean --force

# Remove and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Gemini issues

```bash
# Check API key
cat .env | grep GEMINI_API_KEY

# Test connection
python check_gemini.py --test-api

# List available models
python list_models.py

# Reinstall package
pip uninstall google-generativeai  # Old (deprecated)
pip install google-genai  # New
```

### UV issues

```bash
# Install UV
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal after install
```

### Common Errors

**Error: "ModuleNotFoundError: No module named 'google.genai'"**
```bash
pip install google-genai
```

**Error: "google-generativeai is deprecated"**
```bash
pip uninstall google-generativeai
pip install google-genai
```

**Error: "API key not valid"**
```bash
# Check .env file exists
ls -la .env

# Verify API key
python check_gemini.py
```

**Error: "Port 8000 already in use"**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

## 📚 File tham khảo

| File | Mô tả |
|------|-------|
| **requirements.txt** | Backend dependencies |
| **pyproject.toml** | Python project config |
| **Makefile** | Make commands |
| **setup_all.bat/sh** | Auto setup script |
| **DEPENDENCIES.md** | Chi tiết dependencies |
| **GEMINI_SETUP.md** | Gemini AI setup |
| **INSTALL_COMMANDS.md** | Quick command reference |
| **QUICKSTART.md** | Quick start guide |

## 💡 Best Practices

### 1. Sử dụng Virtual Environment

```bash
# Tạo venv
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

### 2. Keep Dependencies Updated

```bash
# Update backend
uv sync --upgrade
# hoặc
pip install --upgrade -r requirements.txt

# Update frontend
cd frontend && npm update
```

### 3. Pin Versions cho Production

```bash
# Generate lock file
pip freeze > requirements-lock.txt
uv pip compile pyproject.toml -o requirements-lock.txt
```

### 4. Regular Health Checks

```bash
# Weekly checks
make check-gemini
python -m pytest
npm test
```

## 🎯 Next Steps

Sau khi cài đặt thành công:

1. ✅ **Setup Gemini AI** (optional nhưng khuyến nghị)
   ```bash
   setup_gemini.bat  # Windows
   ./setup_gemini.sh # Linux/Mac
   ```

2. ✅ **Start Backend**
   ```bash
   uv run python backend/run.py
   ```

3. ✅ **Start Frontend** (terminal mới)
   ```bash
   cd frontend && npm run dev
   ```

4. ✅ **Open Browser**
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:8000/docs

5. ✅ **Test AI Features**
   - Thực hiện đánh giá từ frontend
   - Xem "Hướng điều trị đề xuất (AI)" trong kết quả

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra **DEPENDENCIES.md** - Troubleshooting section
2. Xem **INSTALL_COMMANDS.md** - Quick fixes
3. Check logs trong terminal
4. Run: `make versions` để xem versions
5. Run: `python check_gemini.py --test-api` để test AI

---

**Last Updated**: January 2026  
**Gemini Version**: 2.5 Flash  
**Python**: 3.10 - 3.13  
**Node.js**: 18+
