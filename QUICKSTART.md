# Quick Start Guide

> **⚠️ Important**: Nếu UV chưa được cài đặt, hãy restart terminal sau khi chạy `install_uv.bat` để UV được load vào PATH.

## 🚀 5-Minute Setup

> **💡 Quick Install**: Chạy `setup_all.bat` (Windows) hoặc `./setup_all.sh` (Linux/Mac) để cài đặt tự động tất cả!

### Option A: Using UV (Recommended - Faster ⚡)

#### 1. Install UV

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or run: `install_uv.bat` (Windows) or `install_uv.sh` (Linux/Mac)

#### 2. Setup Isolated Environment

```bash
# Option A: Use setup script (recommended - ensures isolation)
setup_uv_env.bat  # Windows
./setup_uv_env.sh  # Linux/Mac

# Option B: Manual setup
uv sync --python 3.13

# Frontend (in new terminal)
cd frontend
npm install
```

**Note**: UV tự động tạo `.venv/` riêng trong project root, đảm bảo **không ảnh hưởng** các dự án khác.

#### 3. Start Backend

```bash
# Using UV (recommended)
uv run python backend/run.py

# Or use helper script
uv_run_backend.bat  # Windows
./uv_run_backend.sh  # Linux/Mac
```

Backend runs at: http://localhost:8000

- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/healthz

#### 4. Setup Gemini AI (Optional - Khuyến nghị)

Để bật tính năng tư vấn điều trị cá nhân hóa bằng AI:

**Cách 1: Sử dụng script tự động (Dễ nhất)**

```bash
# Windows
setup_gemini.bat

# Linux/Mac
chmod +x setup_gemini.sh
./setup_gemini.sh
```

**Cách 2: Thủ công**

```bash
# Tạo file .env trong thư mục gốc project
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Sau đó restart backend
```

📖 Xem hướng dẫn chi tiết: [GEMINI_SETUP.md](./GEMINI_SETUP.md)

### Option B: Traditional venv

#### 1. Activate Virtual Environment

```bash
# Windows
call .venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

#### 2. Install Dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend (in new terminal)
cd frontend
npm install
```

#### 3. Start Backend

```bash
python backend/run.py
```

Backend runs at: http://localhost:8000

- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/healthz

### 4. Start Frontend (in new terminal)

```bash
cd frontend
npm run dev
```

Frontend runs at: http://localhost:3000

### 5. Verify Setup

```bash
python scripts/verify_setup.py
```

## 📝 First API Call

Test the API:

```bash
curl -X POST http://localhost:8000/api/assess \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sleep_duration": 7.5,
    "sleep_quality": 3,
    "average_screen_time": 8.0,
    "stress_level": 4
  }'
```

Or use the Swagger UI at http://localhost:8000/docs

## 🎯 Next Steps

1. **Data Pipeline** (optional):

   ```bash
   python scripts/standardize.py --input DryEyeDisease/Dry_Eye_Dataset.csv --output data/standardized/clean_assessments.parquet --report data/standardized/data_quality_report.json
   ```

2. **OLAP Aggregates** (optional):

   ```bash
   python analytics/duckdb/build_agg.py --input data/standardized/clean_assessments.parquet --outdir analytics/duckdb/agg
   ```

3. **Train Models** (future):
   - Replace mock logic in `backend/services/assessment_service.py` with trained models
   - Save models to `modeling/artifacts/`

## 🐛 Troubleshooting

### Backend won't start

- Check Python version: `python --version` (need 3.10+)
- Verify packages: `pip list | findstr fastapi`
- Check imports: `python scripts/verify_setup.py`

### Frontend won't start

- Check Node version: `node --version` (need 18+)
- Clear cache: `rm -rf node_modules package-lock.json && npm install`
- Check API URL in `frontend/.env.local`

### Import errors

- Make sure you're in the project root
- Activate venv before running Python scripts
- Check `sys.path` includes project root

## 📚 Full Documentation

### Installation Guides

- **README_INSTALL.md** - Hướng dẫn cài đặt đầy đủ
- **INSTALL_COMMANDS.md** - Quick command reference
- **DEPENDENCIES.md** - Chi tiết về dependencies

### Setup Guides

- **GEMINI_SETUP.md** - Setup Gemini AI
- **GEMINI_MODEL_OPTIONS.md** - Chọn Gemini model
- **GEMINI_2.5_UPGRADE.md** - Info về Gemini 2.5

### Feature Docs

- **AI_TREATMENT_FEATURE.md** - AI treatment recommendations
- **API_DOCUMENTATION.md** - API reference
- **TESTING_GUIDE.md** - Testing guide

### Others

- **README.md** - Project overview
- **PROJECT_STRUCTURE.md** - Code structure
- **Makefile** - Run `make help` for commands
