# Installation Summary - Dry Eye Assessment

## ✨ Tổng quan

Hệ thống đã được cập nhật với **đầy đủ tài liệu cài đặt** và **scripts tự động** để người dùng có thể setup nhanh chóng.

## 📦 Files đã thêm/cập nhật

### 📄 Documentation Files

| File | Mô tả |
|------|-------|
| **requirements.txt** | ✅ Dependencies đầy đủ (backend + AI) |
| **DEPENDENCIES.md** | ✅ Chi tiết về từng package |
| **README_INSTALL.md** | ✅ Hướng dẫn cài đặt đầy đủ |
| **INSTALL_COMMANDS.md** | ✅ Quick command cheat sheet |
| **GEMINI_SETUP.md** | ✅ Setup Gemini AI (updated) |
| **GEMINI_MODEL_OPTIONS.md** | ✅ Chọn Gemini model |
| **GEMINI_2.5_UPGRADE.md** | ✅ Info về Gemini 2.5 |

### 🔧 Setup Scripts

| File | Platform | Mô tả |
|------|----------|-------|
| **setup_all.bat** | Windows | Auto setup tất cả |
| **setup_all.sh** | Linux/Mac | Auto setup tất cả |
| **setup_gemini.bat** | Windows | Setup Gemini AI |
| **setup_gemini.sh** | Linux/Mac | Setup Gemini AI |

### 🛠️ Build Tools

| File | Mô tả |
|------|-------|
| **Makefile** | Updated với Gemini commands |
| **pyproject.toml** | Updated với google-genai |
| **requirements_ml.txt** | ML dependencies |

### 🔍 Utility Scripts

| File | Mô tả |
|------|-------|
| **check_gemini.py** | Check Gemini config |
| **list_models.py** | List available models |

## 🚀 Cài đặt - 3 Cách

### Cách 1: Auto Setup (Dễ nhất) ⭐

```bash
# Windows
setup_all.bat

# Linux/Mac
chmod +x setup_all.sh && ./setup_all.sh
```

**Thời gian:** 10-15 phút  
**Bao gồm:** Backend + Frontend + Gemini setup (optional)

### Cách 2: Using Make (Recommended for devs) 🛠️

```bash
make install          # Cài tất cả
make setup-gemini     # Setup AI
make check-gemini     # Kiểm tra
make run-backend      # Chạy backend
make run-frontend     # Chạy frontend (terminal mới)
```

**Xem tất cả commands:**
```bash
make help
```

### Cách 3: Manual (Chi tiết nhất) 📝

```bash
# Backend
uv sync  # hoặc: pip install -r requirements.txt

# Frontend
cd frontend && npm install

# Gemini (optional)
echo "GEMINI_API_KEY=your_key" > .env

# Verify
python check_gemini.py --test-api
```

## ✅ Checklist sau khi cài

- [ ] Backend dependencies installed
  ```bash
  pip list | grep fastapi
  pip list | grep google-genai
  ```

- [ ] Frontend dependencies installed
  ```bash
  cd frontend && npm list --depth=0
  ```

- [ ] Gemini AI configured (optional)
  ```bash
  python check_gemini.py
  ```

- [ ] Backend starts successfully
  ```bash
  python backend/run.py
  # Thấy: "✓ Gemini AI service ENABLED"
  ```

- [ ] Frontend starts successfully
  ```bash
  cd frontend && npm run dev
  # Mở: http://localhost:5173
  ```

- [ ] AI feature works
  - Thực hiện đánh giá từ frontend
  - Xem section "Hướng điều trị đề xuất (AI)"

## 📊 Dependencies Overview

### Backend (Total: ~2.5GB)

**Core (500MB)**
- FastAPI 0.128.0+
- Pydantic 2.5.0+
- Uvicorn 0.24.0+

**AI (50MB)** ✨ NEW!
- google-genai 1.56.0+

**ML (2GB)**
- XGBoost, LightGBM, CatBoost
- Scikit-learn, PyTorch

**Data Processing (300MB)**
- Pandas, Polars, DuckDB

### Frontend (Total: ~300MB)

- React 18
- TypeScript 5
- Vite 5
- React Router 6

## 🎯 Quick Commands

```bash
# INSTALL
make install                    # All
setup_all.bat                   # Windows auto

# SETUP GEMINI
make setup-gemini              # Interactive
echo "GEMINI_API_KEY=key" > .env  # Manual

# CHECK
make check-gemini              # Config check
make check-gemini-api          # API test
make versions                  # Show versions

# RUN
make run-backend               # Backend
make run-frontend              # Frontend

# CLEANUP
make clean                     # Remove cache
pip uninstall google-generativeai  # Remove old package
```

## 🔗 Quick Links

### Get Started
- **QUICKSTART.md** - Start here
- **README_INSTALL.md** - Full installation guide
- **INSTALL_COMMANDS.md** - Command cheat sheet

### Gemini AI
- **GEMINI_SETUP.md** - Setup guide
- **GEMINI_MODEL_OPTIONS.md** - Model options
- Get API Key: https://makersuite.google.com/app/apikey

### Dependencies
- **DEPENDENCIES.md** - Package details
- **requirements.txt** - Install list
- **pyproject.toml** - UV config

### Advanced
- **Makefile** - Build commands
- **AI_TREATMENT_FEATURE.md** - Feature docs
- **API_DOCUMENTATION.md** - API reference

## 💡 Best Practices

1. **Luôn dùng virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **Ưu tiên UV cho tốc độ**
   ```bash
   uv sync  # 10x nhanh hơn pip
   ```

3. **Setup Gemini AI** để có AI recommendations
   ```bash
   setup_gemini.bat  # Windows
   ./setup_gemini.sh # Linux/Mac
   ```

4. **Kiểm tra sau mỗi install**
   ```bash
   make check-gemini
   python check_gemini.py --test-api
   ```

5. **Restart backend sau setup Gemini**
   ```bash
   # Ctrl+C để stop
   python backend/run.py  # Start lại
   ```

## 🐛 Common Issues & Fixes

### Issue: google-generativeai deprecated
```bash
pip uninstall google-generativeai
pip install google-genai
```

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Gemini not working
```bash
python check_gemini.py --test-api
# Xem lỗi chi tiết và fix theo hướng dẫn
```

### Issue: UV not found
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 📞 Support

Nếu gặp vấn đề:

1. Check **DEPENDENCIES.md** → Troubleshooting
2. Run `python check_gemini.py --test-api`
3. Check logs in terminal
4. See **README_INSTALL.md** → Detailed guide

## 🎉 Success Criteria

✅ Setup thành công khi:

1. Backend chạy tại http://localhost:8000
2. Frontend chạy tại http://localhost:5173
3. API Docs: http://localhost:8000/docs
4. Gemini log: "✓ Gemini AI service ENABLED"
5. AI recommendations hiển thị trong kết quả đánh giá

---

**Updated**: January 2026  
**Gemini**: 2.5 Flash (Latest)  
**Status**: ✅ Production Ready
