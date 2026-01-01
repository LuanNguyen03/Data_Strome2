# Dependencies Guide

Hướng dẫn chi tiết về các thư viện dependencies của hệ thống.

## 📦 Tổng quan

Hệ thống sử dụng 3 cách quản lý dependencies:

1. **UV** (Khuyến nghị) - Fast, reliable
2. **pip + requirements.txt** - Traditional
3. **pyproject.toml** - Modern Python packaging

## 🔧 Cài đặt nhanh

### Option 1: Sử dụng script tự động (Dễ nhất)

```bash
# Windows
setup_all.bat

# Linux/Mac
chmod +x setup_all.sh
./setup_all.sh
```

### Option 2: Sử dụng Make

```bash
make install       # Cài đặt tất cả
make install-backend   # Chỉ backend
make install-frontend  # Chỉ frontend
```

### Option 3: Thủ công

```bash
# Backend
uv sync                    # Recommended
# or
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

## 📚 Chi tiết Dependencies

### Core Backend

| Package | Version | Purpose |
|---------|---------|---------|
| **fastapi** | >=0.128.0 | Web framework |
| **uvicorn** | >=0.24.0 | ASGI server |
| **pydantic** | >=2.5.0 | Data validation |
| **python-dotenv** | >=1.0.0 | Environment variables |

### AI & Gemini Integration ✨ (NEW)

| Package | Version | Purpose |
|---------|---------|---------|
| **google-genai** | >=1.56.0 | Gemini 2.5 API for AI treatment recommendations |

**Lưu ý quan trọng:**
- ✅ `google-genai` - Package mới, sử dụng
- ❌ `google-generativeai` - DEPRECATED, không dùng

### Data Processing

| Package | Version | Purpose |
|---------|---------|---------|
| **duckdb** | >=0.9.0 | OLAP database |
| **polars** | >=0.19.0 | Fast dataframes |
| **pandas** | >=2.0.0 | Data manipulation |
| **pyarrow** | >=14.0.0 | Columnar data |
| **numpy** | >=1.26.0 | Numerical computing |

### Machine Learning

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | >=1.7.2 | ML algorithms |
| **xgboost** | >=3.1.2 | Gradient boosting |
| **lightgbm** | >=4.6.0 | Fast gradient boosting |
| **catboost** | >=1.2.0 | Categorical boosting |
| **optuna** | >=4.6.0 | Hyperparameter tuning |
| **imbalanced-learn** | >=0.14.1 | Handle imbalanced data |
| **joblib** | >=1.5.3 | Model persistence |

### Deep Learning (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| **pytorch-tabnet** | >=4.1.0 | Tabular deep learning |
| **torch** | >=2.0.0 | PyTorch framework |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **typer** | >=0.9.0 | CLI framework |
| **tqdm** | >=4.65.0 | Progress bars |

### Frontend

| Package | Version | Purpose |
|---------|---------|---------|
| **react** | ^18.2.0 | UI framework |
| **vite** | ^5.0.0 | Build tool |
| **react-router-dom** | ^6.20.0 | Routing |
| **typescript** | ^5.2.2 | Type safety |

## 🔍 Kiểm tra Dependencies

### Kiểm tra Backend

```bash
# Check Python packages
pip list

# Check specific packages
pip show google-genai
pip show fastapi

# Check versions
make versions
```

### Kiểm tra Frontend

```bash
cd frontend
npm list --depth=0
```

### Kiểm tra Gemini AI

```bash
python check_gemini.py
python check_gemini.py --test-api
```

## 🆕 Cập nhật Dependencies

### Cập nhật Backend

```bash
# Using UV
uv sync --upgrade

# Using pip
pip install --upgrade -r requirements.txt
```

### Cập nhật Frontend

```bash
cd frontend
npm update
```

### Cập nhật Gemini Package

```bash
pip install --upgrade google-genai
```

## ❌ Xóa Dependencies cũ (Cleanup)

### Xóa package deprecated

```bash
# Xóa google-generativeai cũ (deprecated)
pip uninstall -y google-generativeai

# Xóa cache
make clean
```

### Reset hoàn toàn

```bash
# Backend
rm -rf .venv
rm -rf __pycache__
uv sync  # hoặc pip install -r requirements.txt

# Frontend
rm -rf frontend/node_modules
cd frontend && npm install
```

## 🐛 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'google.genai'"

```bash
pip install google-genai
```

### Lỗi: Conflict với google-generativeai

```bash
pip uninstall google-generativeai
pip install google-genai
```

### Lỗi: "pip not found" hoặc "python not found"

- Kiểm tra Python đã được cài đặt: `python --version`
- Kiểm tra pip: `pip --version`
- Windows: Thêm Python vào PATH
- Linux/Mac: Sử dụng `python3` và `pip3`

### Lỗi: UV không hoạt động

```bash
# Cài UV
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Lỗi: npm install failed

```bash
# Clear cache
npm cache clean --force

# Xóa node_modules và cài lại
rm -rf node_modules package-lock.json
npm install
```

## 📊 Kích thước Dependencies

| Component | Size | Time to Install |
|-----------|------|----------------|
| Backend (core) | ~500MB | 2-5 min |
| Backend (with ML) | ~2GB | 5-10 min |
| Frontend | ~300MB | 1-3 min |
| Gemini AI | ~50MB | 10-30 sec |

**Tổng**: ~2.5GB, 10-15 phút (lần đầu)

## 🚀 Best Practices

### 1. Sử dụng Virtual Environment

```bash
# Luôn sử dụng venv để tránh conflict
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 2. Pin versions cho Production

```bash
# Generate exact versions
pip freeze > requirements-lock.txt
```

### 3. Kiểm tra security

```bash
# Check for vulnerabilities
pip install safety
safety check
```

### 4. Regular updates

```bash
# Cập nhật monthly
make install
python check_gemini.py --test-api
```

## 📚 Tài liệu tham khảo

- [UV Documentation](https://github.com/astral-sh/uv)
- [Gemini API Docs](https://ai.google.dev/docs)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

## 💡 Tips

1. **Sử dụng UV** khi có thể - nhanh hơn pip nhiều
2. **Cache dependencies** để cài đặt nhanh lần sau
3. **Kiểm tra Gemini** sau khi cài: `python check_gemini.py`
4. **Đọc GEMINI_SETUP.md** để config AI recommendations

---

**Last Updated**: January 2026  
**Gemini Version**: 2.5 Flash (Latest)
