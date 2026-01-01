# Quick Install Commands - Cheat Sheet

## 🚀 Setup nhanh (Một lệnh)

```bash
# Windows
setup_all.bat

# Linux/Mac
chmod +x setup_all.sh && ./setup_all.sh
```

## 📦 Cài đặt từng bước

### 1. Backend

```bash
# Method 1: UV (Khuyến nghị - nhanh hơn)
uv sync

# Method 2: pip
pip install -r requirements.txt

# Method 3: Make
make install-backend
```

### 2. Frontend

```bash
cd frontend
npm install

# Hoặc
make install-frontend
```

### 3. Gemini AI (Optional)

```bash
# Setup wizard
setup_gemini.bat      # Windows
./setup_gemini.sh     # Linux/Mac

# Thủ công
echo "GEMINI_API_KEY=your_key" > .env
```

## ✅ Kiểm tra

```bash
# Kiểm tra backend packages
pip list | grep -E "fastapi|google-genai"

# Kiểm tra Gemini
python check_gemini.py
python check_gemini.py --test-api

# Kiểm tra versions
make versions
```

## 🚀 Chạy Services

```bash
# Backend
uv run python backend/run.py
# hoặc
python backend/run.py
# hoặc
make run-backend

# Frontend (terminal mới)
cd frontend && npm run dev
# hoặc
make run-frontend
```

## 🧹 Dọn dẹp

```bash
# Xóa cache
make clean

# Xóa package cũ deprecated
pip uninstall -y google-generativeai

# Reset hoàn toàn
rm -rf .venv node_modules
```

## 🆘 Fix lỗi thường gặp

```bash
# Lỗi: Module not found
pip install -r requirements.txt

# Lỗi: google-generativeai deprecated
pip uninstall google-generativeai
pip install google-genai

# Lỗi: npm install failed
cd frontend
rm -rf node_modules package-lock.json
npm install

# Lỗi: UV not found
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 📚 Xem thêm

- **DEPENDENCIES.md** - Chi tiết về dependencies
- **GEMINI_SETUP.md** - Setup Gemini AI
- **QUICKSTART.md** - Hướng dẫn đầy đủ
- **Makefile** - Chạy `make help` để xem commands

## 💡 Tips

1. Luôn activate venv trước khi cài:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. Sử dụng UV cho tốc độ:
   ```bash
   uv sync  # 10x nhanh hơn pip
   ```

3. Kiểm tra Gemini sau khi cài:
   ```bash
   python check_gemini.py --test-api
   ```

4. Restart backend sau khi setup Gemini:
   ```bash
   # Ctrl+C để dừng
   uv run python backend/run.py  # Chạy lại
   ```

---

**Quick Links:**
- [Get Gemini API Key](https://makersuite.google.com/app/apikey)
- [Python Downloads](https://www.python.org/downloads/)
- [Node.js Downloads](https://nodejs.org/)
- [UV Installation](https://github.com/astral-sh/uv)
