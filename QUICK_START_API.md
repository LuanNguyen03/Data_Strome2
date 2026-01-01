# Quick Start - API Server

## Cách chạy API Server

### Cách 1: Sử dụng backend/run.py (Khuyên dùng) ✅

```bash
# Từ root directory
python backend/run.py
```

### Cách 2: Sử dụng uvicorn trực tiếp

```bash
# Từ root directory
uvicorn backend.main:app --reload

# HOẶC từ backend directory
cd backend
uvicorn main:app --reload
```

## Lỗi thường gặp

### ❌ Lỗi: "Could not import module 'main'"

**Nguyên nhân**: Đang chạy `uvicorn main:app` từ root directory nhưng `main.py` nằm trong `backend/`

**Giải pháp**: 
- Sử dụng `uvicorn backend.main:app --reload` (từ root)
- Hoặc `python backend/run.py` (từ root)
- Hoặc `cd backend && uvicorn main:app --reload` (từ backend)

## API sẽ chạy tại:
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/healthz
