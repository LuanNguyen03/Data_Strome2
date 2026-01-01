# Hướng dẫn cấu hình Gemini AI

Hệ thống hỗ trợ tích hợp Gemini AI để đưa ra hướng điều trị cá nhân hóa dựa trên thông tin người dùng và kết quả đánh giá.

## 🔑 Lấy API Key

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập bằng tài khoản Google
3. Nhấn **"Create API Key"** hoặc **"Get API Key"**
4. Copy API key (dạng: `AIzaSy...`)

## ⚙️ Cấu hình API Key

### Cách 1: Sử dụng file .env (Khuyến nghị)

1. Tạo file `.env` trong thư mục gốc của project:

```bash
# .env
GEMINI_API_KEY='your api'
```

2. Restart backend:

```bash
# Dừng backend (Ctrl+C)
# Chạy lại
uv run python backend/run.py
```

### Cách 2: Biến môi trường (Tạm thời)

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY="your apiI"
uv run python backend/run.py
```

**Windows (CMD):**

```cmd
set GEMINI_API_KEY="your api"
uv run python backend/run.py
```

**Linux/Mac:**

```bash
export GEMINI_API_KEY="your api"
uv run python backend/run.py
```

**⚠️ Lưu ý:** Với cách 2, biến môi trường chỉ tồn tại trong session terminal hiện tại. Nếu đóng terminal và mở lại, bạn phải set lại.

## ✅ Kiểm tra cấu hình

Khi backend khởi động, xem log:

```
✓ Gemini AI service ENABLED (API key found: AIzaSyAQy...)
```

Nếu thấy:

```
✗ GEMINI_API_KEY not found in environment variables
```

→ API key chưa được load. Hãy kiểm tra lại và restart backend.

## 🧪 Test API

Sau khi cấu hình, thực hiện đánh giá từ frontend. Trong trang kết quả, bạn sẽ thấy:

- **Section mới**: "Hướng điều trị đề xuất (AI)"
- **Button toggle**: "Hiện/Ẩn khuyến nghị"

Nội dung sẽ được tạo tự động bởi Gemini dựa trên:

- Thông tin cá nhân (tuổi, giới tính, BMI)
- Thói quen sinh hoạt (giấc ngủ, màn hình, stress)
- Triệu chứng báo cáo
- Kết quả đánh giá nguy cơ

## 🔒 Bảo mật

- **Không commit** file `.env` vào Git (đã được thêm vào `.gitignore`)
- **Không share** API key công khai
- Sử dụng API key riêng cho mỗi môi trường (dev/staging/production)

## 💰 Chi phí

- Gemini API có **free tier** với giới hạn:
  - **gemini-2.5-flash**: 15 requests/minute, 1,500 requests/day
  - Đủ cho việc demo và development
- **Lưu ý**:
  - Gemini 2.5 là model **mới nhất** (June 2025)
  - 2x nhanh hơn và thông minh hơn Gemini 2.0
  - Gemini 1.5 đã bị **deprecated**
- Xem chi tiết: https://ai.google.dev/pricing

## 📦 Package sử dụng

- **Package**: `google-genai` (phiên bản mới)
- **Lưu ý**: Package cũ `google-generativeai` đã bị deprecated
- Nếu bạn vẫn thấy cảnh báo về package cũ, chạy:
  ```bash
  pip uninstall -y google-generativeai
  pip install google-genai
  ```

## 🐛 Troubleshooting

### Backend không nhận API key

1. Kiểm tra file `.env` có nằm trong thư mục gốc project không
2. Restart lại backend hoàn toàn (kill process và chạy lại)
3. Xem log khi startup để xác nhận status

### Frontend không hiển thị hướng điều trị

1. Mở DevTools (F12) → Console → xem có lỗi gì không
2. Kiểm tra response từ API có field `treatment_recommendations` không
3. Hard refresh trang (Ctrl+Shift+R hoặc Ctrl+F5)

### Lỗi API từ Gemini

- **"API key not valid"**: API key sai hoặc hết hạn
- **"Quota exceeded"**: Đã vượt giới hạn free tier (15/minute hoặc 1,500/day)
- **"Model not found"**: Model name sai (hiện tại dùng `gemini-2.5-flash`)
- **Lưu ý quan trọng**:
  - ✅ Gemini 2.5: Mới nhất, khuyến nghị
  - ⚠️ Gemini 2.0: Vẫn available nhưng cũ hơn
  - ❌ Gemini 1.5: Đã bị deprecated, không còn hỗ trợ

Xem danh sách models: `python list_models.py`

Xem thêm log chi tiết trong terminal chạy backend.

## 📚 Tài liệu tham khảo

- Gemini API Docs: https://ai.google.dev/docs
- Get API Key: https://makersuite.google.com/app/apikey
- Pricing: https://ai.google.dev/pricing
