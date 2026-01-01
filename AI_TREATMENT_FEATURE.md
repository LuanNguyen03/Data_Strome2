# Tính năng Hướng điều trị AI - Tổng quan

## 📋 Mô tả

Hệ thống đã được tích hợp với **Gemini AI** của Google để cung cấp hướng điều trị và lời khuyên cá nhân hóa cho người dùng sau mỗi lần đánh giá nguy cơ.

## ✨ Tính năng

### 1. **Khuyến nghị cá nhân hóa**
- Dựa trên thông tin cá nhân (tuổi, giới tính, BMI)
- Phân tích thói quen sinh hoạt (giấc ngủ, thời gian màn hình, stress)
- Xem xét triệu chứng đã báo cáo
- Căn cứ vào kết quả đánh giá nguy cơ

### 2. **Giao diện thân thiện**
- Section riêng: "Hướng điều trị đề xuất (AI)"
- Button toggle để hiện/ẩn khuyến nghị
- Animation mượt mà khi hiển thị
- Tự động hiển thị nếu có khuyến nghị từ AI

### 3. **Fallback thông minh**
- Nếu không có API key: Hiển thị thông báo hướng dẫn cấu hình
- Nếu có lỗi API: Hiển thị khuyến nghị chung + thông báo lỗi
- Hệ thống vẫn hoạt động bình thường kể cả khi AI bị tắt

## 🎯 Cách sử dụng

### Cho người dùng cuối:

1. Thực hiện đánh giá như bình thường
2. Xem kết quả đánh giá
3. Tìm section **"Hướng điều trị đề xuất (AI)"**
4. Click button **"Hiện khuyến nghị"** để xem chi tiết
5. Click **"Ẩn khuyến nghị"** để thu gọn

### Cho admin/developer:

1. **Cấu hình API key** (xem [GEMINI_SETUP.md](./GEMINI_SETUP.md))
   - Sử dụng script: `setup_gemini.bat` (Windows) hoặc `setup_gemini.sh` (Linux/Mac)
   - Hoặc tạo file `.env` thủ công với `GEMINI_API_KEY=your_key`

2. **Restart backend**
   ```bash
   # Dừng backend hiện tại (Ctrl+C)
   # Chạy lại
   uv run python backend/run.py
   ```

3. **Kiểm tra log**
   Nên thấy: `✓ Gemini AI service ENABLED`

## 📁 Files đã thay đổi

### Backend

1. **`backend/services/gemini_service.py`** (MỚI)
   - Service xử lý gọi API Gemini
   - Build prompt cá nhân hóa dựa trên dữ liệu người dùng
   - Xử lý lỗi và fallback

2. **`backend/services/assessment_service.py`**
   - Chuyển `assess()` sang async
   - Tích hợp GeminiService
   - Gọi API sau khi đánh giá xong

3. **`backend/api/v1/assessments.py`**
   - Cập nhật endpoint thành async
   - Await khi gọi service.assess()

4. **`backend/routers/assessment.py`**
   - Cập nhật endpoint thành async

5. **`backend/main.py`**
   - Thêm log kiểm tra Gemini service khi startup
   - Hiển thị tip nếu chưa cấu hình

### Frontend

1. **`frontend/src/pages/Result.tsx`**
   - Thêm state `showTreatment`
   - Thêm section "Hướng điều trị đề xuất (AI)"
   - Button toggle hiện/ẩn
   - Parse và hiển thị markdown từ AI

2. **`frontend/src/pages/Result.css`**
   - Styles cho treatment section
   - Styles cho toggle button
   - Animation fadeIn

3. **`frontend/src/types.ts`**
   - Thêm field `treatment_recommendations?: string`

### Contracts

1. **`contracts/schemas.py`**
   - Thêm field `treatment_recommendations` vào `AssessmentResponse`

### Documentation

1. **`GEMINI_SETUP.md`** (MỚI)
   - Hướng dẫn chi tiết cấu hình Gemini API
   - Troubleshooting
   - Best practices

2. **`AI_TREATMENT_FEATURE.md`** (MỚI)
   - Tổng quan tính năng
   - Kiến trúc
   - Hướng dẫn sử dụng

3. **`setup_gemini.bat`** (MỚI)
   - Script tự động setup cho Windows

4. **`setup_gemini.sh`** (MỚI)
   - Script tự động setup cho Linux/Mac

5. **`QUICKSTART.md`**
   - Cập nhật section setup Gemini AI
   - Link đến tài liệu chi tiết

## 🏗️ Kiến trúc

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │ Điền form đánh giá
       ▼
┌─────────────────────┐
│  Frontend (React)   │
│  - QuickAssessment  │
│  - Result           │
└──────┬──────────────┘
       │ POST /api/v1/assessments/screening
       │      /api/v1/assessments/triage
       ▼
┌──────────────────────────────┐
│  Backend (FastAPI)           │
│  - assessments.py            │
│    └─> AssessmentService     │
│         ├─> ModelLoader      │
│         │   (ML predictions) │
│         │                    │
│         └─> GeminiService    │
│             (AI treatment)   │
└──────┬───────────────────────┘
       │
       ├─> ML Models (XGBoost/LightGBM)
       │   └─> Risk score + level
       │
       └─> Gemini API (Google)
           └─> Treatment recommendations
```

## 🔐 Bảo mật

- API key được load từ environment variables
- File `.env` đã được thêm vào `.gitignore`
- Không bao giờ commit API key vào Git
- Log chỉ hiển thị 10 ký tự đầu của API key

## 💰 Chi phí

- **Gemini 2.0 Flash Experimental**: Free tier
  - 1,500 requests/day
  - Đủ cho demo và development
  - Hiệu suất tốt hơn so với Gemini 1.5
- Không có chi phí bổ sung trong giai đoạn phát triển

## 🧪 Testing

### Manual Test

1. Không có API key:
   - Section hiện thông báo hướng dẫn cấu hình
   
2. Có API key hợp lệ:
   - Section hiện khuyến nghị từ AI
   - Button toggle hoạt động
   - Nội dung được format đúng

3. API key không hợp lệ/lỗi:
   - Section hiện thông báo lỗi + khuyến nghị chung
   - Hệ thống vẫn hoạt động bình thường

### Automated Test (Future)

```python
# test_gemini_service.py
async def test_gemini_enabled():
    service = GeminiService()
    assert service.enabled == True

async def test_treatment_recommendations():
    result = await service.get_treatment_recommendations(...)
    assert result is not None
    assert len(result) > 0
```

## 📊 Metrics (Future Enhancement)

Có thể theo dõi:
- Số lượng requests tới Gemini API
- Tỷ lệ thành công/thất bại
- Thời gian response trung bình
- User engagement với treatment recommendations

## 🚀 Roadmap

### Phase 2 (Future)
- [ ] Cache recommendations để giảm API calls
- [ ] Support multiple AI providers (OpenAI, Claude)
- [ ] Customize prompt templates
- [ ] A/B testing different prompts
- [ ] Feedback mechanism (helpful/not helpful)
- [ ] Export recommendations as PDF

### Phase 3 (Future)
- [ ] Multi-language support
- [ ] Voice recommendations
- [ ] Integration with telemedicine platforms

## 🐛 Known Issues

Không có known issues tại thời điểm hiện tại.

## 📞 Support

Nếu gặp vấn đề:
1. Xem [GEMINI_SETUP.md](./GEMINI_SETUP.md) - Troubleshooting section
2. Kiểm tra log backend
3. Kiểm tra console frontend (F12)
4. Mở issue trên GitHub (nếu có)

## 📚 References

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [Pricing](https://ai.google.dev/pricing)
