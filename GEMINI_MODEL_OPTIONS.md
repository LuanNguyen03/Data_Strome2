# Gemini Model Options

Hệ thống hỗ trợ nhiều model Gemini khác nhau. Bạn có thể thay đổi model trong file `backend/services/gemini_service.py`.

## 📋 Các Model khả dụng

### Gemini 2.5 (Mới nhất - Khuyến nghị)

**1. gemini-2.5-flash** ✅ (Default - Latest Stable)
- ✅ **Ưu điểm**: 
  - Mới nhất, nhanh nhất
  - Stable (không phải experimental)
  - Thông minh hơn 2.0
  - Quota tốt cho free tier
- **Free tier**: 15 requests/minute, 1,500 requests/day
- **Khuyến nghị**: ✅✅ **Tốt nhất cho mọi use case**

**2. gemini-2.5-pro**
- ✅ Chất lượng cao nhất
- ⚠️ Chậm hơn, quota thấp hơn
- **Khuyến nghị**: Production với chất lượng tối đa

### Gemini 2.0

**1. gemini-2.0-flash-exp** (Experimental - Default)
- ✅ **Ưu điểm**: Nhanh nhất, thông minh nhất
- ⚠️ **Nhược điểm**: 
  - Quota thấp hơn (có thể bị giới hạn)
  - Experimental (chưa stable)
- **Free tier**: 1,500 requests/day
- **Khuyến nghị**: Tốt cho production khi ra stable

**2. gemini-2.0-flash-thinking-exp-01-21**
- ✅ Phiên bản thinking, phù hợp với các task phân tích phức tạp
- ⚠️ Có thể chậm hơn

### Gemini 1.5 (Stable)

**3. gemini-1.5-flash** (Stable - Recommended nếu quota limited)
- ✅ **Ưu điểm**: 
  - Stable, đã test kỹ
  - Quota cao hơn
  - Nhanh, đáng tin cậy
- **Free tier**: 15 requests/minute, 1,500 requests/day
- **Khuyến nghị**: ✅ **Tốt nhất cho demo và development**

**4. gemini-1.5-pro**
- ✅ Chất lượng cao hơn flash
- ⚠️ Chậm hơn, quota thấp hơn
- **Free tier**: 2 requests/minute

**5. gemini-1.5-flash-8b**
- ✅ Nhẹ nhất, nhanh nhất
- ⚠️ Chất lượng thấp hơn

## 🔧 Cách thay đổi Model

### Bước 1: Sửa file gemini_service.py

Mở file `backend/services/gemini_service.py` và tìm dòng:

```python
response = await self.client.aio.models.generate_content(
    model='gemini-2.0-flash-exp',  # <-- Thay đổi ở đây
    contents=prompt
)
```

Thay đổi thành model bạn muốn:

```python
# Option 1: Gemini 2.0 (nếu có quota)
model='gemini-2.0-flash-exp'

# Option 2: Gemini 1.5 Flash (Stable - Khuyến nghị)
model='gemini-1.5-flash'

# Option 3: Gemini 1.5 Pro (Chất lượng cao)
model='gemini-1.5-pro'

# Option 4: Gemini 1.5 Flash 8B (Nhanh nhất)
model='gemini-1.5-flash-8b'
```

### Bước 2: Restart Backend

```bash
# Dừng backend (Ctrl+C)
# Chạy lại
uv run python backend/run.py
```

## ⚠️ Xử lý lỗi Quota Exceeded

Nếu bạn gặp lỗi:
```
429 RESOURCE_EXHAUSTED
Quota exceeded for metric
```

### Giải pháp:

**1. Đợi quota reset**
- Free tier reset mỗi phút/ngày
- Check usage tại: https://ai.dev/usage

**2. Chuyển sang model khác**
```python
# Từ gemini-2.0-flash-exp (quota thấp)
# Sang gemini-1.5-flash (quota cao hơn)
model='gemini-1.5-flash'
```

**3. Upgrade account** (nếu cần)
- Pay-as-you-go: $0.075/1M tokens (input)
- Không giới hạn quota

## 📊 So sánh Performance

| Model | Speed | Quality | Quota (Free) | Recommend |
|-------|-------|---------|--------------|-----------|
| **gemini-2.5-flash** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 15/min, 1,500/day | ✅✅✅ Best |
| gemini-2.5-pro | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 2/min | 💼 High quality |
| gemini-2.0-flash | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 15/min, 1,500/day | ✅ Stable |
| gemini-2.0-flash-exp | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Limited | ⚠️ Experimental |
| ~~gemini-1.5-flash~~ | - | - | - | ❌ Deprecated |

## 🎯 Khuyến nghị theo Use Case

### Demo & Development ✅ (Default)
```python
model='gemini-2.5-flash'  # Mới nhất, stable, quota tốt
```

### Production
```python
model='gemini-2.5-flash'  # Hoặc gemini-2.5-pro nếu cần quality cao hơn
```

### High Volume với budget
```python
model='gemini-2.5-flash'  # Best balance giữa speed và quality
```

### Testing Gemini 2.0
```python
model='gemini-2.0-flash-exp'  # Khi muốn thử experimental features
```

## 🆕 So sánh Gemini 2.5 vs 2.0

| Feature | Gemini 2.5 | Gemini 2.0 |
|---------|------------|------------|
| Speed | 2x faster | Fast |
| Multimodal | ✅ Better | ✅ Good |
| Reasoning | ✅ Improved | Good |
| Context window | 2M tokens | 1M tokens |
| Stability | ✅ Stable | Stable |
| Release | June 2025 | Dec 2024 |

## 🔍 Check Quota hiện tại

```bash
# Truy cập
https://ai.dev/usage?tab=rate-limit

# Hoặc chạy test
python check_gemini.py --test-api
```

## 📚 Tài liệu

- [Model comparison](https://ai.google.dev/gemini-api/docs/models/gemini)
- [Pricing & Quotas](https://ai.google.dev/pricing)
- [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
