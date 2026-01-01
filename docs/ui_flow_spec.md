# UI Flow Spec — Luồng giao diện (Stage A → hỏi Symptom → Stage B)

Mục tiêu:

- UX rõ ràng, không gây hoảng loạn
- Đúng chuẩn y tế: “screening/triage support”
- Hỗ trợ input thiếu trường nhưng có confidence và gợi ý bổ sung

---

## 1) Trang / Flow tổng

### Screen 1: Quick Assessment (Stage A)

- Nhập thông tin tối thiểu cho screening
- Nhấn “Xem kết quả sàng lọc”

### Screen 2: Symptoms (Stage B trigger)

- Chỉ xuất hiện khi:
  - user chủ động chọn “Tôi có triệu chứng”
  - hoặc hệ thống trigger vì risk_A cao

### Screen 3: Result

- Hiển thị kết quả theo mode:
  - A_only_screening hoặc B_with_symptoms
- Kèm top_factors + next_step + disclaimers

---

## 2) Screen 1 — Quick Assessment (Stage A)

### 2.1 Nhóm field hiển thị (gợi ý UX)

**Nhóm bắt buộc (recommended):**

- average_screen_time (giờ/ngày)
- sleep_duration (giờ)
- sleep_quality (1–5)
- stress_level (1–5)

**Nhóm tùy chọn nhưng hữu ích:**

- age, gender
- smart_device_before_bed, bluelight_filter
- caffeine/alcohol/smoking
- daily_steps, physical_activity
- heart_rate
- blood_pressure_raw (hoặc tách systolic/diastolic)

### 2.2 UX rule

- Mỗi nhóm có tooltip 1 dòng: “để tăng độ chính xác”
- Nếu bỏ trống field quan trọng → vẫn cho submit nhưng cảnh báo nhẹ:
  - “Thiếu sleep_quality, kết quả sẽ kém chắc hơn”

### 2.3 Output ngay sau Stage A

Hiển thị:

- risk_score + risk_level
- confidence badge
- top_factors (3–5 gạch đầu dòng)
- next_step (2–4 gạch đầu dòng)
- nút CTA:
  - “Trả lời thêm triệu chứng để phân loại rõ hơn” (nếu trigger_symptom=true)

---

## 3) Trigger hỏi symptom (router)

### 3.1 Khi nào hỏi symptom

- Nếu Stage A risk_score >= 65 (default demo)
- Hoặc user tick “Tôi đang có triệu chứng”

### 3.2 Cách hỏi để không gây hoảng

- “Để tăng độ chắc của phân loại, bạn có gặp các triệu chứng sau không?”
- Không dùng từ “bệnh nặng/đáng sợ”

---

## 4) Screen 2 — Symptoms (Stage B)

### 4.1 Field

- discomfort_eyestrain (0/1)
- redness_in_eye (0/1)
- itchiness_irritation_in_eye (0/1)

### 4.2 UX

- Mỗi symptom có mô tả 1 dòng (đời thường)
- Có nút “Bỏ qua” (vẫn giữ kết quả Stage A)

---

## 5) Screen 3 — Result (A hoặc B)

### 5.1 Khối hiển thị chính

- Score (0–100) + level
- Confidence badge
- Mode label nhỏ:
  - “Sàng lọc (không triệu chứng)” hoặc “Phân loại (có triệu chứng)”

### 5.2 Top factors (giải thích)

- Hiển thị dạng bullet ngắn:
  - “Screen time cao”
  - “Chất lượng ngủ thấp”
  - “Symptom score cao” (chỉ mode B)

### 5.3 Next step (hành động)

Mapping theo level:

- Low:
  - “Theo dõi + nghỉ mắt định kỳ”
- Medium:
  - “Theo dõi + cải thiện thói quen + cân nhắc khám nếu kéo dài”
- High:
  - “Khuyến nghị cân nhắc khám chuyên khoa (đặc biệt nếu kéo dài)”

### 5.4 Disclaimers (bắt buộc)

- “Kết quả chỉ hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.”
- “Nếu triệu chứng kéo dài/nặng, nên tham khảo bác sĩ.”

---

## 6) Handling missing data & confidence trong UI

### 6.1 Hiển thị missing_fields

- Nếu confidence != High, hiển thị:
  - “Thiếu thông tin: sleep_quality, average_screen_time”
  - “Bạn có thể bổ sung để tăng độ chắc”

### 6.2 Gợi ý “đáng điền nhất”

- Nếu Stage A risk cao: gợi ý symptom
- Nếu thiếu screen/sleep: gợi ý 2 field đó

---

## 7) Copywriting style (quan trọng)

Nên dùng:

- “nguy cơ”, “sàng lọc”, “phân loại”, “theo dõi”, “cân nhắc khám”

Tránh dùng:

- “chẩn đoán”, “chắc chắn mắc”, “kết luận bệnh”

---

## 8) Definition of Done (UI)

- [ ] Stage A form submit được dù thiếu field (có cảnh báo nhẹ)
- [ ] Confidence badge + missing_fields hiển thị rõ
- [ ] Router trigger symptom rõ ràng
- [ ] Mode A/B hiển thị khác nhau, có disclaimers bắt buộc
