# 03) Medical-Standard Modeling Plan — 2-Stage (Screening → Triage)

Ngày: 26/12/2025  
Mục tiêu: Thiết kế mô hình “đúng bài y tế” tách rõ nhiệm vụ, tránh leakage, xử lý thiếu dữ liệu tốt và trả output có trách nhiệm.

---

## 1. Nguyên tắc y tế

- Screening (dự báo sớm) và Triage (phân loại khi có symptom) là 2 nhiệm vụ khác nhau.
- Không dùng symptom cho Screening để tránh “ăn gian” (leakage).
- Output phải đi kèm confidence và next_step, không “phán bệnh”.

---

## 2. Tổng quan 2-stage cascade

### Stage A — Screening (No symptoms)

- Dự báo nguy cơ dựa trên hành vi và lối sống.

### Stage B — Triage (With symptoms)

- Khi có symptom, hỗ trợ phân loại tốt hơn.

### Router

- Nếu thiếu symptom → chạy A.
- Nếu risk_A cao → hỏi symptom → chạy B.
- Nếu đã có symptom → ưu tiên B.

---

## 3. Stage A — Screening (không symptom)

### 3.1 Mục tiêu

- Phát hiện nguy cơ sớm (ưu tiên recall).
- Trả về risk_score dễ hiểu.

### 3.2 Feature groups (dùng)

- Sleep: sleep_duration, sleep_quality, sleep_disorder, wake_up_during_night, feel_sleepy_during_day
- Device: average_screen_time, smart_device_before_bed, bluelight_filter
- Lifestyle: stress_level, daily_steps, physical_activity, caffeine/alcohol/smoking
- Person: age, gender, bmi (hoặc height/weight)
- Vitals: systolic, diastolic, heart_rate
- Medical: medical_issue, ongoing_medication

### 3.3 Feature groups (không dùng)

- discomfort_eyestrain, redness_in_eye, itchiness_irritation_in_eye
- symptom_score

### 3.4 Output schema (bắt buộc)

- mode_used = "A_only_screening"
- risk_score (0–100)
- risk_level: Low/Medium/High
- confidence: High/Medium/Low
- missing_fields: [...]
- top_factors: 3–5 yếu tố
- next_step:
  - “theo dõi”
  - “điền thêm symptom”
  - “cân nhắc khám” (nếu risk rất cao)

---

## 4. Stage B — Triage (có symptom)

### 4.1 Mục tiêu

- Phân loại khi có symptom, cân bằng precision/recall.
- Giải thích rõ hơn vì đã có thông tin “gần bệnh”.

### 4.2 Feature groups (dùng)

- Toàn bộ Stage A + symptom:
  - discomfort_eyestrain
  - redness_in_eye
  - itchiness_irritation_in_eye
  - symptom_score (0..3) (optional)

### 4.3 Output schema (bắt buộc)

- mode_used = "B_with_symptoms"
- probability (0–100)
- triage_level: Low/Medium/High
- confidence: High/Medium/Low
- missing_fields: [...]
- top_factors
- next_step:
  - “theo dõi + điều chỉnh thói quen”
  - “nên khám” khi vượt ngưỡng triage

---

## 5. Router logic (chi tiết)

### 5.1 Trigger để hỏi symptom

- Nếu Stage A risk_score >= threshold_trigger (gợi ý 65/100):
  - yêu cầu nhập 3 symptom để chạy Stage B
- Nếu user chủ động nhập symptom:
  - chạy Stage B ngay

### 5.2 Lý do y tế

- Risk cao nhưng chưa có symptom: cần thêm thông tin để tăng độ chắc, tránh báo động giả.

---

## 6. Missing data policy + confidence

### 6.1 Nguyên tắc

- Không chặn dự đoán vì thiếu dữ liệu (demo thực tế).
- Trả missing_fields + confidence.

### 6.2 Confidence policy (gợi ý)

- High: thiếu <= 10% trường quan trọng
- Medium: thiếu 10–30%
- Low: thiếu > 30% hoặc thiếu nhóm chủ lực (sleep/screen)

### 6.3 Gợi ý field nên bổ sung

- Nếu đang A và risk cao: ưu tiên symptom
- Nếu thiếu sleep_quality hoặc screen_time: ưu tiên 2 field này

---

## 7. Evaluation plan (demo đúng bài)

### 7.1 Metrics

- ROC-AUC, PR-AUC
- Recall(Y) cho Stage A
- Precision/Recall cân bằng cho Stage B

### 7.2 Thresholding

- Stage A: chọn threshold ưu tiên recall (screening không bỏ sót)
- Stage B: threshold cân bằng theo mục tiêu triage

### 7.3 Calibration (nếu trả %)

- Có kế hoạch calibrate để xác suất đáng tin (đặc biệt nếu nói “70%”).

---

## 8. Definition of Done (DoD)

- [ ] Stage A/B tách nhiệm vụ rõ, tránh leakage
- [ ] Router logic rõ, giải thích được
- [ ] Output schema có confidence + next_step
- [ ] Có plan đánh giá và chọn threshold
