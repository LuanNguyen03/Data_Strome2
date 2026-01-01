# 01) Data Standardization — Kế hoạch chuẩn hoá dữ liệu (Demo)

Ngày: 26/12/2025  
Mục tiêu: Chuẩn hoá dữ liệu để dùng thống nhất cho OLAP (DuckDB) và mô hình 2-stage (Screening → Triage), cho phép input thiếu trường nhưng vẫn dự đoán được.

---

## 1. Mục tiêu & nguyên tắc

### 1.1 Mục tiêu

- Tạo **single source of truth**: `clean_assessments.parquet`
- Đảm bảo dữ liệu:
  - dễ query OLAP nhanh (DuckDB + Parquet)
  - dễ train & infer ML (schema ổn định)
  - có audit chất lượng dữ liệu (missing/validity/bp parse)

### 1.2 Nguyên tắc

- Không “sửa bừa” dữ liệu: out-of-range → NULL + flag
- Không impute trong OLAP: giữ NULL để phân tích thật
- Derived columns làm ngay từ đầu để OLAP/model dùng thống nhất

---

## 2. Input/Output

### 2.1 Input

- `Dry_Eye_Dataset.csv` (raw)

### 2.2 Output bắt buộc

1. `data/standardized/clean_assessments.parquet`
2. `data/standardized/data_quality_report.json` (và/hoặc `.md`)

---

## 3. Chuẩn hoá schema

### 3.1 Naming convention

- Đổi tất cả tên cột sang `snake_case`
- Quy tắc:
  - chữ thường
  - khoảng trắng → `_`
  - bỏ ký tự đặc biệt
  - không đổi nghĩa, chỉ đổi format

### 3.2 Chuẩn hoá kiểu dữ liệu (data types)

#### A) Binary Y/N → 0/1

Áp dụng cho toàn bộ cột nhị phân:

- sleep_disorder, wake_up_during_night, feel_sleepy_during_day
- caffeine_consumption, alcohol_consumption, smoking
- medical_issue, ongoing_medication
- smart_device_before_bed, bluelight_filter
- discomfort_eyestrain, redness_in_eye, itchiness_irritation_in_eye
- dry_eye_disease (target)

Mapping:

- N → 0
- Y → 1

#### B) Gender

- F → 0
- M → 1

#### C) Numeric

Ép kiểu rõ ràng:

- age (int)
- height (int, cm)
- weight (int, kg)
- sleep_duration (float, hours)
- average_screen_time (float, hours/day)
- sleep_quality (int 1–5)
- stress_level (int 1–5)
- heart_rate (int)
- daily_steps (int)
- physical_activity (int)

---

## 4. Chuẩn hoá Blood Pressure

### 4.1 Lưu raw để audit

- `blood_pressure_raw`: string (giữ nguyên dữ liệu gốc)

### 4.2 Parse sang numeric

- systolic (int)
- diastolic (int)
- bp_parse_ok (0/1)

### 4.3 Hành vi khi parse fail

- systolic/diastolic = NULL
- bp_parse_ok = 0
- báo cáo bp_parse_ok_rate trong data_quality_report

---

## 5. Validation rules & flags

### 5.1 Mục tiêu

- Phát hiện dữ liệu “không hợp lý”
- Không chỉnh sửa “ép” → dùng NULL + flag để minh bạch

### 5.2 Range rules (demo-friendly)

- age: 18–45
- sleep_quality: 1–5
- stress_level: 1–5
- sleep_duration: 0–24
- average_screen_time: 0–24
- heart_rate: 40–220
- daily_steps: 0–50,000
- physical_activity: 0–600
- height: 120–230
- weight: 30–250
- systolic: 70–250
- diastolic: 40–150

### 5.3 Khi out-of-range

- set NULL
- tạo flag `<col>_in_range` (0/1)

### 5.4 validity_ratio

- validity_ratio = mean(all `<col>_in_range`)
- dùng để:
  - hiểu “độ sạch” từng dòng
  - giải thích confidence của model

---

## 6. Derived columns (chuẩn cho OLAP + model)

### 6.1 BMI

- bmi = weight / (height_m^2)

### 6.2 Banding (chuẩn cho pivot/heatmap)

- age_band:
  - 18–24, 25–29, 30–34, 35–39, 40–45
- screen_time_band:
  - 0–2, 2–4, 4–6, 6–8, 8–10, 10+
- sleep_duration_band (gợi ý):
  - <6, 6–7, 7–8, 8–9, 9+
- symptom_score:
  - symptom_score = discomfort_eyestrain + redness_in_eye + itchiness_irritation_in_eye (0..3)

---

## 7. Missing data policy

### 7.1 OLAP/BI

- KHÔNG impute
- Giữ NULL, báo cáo missing rate theo cột và theo nhóm

### 7.2 ML (để pipeline ML xử lý)

- numeric: median
- binary: mode
- missing indicators:
  - miss_bp (bp_parse_ok=0)
  - miss_screen_time
  - miss_sleep_quality
  - miss_symptom_group (thiếu bất kỳ symptom nào)

---

## 8. Data quality report (nội dung phải có)

- rows, cols
- class balance: DED positive rate
- bp_parse_ok_rate
- missing rate theo từng cột (top 10)
- avg_validity_ratio + phân phối validity_ratio
- out-of-range counts theo từng cột

---

## 9. Definition of Done (DoD)

- [ ] 100% cột đã snake_case
- [ ] Binary không còn Y/N text
- [ ] Có systolic/diastolic/bp_parse_ok
- [ ] Có derived columns: bmi, bands, symptom_score
- [ ] Có data_quality_report đầy đủ để demo
