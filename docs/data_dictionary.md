# Data Dictionary — Từ điển dữ liệu (Chuẩn hoá + OLAP + Modeling)

Mục tiêu:

- Thống nhất ý nghĩa từng cột
- Quy định kiểu dữ liệu, range, missing policy
- Làm nền cho data contract và UI form

---

## 1) Nhóm định danh & audit (không định danh cá nhân)

### request_id (optional, production)

- Type: string
- Ý nghĩa: mã yêu cầu để audit log
- Missing: cho demo có thể không cần
- Ghi chú: production nên có

### timestamp (optional, production)

- Type: string (ISO8601)
- Ý nghĩa: thời điểm ghi nhận
- Missing: demo có thể bỏ

---

## 2) Nhóm nhân khẩu học (Person)

### age

- Type: int
- Range: 18–45 (dataset demo)
- Missing handling:
  - OLAP: giữ NULL
  - ML: impute median + flag miss_age
- Ghi chú: nếu dùng rộng hơn, mở range

### gender

- Type: int (0/1)
- Mapping: F=0, M=1
- Missing:
  - OLAP: NULL
  - ML: impute mode + flag miss_gender

### height

- Type: int (cm)
- Range: 120–230
- Missing: OLAP NULL, ML median + flag

### weight

- Type: int (kg)
- Range: 30–250
- Missing: OLAP NULL, ML median + flag

### bmi (derived)

- Type: float
- Formula: weight / (height_m^2)
- Missing: nếu height/weight missing → bmi NULL

### age_band (derived)

- Type: category/string
- Buckets: 18–24, 25–29, 30–34, 35–39, 40–45
- Missing: nếu age NULL → age_band NULL

---

## 3) Nhóm ngủ (Sleep)

### sleep_duration

- Type: float (hours)
- Range: 0–24
- Missing: OLAP NULL, ML median + miss_sleep_duration

### sleep_quality

- Type: int (1–5)
- Range: 1–5
- Missing: OLAP NULL, ML median/mode + miss_sleep_quality
- Ghi chú: biến quan trọng cho screening

### sleep_disorder

- Type: int (0/1)
- Meaning: có rối loạn giấc ngủ không
- Missing: OLAP NULL, ML mode + flag

### wake_up_during_night

- Type: int (0/1)
- Meaning: hay thức giấc ban đêm
- Missing: OLAP NULL, ML mode + flag

### feel_sleepy_during_day

- Type: int (0/1)
- Meaning: buồn ngủ ban ngày
- Missing: OLAP NULL, ML mode + flag

### sleep_duration_band (derived)

- Type: category/string
- Buckets gợi ý: <6, 6–7, 7–8, 8–9, 9+
- Missing: nếu sleep_duration NULL → NULL

---

## 4) Nhóm màn hình & thiết bị (Device / Screen)

### average_screen_time

- Type: float (hours/day)
- Range: 0–24
- Missing: OLAP NULL, ML median + miss_screen_time
- Ghi chú: biến quan trọng cho screening

### smart_device_before_bed

- Type: int (0/1)
- Meaning: dùng thiết bị trước khi ngủ
- Missing: OLAP NULL, ML mode + flag

### bluelight_filter

- Type: int (0/1)
- Meaning: dùng lọc ánh sáng xanh
- Missing: OLAP NULL, ML mode + flag

### screen_time_band (derived)

- Type: category/string
- Buckets: 0–2, 2–4, 4–6, 6–8, 8–10, 10+
- Missing: nếu average_screen_time NULL → NULL

---

## 5) Nhóm thói quen & vận động (Lifestyle)

### stress_level

- Type: int (1–5)
- Range: 1–5
- Missing: OLAP NULL, ML median + miss_stress_level

### daily_steps

- Type: int
- Range: 0–50,000
- Missing: OLAP NULL, ML median + miss_daily_steps

### physical_activity

- Type: int
- Range: 0–600 (dataset demo)
- Missing: OLAP NULL, ML median + miss_physical_activity

### caffeine_consumption

- Type: int (0/1)
- Missing: OLAP NULL, ML mode + flag

### alcohol_consumption

- Type: int (0/1)
- Missing: OLAP NULL, ML mode + flag

### smoking

- Type: int (0/1)
- Missing: OLAP NULL, ML mode + flag

---

## 6) Nhóm sinh hiệu (Vitals)

### blood_pressure_raw

- Type: string
- Meaning: raw input (audit)
- Missing: ok

### systolic (derived)

- Type: int
- Range: 70–250
- Missing: nếu parse fail → NULL

### diastolic (derived)

- Type: int
- Range: 40–150
- Missing: nếu parse fail → NULL

### bp_parse_ok (derived)

- Type: int (0/1)
- Meaning: parse huyết áp thành công
- Missing: không (luôn có 0/1)

### heart_rate

- Type: int
- Range: 40–220
- Missing: OLAP NULL, ML median + miss_heart_rate

---

## 7) Nhóm bệnh nền & thuốc (Medical)

### medical_issue

- Type: int (0/1)
- Meaning: có bệnh nền liên quan
- Missing: OLAP NULL, ML mode + flag

### ongoing_medication

- Type: int (0/1)
- Meaning: đang dùng thuốc
- Missing: OLAP NULL, ML mode + flag

---

## 8) Nhóm triệu chứng mắt (Symptoms)

> Chỉ dùng cho Stage B (triage). Stage A không dùng để tránh leakage.

### discomfort_eyestrain

- Type: int (0/1)
- Missing: OLAP NULL, ML mode + miss_symptom_group (nếu thiếu)

### redness_in_eye

- Type: int (0/1)
- Missing: tương tự

### itchiness_irritation_in_eye

- Type: int (0/1)
- Missing: tương tự

### symptom_score (derived)

- Type: int (0..3)
- Formula: sum(3 symptom)
- Missing: nếu thiếu symptom → symptom_score NULL (hoặc compute với rule rõ ràng)

---

## 9) Nhãn mục tiêu (Target)

### dry_eye_disease

- Type: int (0/1)
- Meaning: label trong dataset
- Note: production không có label realtime (chỉ dùng đánh giá nếu có follow-up)

---

## 10) Các cột chất lượng dữ liệu (Data Quality)

### <col>\_in_range (flags)

- Type: int (0/1)
- Meaning: giá trị hợp lệ theo range hoặc NULL

### validity_ratio

- Type: float (0..1)
- Meaning: tỷ lệ hợp lệ của dòng
- Use: ảnh hưởng confidence

---

## 11) Data Contract summary (gợi ý)

- Bắt buộc tối thiểu để Stage A có giá trị:
  - age, gender (nếu có)
  - sleep_duration OR sleep_quality (tốt nhất cả 2)
  - average_screen_time
  - stress_level (nếu có)
- Nếu thiếu các nhóm chủ lực → confidence giảm và gợi ý điền thêm
