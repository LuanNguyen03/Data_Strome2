# Metrics & Reporting Spec — Dashboard, KPI, và cách diễn giải

Mục tiêu:

- Xác định dashboard cần gì để demo thuyết phục
- KPI nào là “core”, KPI nào là “support”
- Quy chuẩn biểu đồ + cách đọc để tránh hiểu sai (nhân quả vs tương quan)

---

## 1) Nguyên tắc trình bày

- KPI phải trả lời câu hỏi cụ thể
- Số liệu phải có n (mẫu) để tránh hiểu sai
- Luôn nhấn mạnh: đây là xu hướng trong dataset demo, không kết luận nhân quả
- Nên có 1 block “data quality” để tăng niềm tin

---

## 2) Bộ dashboard đề xuất (3 tab)

### Tab A — Overview

Mục tiêu: 60 giây hiểu dataset và mức độ rủi ro chung

- Total records (n)
- DED positive rate (%)
- Missing rate top-5
- BP parse ok rate
- Avg validity_ratio

Biểu đồ:

- Bar: DED rate theo age_band
- Bar: DED rate theo gender (hoặc stacked age×gender)

---

### Tab B — Risk Drivers (Screening domain)

Mục tiêu: giải thích vì sao Stage A tập trung sleep + screen
Biểu đồ:

- Heatmap: screen_time_band × sleep_quality (rate + n)
- Heatmap: stress_level × sleep_duration_band (rate + n)
- Bar: DED rate theo screen_time_band
- Bar: DED rate theo sleep_quality

Ghi chú trình bày:

- Nếu n nhỏ trong ô heatmap → hiển thị cảnh báo “sample nhỏ”
- Có thể gộp band để ổn định

---

### Tab C — Symptom & Triage

Mục tiêu: justify Stage B khi có symptom
Biểu đồ:

- Bar/Line: DED rate theo symptom_score (0..3)
- Stacked: tỷ lệ symptom (mỗi symptom) theo DED (nếu muốn)
- Table: symptom_score vs n vs rate

Ghi chú trình bày:

- “Symptom gần nhãn” → triage mạnh hơn, không dùng cho dự báo sớm

---

## 3) KPI catalog (core vs support)

### 3.1 Core KPI (bắt buộc demo)

1. DED positive rate (overall)
2. DED rate by age_band × gender
3. Heatmap: screen_time_band × sleep_quality
4. DED rate by symptom_score
5. Data quality: missing rate + validity_ratio

### 3.2 Support KPI (nếu còn thời gian)

- DED rate by stress_level
- DED rate by sleep_duration_band
- DED rate by screen_time_band riêng lẻ
- Missing by group (age_band/gender) chi tiết hơn

---

## 4) Reporting chuẩn cho mỗi chart

Mỗi biểu đồ nên có:

- Title: nêu câu hỏi (không chỉ “Biểu đồ A”)
- Subtitle: ghi “n=” và note về missing
- Tooltip (nếu UI có): rate + positives + n
- Footnote: “tương quan, không kết luận nhân quả”

---

## 5) Data Quality Panel (đề xuất hiển thị)

- Missing rate top-5 (bar)
- BP parse ok rate (single KPI card)
- Avg validity_ratio (card)
- Missing by group (table hoặc bar)
- Alert rules:
  - Nếu missing_rate_screen_time > X% → warning
  - Nếu bp_parse_ok_rate < Y% → warning

---

## 6) Model reporting (khi có metrics)

Khi bạn chạy training thật (sau demo plan), nên có:

- ROC-AUC, PR-AUC cho Stage A và Stage B
- Recall(Y) cho Stage A tại threshold
- Precision/Recall cho Stage B tại threshold
- Calibration note (nếu hiển thị %)

---

## 7) Definition of Done (Reporting)

- [ ] Có 3 tab (Overview, Risk Drivers, Symptom & Triage)
- [ ] Mỗi chart có n + rate + note
- [ ] Có Data Quality panel để tăng độ tin cậy
- [ ] Người nghe hiểu: “Stage A = risk sớm”, “Stage B = triage”
