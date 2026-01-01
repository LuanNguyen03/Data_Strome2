# Demo Storyboard (10–15 phút)

Mục tiêu: demo mạch lạc, không bị “AI nói lan man”, nhấn đúng chuẩn y tế.

---

## 1) Mở đầu (30–45s)

- “Bài toán y tế có 2 phần: Screening sớm và Triage khi có triệu chứng.”
- “Demo này hỗ trợ sàng lọc/triage, không thay thế chẩn đoán bác sĩ.”

---

## 2) Chuẩn hoá dữ liệu (2 phút)

- Cho thấy pipeline: Raw CSV → clean Parquet
- Trình bày data quality:
  - missing rate
  - bp parse ok rate
  - validity_ratio
  - class balance

Key message:

- “Dữ liệu sạch và minh bạch giúp mô hình đáng tin hơn.”

---

## 3) OLAP (DuckDB) — Insight nhanh (4 phút)

- KPI 1: age_band × gender
- KPI 2: heatmap screen×sleep
- KPI 3: symptom_score
- KPI 5: data quality by group

Key message:

- “OLAP giúp giải thích mô hình, không phải AI nói theo cảm giác.”

---

## 4) Demo Stage A (Screening) (3 phút)

- Input: sleep + screen + stress + age/gender…
- Output: risk_score + confidence + top_factors
- Nếu risk cao: hệ thống gợi ý nhập symptom

Key message:

- “Stage A không dùng symptom để tránh ăn gian.”

---

## 5) Router → hỏi symptom → Stage B (Triage) (3 phút)

- User nhập 3 symptom
- Output: probability/triage + confidence + next_step

Key message:

- “Stage B là triage, không phải dự báo sớm.”

---

## 6) Kết thúc (1 phút)

- “Muốn lên production: external validation + clinical review + monitoring drift/calibration.”
