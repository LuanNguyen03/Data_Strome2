# 02) OLAP Demo bằng DuckDB (Không dùng SQL server)

Ngày: 26/12/2025  
Mục tiêu: Dùng DuckDB đọc Parquet để tạo KPI aggregates nhanh, phục vụ dashboard và giải thích mô hình.

---

## 1. Mục tiêu OLAP demo

- Tạo “mini analytical warehouse” để:
  - pivot/heatmap nhanh
  - giải thích xu hướng risk (screen + sleep + stress)
  - justify triage bằng symptom_score
  - kiểm soát chất lượng dữ liệu theo nhóm (missing/validity)

---

## 2. Vì sao chọn DuckDB

- Embedded OLAP engine: không cần server, chạy local
- Query Parquet nhanh, pushdown tốt
- Xuất kết quả ra parquet/csv cho dashboard
- Giữ tư duy SQL/OLAP để dễ chuyển sang DWH thật

---

## 3. Chiến lược triển khai (Chốt Option A)

### Option A (demo nhanh, ít rủi ro)

- Query trực tiếp `clean_assessments.parquet`
- Sinh aggregates file-based
- Dashboard đọc trực tiếp aggregates

Lý do chọn:

- Demo cần ổn định, không cần join star schema phức tạp
- Vẫn đúng OLAP: KPI = aggregation theo dimension/band

---

## 4. Bộ KPI bắt buộc (để dashboard vẽ ngay)

Quy ước output của mọi KPI:

- n = count(\*)
- positives = sum(dry_eye_disease)
- rate = positives / n

---

### KPI 1 — DED rate by age_band × gender

**Câu hỏi:** Nhóm tuổi nào (theo giới) có tỷ lệ cao hơn?  
**Insight mong muốn:** Có phân tầng rủi ro theo nhóm.

**Output fields:**

- age_band, gender, n, positives, rate

---

### KPI 2 — Heatmap: screen_time_band × sleep_quality

**Câu hỏi:** Screen cao + ngủ kém có xu hướng tăng DED không?  
**Insight mong muốn:** Trình bày “risk domain” dễ hiểu.

**Output fields:**

- screen_time_band, sleep_quality, n, positives, rate

---

### KPI 3 — DED rate by symptom_score (0..3)

**Câu hỏi:** symptom càng nhiều tỷ lệ DED càng tăng?  
**Insight mong muốn:** justify Stage B triage.

**Output fields:**

- symptom_score, n, positives, rate

---

### KPI 4 — DED rate by stress_level × sleep_duration_band

**Câu hỏi:** Stress cao kết hợp ngủ ít có trend tăng không?  
**Insight mong muốn:** gợi ý can thiệp hành vi.

**Output fields:**

- stress_level, sleep_duration_band, n, positives, rate

---

### KPI 5 — Data quality by group

**Câu hỏi:** Nhóm nào thiếu dữ liệu nhiều? validity thấp?  
**Insight mong muốn:** giải thích confidence, tránh bias do missing.

**Output fields (gợi ý):**

- age_band, gender
- missing_rate_screen_time
- missing_rate_sleep_quality
- missing_rate_bp
- avg_validity_ratio
- n

---

## 5. Deliverables OLAP demo

- analytics/duckdb/agg/agg_ded_by_age_gender.parquet
- analytics/duckdb/agg/agg_ded_by_screen_sleep.parquet
- analytics/duckdb/agg/agg_ded_by_symptom_score.parquet
- analytics/duckdb/agg/agg_ded_by_stress_sleepband.parquet
- analytics/duckdb/agg/agg_data_quality_by_group.parquet

Kèm theo:

- docs/olap_queries.md (KPI → câu hỏi → logic → ý nghĩa)
- docs/olap_summary.md (5–8 insight slide-ready)

---

## 6. Definition of Done (DoD)

- [ ] Có đủ 5 KPI dataset
- [ ] Heatmap dataset chạy ổn (không quá thưa)
- [ ] Có insight ngắn gọn: 3 insight chính + 2 insight phụ
- [ ] Có bảng data quality by group để giải thích confidence
