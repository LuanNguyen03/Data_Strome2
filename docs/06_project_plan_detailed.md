# Project Plan Detailed (Demo)

---

## 1) Scope

### In-scope

- Standardization → clean Parquet + data quality report
- DuckDB OLAP → aggregates parquet/csv + olap summary
- Medical-standard modeling plan → stage A/B + router + evaluation plan

### Out-of-scope

- Production backend full
- SQL server / DWH thật
- Clinical deployment

---

## 2) Deliverables

### Data

- clean_assessments.parquet
- data_quality_report.json/md

### OLAP

- 5 agg parquet/csv
- olap_queries.md
- olap_summary.md

### Modeling plan

- 03_medical_modeling_plan.md
- threshold strategy notes (optional)
- output schema spec (optional)

---

## 3) Timeline (gợi ý 1–2 ngày)

- Day 1: schema + standardization plan + data quality report template
- Day 2: OLAP KPI plan + olap_summary + storyboard

---

## 4) Definition of Done

- Demo chạy theo storyboard, có đủ file agg, có insight, có output schema A/B rõ ràng
