# OLAP Analytics Overview - DuckDB Implementation

## 📋 Tổng quan

Hệ thống sử dụng **DuckDB** (embedded OLAP engine) để tạo các KPI aggregates và phân tích dữ liệu nhanh chóng mà không cần SQL server.

---

## 🎯 Mục tiêu OLAP

### 1. Analytical Warehouse

Tạo "mini analytical warehouse" để:

- ✅ **Pivot/Heatmap nhanh**: Phân tích tỷ lệ khô mắt theo các dimensions
- ✅ **Giải thích xu hướng risk**: Screen time, sleep quality, stress levels
- ✅ **Justify triage**: Mối quan hệ giữa symptom score và DED rate
- ✅ **Data quality monitoring**: Missing data và validity theo nhóm

### 2. Use Cases

- **Dashboard Visualization**: Hiển thị charts và tables
- **Risk Analysis**: Phân tích các yếu tố nguy cơ
- **Model Explanation**: Giải thích tại sao model đưa ra predictions
- **Data Quality Assessment**: Đánh giá chất lượng dữ liệu

---

## 🔧 Vì sao chọn DuckDB?

### Ưu điểm

1. **Embedded Engine**: Không cần server, chạy local
2. **Fast Queries**: Query Parquet files rất nhanh
3. **SQL Support**: Dùng SQL quen thuộc
4. **Pushdown Optimization**: Tối ưu query tự động
5. **No Dependencies**: Không cần setup database server
6. **Export Format**: Xuất kết quả ra Parquet/CSV dễ dàng

### So sánh với các options khác

| Option | Pros | Cons | Chọn? |
|--------|------|------|-------|
| **DuckDB** | Embedded, fast, SQL | Limited concurrent writes | ✅ **Chọn** |
| PostgreSQL | Full-featured, scalable | Cần server, setup phức tạp | ❌ |
| SQLite | Embedded, simple | Slower với analytics | ❌ |
| Pandas | Python native | Memory intensive, slower | ❌ |

---

## 📊 5 KPI Aggregates Chính

Hệ thống tạo 5 KPI aggregates chính, mỗi KPI có format chuẩn:

### Format chuẩn

Mọi KPI output đều có:
- `n`: Count of records
- `positives`: Sum of `dry_eye_disease` (count of positives)
- `rate`: `positives / n` (DED positive rate)

### KPI 1: DED Rate by Age × Gender

**Mục đích**: Phân tích tỷ lệ khô mắt theo nhóm tuổi và giới tính

**Câu hỏi**: Nhóm tuổi nào (theo giới) có tỷ lệ cao hơn?

**Output Fields**:
- `age_band`: 18-24, 25-29, 30-34, 35-39, 40-45
- `gender`: 0 (Female), 1 (Male)
- `n`: Số lượng records
- `positives`: Số lượng positive cases
- `rate`: DED positive rate (0-1)

**File Output**: `analytics/duckdb/agg/agg_ded_by_age_gender.parquet`

**Visualization**: Pivot table hoặc Stacked bar chart

**Insight Expected**: Có phân tầng rủi ro theo nhóm (ví dụ: nữ 40-45 tuổi có rate cao hơn)

---

### KPI 2: Heatmap - Screen Time × Sleep Quality

**Mục đích**: Mối quan hệ giữa thời gian màn hình và chất lượng giấc ngủ

**Câu hỏi**: Screen cao + ngủ kém có xu hướng tăng DED không?

**Output Fields**:
- `screen_time_band`: 0-2, 2-4, 4-6, 6-8, 8-10, 10+
- `sleep_quality`: 1, 2, 3, 4, 5
- `n`: Số lượng records
- `positives`: Số lượng positive cases
- `rate`: DED positive rate

**File Output**: `analytics/duckdb/agg/agg_ded_by_screen_sleep.parquet`

**Visualization**: Heatmap với color scale (rate) và size (n)

**Insight Expected**: Trình bày "risk domain" dễ hiểu (ví dụ: screen > 8h và sleep quality < 3 có rate cao)

**Note**: Nếu n nhỏ trong ô heatmap → hiển thị cảnh báo "sample nhỏ"

---

### KPI 3: DED Rate by Symptom Score

**Mục đích**: Mối quan hệ giữa số lượng triệu chứng và tỷ lệ khô mắt

**Câu hỏi**: Symptom càng nhiều tỷ lệ DED càng tăng?

**Output Fields**:
- `symptom_score`: 0, 1, 2, 3
- `n`: Số lượng records
- `positives`: Số lượng positive cases
- `rate`: DED positive rate

**File Output**: `analytics/duckdb/agg/agg_ded_by_symptom_score.parquet`

**Visualization**: Bar chart hoặc Line chart

**Insight Expected**: Justify Stage B triage (symptom_score tăng → rate tăng)

**Interpretation**: 
- symptom_score = 0: Không triệu chứng → rate thấp
- symptom_score = 3: Đầy đủ triệu chứng → rate cao
- → Hợp lý để dùng symptoms cho triage (Stage B)

---

### KPI 4: Stress Level × Sleep Duration Band

**Mục đích**: Mối quan hệ giữa stress và thời lượng ngủ

**Câu hỏi**: Stress cao kết hợp ngủ ít có trend tăng không?

**Output Fields**:
- `stress_level`: 1, 2, 3, 4, 5
- `sleep_duration_band`: <6, 6-7, 7-8, 8-9, 9+
- `n`: Số lượng records
- `positives`: Số lượng positive cases
- `rate`: DED positive rate

**File Output**: `analytics/duckdb/agg/agg_ded_by_stress_sleepband.parquet`

**Visualization**: Heatmap

**Insight Expected**: Gợi ý can thiệp hành vi (ví dụ: stress=5 và sleep<6h có rate cao)

---

### KPI 5: Data Quality by Group

**Mục đích**: Phân tích chất lượng dữ liệu theo nhóm

**Câu hỏi**: Nhóm nào thiếu dữ liệu nhiều? Validity thấp?

**Output Fields**:
- `age_band`: Age band
- `gender`: 0/1
- `missing_rate_screen_time`: Missing rate của screen_time
- `missing_rate_sleep_quality`: Missing rate của sleep_quality
- `missing_rate_bp`: Missing rate của blood pressure
- `avg_validity_ratio`: Average validity ratio của nhóm
- `n`: Số lượng records

**File Output**: `analytics/duckdb/agg/agg_data_quality_by_group.parquet`

**Visualization**: Table hoặc Bar chart

**Insight Expected**: Giải thích confidence, tránh bias do missing data

**Use Case**: 
- Nếu nhóm nào missing nhiều → confidence thấp hơn
- Cảnh báo nếu missing_rate > threshold

---

## 🔄 Quy trình xây dựng OLAP

### Input

**File**: `data/standardized/clean_assessments.parquet`

**Format**: Parquet với đầy đủ features và derived columns

### Processing

**Script**: `backend/scripts/olap_build.py`

**Quy trình**:

1. Connect DuckDB
2. Load Parquet file
3. Execute SQL queries cho từng KPI
4. Aggregate theo dimensions
5. Calculate n, positives, rate
6. Export to Parquet

**SQL Example** (KPI 1):

```sql
SELECT 
    age_band,
    gender,
    COUNT(*) as n,
    SUM(dry_eye_disease) as positives,
    CAST(SUM(dry_eye_disease) AS DOUBLE) / COUNT(*) as rate
FROM 'data/standardized/clean_assessments.parquet'
GROUP BY age_band, gender
ORDER BY age_band, gender
```

### Output

**Location**: `analytics/duckdb/agg/`

**Files**:
- `agg_ded_by_age_gender.parquet`
- `agg_ded_by_screen_sleep.parquet`
- `agg_ded_by_symptom_score.parquet`
- `agg_ded_by_stress_sleepband.parquet`
- `agg_data_quality_by_group.parquet`

**Format**: Parquet (optimized for fast reading)

---

## 📈 Usage trong Dashboard

### Tab 1: Overview

- **KPI 1**: DED Rate by Age × Gender (Stacked bar chart)
- **KPI 5**: Data Quality by Group (Table)

### Tab 2: Risk Drivers

- **KPI 2**: Screen Time × Sleep Quality (Heatmap)
- **KPI 4**: Stress × Sleep Duration (Heatmap)

### Tab 3: Symptom & Triage

- **KPI 3**: DED Rate by Symptom Score (Bar chart)

---

## 🔍 Query Performance

### Benchmarks

- **Query Time**: < 1 second cho mỗi KPI
- **File Size**: ~50KB mỗi aggregate file
- **Memory Usage**: Minimal (DuckDB optimized)

### Optimization

- ✅ **Parquet Format**: Columnar storage, fast reads
- ✅ **Pushdown Filters**: DuckDB pushes filters to Parquet reader
- ✅ **Aggregation Optimization**: Efficient GROUP BY
- ✅ **No Joins**: Simple aggregations, no complex joins needed

---

## 📊 Visualization Guidelines

### Heatmaps

- **Color Scale**: Rate (0-1) mapped to color gradient
- **Size**: n mapped to marker size (optional)
- **Tooltip**: Show n, positives, rate
- **Warning**: Nếu n < threshold → show "small sample" warning

### Bar Charts

- **Y-axis**: Rate (0-1) hoặc percentage (0-100%)
- **X-axis**: Dimension values (age_band, symptom_score, etc.)
- **Tooltip**: n, positives, rate
- **Error Bars**: Optional (confidence intervals)

### Tables

- **Columns**: Dimension columns + n + positives + rate
- **Sorting**: Default sort by rate (descending)
- **Formatting**: Rate as percentage (e.g., "65.5%")

---

## 🎯 Insights từ OLAP

### Insight 1: Age and Gender Patterns

**Finding**: Nữ giới 40-45 tuổi có DED rate cao nhất

**Action**: Focus screening efforts on this group

### Insight 2: Screen Time and Sleep Interaction

**Finding**: Screen > 8h và sleep quality < 3 → rate tăng mạnh

**Action**: Recommend screen time limits và sleep hygiene

### Insight 3: Symptom Score Gradient

**Finding**: symptom_score tăng → rate tăng rõ rệt (0→1→2→3)

**Action**: Justify Stage B triage approach

### Insight 4: Stress and Sleep Duration

**Finding**: High stress + low sleep duration → high rate

**Action**: Stress management và sleep duration recommendations

### Insight 5: Data Quality Variation

**Finding**: Một số nhóm có missing rate cao

**Action**: Improve data collection cho các nhóm này

---

## 📚 Related Documentation

- [02_olap_duckdb_plan.md](./docs/02_olap_duckdb_plan.md) - Detailed specification
- [olap_queries.md](./docs/olap_queries.md) - SQL queries
- [olap_summary.md](./docs/olap_summary.md) - Insights summary
- [metrics_and_reporting.md](./docs/metrics_and_reporting.md) - Reporting guidelines

---

## 🔧 Technical Implementation

### DuckDB Connection

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL parquet;")
conn.execute("LOAD parquet;")
```

### Query Execution

```python
# Load data
df = conn.execute("""
    SELECT * FROM 'data/standardized/clean_assessments.parquet'
""").df()

# Aggregate
result = conn.execute("""
    SELECT 
        age_band,
        gender,
        COUNT(*) as n,
        SUM(dry_eye_disease) as positives,
        CAST(SUM(dry_eye_disease) AS DOUBLE) / COUNT(*) as rate
    FROM 'data/standardized/clean_assessments.parquet'
    GROUP BY age_band, gender
""").df()

# Export
result.to_parquet('analytics/duckdb/agg/agg_ded_by_age_gender.parquet')
```

---

## ✅ Best Practices

### 1. Regular Updates

- Rebuild aggregates sau khi update data
- Version control aggregates
- Document changes

### 2. Validation

- Check n > threshold (đảm bảo sample size đủ)
- Validate rate trong range (0-1)
- Check consistency với source data

### 3. Performance

- Cache aggregates (không cần rebuild mỗi lần)
- Use Parquet format (fast reads)
- Optimize SQL queries

---

**Last Updated**: January 2026  
**Engine**: DuckDB (Embedded)  
**KPIs**: 5 aggregates  
**Format**: Parquet
