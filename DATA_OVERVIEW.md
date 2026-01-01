# Data Overview - Dataset và Quy trình Chuẩn hóa

## 📋 Tổng quan

Hệ thống sử dụng dataset **Dry Eye Disease** với 20,000 bệnh nhân để đánh giá nguy cơ khô mắt. Dataset được chuẩn hóa từ raw CSV sang clean Parquet format để đảm bảo chất lượng và tính nhất quán.

---

## 📊 Dataset Characteristics

### Thông tin cơ bản

| Attribute | Value | Mô tả |
|-----------|-------|-------|
| **Sample Size** | 20,000 | Tổng số bệnh nhân |
| **Features** | 48 | 26 original + 22 engineered |
| **Class Balance** | 65% positive / 35% negative | Imbalanced nhưng manageable |
| **Missing Data** | 0% | Sau khi chuẩn hóa |
| **Format** | Parquet | Optimized for analytics |

### Target Variable

- **dry_eye_disease**: Binary (0 = No, 1 = Yes)
- **Positive Rate**: 65%
- **Distribution**: Imbalanced nhưng đủ để train model

---

## 📁 Data Structure

### Input Data (Raw)

**File**: `Dry_Eye_Dataset.csv`

**Format**: CSV với các đặc điểm:
- Column names: Mixed case, có spaces
- Data types: Mixed (text, numeric)
- Binary fields: Y/N format
- Blood pressure: String format (e.g., "120/80")

### Output Data (Standardized)

**File**: `data/standardized/clean_assessments.parquet`

**Format**: Parquet với:
- ✅ Snake_case column names
- ✅ Normalized data types
- ✅ Binary fields: 0/1
- ✅ Parsed blood pressure: systolic/diastolic (numeric)
- ✅ Derived features: BMI, bands, symptom_score
- ✅ Validation flags: in_range flags, validity_ratio

---

## 🔄 Quy trình Chuẩn hóa

### Bước 1: Naming Convention

**Mục tiêu**: Chuẩn hóa tất cả column names sang `snake_case`

**Ví dụ**:
```
"Sleep Duration" → "sleep_duration"
"Average Screen Time" → "average_screen_time"
"Blood Pressure" → "blood_pressure_raw"
```

**Quy tắc**:
- Chữ thường
- Khoảng trắng → `_`
- Bỏ ký tự đặc biệt
- Giữ nguyên nghĩa

### Bước 2: Data Type Normalization

#### Binary Fields (Y/N → 0/1)

Các fields được convert:

- `sleep_disorder`
- `wake_up_during_night`
- `feel_sleepy_during_day`
- `caffeine_consumption`
- `alcohol_consumption`
- `smoking`
- `medical_issue`
- `ongoing_medication`
- `smart_device_before_bed`
- `bluelight_filter`
- `discomfort_eyestrain`
- `redness_in_eye`
- `itchiness_irritation_in_eye`
- `dry_eye_disease` (target)

**Mapping**: N → 0, Y → 1

#### Gender

- F → 0
- M → 1

#### Numeric Fields

Ép kiểu rõ ràng:
- `age`: int
- `height`: int (cm)
- `weight`: int (kg)
- `sleep_duration`: float (hours)
- `average_screen_time`: float (hours/day)
- `sleep_quality`: int (1-5)
- `stress_level`: int (1-5)
- `heart_rate`: int (bpm)
- `daily_steps`: int
- `physical_activity`: int (minutes)

### Bước 3: Blood Pressure Parsing

**Input**: String format (e.g., "120/80", "140/90", "N/A")

**Processing**:
1. Parse systolic và diastolic
2. Validate ranges (systolic: 70-250, diastolic: 40-150)
3. Set NULL nếu parse fail hoặc out-of-range

**Output Fields**:
- `blood_pressure_raw`: Original string (để audit)
- `systolic`: int (70-250) hoặc NULL
- `diastolic`: int (40-150) hoặc NULL
- `bp_parse_ok`: 0/1 flag

**Example**:
```
"120/80" → systolic=120, diastolic=80, bp_parse_ok=1
"N/A" → systolic=NULL, diastolic=NULL, bp_parse_ok=0
"300/200" → systolic=NULL, diastolic=NULL, bp_parse_ok=0 (out-of-range)
```

### Bước 4: Range Validation

**Nguyên tắc**: Out-of-range values → NULL + flag

**Validation Rules**:

| Field | Range | Action if Out-of-Range |
|-------|-------|------------------------|
| age | 18-45 | → NULL + `age_in_range` = 0 |
| sleep_quality | 1-5 | → NULL + `sleep_quality_in_range` = 0 |
| stress_level | 1-5 | → NULL + `stress_level_in_range` = 0 |
| sleep_duration | 0-24 | → NULL + `sleep_duration_in_range` = 0 |
| average_screen_time | 0-24 | → NULL + `average_screen_time_in_range` = 0 |
| heart_rate | 40-220 | → NULL + `heart_rate_in_range` = 0 |
| daily_steps | 0-50,000 | → NULL + `daily_steps_in_range` = 0 |
| physical_activity | 0-600 | → NULL + `physical_activity_in_range` = 0 |
| height | 120-230 | → NULL + `height_in_range` = 0 |
| weight | 30-250 | → NULL + `weight_in_range` = 0 |
| systolic | 70-250 | → NULL + `systolic_in_range` = 0 |
| diastolic | 40-150 | → NULL + `diastolic_in_range` = 0 |

**Validity Ratio**: Mean of all `*_in_range` flags per record

### Bước 5: Derived Features

#### BMI (Body Mass Index)

```python
bmi = weight / (height/100) ** 2
```

#### Bands (for OLAP analytics)

**Age Band**:
- 18-24
- 25-29
- 30-34
- 35-39
- 40-45

**Screen Time Band**:
- 0-2 hours
- 2-4 hours
- 4-6 hours
- 6-8 hours
- 8-10 hours
- 10+ hours

**Sleep Duration Band**:
- < 6 hours
- 6-7 hours
- 7-8 hours
- 8-9 hours
- 9+ hours

#### Symptom Score

```python
symptom_score = (
    discomfort_eyestrain +
    redness_in_eye +
    itchiness_irritation_in_eye
)
# Range: 0-3
```

---

## 📈 Data Quality Metrics

### Quality Report Output

**File**: `data/standardized/data_quality_report.json`

**Nội dung**:

```json
{
  "summary": {
    "total_rows": 20000,
    "total_cols": 48,
    "ded_positive_rate": 0.65
  },
  "bp_parsing": {
    "parse_ok_rate": 0.95,
    "parse_fail_count": 1000
  },
  "missing_rates": {
    "screen_time": 0.02,
    "sleep_quality": 0.01,
    "systolic": 0.05,
    ...
  },
  "validity": {
    "avg_validity_ratio": 0.92,
    "validity_distribution": {...}
  },
  "out_of_range": {
    "age": 50,
    "sleep_quality": 30,
    ...
  }
}
```

### Key Metrics

1. **Missing Rate**: Tỷ lệ missing data theo từng field
2. **BP Parse OK Rate**: Tỷ lệ parse thành công blood pressure
3. **Average Validity Ratio**: Trung bình validity ratio của tất cả records
4. **Out-of-Range Counts**: Số lượng values out-of-range theo từng field

---

## 🔍 Feature Groups

### 1. Personal Information

- `age` (int, 18-45)
- `gender` (int, 0=F, 1=M)
- `height` (int, cm)
- `weight` (int, kg)
- `bmi` (float, derived)

### 2. Sleep Features

- `sleep_duration` (float, hours)
- `sleep_quality` (int, 1-5)
- `sleep_disorder` (int, 0/1)
- `wake_up_during_night` (int, 0/1)
- `feel_sleepy_during_day` (int, 0/1)
- `sleep_duration_band` (string, derived)

### 3. Screen/Device Usage

- `average_screen_time` (float, hours/day)
- `smart_device_before_bed` (int, 0/1)
- `bluelight_filter` (int, 0/1)
- `screen_time_band` (string, derived)

### 4. Lifestyle

- `stress_level` (int, 1-5)
- `daily_steps` (int)
- `physical_activity` (int, minutes)
- `caffeine_consumption` (int, 0/1)
- `alcohol_consumption` (int, 0/1)
- `smoking` (int, 0/1)

### 5. Vitals

- `systolic` (int, mmHg)
- `diastolic` (int, mmHg)
- `heart_rate` (int, bpm)
- `blood_pressure_raw` (string, original)
- `bp_parse_ok` (int, 0/1)

### 6. Medical History

- `medical_issue` (int, 0/1)
- `ongoing_medication` (int, 0/1)

### 7. Symptoms (Stage B only)

- `discomfort_eyestrain` (int, 0/1)
- `redness_in_eye` (int, 0/1)
- `itchiness_irritation_in_eye` (int, 0/1)
- `symptom_score` (int, 0-3, derived)

### 8. Target

- `dry_eye_disease` (int, 0/1)

---

## 🔧 Data Processing Script

### Standardization Script

**Location**: `backend/scripts/standardize.py`

**Usage**:

```bash
python backend/scripts/standardize.py \
  --input DryEyeDisease/Dry_Eye_Dataset.csv \
  --output data/standardized/clean_assessments.parquet \
  --report data/standardized/data_quality_report.json
```

**What it does**:

1. Load raw CSV
2. Apply naming convention (snake_case)
3. Normalize data types
4. Parse blood pressure
5. Validate ranges
6. Create derived features (BMI, bands, symptom_score)
7. Calculate validity ratios
8. Save to Parquet
9. Generate quality report

---

## 📊 Data Statistics

### Distribution

**Age Distribution**:
- Mean: ~31 years
- Range: 18-45 years

**Gender Distribution**:
- Female (0): ~50%
- Male (1): ~50%

**Class Distribution**:
- Positive (dry_eye_disease=1): 65%
- Negative (dry_eye_disease=0): 35%

**Screen Time**:
- Mean: ~7.5 hours/day
- Range: 0-24 hours

**Sleep Duration**:
- Mean: ~7.0 hours/night
- Range: 3-12 hours

### Missing Data (After Standardization)

- **Overall**: 0% missing (imputed hoặc validated)
- **Top Missing Fields**:
  - Blood pressure: ~5%
  - Screen time: ~2%
  - Sleep quality: ~1%

---

## ✅ Data Quality Assurance

### Validation Rules

1. ✅ **Type Checking**: Tất cả fields có đúng type
2. ✅ **Range Validation**: Values trong acceptable ranges
3. ✅ **BP Parsing**: Blood pressure được parse correctly
4. ✅ **Consistency**: No conflicting values
5. ✅ **Completeness**: Missing rates tracked

### Quality Flags

- `*_in_range`: Flag cho mỗi field (0/1)
- `validity_ratio`: Mean of all in_range flags (0-1)
- `bp_parse_ok`: Blood pressure parse success (0/1)

---

## 🔄 Data Pipeline

### Input → Output Flow

```
Dry_Eye_Dataset.csv (Raw)
    ↓
[Standardization Script]
    ↓
clean_assessments.parquet (Standardized)
    ↓
[OLAP Build Script]
    ↓
analytics/duckdb/agg/*.parquet (Aggregates)
    ↓
[ML Training]
    ↓
modeling/artifacts/*.pkl (Models)
```

---

## 📚 Related Documentation

- [01_data_standardization.md](./docs/01_data_standardization.md) - Detailed specification
- [data_dictionary.md](./docs/data_dictionary.md) - Complete field dictionary
- [OLAP_OVERVIEW.md](./OLAP_OVERVIEW.md) - How data is used in OLAP
- [AI_MODELS.md](./AI_MODELS.md) - How data is used in ML models

---

## 🎯 Best Practices

### For Data Updates

1. **Keep Raw Data**: Luôn giữ raw CSV để trace back
2. **Version Control**: Track versions của standardized data
3. **Quality Reports**: Generate reports sau mỗi standardization
4. **Validation**: Re-validate sau khi update data

### For Analysis

1. **Use Parquet**: Parquet format cho analytics (nhanh hơn CSV)
2. **Check Quality**: Xem data_quality_report.json trước khi analyze
3. **Handle Missing**: Luôn check missing rates
4. **Validity Ratio**: Xem validity_ratio để đánh giá data quality

---

**Last Updated**: January 2026  
**Dataset Version**: 1.0.0  
**Records**: 20,000  
**Format**: Parquet
