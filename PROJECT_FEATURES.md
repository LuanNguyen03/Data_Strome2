# Tính năng và Khả năng của Hệ thống

## 📋 Tổng quan

Hệ thống **Dry Eye Disease Risk Assessment** là một ứng dụng y tế kỹ thuật số được thiết kế để sàng lọc và phân loại nguy cơ khô mắt, với các tính năng nổi bật:

---

## ✨ Tính năng chính

### 1. 🔍 2-Stage Assessment System

Hệ thống sử dụng kiến trúc **2-stage** để đánh giá nguy cơ một cách chính xác và tuân thủ nguyên tắc y tế.

#### Stage A: Screening (Sàng lọc không triệu chứng)

**Mục đích**: Phát hiện nguy cơ sớm dựa trên hành vi và lối sống

**Input**:
- Thông tin cá nhân (tuổi, giới tính, BMI)
- Thói quen giấc ngủ (thời lượng, chất lượng, rối loạn)
- Thời gian sử dụng màn hình
- Lối sống (stress, hoạt động thể chất, caffeine, alcohol, thuốc lá)
- Dấu hiệu sinh tồn (huyết áp, nhịp tim)
- Tiền sử y tế

**Đặc điểm**:
- ✅ **KHÔNG sử dụng triệu chứng** để tránh leakage
- ✅ Cho phép thiếu dữ liệu (graceful degradation)
- ✅ Ưu tiên recall để không bỏ sót ca nguy cơ cao

**Output**:
- Risk score (0-100)
- Risk level (Low/Medium/High)
- Confidence level (High/Medium/Low)
- Top contributing factors
- Next steps recommendations

#### Stage B: Triage (Phân loại với triệu chứng)

**Mục đích**: Phân loại chính xác hơn khi đã có triệu chứng

**Input**:
- Tất cả thông tin Stage A
- **+ Triệu chứng**:
  - Khó chịu/Mỏi mắt
  - Đỏ mắt
  - Ngứa/Kích ứng mắt

**Đặc điểm**:
- ✅ Sử dụng triệu chứng để tăng độ chính xác
- ✅ Cân bằng precision và recall
- ✅ Phù hợp cho triage

**Output**:
- Triage score (0-100)
- Triage level (Low/Medium/High)
- Confidence level
- Top contributing factors (bao gồm symptoms)
- Next steps (có thể khuyến nghị khám bác sĩ)

#### Router Logic (Chuyển đổi giữa Stage A và B)

Hệ thống tự động quyết định:

- Nếu **chưa có triệu chứng** → Chạy Stage A
- Nếu Stage A **risk_score >= 65** → Yêu cầu nhập triệu chứng → Chạy Stage B
- Nếu **đã có triệu chứng** → Chạy Stage B trực tiếp

---

### 2. 🤖 AI-Powered Treatment Recommendations

Tích hợp **Google Gemini 2.5 Flash** để đưa ra hướng điều trị cá nhân hóa.

#### Tính năng

- ✅ **Personalized Recommendations**: Dựa trên thông tin cá nhân của từng người dùng
- ✅ **Context-Aware**: Xem xét lối sống, triệu chứng và kết quả đánh giá
- ✅ **Professional**: Ngôn ngữ chuyên môn y khoa nhưng dễ hiểu
- ✅ **Actionable**: Đưa ra các bước cụ thể có thể thực hiện ngay

#### Thông tin được sử dụng

1. **Thông tin cá nhân**:
   - Tuổi, giới tính
   - BMI (tính từ chiều cao/cân nặng)

2. **Thói quen sinh hoạt**:
   - Giấc ngủ (thời lượng, chất lượng)
   - Thời gian dùng màn hình
   - Mức độ căng thẳng

3. **Triệu chứng báo cáo**:
   - Các triệu chứng mắt đã nhập

4. **Kết quả đánh giá**:
   - Risk score
   - Risk level

#### Output Format

- Danh sách 3-5 hướng điều trị cụ thể
- Giải thích ngắn gọn lý do
- Nhấn mạnh việc khám bác sĩ nếu nguy cơ cao
- Không đưa ra đơn thuốc cụ thể

📖 [Chi tiết setup Gemini AI →](./GEMINI_SETUP.md)  
📖 [AI Treatment Feature Docs →](./AI_TREATMENT_FEATURE.md)

---

### 3. 📊 OLAP Analytics với DuckDB

Hệ thống tích hợp **DuckDB** (embedded OLAP engine) để phân tích dữ liệu nhanh chóng.

#### 5 KPI Aggregates chính

1. **DED Rate by Age × Gender**
   - Phân tích tỷ lệ khô mắt theo nhóm tuổi và giới tính
   - Format: Pivot table với n, positives, rate

2. **Heatmap: Screen Time × Sleep Quality**
   - Mối quan hệ giữa thời gian màn hình và chất lượng giấc ngủ
   - Format: Heatmap với rate và n

3. **DED Rate by Symptom Score**
   - Mối quan hệ giữa số lượng triệu chứng và tỷ lệ khô mắt
   - Format: Bar chart với symptom_score (0-3) vs rate

4. **Stress Level × Sleep Duration**
   - Mối quan hệ giữa stress và thời lượng ngủ
   - Format: Heatmap

5. **Data Quality by Group**
   - Phân tích missing data và validity theo nhóm
   - Format: Table với missing rates và validity ratios

#### Lợi ích

- ✅ **Fast Queries**: DuckDB query Parquet nhanh
- ✅ **No Server Required**: Embedded engine
- ✅ **Pivot Tables**: Dễ dàng tạo pivot và heatmap
- ✅ **Dashboard Ready**: Output format sẵn sàng cho dashboard

📖 [Chi tiết về OLAP →](./OLAP_OVERVIEW.md)

---

### 4. 🧠 Machine Learning Models

Hệ thống sử dụng **stacking ensemble** với nhiều thuật toán ML.

#### Model Architecture

**Stacking Ensemble gồm:**

1. **XGBoost** - Gradient boosting
2. **LightGBM** - Fast gradient boosting
3. **CatBoost** - Categorical handling
4. **HistGradientBoosting** - Sklearn native
5. **ExtraTrees** - Randomized trees
6. **RandomForest** - Bagging
7. **TabNet** (optional) - Deep learning

**Meta-learner**: Neural Network (MLP 128-64-32)

#### Feature Engineering

- ✅ **118 engineered features**
- ✅ Polynomial features (squares, cubes, sqrt, log)
- ✅ Interaction terms (48+ combinations)
- ✅ Medical domain features (MAP, pulse pressure)
- ✅ Ratio features và composites
- ✅ Binning và categorization

#### Preprocessing

- ✅ Stratified train/val/test split
- ✅ SMOTE-Tomek for imbalance
- ✅ RobustScaler (handles outliers)
- ✅ Feature selection
- ✅ Probability calibration

📖 [Chi tiết về AI Models →](./AI_MODELS.md)

---

### 5. 📁 Data Standardization Pipeline

Quy trình chuẩn hóa dữ liệu từ raw CSV sang clean Parquet.

#### Quy trình

1. **Input**: `Dry_Eye_Dataset.csv` (raw)
2. **Processing**:
   - Convert naming convention (snake_case)
   - Normalize data types (binary Y/N → 0/1)
   - Parse blood pressure (systolic/diastolic)
   - Validate ranges (out-of-range → NULL)
   - Create derived features (BMI, bands, symptom_score)
3. **Output**: 
   - `clean_assessments.parquet` (standardized data)
   - `data_quality_report.json` (quality metrics)

#### Data Quality Features

- ✅ Missing data tracking
- ✅ Range validation flags
- ✅ Validity ratio per record
- ✅ BP parse success rate
- ✅ Data quality report

📖 [Chi tiết về Data →](./DATA_OVERVIEW.md)

---

### 6. 🌐 Modern Web Interface

Frontend React + TypeScript với UX tối ưu.

#### Pages

1. **Quick Assessment** (`/quick-assessment`)
   - Form nhập liệu Stage A
   - Gentle warnings cho missing fields
   - Allow submission với incomplete data

2. **Symptoms** (`/symptoms`)
   - Form nhập 3 triệu chứng
   - Load previous form data
   - Skip option

3. **Result** (`/result`)
   - Display assessment results
   - Show AI treatment recommendations
   - Next steps và disclaimers
   - Toggle để hiện/ẩn AI recommendations

#### Features

- ✅ Responsive design
- ✅ State persistence (localStorage)
- ✅ Smooth transitions
- ✅ Error handling
- ✅ Loading states

---

### 7. 📡 RESTful API

FastAPI backend với contract compliance.

#### Endpoints

- `GET /api/v1/healthz` - Health check
- `POST /api/v1/assessments/screening` - Stage A screening
- `POST /api/v1/assessments/triage` - Stage B triage
- `GET /api/v1/olap/kpis` - List OLAP KPIs
- `GET /api/v1/models/info` - Model information

#### API Features

- ✅ Strict contract compliance
- ✅ Versioned API (v1)
- ✅ Comprehensive error handling
- ✅ Audit logging
- ✅ Disclaimers in every response
- ✅ Model versioning

📖 [API Documentation →](./API_DOCUMENTATION.md)

---

### 8. 🔒 Medical Governance Compliance

Hệ thống tuân thủ các nguyên tắc y tế nghiêm ngặt.

#### Compliance Features

- ✅ **No Diagnosis**: Không đưa ra chẩn đoán
- ✅ **No Leakage**: Stage A không sử dụng triệu chứng
- ✅ **Disclaimers**: Luôn có trong mọi response
- ✅ **Confidence Levels**: Phản ánh độ tin cậy
- ✅ **Next Steps**: Hướng dẫn rõ ràng
- ✅ **Audit Logging**: Ghi log tất cả assessments

#### Safety Measures

- ✅ Confidence calculation dựa trên missing data
- ✅ Graceful degradation khi thiếu dữ liệu
- ✅ Clear separation giữa screening và triage
- ✅ Medical disclaimers

📖 [Clinical Governance Checklist →](./docs/clinical_governance_checklist.md)

---

## 🎯 Use Cases

### 1. Self-Assessment (Người dùng cá nhân)

- Người dùng điền form online
- Nhận được đánh giá nguy cơ
- Xem hướng điều trị từ AI
- Quyết định có nên khám bác sĩ không

### 2. Clinical Support (Hỗ trợ y tế)

- Nhân viên y tế nhập thông tin bệnh nhân
- Hệ thống hỗ trợ đánh giá nguy cơ
- Bác sĩ xem xét kết quả + AI recommendations
- Ra quyết định điều trị

### 3. Population Screening (Sàng lọc quần thể)

- Tích hợp vào hệ thống health check
- Sàng lọc hàng loạt
- OLAP analytics để phân tích xu hướng
- Identify high-risk groups

### 4. Research & Analytics

- OLAP aggregates cho research
- Model performance tracking
- Data quality monitoring
- Trend analysis

---

## 🔄 Workflow

### Assessment Flow

```
1. User nhập thông tin Stage A
   ↓
2. System đánh giá Stage A
   ↓
3. Nếu risk cao → Yêu cầu nhập triệu chứng
   ↓
4. User nhập triệu chứng (optional)
   ↓
5. System đánh giá Stage B (nếu có triệu chứng)
   ↓
6. System gọi Gemini AI → Tạo recommendations
   ↓
7. Display results + AI recommendations
   ↓
8. User quyết định next steps
```

---

## 📈 Performance Characteristics

### Response Time

- **Stage A Assessment**: < 100ms
- **Stage B Assessment**: < 150ms
- **AI Recommendations**: 1-3 seconds (depends on Gemini API)

### Accuracy

- **Stage A**: AUC = 0.5077 (near random - do dataset limitations)
- **Stage B**: AUC = 0.5982 (best performance)
- **Best Model**: AUC = 0.6047 (stacking ensemble)

**Note**: Performance bị giới hạn bởi dataset (thiếu clinical features).  
Với clinical-grade data, expected AUC > 0.90.

📖 [Chi tiết về Results và Metrics →](./RESULTS_AND_METRICS.md)

---

## 🚀 Future Enhancements

### Planned Features

- [ ] Dashboard visualization cho OLAP KPIs
- [ ] Export PDF reports
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Integration với EMR systems
- [ ] Real-time monitoring và alerts

### Model Improvements

- [ ] Collect clinical-grade data (Schirmer test, TBUT, etc.)
- [ ] External validation với dataset khác
- [ ] Model calibration improvements
- [ ] Drift detection và monitoring

---

## 📚 Related Documentation

- [README.md](./README.md) - Tổng quan dự án
- [DATA_OVERVIEW.md](./DATA_OVERVIEW.md) - Chi tiết về dataset
- [AI_MODELS.md](./AI_MODELS.md) - Chi tiết về ML models
- [RESULTS_AND_METRICS.md](./RESULTS_AND_METRICS.md) - Kết quả và metrics
- [OLAP_OVERVIEW.md](./OLAP_OVERVIEW.md) - Chi tiết về OLAP
- [GEMINI_SETUP.md](./GEMINI_SETUP.md) - Setup AI recommendations

---

**Last Updated**: January 2026  
**Version**: 1.0.0
