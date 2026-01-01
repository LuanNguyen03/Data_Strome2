# Dry Eye Disease Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![Gemini AI](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)

Hệ thống đánh giá nguy cơ khô mắt (Dry Eye Disease) theo chuẩn y tế, sử dụng **2-stage screening và triage**, tích hợp **AI tư vấn điều trị** với Google Gemini 2.5.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Tính năng chính](#-tính-năng-chính)
- [Quick Start](#-quick-start)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Chạy dự án](#-chạy-dự-án)
- [Tài liệu](#-tài-liệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Disclaimer](#-medical-disclaimer)

---

## 🎯 Tổng quan

Hệ thống **Dry Eye Disease Risk Assessment** là một ứng dụng y tế kỹ thuật số được thiết kế để:

- ✅ **Sàng lọc nguy cơ sớm (Stage A)**: Đánh giá nguy cơ dựa trên hành vi và lối sống
- ✅ **Phân loại khi có triệu chứng (Stage B)**: Hỗ trợ triage khi người dùng có triệu chứng
- ✅ **Tư vấn điều trị AI**: Đưa ra hướng điều trị cá nhân hóa với Google Gemini 2.5
- ✅ **OLAP Analytics**: Phân tích dữ liệu với DuckDB
- ✅ **ML Models**: Mô hình học máy 2-stage với XGBoost, LightGBM, CatBoost

### Đối tượng sử dụng

- 👥 Người dùng phổ thông (self-assessment)
- 👨‍⚕️ Nhân viên y tế (hỗ trợ đánh giá)
- 🏥 Phòng khám (tích hợp hệ thống)

---

## ✨ Tính năng chính

### 1. **2-Stage Assessment System**

- **Stage A (Screening)**: Không sử dụng triệu chứng, chỉ dựa trên lối sống
- **Stage B (Triage)**: Sử dụng triệu chứng để phân loại chính xác hơn
- **Router Logic**: Tự động chuyển từ Stage A sang B khi cần

📖 [Xem chi tiết tính năng →](./PROJECT_FEATURES.md)

### 2. **AI-Powered Treatment Recommendations**

- Tích hợp **Google Gemini 2.5 Flash**
- Tư vấn điều trị cá nhân hóa dựa trên:
  - Thông tin cá nhân (tuổi, giới tính, BMI)
  - Thói quen sinh hoạt (giấc ngủ, màn hình, stress)
  - Triệu chứng báo cáo
  - Kết quả đánh giá nguy cơ

📖 [Hướng dẫn setup Gemini AI →](./GEMINI_SETUP.md)

### 3. **Data Standardization Pipeline**

- Chuẩn hóa dữ liệu từ CSV sang Parquet
- Validation và quality checks
- Derived features (BMI, bands, symptom scores)

📖 [Chi tiết về Data →](./DATA_OVERVIEW.md)

### 4. **OLAP Analytics với DuckDB**

- 5 KPI aggregates chính
- Heatmaps và pivot tables
- Data quality monitoring

📖 [Chi tiết về OLAP →](./OLAP_OVERVIEW.md)

### 5. **Machine Learning Models**

- Stacking ensemble (XGBoost, LightGBM, CatBoost)
- Feature engineering nâng cao
- 2-stage model architecture

📖 [Chi tiết về AI Models →](./AI_MODELS.md)

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 - 3.13
- **Node.js**: 18+
- **UV** (khuyến nghị) hoặc pip

### Cài đặt nhanh (1 lệnh)

```bash
# Windows
setup_all.bat

# Linux/Mac
chmod +x setup_all.sh && ./setup_all.sh
```

### Cài đặt từng bước

#### 1. Backend

```bash
# Sử dụng UV (khuyến nghị - nhanh hơn 10x)
uv sync

# Hoặc sử dụng pip
pip install -r requirements.txt
```

#### 2. Frontend

```bash
cd frontend
npm install
```

#### 3. Setup Gemini AI (Optional nhưng khuyến nghị)

```bash
# Interactive setup
setup_gemini.bat      # Windows
./setup_gemini.sh     # Linux/Mac

# Hoặc thủ công
echo "GEMINI_API_KEY=your_api_key" > .env
```

📖 [Hướng dẫn cài đặt đầy đủ →](./README_INSTALL.md)  
📖 [Quick commands reference →](./INSTALL_COMMANDS.md)

---

## ⚙️ Cấu hình

### Environment Variables

Tạo file `.env` trong thư mục gốc:

```bash
# Gemini AI (Optional)
GEMINI_API_KEY=your_api_key_here

# Backend (Optional)
API_HOST=0.0.0.0
API_PORT=8000
```

📖 [Chi tiết cấu hình Gemini →](./GEMINI_SETUP.md)

### Kiểm tra cấu hình

```bash
# Check Gemini configuration
python check_gemini.py

# Test Gemini API
python check_gemini.py --test-api

# Check all versions
make versions
```

---

## 🏃 Chạy dự án

### 1. Start Backend

```bash
# Sử dụng UV (recommended)
uv run python backend/run.py

# Hoặc trực tiếp
python backend/run.py

# Hoặc Make
make run-backend
```

Backend chạy tại: **http://localhost:8000**
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/healthz

### 2. Start Frontend (Terminal mới)

```bash
cd frontend
npm run dev

# Hoặc Make
make run-frontend
```

Frontend chạy tại: **http://localhost:5173**

### 3. Test API

```bash
# Health check
curl http://localhost:8000/api/v1/healthz

# Screening assessment
curl -X POST http://localhost:8000/api/v1/assessments/screening \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sleep_duration": 7.5,
    "sleep_quality": 3,
    "average_screen_time": 8.0,
    "stress_level": 4
  }'
```

📖 [API Documentation →](./API_DOCUMENTATION.md)  
📖 [Testing Guide →](./TESTING_GUIDE.md)

---

## 📚 Tài liệu

> 📖 **Xem [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)** để có mục lục đầy đủ tất cả tài liệu

### 📖 Quick Links

| Tài liệu | Mô tả | Link |
|----------|-------|------|
| **DOCUMENTATION_INDEX.md** | 📑 **Mục lục tất cả tài liệu** | [→ Xem](./DOCUMENTATION_INDEX.md) |
| **PROJECT_FEATURES.md** | Tính năng và khả năng của hệ thống | [→ Xem](./PROJECT_FEATURES.md) |
| **DATA_OVERVIEW.md** | Giới thiệu về dataset và quy trình chuẩn hóa | [→ Xem](./DATA_OVERVIEW.md) |
| **OLAP_OVERVIEW.md** | OLAP analytics với DuckDB | [→ Xem](./OLAP_OVERVIEW.md) |
| **AI_MODELS.md** | Mô hình AI và kiến trúc 2-stage | [→ Xem](./AI_MODELS.md) |
| **RESULTS_AND_METRICS.md** | Kết quả đạt được, metrics và đánh giá | [→ Xem](./RESULTS_AND_METRICS.md) |

### 🔧 Setup & Installation

| Tài liệu | Mô tả | Link |
|----------|-------|------|
| **README_INSTALL.md** | Hướng dẫn cài đặt đầy đủ | [→ Xem](./README_INSTALL.md) |
| **QUICKSTART.md** | Quick start guide | [→ Xem](./QUICKSTART.md) |
| **INSTALL_COMMANDS.md** | Command cheat sheet | [→ Xem](./INSTALL_COMMANDS.md) |
| **DEPENDENCIES.md** | Chi tiết dependencies | [→ Xem](./DEPENDENCIES.md) |
| **GEMINI_SETUP.md** | Setup Gemini AI | [→ Xem](./GEMINI_SETUP.md) |

### 📊 Technical Documentation

| Tài liệu | Mô tả | Link |
|----------|-------|------|
| **API_DOCUMENTATION.md** | API endpoints và schemas | [→ Xem](./API_DOCUMENTATION.md) |
| **PROJECT_STRUCTURE.md** | Cấu trúc code và modules | [→ Xem](./PROJECT_STRUCTURE.md) |
| **TESTING_GUIDE.md** | Testing và QA | [→ Xem](./TESTING_GUIDE.md) |
| **FINAL_ASSESSMENT.md** | Technical assessment | [→ Xem](./FINAL_ASSESSMENT.md) |

### 🤖 AI & ML

| Tài liệu | Mô tả | Link |
|----------|-------|------|
| **AI_TREATMENT_FEATURE.md** | AI treatment recommendations | [→ Xem](./AI_TREATMENT_FEATURE.md) |
| **GEMINI_MODEL_OPTIONS.md** | Gemini model options | [→ Xem](./GEMINI_MODEL_OPTIONS.md) |
| **GEMINI_2.5_UPGRADE.md** | Gemini 2.5 upgrade info | [→ Xem](./GEMINI_2.5_UPGRADE.md) |

---

## 🏗️ Cấu trúc dự án

```
DataStrome2/
├── backend/              # FastAPI backend
│   ├── api/v1/          # API endpoints
│   ├── services/        # Business logic (ML, Gemini)
│   ├── scripts/         # Data pipeline scripts
│   └── utils/           # Utilities (audit, disclaimers)
│
├── frontend/            # React + TypeScript frontend
│   ├── src/
│   │   ├── pages/       # Main pages (Assessment, Result)
│   │   ├── components/  # Reusable components
│   │   └── api/         # API client
│   └── package.json
│
├── docs/                # Technical specifications
│   ├── 01_data_standardization.md
│   ├── 02_olap_duckdb_plan.md
│   └── 03_medical_modeling_plan.md
│
├── data/                # Data directories
│   ├── raw/             # Raw input data
│   └── standardized/    # Clean standardized data
│
├── analytics/           # OLAP DuckDB aggregates
│   └── duckdb/agg/      # KPI aggregates
│
├── modeling/            # ML models
│   ├── artifacts/       # Saved models
│   ├── reports/         # Evaluation reports
│   └── registry/        # Model registry
│
├── scripts/             # Utility scripts
├── contracts/           # Shared API contracts
└── requirements.txt     # Python dependencies
```

📖 [Chi tiết cấu trúc →](./PROJECT_STRUCTURE.md)

---

## 🔬 Kết quả và Metrics

### Model Performance

- **Stage A (Screening)**: AUC = 0.5077 (near random)
- **Stage B (Triage)**: AUC = 0.5982 (best performance)
- **Best Model**: Stacking ensemble với 118 features → AUC = 0.6047

### Dataset Characteristics

- **Samples**: 20,000 patients
- **Features**: 48 (26 original + 22 engineered)
- **Class Balance**: 65% positive / 35% negative
- **Missing Data**: 0% (after standardization)

### OLAP KPIs

- 5 KPI aggregates đã được tạo
- Heatmaps và pivot tables
- Data quality monitoring

📖 [Chi tiết kết quả và metrics →](./RESULTS_AND_METRICS.md)

---

## 🏥 Medical Disclaimer

### Intended Use

Hệ thống này được thiết kế để:

- ✅ **Screening nguy cơ sớm**: Đánh giá nguy cơ dựa trên hành vi và lối sống
- ✅ **Triage khi có triệu chứng**: Hỗ trợ phân loại khi người dùng đã có triệu chứng
- ✅ **Hỗ trợ quyết định**: Cung cấp thông tin để người dùng/bác sĩ quyết định

### Not Intended Use

**Hệ thống này KHÔNG được thiết kế để:**

- ❌ Chẩn đoán xác định bệnh khô mắt
- ❌ Quyết định điều trị
- ❌ Thay thế đánh giá của bác sĩ chuyên khoa
- ❌ Sử dụng trong các tình huống cấp cứu

### Important Notes

- Kết quả chỉ mang tính **hỗ trợ sàng lọc và phân loại**
- Luôn có disclaimers trong mọi response
- Confidence level phản ánh độ đầy đủ của dữ liệu đầu vào
- Người dùng nên tham khảo bác sĩ nếu triệu chứng kéo dài hoặc nặng

📖 [Clinical Governance Checklist →](./docs/clinical_governance_checklist.md)

---

## 🛠️ Technology Stack

### Backend

- **Framework**: FastAPI 0.128+
- **AI/ML**: 
  - Google Gemini 2.5 Flash (AI recommendations)
  - XGBoost, LightGBM, CatBoost (ML models)
- **Data Processing**: Polars, Pandas, DuckDB
- **API**: RESTful v1 (`/api/v1/`)

### Frontend

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5
- **Routing**: React Router 6
- **Styling**: CSS Modules

### Data & Analytics

- **Storage**: Parquet files
- **OLAP**: DuckDB (embedded)
- **Processing**: Polars

---

## 📈 Development Workflow

### Make Commands

```bash
make help              # Show all commands
make install           # Install all dependencies
make setup-gemini      # Setup Gemini AI
make check-gemini      # Check Gemini config
make run-backend       # Start backend
make run-frontend      # Start frontend
make clean            # Clean cache
```

📖 [Xem tất cả commands →](./INSTALL_COMMANDS.md)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

Internal project - Medical use only

---

## 🙏 Acknowledgments

- Google Gemini API for AI-powered recommendations
- FastAPI team for excellent framework
- React team for frontend framework
- DuckDB for embedded OLAP engine

---

## 📞 Support

Nếu gặp vấn đề:

1. Xem [Troubleshooting Guide](./README_INSTALL.md#troubleshooting)
2. Check [Issues](../../issues)
3. Đọc [Documentation](./docs/)

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Gemini Model**: 2.5 Flash (Latest)  
**Status**: ✅ Production Ready
