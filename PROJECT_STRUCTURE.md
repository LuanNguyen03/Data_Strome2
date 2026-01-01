# Project Structure Overview

## 📁 Monorepo Layout

```
DataStrome2/
├── backend/                 # FastAPI backend application
│   ├── main.py              # FastAPI app entry point
│   ├── run.py               # Server runner script
│   ├── requirements.txt     # Backend dependencies
│   ├── routers/             # API route handlers
│   │   ├── health.py        # Health check endpoint
│   │   └── assessment.py   # Main assessment endpoint
│   └── services/            # Business logic layer
│       └── assessment_service.py  # 2-stage assessment logic
│
├── frontend/                # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx          # Main app component
│   │   ├── main.tsx         # React entry point
│   │   ├── types.ts         # TypeScript type definitions
│   │   ├── api/
│   │   │   └── client.ts    # API client (axios)
│   │   └── components/
│   │       ├── AssessmentForm.tsx    # Input form (Stage A)
│   │       └── ResultPage.tsx        # Results display
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Vite configuration
│
├── contracts/               # Shared API contracts
│   ├── schemas.py          # Pydantic models (backend)
│   └── types.ts            # TypeScript types (frontend)
│
├── scripts/                # Utility scripts
│   ├── standardize.py      # Data standardization pipeline
│   ├── verify_setup.py      # Setup verification
│   └── run_all.bat         # Windows task runner
│
├── analytics/              # OLAP analytics
│   └── duckdb/
│       ├── build_agg.py    # Generate OLAP aggregates
│       └── agg/            # Output aggregates (Parquet)
│
├── data/                   # Data directories
│   ├── raw/                # Raw input data
│   └── standardized/       # Clean standardized data
│
├── modeling/               # ML model artifacts
│   ├── artifacts/          # Saved models
│   └── reports/            # Model evaluation reports
│
├── docs/                   # Documentation specs
│   ├── 01_data_standardization.md
│   ├── 02_olap_duckdb_plan.md
│   ├── 03_medical_modeling_plan.md
│   ├── output_contract.md
│   ├── threshold_notes.md
│   ├── ui_flow_spec.md
│   └── ... (other specs)
│
├── requirements.txt        # Root Python dependencies
├── Makefile               # Build automation
├── README.md              # Main documentation
└── QUICKSTART.md          # Quick start guide
```

## 🔄 Data Flow

```
1. Raw Data (CSV)
   ↓
2. Standardization (scripts/standardize.py)
   → clean_assessments.parquet
   → data_quality_report.json
   ↓
3. OLAP Aggregates (analytics/duckdb/build_agg.py)
   → 5 KPI Parquet files
   ↓
4. Backend API (FastAPI)
   → /api/assess endpoint
   → AssessmentService (2-stage logic)
   → AssessmentResponse (JSON)
   ↓
5. Frontend (React)
   → AssessmentForm → API call
   → ResultPage → Display results
```

## 🏗️ Architecture

### Backend (FastAPI)

- **Routers**: Handle HTTP requests/responses
- **Services**: Business logic (assessment, routing)
- **Contracts**: Shared Pydantic schemas for validation

### Frontend (React + TypeScript)

- **Components**: UI components (form, results)
- **API Client**: Axios-based HTTP client
- **Types**: TypeScript types matching backend contracts

### Shared Contracts

- **schemas.py**: Pydantic models for backend
- **types.ts**: TypeScript types for frontend
- Ensures type safety across stack

## 🎯 Key Features

1. **2-Stage Assessment**:
   - Stage A: Screening (no symptoms)
   - Stage B: Triage (with symptoms)
   - Router: Auto-trigger symptom questions

2. **Medical Governance**:
   - No leakage (symptoms not in Stage A)
   - Disclaimers always shown
   - Confidence based on missing data

3. **File-Based**:
   - No SQL server required
   - DuckDB for OLAP
   - Parquet for data storage

## 📦 Dependencies

### Backend
- FastAPI: Web framework
- Pydantic: Data validation
- Uvicorn: ASGI server

### Frontend
- React: UI framework
- TypeScript: Type safety
- Vite: Build tool
- Axios: HTTP client

### Data Pipeline
- Polars: Data processing
- DuckDB: OLAP engine
- PyArrow: Parquet support

## 🚀 Running the System

See `QUICKSTART.md` for step-by-step instructions.

