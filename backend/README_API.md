# FastAPI Implementation - Complete Guide

## ✅ Implementation Status: COMPLETE

All required endpoints have been implemented with **strict contract compliance** per `docs/output_contract.md`.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib duckdb
```

### 2. Start Server

```bash
# From project root
cd backend
uvicorn main:app --reload

# Or from project root
uvicorn backend.main:app --reload
```

### 3. Access API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/healthz
- **Root**: http://localhost:8000/

## 📋 All Endpoints

### Health & Status

**GET /api/v1/healthz**
- Returns service health and model status
- Response includes `model_version` and `use_ml_models`

### Assessments

**POST /api/v1/assessments/screening**
- Stage A: Screening (no symptoms)
- Input: `ScreeningRequest` (all fields optional)
- Output: `AssessmentResponse` with `mode_used="A_only_screening"`
- **trigger_symptom=true** if `risk_score >= 65`

**POST /api/v1/assessments/triage**
- Stage B: Triage (with symptoms)
- Input: `TriageRequest` (Stage A + symptoms)
- Output: `AssessmentResponse` with `mode_used="B_with_symptoms"`

### Model Metadata

**GET /api/v1/models/latest**
- Returns latest model metadata summary

**GET /api/v1/models/latest/metrics**
- Returns performance metrics from registry

**GET /api/v1/models/latest/calibration**
- Returns calibration information

### OLAP KPIs

**GET /api/v1/olap/kpis**
- Lists available KPI datasets

**GET /api/v1/olap/kpis/{name}**
- Returns paginated JSON from parquet files
- Query params: `page`, `page_size`

## 🔧 Model Loading

**On Startup:**
1. Loads latest `model_version` from `modeling/registry/registry.json`
2. Attempts to load `model_A`, `model_B` from artifacts
3. Loads preprocessors (scalers) and feature lists
4. **If missing → Falls back to rule-based** with `model_version="fallback_rule_v1"`

**The API works with or without trained models!**

## 📝 Contract Compliance

### Every Response Includes:

✅ `model_version` - Always present  
✅ `disclaimers` - Always present (array)  
✅ `request_id` - UUID  
✅ `timestamp` - ISO8601  
✅ `mode_used` - "A_only_screening" or "B_with_symptoms"  
✅ `risk_score` - 0-100  
✅ `risk_level` - Low/Medium/High  
✅ `confidence` - High/Medium/Low  
✅ `missing_fields` - string[]  
✅ `top_factors` - array  
✅ `next_step` - object  

### Mode-Specific:

**Screening (Stage A):**
- `screening.risk_A` - 0-100
- `screening.trigger_symptom` - boolean (true if risk_score >= 65)

**Triage (Stage B):**
- `triage.prob_B` - 0-1
- `triage.triage_level` - Low/Medium/High

## 📊 Audit Logging

**Location**: `backend/logs/audit.jsonl`

**Logged per request:**
- request_id
- timestamp
- mode_used
- model_version
- risk_score
- risk_level
- confidence
- missing_fields_count
- trigger_symptom (Stage A) or triage_level (Stage B)

**Format**: Append-only JSONL (one JSON object per line)

## 🧪 Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/api/v1/healthz

# Screening
curl -X POST http://localhost:8000/api/v1/assessments/screening \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "average_screen_time": 8.5,
    "sleep_quality": 3,
    "sleep_duration": 6.5,
    "stress_level": 4
  }'

# Triage
curl -X POST http://localhost:8000/api/v1/assessments/triage \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "average_screen_time": 8.5,
    "discomfort_eyestrain": 1,
    "redness_in_eye": 1
  }'
```

### Using Python test script

```bash
# Make sure server is running first
python backend/test_api.py
```

## 📁 File Structure

```
backend/
├── main.py                          # FastAPI app (startup event loads models)
├── api/v1/
│   ├── assessments.py              # POST /screening, /triage
│   ├── health.py                   # GET /healthz
│   ├── models.py                   # GET /models/latest/*
│   ├── olap.py                     # GET /olap/kpis/*
│   ├── schemas.py                  # Request schemas
│   └── response_wrapper.py         # Adds model_version
├── services/
│   ├── model_loader.py             # Model loading + fallback
│   └── assessment_service.py       # Assessment logic
└── utils/
    ├── audit.py                    # Audit logging
    └── disclaimers.py              # Standard disclaimers
```

## ⚠️ Important Notes

1. **Model Loading**: Happens on startup via `@app.on_event("startup")`
2. **Fallback**: If models not found, uses rule-based scoring (always works)
3. **Feature Engineering**: Matches training pipeline (BMI, interactions, polynomials)
4. **Contract**: All responses include `model_version` and `disclaimers`
5. **No Diagnosis Language**: Uses "sàng lọc", "phân loại", "nguy cơ" only

## 🔍 Troubleshooting

### Models not loading?

- Check `modeling/registry/registry.json` exists
- Check artifact paths in registry are correct
- API will fallback to rule-based (still works!)

### Import errors?

- Make sure you're running from project root or have PYTHONPATH set
- Contracts module should be in project root

### Audit log not created?

- Check `backend/logs/` directory exists
- Check write permissions

## ✅ Definition of Done

- [x] All 8 endpoints implemented
- [x] Model loading on startup
- [x] Fallback to rule-based if models missing
- [x] Contract compliance (model_version, disclaimers)
- [x] Audit logging per request
- [x] No diagnosis language
- [x] trigger_symptom logic (>= 65)
- [x] Works with and without trained artifacts

## 📚 Documentation

- Contract: `docs/output_contract.md`
- UI Flow: `docs/ui_flow_spec.md`
- Risk Copywriting: `docs/risk_copywriting_library.md`
- Clinical Governance: `docs/clinical_governance_checklist.md`
- Implementation Summary: `backend/API_IMPLEMENTATION_SUMMARY.md`
