# Testing Guide - Complete Test Suite

## Overview

Test suite implementing requirements from `docs/clinical_governance_checklist.md`:
- Backend API contract tests
- Data leakage prevention tests
- Frontend component tests
- One-shot orchestration with smoke tests

## Backend Tests

### 1. Standardization Tests ✅ (Already Exists)
**File**: `backend/scripts/tests/test_standardize.py`

Tests:
- BP parsing logic
- Banding (age, screen_time, sleep_duration)
- Range validation
- Symptom score calculation

**Run**:
```bash
python backend/scripts/tests/test_standardize.py
```

### 2. API Contract Tests ✅ NEW
**File**: `backend/scripts/tests/test_api_contract.py`

Tests:
- ✅ Screening response always includes `disclaimers` and `model_version`
- ✅ Triage response always includes `disclaimers` and `model_version`
- ✅ Disclaimers contain required medical language
- ✅ Model version always present (even with fallback)

**Requirements**:
- API server must be running: `uvicorn backend.main:app --reload`

**Run**:
```bash
python backend/scripts/tests/test_api_contract.py
```

### 3. Leakage Tests ✅ NEW
**File**: `backend/scripts/tests/test_leakage.py`

Tests:
- ✅ Stage A feature list excludes symptom columns
- ✅ STAGE_A_EXCLUDED list properly defined
- ✅ Stage A features are safe (no symptom leakage)
- ✅ Stage B correctly includes symptoms (sanity check)

**Run**:
```bash
python backend/scripts/tests/test_leakage.py
```

### Run All Backend Tests

```bash
python backend/scripts/tests/test_all.py
```

## Frontend Tests

### Component Test ✅ NEW
**File**: `frontend/src/pages/__tests__/Result.test.tsx`

Tests:
- ✅ Disclaimers always render on Result page (Stage A)
- ✅ Disclaimers always render on Result page (Stage B)
- ✅ All disclaimers from response are rendered
- ✅ Disclaimers section exists even with empty array

**Run**:
```bash
cd frontend
npm install  # Install @testing-library/react if needed
npm test
```

## One-Shot Orchestration ✅ NEW

### Script: `backend/scripts/run_all.py`

**Purpose**: Run complete pipeline and smoke tests

**Steps**:
1. Data Standardization (`standardize.py`)
2. OLAP KPI Generation (`olap_build.py`)
3. Model Training (optional, use `--train` flag)
4. API Smoke Tests

**Usage**:
```bash
# Run without training (faster)
python backend/scripts/run_all.py

# Run with training (may take hours)
python backend/scripts/run_all.py --train
```

**Smoke Tests**:
- ✅ `GET /api/v1/healthz`
- ✅ `POST /api/v1/assessments/screening` (checks disclaimers, model_version)
- ✅ `POST /api/v1/assessments/triage` (if trigger_symptom=true)
- ✅ `GET /api/v1/olap/kpis/age_gender`
- ✅ `GET /api/v1/models/latest`

**Output**:
- ✅ PASS/FAIL summary
- ✅ Key artifact locations
- ✅ Color-coded terminal output

## Test Execution Order

### Recommended Workflow:

1. **Run Backend Tests** (no API server needed for most):
   ```bash
   python backend/scripts/tests/test_leakage.py
   python backend/scripts/tests/test_standardize.py
   ```

2. **Start API Server**:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Run API Contract Tests**:
   ```bash
   python backend/scripts/tests/test_api_contract.py
   ```

4. **Run Orchestration** (full pipeline):
   ```bash
   python backend/scripts/run_all.py
   ```

5. **Run Frontend Tests**:
   ```bash
   cd frontend
   npm test
   ```

## Expected Results

### Backend Tests
```
✅ All API contract tests passed!
✅ All leakage tests passed! No data leakage detected.
✅ All standardization tests passed!
```

### Orchestration Script
```
======================================================================
Summary
======================================================================
✅ ALL TESTS PASSED

Key Artifact Locations:
  • Standardized data: data/standardized/clean_assessments.parquet
  • OLAP KPIs: analytics/duckdb/agg/*.parquet
  • Model registry: modeling/registry/registry.json
  • Model artifacts: modeling/artifacts/*.joblib
  • Audit logs: backend/logs/audit.jsonl
```

## Definition of Done

- [x] All backend tests pass
- [x] API contract tests verify disclaimers and model_version
- [x] Leakage test confirms Stage A excludes symptoms
- [x] Frontend test verifies disclaimers always render
- [x] run_all.py prints PASS and key artifact locations
- [x] Smoke tests verify all endpoints work
- [x] Tests can run independently or together

## Troubleshooting

### API Tests Fail
- Ensure API server is running: `uvicorn backend.main:app --reload`
- Check server logs for errors
- Verify port 8000 is available

### Leakage Test Fails
- Check `backend/scripts/train_models_improved.py` for STAGE_A_FEATURES
- Verify STAGE_A_EXCLUDED list includes all symptom columns
- Ensure no symptom columns are accidentally included

### Frontend Tests Fail
- Install dependencies: `cd frontend && npm install`
- Check vitest config in `vite.config.ts`
- Verify test setup file exists: `frontend/src/test/setup.ts`

### Orchestration Script Fails
- Check that raw data exists: `data/raw/Dry_Eye_Dataset.csv`
- Verify Python environment has all dependencies
- Check file permissions for output directories
