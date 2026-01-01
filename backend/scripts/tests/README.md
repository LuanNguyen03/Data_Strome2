# Backend Tests

## Test Files

### 1. `test_standardize.py`
Tests for data standardization pipeline (already exists)
- BP parsing
- Banding logic
- Range validation

### 2. `test_api_contract.py` ✅ NEW
API contract compliance tests per `docs/clinical_governance_checklist.md`
- Screening response always includes disclaimers and model_version
- Triage response always includes disclaimers and model_version
- Disclaimers contain required medical language
- Model version always present (even with fallback)

### 3. `test_leakage.py` ✅ NEW
Data leakage prevention tests
- Stage A feature list excludes symptom columns
- STAGE_A_EXCLUDED list properly defined
- Stage A features are safe (no symptom leakage)
- Stage B correctly includes symptoms (sanity check)

## Running Tests

### Run Individual Tests

```bash
# Standardization tests
python backend/scripts/tests/test_standardize.py

# API contract tests (requires API server running)
python backend/scripts/tests/test_api_contract.py

# Leakage tests
python backend/scripts/tests/test_leakage.py
```

### Run All Tests

```bash
python backend/scripts/tests/test_all.py
```

### Using pytest

```bash
pytest backend/scripts/tests/
```

## Test Requirements

### API Contract Tests
- Requires FastAPI server running on `http://localhost:8000`
- Start server: `uvicorn backend.main:app --reload`
- Tests verify contract compliance per `docs/output_contract.md`

### Leakage Tests
- No external dependencies
- Tests feature lists from `train_models_improved.py`
- Ensures Stage A never uses symptom data

## Expected Output

```
Running API Contract Tests...
✅ Screening contract test passed
✅ Triage contract test passed
✅ Disclaimers content test passed
✅ Model version test passed

✅ All API contract tests passed!
```
