# Quick Test Runner Guide

## Running Tests

### Backend Tests

#### 1. Leakage Test (Fast, No Dependencies)
```bash
python backend/scripts/tests/test_leakage.py
```

**Expected Output**:
```
Running Leakage Tests...
======================================================================
✅ No symptom columns found in STAGE_A_FEATURES
✅ STAGE_A_EXCLUDED list properly defined
✅ Stage A features are safe (no symptom leakage)
✅ Stage B correctly includes symptoms: ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
======================================================================
✅ All leakage tests passed! No data leakage detected.
```

#### 2. Standardization Test
```bash
python backend/scripts/tests/test_standardize.py
```

#### 3. API Contract Test (Requires API Server)
```bash
# Terminal 1: Start API server
uvicorn backend.main:app --reload

# Terminal 2: Run tests
python backend/scripts/tests/test_api_contract.py
```

**Expected Output**:
```
Running API Contract Tests...
✅ Screening contract test passed
✅ Triage contract test passed
✅ Disclaimers content test passed
✅ Model version test passed

✅ All API contract tests passed!
```

#### 4. Run All Backend Tests
```bash
python backend/scripts/tests/test_all.py
```

### Frontend Tests

```bash
cd frontend
npm install  # Install test dependencies if needed
npm test
```

**Expected Output**:
```
✓ Result Page - Disclaimers Always Render (4)
  ✓ should always render disclaimers for Stage A response
  ✓ should always render disclaimers for Stage B response
  ✓ should render all disclaimers from response
  ✓ should render disclaimers section even with empty disclaimers array

Test Files  1 passed (1)
     Tests  4 passed (4)
```

### One-Shot Orchestration

```bash
# Without training (faster)
python backend/scripts/run_all.py

# With training (may take hours)
python backend/scripts/run_all.py --train
```

**Expected Output** (on success):
```
======================================================================
Summary
======================================================================

Key Artifact Locations:
  • Standardized data: data/standardized/clean_assessments.parquet
  • OLAP KPIs: analytics/duckdb/agg/*.parquet
  • Model registry: modeling/registry/registry.json
  • Model artifacts: modeling/artifacts/*.joblib
  • Audit logs: backend/logs/audit.jsonl

✅ ALL TESTS PASSED
```

## Test Checklist

Before committing/deploying, run:

- [ ] `python backend/scripts/tests/test_leakage.py` - PASS
- [ ] `python backend/scripts/tests/test_standardize.py` - PASS
- [ ] `python backend/scripts/tests/test_api_contract.py` - PASS (with server)
- [ ] `cd frontend && npm test` - PASS
- [ ] `python backend/scripts/run_all.py` - PASS (orchestration)

## Troubleshooting

### API Contract Tests Fail
- Ensure API server is running: `uvicorn backend.main:app --reload`
- Check server logs for errors
- Verify endpoints are accessible: `curl http://localhost:8000/api/v1/healthz`

### Frontend Tests Fail
- Install dependencies: `cd frontend && npm install`
- Check `vite.config.ts` has test configuration
- Verify `frontend/src/test/setup.ts` exists

### Orchestration Script Fails
- Check raw data exists: `data/raw/Dry_Eye_Dataset.csv`
- Verify Python environment has all dependencies
- Check file permissions for output directories
- API server not required (smoke tests skip gracefully)
