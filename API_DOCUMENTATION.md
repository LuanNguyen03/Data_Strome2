# API v1 Documentation

## Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### 1. Health Check

**GET** `/api/v1/healthz`

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "dry-eye-assessment-api",
  "version": "v1"
}
```

---

### 2. Screening Assessment (Stage A)

**POST** `/api/v1/assessments/screening`

Performs Stage A screening assessment without using symptoms (no leakage).

**Request Body:**
```json
{
  "age": 30,
  "gender": 1,
  "sleep_duration": 7.5,
  "sleep_quality": 3,
  "average_screen_time": 8.0,
  "stress_level": 4,
  "sleep_disorder": 0,
  "smart_device_before_bed": 1,
  ...
}
```

**Note:** Symptoms fields are optional but **ignored** in Stage A.

**Response:** (See `docs/output_contract.md` for full schema)
```json
{
  "request_id": "req_abc123",
  "timestamp": "2025-12-29T20:00:00",
  "mode_used": "A_only_screening",
  "risk_score": 68.0,
  "risk_level": "Medium",
  "confidence": "Medium",
  "missing_fields": ["sleep_quality"],
  "top_factors": [...],
  "next_step": {...},
  "screening": {
    "risk_A": 68.0,
    "trigger_symptom": true
  },
  "disclaimers": [...]
}
```

---

### 3. Triage Assessment (Stage B)

**POST** `/api/v1/assessments/triage`

Performs Stage B triage assessment with symptoms.

**Request Body:**
```json
{
  "age": 30,
  "sleep_duration": 7.5,
  "sleep_quality": 3,
  "average_screen_time": 8.0,
  "discomfort_eyestrain": 1,
  "redness_in_eye": 1,
  "itchiness_irritation_in_eye": 0,
  ...
}
```

**Response:**
```json
{
  "request_id": "req_xyz789",
  "timestamp": "2025-12-29T20:00:00",
  "mode_used": "B_with_symptoms",
  "risk_score": 75.0,
  "risk_level": "High",
  "confidence": "High",
  "missing_fields": [],
  "top_factors": [...],
  "next_step": {...},
  "triage": {
    "prob_B": 75.0,
    "triage_level": "High"
  },
  "disclaimers": [...]
}
```

---

### 4. List OLAP KPIs

**GET** `/api/v1/olap/kpis`

Returns list of available KPI datasets.

**Response:**
```json
{
  "kpis": [
    {
      "name": "age_gender",
      "filename": "agg_ded_by_age_gender.parquet",
      "available": true,
      "row_count": 10
    },
    ...
  ],
  "total": 5
}
```

---

### 5. Get OLAP KPI Data

**GET** `/api/v1/olap/kpis/{name}`

Returns paginated KPI data.

**Path Parameters:**
- `name`: One of: `age_gender`, `screen_sleep`, `symptom_score`, `stress_sleepband`, `data_quality_group`

**Query Parameters:**
- `page` (default: 1): Page number (1-indexed)
- `page_size` (default: 100, max: 1000): Items per page

**Example:**
```
GET /api/v1/olap/kpis/age_gender?page=1&page_size=50
```

**Response:**
```json
{
  "name": "age_gender",
  "filename": "agg_ded_by_age_gender.parquet",
  "page": 1,
  "page_size": 50,
  "total_rows": 10,
  "total_pages": 1,
  "has_next": false,
  "has_prev": false,
  "data": [
    {
      "age_band": "18-24",
      "gender": 0,
      "n": 150,
      "positives": 45,
      "rate": 0.3
    },
    ...
  ]
}
```

## Request Validation

- Numeric fields have range validation (soft: allows missing, rejects insane values)
- All fields are optional except where noted
- Symptoms in screening request are ignored (no leakage)

## Response Format

All assessment responses follow the contract in `docs/output_contract.md`:
- Always includes `disclaimers` (per clinical governance)
- Always includes `request_id`, `timestamp`, `mode_used`
- `confidence` calculated from missing fields
- `top_factors` explain risk contributors
- `next_step` provides actionable recommendations

## Audit Logging

All assessment requests are logged to `backend/logs/audit.jsonl` (append-only JSONL):
- `request_id`
- `timestamp`
- `mode_used`
- `risk_level`
- `confidence`
- `missing_fields_count`
- `model_version`

No personal identifiers are logged.

## Error Responses

**400 Bad Request:** Invalid input (range violations, etc.)
**404 Not Found:** KPI not found
**500 Internal Server Error:** Assessment or data access failure

## CORS

CORS enabled for:
- `http://localhost:3000` (React dev)
- `http://localhost:5173` (Vite dev)

