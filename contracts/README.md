# Shared Contracts

This folder contains shared API contracts used by both backend (FastAPI/Pydantic) and frontend (TypeScript).

## Files

- `schemas.py` - Pydantic models for backend validation and serialization
- `types.ts` - TypeScript types for frontend type safety

## Usage

### Backend

```python
from contracts import AssessmentRequest, AssessmentResponse
```

### Frontend

```typescript
import { AssessmentRequest, AssessmentResponse } from '../types'
```

## Contract Specification

See `docs/output_contract.md` for the full API contract specification.

