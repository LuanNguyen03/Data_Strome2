# React TypeScript Frontend - Dry Eye Assessment

## Overview

Three-page React application implementing the assessment flow per `docs/ui_flow_spec.md`:

1. **`/quick-assessment`** - Stage A screening form
2. **`/symptoms`** - Stage B symptom form
3. **`/result`** - Results display with contract compliance

## Features

### ✅ Hard Requirements Met

1. **Stage A form allows missing values** - Shows gentle warnings for key fields
2. **Screening response display** - Shows score, level, confidence, top_factors, next_step, disclaimers
3. **trigger_symptom CTA** - Shows CTA to `/symptoms` if `trigger_symptom=true`
4. **Stage B page** - 3 symptom toggles with descriptions
5. **State persistence** - Uses localStorage to persist form data and responses
6. **Copywriting compliance** - Matches `docs/risk_copywriting_library.md` (no diagnosis words)
7. **Disclaimers always shown** - Result page always displays disclaimers

## Setup

```bash
cd frontend
npm install
npm run dev
```

## Pages

### `/quick-assessment` (Stage A)

- Form with recommended and optional fields
- Gentle warnings for missing critical fields (sleep_quality, average_screen_time, sleep_duration, stress_level)
- Allows submission even with missing fields
- Saves form data to localStorage
- On submit: calls `/api/v1/assessments/screening`, saves response, navigates to `/result`

### `/symptoms` (Stage B)

- 3 symptom toggles with descriptions:
  - Khó chịu / Mỏi mắt
  - Đỏ mắt
  - Ngứa / Kích ứng
- Loads previous form data from localStorage
- "Bỏ qua" button to return to result
- On submit: calls `/api/v1/assessments/triage`, saves response, navigates to `/result`

### `/result`

- Displays full assessment response contract
- Shows:
  - Score (0-100) with risk level badge
  - Confidence badge with explanation
  - Missing fields warning (if any)
  - Top factors list
  - Next steps with actions
  - **CTA to symptoms** (if `trigger_symptom=true`)
  - **Disclaimers** (always shown)
- Persists in localStorage (survives refresh)
- Action buttons: "Đánh giá lại", "Thêm triệu chứng"

## State Management

- **Form data**: Saved to `localStorage.getItem('assessment_form_data')`
- **Response**: Saved to `localStorage.getItem('assessment_response')`
- Both persist across page refreshes

## Copywriting

All copywriting follows `docs/risk_copywriting_library.md`:

- ✅ Uses: "nguy cơ", "sàng lọc", "phân loại", "theo dõi", "cân nhắc khám"
- ❌ Avoids: "chẩn đoán", "chắc chắn mắc", "kết luận bệnh"
- Confidence messages match library exactly
- CTA text matches library
- Disclaimers match library

## API Integration

- Base URL: `http://localhost:8000` (or `VITE_API_URL` env var)
- Endpoints:
  - `POST /api/v1/assessments/screening`
  - `POST /api/v1/assessments/triage`
- All responses include `model_version` (added by backend)

## Styling

- Modern, clean UI with proper spacing
- Responsive design (mobile-friendly)
- Color-coded risk levels (Low/Medium/High)
- Visual hierarchy for important information
- Accessible form controls

## Testing

1. Start backend: `uvicorn backend.main:app --reload`
2. Start frontend: `npm run dev`
3. Navigate to `http://localhost:5173`
4. Test flow:
   - Fill Stage A form (can skip fields)
   - Submit → see result
   - If `trigger_symptom=true`, click CTA → fill symptoms → see updated result
   - Refresh page → result persists

## Definition of Done

- [x] End-to-end flow works against backend
- [x] Result page always shows disclaimers
- [x] Stage A form allows missing values with warnings
- [x] trigger_symptom CTA works
- [x] State persists in localStorage
- [x] Copywriting matches library
- [x] All three pages implemented
- [x] Routing configured
- [x] Responsive design
