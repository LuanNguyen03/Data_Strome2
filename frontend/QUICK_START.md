# Quick Start Guide - React Frontend

## Prerequisites

- Node.js 18+ installed
- Backend API running on `http://localhost:8000`

## Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

## Testing the Flow

### 1. Stage A (Screening)

1. Navigate to `http://localhost:5173/quick-assessment`
2. Fill in the form:
   - **Recommended fields**: average_screen_time, sleep_duration, sleep_quality, stress_level
   - **Optional fields**: age, gender, height, weight, etc.
3. Notice: You can submit even with missing fields (warnings will show)
4. Click "Xem kết quả sàng lọc"
5. You'll be redirected to `/result` with the assessment response

### 2. Stage B (Symptoms) - If Triggered

1. If `trigger_symptom=true` in the result, you'll see a CTA button
2. Click "Trả lời thêm triệu chứng (30 giây)"
3. You'll be redirected to `/symptoms`
4. Toggle the 3 symptom checkboxes:
   - Khó chịu / Mỏi mắt
   - Đỏ mắt
   - Ngứa / Kích ứng
5. Click "Phân loại với triệu chứng"
6. You'll see an updated result with Stage B assessment

### 3. Result Page

The result page shows:
- **Score** (0-100) with visual circle
- **Risk Level** badge (Low/Medium/High)
- **Confidence** badge with explanation
- **Missing Fields** warning (if any)
- **Top Factors** list
- **Next Steps** with actions
- **CTA to Symptoms** (if trigger_symptom=true)
- **Disclaimers** (always shown)

## Features to Test

### ✅ Missing Fields Handling
- Submit form with missing critical fields
- See warnings: "Thiếu thông tin: sleep_quality, average_screen_time"
- Form still submits successfully
- Result shows lower confidence

### ✅ State Persistence
- Fill form → refresh page → form data persists
- Get result → refresh page → result persists
- Navigate between pages → form data persists

### ✅ trigger_symptom CTA
- Submit Stage A with risk_score >= 65
- See CTA: "Trả lời thêm triệu chứng (30 giây)"
- Click CTA → navigate to symptoms page
- Fill symptoms → see updated result

### ✅ Disclaimers
- Result page always shows disclaimers
- Disclaimers match `docs/risk_copywriting_library.md`

## Troubleshooting

### API Connection Issues

If you see API errors:
1. Check backend is running: `curl http://localhost:8000/api/v1/healthz`
2. Check CORS settings in backend
3. Verify `VITE_API_URL` in `.env` (if using custom URL)

### localStorage Issues

If data doesn't persist:
1. Check browser console for errors
2. Verify localStorage is enabled
3. Check browser storage limits

### Routing Issues

If pages don't load:
1. Verify React Router is installed: `npm list react-router-dom`
2. Check browser console for errors
3. Verify all page components are imported correctly

## Environment Variables

Create `.env` file in `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

## Build for Production

```bash
npm run build
```

Output will be in `frontend/dist/`

## Code Structure

```
frontend/src/
├── pages/
│   ├── QuickAssessment.tsx  # Stage A form
│   ├── Symptoms.tsx          # Stage B symptom form
│   └── Result.tsx            # Result display
├── App.tsx                   # Router + layout
├── api/
│   └── client.ts            # API client
└── types.ts                  # TypeScript types
```

## Next Steps

1. Test end-to-end flow
2. Verify all copywriting matches library
3. Test on mobile devices
4. Verify accessibility
5. Test with different risk levels
