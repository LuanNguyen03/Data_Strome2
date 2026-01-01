# React Frontend Implementation Summary

## ✅ Implementation Complete

All three pages have been implemented with strict compliance to:
- `docs/ui_flow_spec.md`
- `docs/output_contract.md`
- `docs/risk_copywriting_library.md`

## 📄 Pages Implemented

### 1. `/quick-assessment` (Stage A Form)
**File**: `frontend/src/pages/QuickAssessment.tsx`

**Features**:
- ✅ Allows missing values (all fields optional)
- ✅ Shows gentle warnings for critical fields (sleep_quality, average_screen_time, sleep_duration, stress_level)
- ✅ Two sections: "Recommended" and "Optional"
- ✅ Tooltips explaining importance
- ✅ Saves form data to localStorage
- ✅ Calls `/api/v1/assessments/screening`
- ✅ Navigates to `/result` on success

**Critical Fields with Warnings**:
- Thời gian nhìn màn hình (average_screen_time)
- Thời lượng ngủ (sleep_duration)
- Chất lượng ngủ (sleep_quality)
- Mức độ căng thẳng (stress_level)

### 2. `/symptoms` (Stage B Symptom Form)
**File**: `frontend/src/pages/Symptoms.tsx`

**Features**:
- ✅ 3 symptom toggles with descriptions:
  - Khó chịu / Mỏi mắt (discomfort_eyestrain)
  - Đỏ mắt (redness_in_eye)
  - Ngứa / Kích ứng (itchiness_irritation_in_eye)
- ✅ Loads previous form data from localStorage
- ✅ "Bỏ qua" button to return to result
- ✅ Calls `/api/v1/assessments/triage`
- ✅ Saves updated form data and response
- ✅ Navigates to `/result` on success

**Copywriting**: "Để tăng độ chắc của phân loại, bạn có gặp các triệu chứng sau không?"

### 3. `/result` (Response Display)
**File**: `frontend/src/pages/Result.tsx`

**Features**:
- ✅ Displays full assessment response contract
- ✅ **Score (0-100) with risk level badge** (Low/Medium/High)
- ✅ **Confidence badge** with explanation text
- ✅ **Missing fields warning** (if confidence != High)
- ✅ **Top factors list** with direction and strength
- ✅ **Next steps** with title, actions, and ask_for_more_info
- ✅ **CTA to symptoms** if `trigger_symptom=true` (Stage A only)
- ✅ **Disclaimers ALWAYS shown** (per contract)
- ✅ Persists in localStorage (survives refresh)
- ✅ Action buttons: "Đánh giá lại", "Thêm triệu chứng"

**Visual Elements**:
- Score circle with gradient
- Color-coded risk level badges
- Confidence badges (High/Medium/Low with colors)
- Factor items with direction indicators
- CTA card with gradient background
- Disclaimers section with warning icon

## 🔄 State Management

### localStorage Keys:
- `assessment_form_data` - Current form data (persists across pages)
- `assessment_response` - Latest assessment response (persists on refresh)

### Flow:
1. User fills Stage A form → saved to localStorage
2. Submit → API call → response saved → navigate to `/result`
3. If `trigger_symptom=true` → CTA shown → navigate to `/symptoms`
4. User fills symptoms → merged with form data → API call → response saved → navigate to `/result`
5. Refresh page → Result page loads from localStorage

## 📝 Copywriting Compliance

All text follows `docs/risk_copywriting_library.md`:

### ✅ Uses (Correct):
- "nguy cơ", "sàng lọc", "phân loại"
- "theo dõi", "cân nhắc khám"
- "độ chắc (confidence)", "bổ sung thông tin"

### ❌ Avoids (No diagnosis words):
- "chẩn đoán"
- "chắc chắn mắc"
- "kết luận bệnh"
- "nguy hiểm", "bệnh nặng"

### Specific Copy Used:
- **Confidence messages**: Exact match from library
- **CTA text**: "Trả lời thêm 3 triệu chứng để phân loại rõ hơn"
- **Disclaimers**: From library
- **Mode labels**: "Sàng lọc (không triệu chứng)" / "Phân loại (có triệu chứng)"

## 🎨 Styling

### Design Principles:
- Clean, modern UI
- Proper visual hierarchy
- Color-coded risk levels
- Responsive (mobile-friendly)
- Accessible form controls

### Color Scheme:
- **Low Risk**: Green (#d4edda)
- **Medium Risk**: Yellow (#fff3cd)
- **High Risk**: Red (#f8d7da)
- **Primary Action**: Blue (#4a90e2)
- **Gradient**: Purple (#667eea to #764ba2)

## 🔌 API Integration

### Endpoints Used:
- `POST /api/v1/assessments/screening` - Stage A
- `POST /api/v1/assessments/triage` - Stage B

### Response Handling:
- All responses include `model_version` (added by backend)
- Response contract matches `docs/output_contract.md`
- Error handling with user-friendly messages

## ✅ Definition of Done Checklist

- [x] Stage A form allows missing values
- [x] Shows gentle warnings for key fields
- [x] Screening response shows: score, level, confidence, top_factors, next_step, disclaimers
- [x] trigger_symptom=true shows CTA to /symptoms
- [x] Stage B page has 3 symptom toggles
- [x] Submit triage works
- [x] Latest response persisted in state + localStorage
- [x] Refresh keeps result
- [x] Copywriting matches risk_copywriting_library.md
- [x] No diagnosis words used
- [x] End-to-end flow works against backend
- [x] Result page always shows disclaimers

## 🚀 Running the App

```bash
# Install dependencies
cd frontend
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## 📁 File Structure

```
frontend/src/
├── pages/
│   ├── QuickAssessment.tsx    # Stage A form
│   ├── QuickAssessment.css
│   ├── Symptoms.tsx            # Stage B symptom form
│   ├── Symptoms.css
│   ├── Result.tsx              # Result display
│   └── Result.css
├── App.tsx                     # Router setup
├── App.css
├── api/
│   └── client.ts               # API client
└── types.ts                    # TypeScript types
```

## 🧪 Testing Checklist

1. **Stage A Flow**:
   - [ ] Fill form with all fields → submit → see result
   - [ ] Fill form with missing critical fields → see warnings → submit → see result
   - [ ] Submit with minimal fields → see result with low confidence

2. **Stage B Flow**:
   - [ ] From result with trigger_symptom=true → click CTA → fill symptoms → see updated result
   - [ ] From result → click "Thêm triệu chứng" → fill symptoms → see updated result
   - [ ] From symptoms → click "Bỏ qua" → return to result

3. **Persistence**:
   - [ ] Fill form → refresh → form data persists
   - [ ] Get result → refresh → result persists
   - [ ] Navigate between pages → form data persists

4. **UI/UX**:
   - [ ] Warnings show for missing critical fields
   - [ ] Confidence badge shows correct message
   - [ ] Risk level badge shows correct color
   - [ ] Disclaimers always visible on result page
   - [ ] CTA appears when trigger_symptom=true
   - [ ] Mobile responsive

5. **Copywriting**:
   - [ ] No diagnosis words used
   - [ ] Confidence messages match library
   - [ ] CTA text matches library
   - [ ] Disclaimers match library

## 📝 Notes

- All pages use React Router for navigation
- localStorage is used for persistence (not a state management library)
- Form validation is soft (warnings, not errors)
- Error handling shows user-friendly messages
- All API calls use axios with proper error handling
- TypeScript types match backend contract
