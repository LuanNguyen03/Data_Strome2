# Dashboard Implementation Summary

## ✅ Implementation Complete

Dashboard page with 3 tabs implemented per `docs/metrics_and_reporting.md` and `docs/02_olap_duckdb_plan.md`.

## 📊 Dashboard Features

### Route: `/dashboard`

### Three Tabs:

1. **Tổng quan (Overview)**
   - Summary cards: Total records, DED positive rate, Avg validity ratio
   - Bar chart: DED rate by age_band × gender
   - Data quality chart: Top 5 missing fields

2. **Yếu tố nguy cơ (Risk Drivers)**
   - Heatmap: screen_time_band × sleep_quality (rate + n)
   - Heatmap: stress_level × sleep_duration_band (rate + n)
   - Bar chart: DED rate by screen_time_band

3. **Triệu chứng & Phân loại (Symptom & Triage)**
   - Bar chart: DED rate by symptom_score (0-3)
   - Table: Symptom score details (n, positives, rate)

## 🔌 API Integration

**Fetches from:**
- `GET /api/v1/olap/kpis/age_gender`
- `GET /api/v1/olap/kpis/screen_sleep`
- `GET /api/v1/olap/kpis/symptom_score`
- `GET /api/v1/olap/kpis/stress_sleepband`
- `GET /api/v1/olap/kpis/data_quality_group`

**Data Format:**
- Each KPI returns paginated JSON with `data` array
- Each row contains: dimensions, `n`, `positives`, `rate`

## 📈 Chart Requirements Met

### ✅ Every Chart/Table Shows:
- **n** (count) - Always displayed
- **rate** (percentage) - Always displayed
- **Footnote**: "Tương quan, không kết luận nhân quả" - On every chart

### ✅ Heatmap Tables:
- Show rate + n in each cell
- Color intensity based on rate
- Tooltip on hover with full details
- Row totals shown

### ✅ Bar Charts:
- Show rate and n in bar labels
- Responsive width based on max rate
- Clear axis labels

## 🎨 Visual Design

### Heatmap Features:
- Color gradient: Light red (low rate) → Dark red (high rate)
- Text color adjusts for readability (white on dark, black on light)
- Cell shows: rate (large) + n (small)
- Row headers show totals

### Bar Chart Features:
- Color-coded bars
- Value labels on bars: "XX.X% (n=YYYY)"
- Responsive layout

### Summary Cards:
- Gradient background
- Large numbers
- Clear labels

## 📝 Copywriting Compliance

- ✅ All footnotes: "Tương quan, không kết luận nhân quả"
- ✅ Chart titles are questions/descriptions (not just "Chart A")
- ✅ Subtitles show n and missing info
- ✅ No diagnosis language

## 🔧 Technical Implementation

### Components:
- `Dashboard.tsx` - Main component with tabs
- `OverviewTab` - Overview content
- `RiskDriversTab` - Risk drivers content
- `SymptomTriageTab` - Symptom & triage content
- Helper components: `AgeGenderChart`, `HeatmapTable`, `BarChart`, `DataQualityChart`, `SymptomScoreTable`

### State Management:
- Loads all KPIs on mount
- Error handling with retry
- Loading states
- Handles missing data gracefully

### Data Processing:
- Aggregates data for bar charts
- Formats rates as percentages
- Formats numbers with locale
- Handles null/undefined values

## ✅ Definition of Done

- [x] Dashboard renders using real KPI parquet outputs
- [x] Heatmap-like tables show rate + n
- [x] Every chart displays n and rate
- [x] Footnote on every chart: "tương quan, không kết luận nhân quả"
- [x] Three tabs implemented (Overview, Risk Drivers, Symptom & Triage)
- [x] Fetches from `/api/v1/olap/kpis/{name}`
- [x] Responsive design
- [x] Error handling

## 🚀 Usage

1. Ensure backend OLAP KPI files are generated
2. Start backend: `uvicorn backend.main:app --reload`
3. Start frontend: `npm run dev`
4. Navigate to: `http://localhost:5173/dashboard`

## 📁 Files Created

```
frontend/src/pages/
├── Dashboard.tsx    ✅ NEW
└── Dashboard.css    ✅ NEW
```

## 🧪 Testing

1. **Test with real data:**
   - Ensure OLAP parquet files exist in `analytics/duckdb/agg/`
   - Check backend `/api/v1/olap/kpis` returns available KPIs
   - Navigate to dashboard and verify all tabs load

2. **Test with missing data:**
   - Dashboard should handle missing KPIs gracefully
   - Shows "no data" messages where appropriate

3. **Test responsiveness:**
   - Check on mobile/tablet
   - Heatmaps should scroll horizontally
   - Charts should stack vertically

## 📊 KPI Data Structure

Each KPI dataset contains rows with:
- Dimension fields (age_band, gender, screen_time_band, etc.)
- `n`: Count of records
- `positives`: Count of positive cases
- `rate`: positives / n

Example:
```json
{
  "age_band": "18-25",
  "gender": 0,
  "n": 1250,
  "positives": 815,
  "rate": 0.652
}
```

## 🎯 Key Features

1. **Real-time data**: Fetches from backend API
2. **Visual heatmaps**: Color-coded by rate intensity
3. **Comprehensive charts**: Bar charts, tables, heatmaps
4. **Data quality**: Shows missing rates and validity
5. **Responsive**: Works on all screen sizes
6. **Error handling**: Graceful degradation
7. **Loading states**: User feedback during data fetch
