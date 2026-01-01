# OLAP Implementation Guide

## Overview

OLAP aggregates are generated using DuckDB from standardized parquet data. The implementation follows the specs in:
- `docs/02_olap_duckdb_plan.md`
- `docs/olap_queries.md`
- `docs/metrics_and_reporting.md`

## Usage

### CLI Entrypoint (Recommended)

```bash
python -m backend.scripts.olap_build \
  --input data/standardized/clean_assessments.parquet \
  --outdir analytics/duckdb/agg \
  --csv  # Optional: also export CSV files
```

### Legacy Script (Backward Compatible)

```bash
python analytics/duckdb/build_agg.py \
  --input data/standardized/clean_assessments.parquet \
  --outdir analytics/duckdb/agg
```

## Output Files

The script generates 5 KPI parquet files:

1. **agg_ded_by_age_gender.parquet**
   - Dimensions: `age_band`, `gender`
   - Metrics: `n`, `positives`, `rate`

2. **agg_ded_by_screen_sleep.parquet**
   - Dimensions: `screen_time_band`, `sleep_quality`
   - Metrics: `n`, `positives`, `rate`

3. **agg_ded_by_symptom_score.parquet**
   - Dimensions: `symptom_score`
   - Metrics: `n`, `positives`, `rate`

4. **agg_ded_by_stress_sleepband.parquet**
   - Dimensions: `stress_level`, `sleep_duration_band`
   - Metrics: `n`, `positives`, `rate`

5. **agg_data_quality_by_group.parquet**
   - Dimensions: `age_band`, `gender`
   - Metrics: `n`, `missing_rate_screen_time`, `missing_rate_sleep_quality`, `missing_rate_bp`, `avg_validity_ratio`

## Snapshot Report

The script also generates `olap_snapshot.json` with:
- Generation timestamp
- Input file path
- Total rows processed
- Summary statistics for each KPI:
  - Number of cells
  - Total n, positives, rate
  - Min/max n per cell
  - Warning flags for low-n cells (< 10)

## Key Features

- **No Server Dependency**: Uses embedded DuckDB
- **Handles NULLs**: Filters out NULL dimensions (keeps data quality intact)
- **Low-n Warnings**: Flags cells with n < 10 for UI warnings
- **Preserves All Cells**: Does NOT drop small cells (per requirements)
- **Type Safety**: Proper casting for rate calculations

## Query Logic

All queries follow the pattern:
```sql
SELECT
    dimension1, dimension2, ...
    COUNT(*) AS n,
    SUM(COALESCE(dry_eye_disease, 0)) AS positives,
    CASE 
        WHEN COUNT(*) = 0 THEN NULL 
        ELSE CAST(SUM(COALESCE(dry_eye_disease, 0)) AS DOUBLE) / COUNT(*) 
    END AS rate
FROM input_data
WHERE dimension1 IS NOT NULL AND dimension2 IS NOT NULL
GROUP BY dimension1, dimension2, ...
ORDER BY dimension1, dimension2, ...;
```

## Data Quality KPI

The data quality KPI uses different metrics:
- `missing_rate_screen_time`: % of NULL `average_screen_time`
- `missing_rate_sleep_quality`: % of NULL `sleep_quality`
- `missing_rate_bp`: % where `bp_parse_ok = 0` or NULL
- `avg_validity_ratio`: Average validity ratio per group

## Integration with Dashboard

The generated parquet files can be:
1. Read directly by DuckDB in dashboard queries
2. Converted to CSV for Excel/BI tools
3. Loaded into pandas/polars for Python analysis

Example dashboard query:
```python
import duckdb
con = duckdb.connect()
df = con.execute(
    "SELECT * FROM read_parquet('analytics/duckdb/agg/agg_ded_by_age_gender.parquet')"
).fetchdf()
```

## Troubleshooting

### Input file not found
- Run standardization first: `python scripts/standardize.py`

### Empty aggregates
- Check that standardized data has the required derived columns:
  - `age_band`, `screen_time_band`, `sleep_duration_band`, `symptom_score`
- Verify data quality report shows these columns exist

### Low-n warnings
- This is expected for sparse combinations
- UI should display warnings when n < 10
- Consider aggregating bands if too sparse

