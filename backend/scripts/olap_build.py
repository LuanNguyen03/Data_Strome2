"""
OLAP generation using DuckDB.
Generates 5 KPI datasets from standardized parquet data.

Input: data/standardized/clean_assessments.parquet
Output: analytics/duckdb/agg/*.parquet (and optionally CSV)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import duckdb


# KPI 1: DED rate by age_band × gender
QUERY_AGE_GENDER = """
    SELECT
        age_band,
        gender,
        COUNT(*) AS n,
        SUM(COALESCE(dry_eye_disease, 0)) AS positives,
        CASE 
            WHEN COUNT(*) = 0 THEN NULL 
            ELSE CAST(SUM(COALESCE(dry_eye_disease, 0)) AS DOUBLE) / COUNT(*) 
        END AS rate
    FROM input_data
    WHERE age_band IS NOT NULL AND gender IS NOT NULL
    GROUP BY age_band, gender
    ORDER BY age_band, gender;
"""

# KPI 2: Heatmap screen_time_band × sleep_quality
QUERY_SCREEN_SLEEP = """
    SELECT
        screen_time_band,
        sleep_quality,
        COUNT(*) AS n,
        SUM(COALESCE(dry_eye_disease, 0)) AS positives,
        CASE 
            WHEN COUNT(*) = 0 THEN NULL 
            ELSE CAST(SUM(COALESCE(dry_eye_disease, 0)) AS DOUBLE) / COUNT(*) 
        END AS rate
    FROM input_data
    WHERE screen_time_band IS NOT NULL AND sleep_quality IS NOT NULL
    GROUP BY screen_time_band, sleep_quality
    ORDER BY screen_time_band, sleep_quality;
"""

# KPI 3: DED rate by symptom_score
QUERY_SYMPTOM_SCORE = """
    SELECT
        symptom_score,
        COUNT(*) AS n,
        SUM(COALESCE(dry_eye_disease, 0)) AS positives,
        CASE 
            WHEN COUNT(*) = 0 THEN NULL 
            ELSE CAST(SUM(COALESCE(dry_eye_disease, 0)) AS DOUBLE) / COUNT(*) 
        END AS rate
    FROM input_data
    WHERE symptom_score IS NOT NULL
    GROUP BY symptom_score
    ORDER BY symptom_score;
"""

# KPI 4: DED rate by stress_level × sleep_duration_band
QUERY_STRESS_SLEEPBAND = """
    SELECT
        stress_level,
        sleep_duration_band,
        COUNT(*) AS n,
        SUM(COALESCE(dry_eye_disease, 0)) AS positives,
        CASE 
            WHEN COUNT(*) = 0 THEN NULL 
            ELSE CAST(SUM(COALESCE(dry_eye_disease, 0)) AS DOUBLE) / COUNT(*) 
        END AS rate
    FROM input_data
    WHERE stress_level IS NOT NULL AND sleep_duration_band IS NOT NULL
    GROUP BY stress_level, sleep_duration_band
    ORDER BY stress_level, sleep_duration_band;
"""

# KPI 5: Data quality by group
QUERY_DATA_QUALITY = """
    SELECT
        age_band,
        gender,
        COUNT(*) AS n,
        CAST(AVG(CASE WHEN average_screen_time IS NULL THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS missing_rate_screen_time,
        CAST(AVG(CASE WHEN sleep_quality IS NULL THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS missing_rate_sleep_quality,
        CAST(AVG(CASE WHEN bp_parse_ok = 0 OR bp_parse_ok IS NULL THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS missing_rate_bp,
        CAST(AVG(COALESCE(validity_ratio, 0.0)) AS DOUBLE) AS avg_validity_ratio
    FROM input_data
    WHERE age_band IS NOT NULL AND gender IS NOT NULL
    GROUP BY age_band, gender
    ORDER BY age_band, gender;
"""


KPI_CONFIGS = [
    {
        "filename": "agg_ded_by_age_gender.parquet",
        "query": QUERY_AGE_GENDER,
        "description": "DED rate by age_band × gender",
        "dimensions": ["age_band", "gender"],
    },
    {
        "filename": "agg_ded_by_screen_sleep.parquet",
        "query": QUERY_SCREEN_SLEEP,
        "description": "Heatmap: screen_time_band × sleep_quality",
        "dimensions": ["screen_time_band", "sleep_quality"],
    },
    {
        "filename": "agg_ded_by_symptom_score.parquet",
        "query": QUERY_SYMPTOM_SCORE,
        "description": "DED rate by symptom_score",
        "dimensions": ["symptom_score"],
    },
    {
        "filename": "agg_ded_by_stress_sleepband.parquet",
        "query": QUERY_STRESS_SLEEPBAND,
        "description": "DED rate by stress_level × sleep_duration_band",
        "dimensions": ["stress_level", "sleep_duration_band"],
    },
    {
        "filename": "agg_data_quality_by_group.parquet",
        "query": QUERY_DATA_QUALITY,
        "description": "Data quality metrics by age_band × gender",
        "dimensions": ["age_band", "gender"],
    },
]


def build_olap_aggregates(input_path: Path, output_dir: Path, export_csv: bool = False) -> Dict[str, Any]:
    """
    Build all OLAP aggregates using DuckDB.
    
    Returns summary statistics for snapshot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to DuckDB
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false;")
    
    # Create view from parquet
    print(f"Loading data from {input_path}...")
    # DuckDB doesn't support prepared statements for CREATE VIEW with read_parquet
    # Use string formatting instead (input_path is validated to be a Path object)
    input_path_str = str(input_path).replace("\\", "/")  # Normalize path separators
    con.execute(
        f"CREATE OR REPLACE VIEW input_data AS SELECT * FROM read_parquet('{input_path_str}');"
    )
    
    # Get total rows for summary
    total_rows = con.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    print(f"Total rows: {total_rows:,}")
    
    snapshot_data = {
        "generated_at": datetime.now().isoformat(),
        "input_file": str(input_path),
        "total_rows": total_rows,
        "kpis": [],
    }
    
    # Build each KPI
    for config in KPI_CONFIGS:
        filename = config["filename"]
        query = config["query"]
        description = config["description"]
        
        print(f"\nBuilding {filename}...")
        print(f"  Description: {description}")
        
        # Execute query and get results
        result = con.execute(query).fetchdf()
        
        # Get summary stats
        total_n = result["n"].sum() if "n" in result.columns and len(result) > 0 else 0
        total_positives = result["positives"].sum() if "positives" in result.columns and len(result) > 0 else 0
        overall_rate = total_positives / total_n if total_n > 0 else None
        num_cells = len(result)
        # Handle NaN values from empty results
        import math
        min_n_val = result["n"].min() if "n" in result.columns and len(result) > 0 else None
        max_n_val = result["n"].max() if "n" in result.columns and len(result) > 0 else None
        min_n = int(min_n_val) if min_n_val is not None and not math.isnan(min_n_val) else 0
        max_n = int(max_n_val) if max_n_val is not None and not math.isnan(max_n_val) else 0
        
        # Write parquet
        parquet_path = output_dir / filename
        parquet_path_str = str(parquet_path).replace("\\", "/")  # Normalize path separators
        # Remove trailing semicolon from query for COPY statement
        query_clean = query.rstrip().rstrip(";")
        con.execute(
            f"COPY ({query_clean}) TO '{parquet_path_str}' (FORMAT 'parquet');"
        )
        print(f"  ✓ Written: {parquet_path}")
        
        # Optionally write CSV
        if export_csv:
            csv_path = output_dir / filename.replace(".parquet", ".csv")
            result.to_csv(csv_path, index=False)
            print(f"  ✓ Written: {csv_path}")
        
        # Store snapshot info
        snapshot_data["kpis"].append({
            "filename": filename,
            "description": description,
            "dimensions": config["dimensions"],
            "num_cells": num_cells,
            "total_n": int(total_n),
            "total_positives": int(total_positives),
            "overall_rate": float(overall_rate) if overall_rate is not None else None,
            "min_n": int(min_n),
            "max_n": int(max_n),
            "has_low_n_cells": min_n < 10,  # Flag if any cell has n < 10
        })
        
        print(f"  Summary: {num_cells} cells, n={total_n:,}, positives={total_positives:,}, rate={overall_rate:.2%}" if overall_rate else f"  Summary: {num_cells} cells, n={total_n:,}")
        if min_n < 10:
            print(f"  ⚠ Warning: Some cells have n < 10 (min={min_n})")
    
    con.close()
    
    # Write snapshot JSON
    snapshot_path = output_dir / "olap_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Snapshot written: {snapshot_path}")
    
    print(f"\n✅ All aggregates written to {output_dir}")
    return snapshot_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OLAP aggregates using DuckDB from standardized parquet data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/standardized/clean_assessments.parquet"),
        help="Path to standardized parquet file",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analytics/duckdb/agg"),
        help="Output directory for aggregate parquet files",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export CSV files alongside parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        print("   Run standardization first: python scripts/standardize.py")
        return
    
    try:
        snapshot = build_olap_aggregates(args.input, args.outdir, export_csv=args.csv)
        print("\n✅ OLAP generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

