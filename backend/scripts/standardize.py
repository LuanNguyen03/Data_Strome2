"""
Deterministic standardization pipeline.
Follows docs/01_data_standardization.md and docs/data_dictionary.md exactly.

Input: data/raw/Dry_Eye_Dataset.csv
Output: 
  - data/standardized/clean_assessments.parquet
  - data/standardized/data_quality_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl


# Mapping rules from docs
BINARY_MAP = {"Y": 1, "N": 0, "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0}
GENDER_MAP = {"F": 0, "M": 1}

# Range rules from docs/01_data_standardization.md
RANGE_RULES = {
    "age": (18, 45),
    "sleep_quality": (1, 5),
    "stress_level": (1, 5),
    "sleep_duration": (0, 24),
    "average_screen_time": (0, 24),
    "heart_rate": (40, 220),
    "daily_steps": (0, 50000),
    "physical_activity": (0, 600),
    "height": (120, 230),
    "weight": (30, 250),
    "systolic": (70, 250),
    "diastolic": (40, 150),
}

BINARY_COLS = [
    "sleep_disorder", "wake_up_during_night", "feel_sleepy_during_day",
    "caffeine_consumption", "alcohol_consumption", "smoking",
    "medical_issue", "ongoing_medication", "smart_device_before_bed",
    "blue_light_filter", "discomfort_eye_strain", "redness_in_eye",
    "itchiness_irritation_in_eye", "dry_eye_disease"
]

def snake_case(name: str) -> str:
    """Convert column name to snake_case"""
    s = name.strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    return s.strip("_").lower()

def parse_blood_pressure(bp_series: pl.Series) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """Parse '120/80' into systolic, diastolic, and success flag"""
    # Pattern: one or more digits, optional separator, one or more digits
    pattern = r"(\d+)\s*[/-]\s*(\d+)"
    
    systolic = bp_series.str.extract(pattern, 1).cast(pl.Int64, strict=False)
    diastolic = bp_series.str.extract(pattern, 2).cast(pl.Int64, strict=False)
    
    # Success flag: both are not null
    ok = (systolic.is_not_null() & diastolic.is_not_null()).cast(pl.Int64)
    
    return systolic.alias("systolic"), diastolic.alias("diastolic"), ok.alias("bp_parse_ok")

def create_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add bmi, bands, and symptom_score"""
    
    # 1. BMI = weight / (height/100)^2
    df = df.with_columns([
        (pl.col("weight") / ((pl.col("height") / 100.0) ** 2)).alias("bmi")
    ])
    
    # 2. Age bands: 18-24, 25-29, 30-34, 35-39, 40-45
    df = df.with_columns([
        pl.when(pl.col("age") < 25).then(pl.lit("18-24"))
        .when(pl.col("age") < 30).then(pl.lit("25-29"))
        .when(pl.col("age") < 35).then(pl.lit("30-34"))
        .when(pl.col("age") < 40).then(pl.lit("35-39"))
        .otherwise(pl.lit("40-45"))
        .alias("age_band")
    ])
    
    # 3. Screen time bands: 0-2, 2-4, 4-6, 6-8, 8-10, 10+
    df = df.with_columns([
        pl.when(pl.col("average_screen_time") < 2).then(pl.lit("0-2"))
        .when(pl.col("average_screen_time") < 4).then(pl.lit("2-4"))
        .when(pl.col("average_screen_time") < 6).then(pl.lit("4-6"))
        .when(pl.col("average_screen_time") < 8).then(pl.lit("6-8"))
        .when(pl.col("average_screen_time") < 10).then(pl.lit("8-10"))
        .otherwise(pl.lit("10+"))
        .alias("screen_time_band")
    ])
    
    # 4. Sleep duration bands: <6, 6-7, 7-8, 8-9, 9+
    df = df.with_columns([
        pl.when(pl.col("sleep_duration") < 6).then(pl.lit("<6"))
        .when(pl.col("sleep_duration") < 7).then(pl.lit("6-7"))
        .when(pl.col("sleep_duration") < 8).then(pl.lit("7-8"))
        .when(pl.col("sleep_duration") < 9).then(pl.lit("8-9"))
        .otherwise(pl.lit("9+"))
        .alias("sleep_duration_band")
    ])
    
    # 5. Symptom score = sum of 3 symptoms
    symptom_cols = ["discomfort_eye_strain", "redness_in_eye", "itchiness_irritation_in_eye"]
    available_symptoms = [c for c in symptom_cols if c in df.columns]
    if available_symptoms:
        df = df.with_columns([
            pl.sum_horizontal(available_symptoms).alias("symptom_score")
        ])
    else:
        df = df.with_columns([pl.lit(None).cast(pl.Int64).alias("symptom_score")])
        
    return df

def standardize(input_path: Path, output_dir: Path):
    print(f"Reading {input_path}...")
    # Use Pandas for initial read to handle potential BOM or weird CSV formats
    df_raw = pd.read_csv(input_path)
    
    # 1. snake_case all columns
    df_raw.columns = [snake_case(c) for c in df_raw.columns]
    
    # Map 'discomfort_eye_strain' if it was named slightly differently (e.g. 'discomfort_eyestrain')
    if 'discomfort_eye_strain' not in df_raw.columns and 'discomfort_eyestrain' in df_raw.columns:
        df_raw = df_raw.rename(columns={'discomfort_eyestrain': 'discomfort_eye_strain'})
    
    # Convert to Polars for faster deterministic processing
    df = pl.from_pandas(df_raw)
    
    # Store raw BP for audit
    if "blood_pressure" in df.columns:
        df = df.with_columns([pl.col("blood_pressure").alias("blood_pressure_raw")])
    
    # 2. Map Binary Y/N
    for col in BINARY_COLS:
        if col in df.columns:
            # Handle mixed types by converting to string first
            df = df.with_columns([
                pl.col(col).cast(pl.String).str.to_uppercase().replace(BINARY_MAP, default=None).cast(pl.Int64).alias(col)
            ])
            
    # 3. Map Gender
    if "gender" in df.columns:
        df = df.with_columns([
            pl.col("gender").cast(pl.String).str.to_uppercase().replace(GENDER_MAP, default=None).cast(pl.Int64)
        ])
        
    # 4. Parse Blood Pressure
    if "blood_pressure" in df.columns:
        sys, dia, ok = parse_blood_pressure(df["blood_pressure"])
        df = df.with_columns([sys, dia, ok])
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Int64).alias("systolic"),
            pl.lit(None).cast(pl.Int64).alias("diastolic"),
            pl.lit(0).cast(pl.Int64).alias("bp_parse_ok")
        ])
        
    # 5. Apply Range Rules
    in_range_cols = []
    for col, (low, high) in RANGE_RULES.items():
        if col in df.columns:
            flag_col = f"{col}_in_range"
            # Flag = 1 if in range AND not null
            df = df.with_columns([
                pl.when((pl.col(col) >= low) & (pl.col(col) <= high))
                .then(1).otherwise(0).alias(flag_col)
            ])
            # Set to NULL if out of range
            df = df.with_columns([
                pl.when(pl.col(flag_col) == 1).then(pl.col(col)).otherwise(None).alias(col)
            ])
            in_range_cols.append(flag_col)
            
    # 6. Validity Ratio
    if in_range_cols:
        df = df.with_columns([
            (pl.sum_horizontal(in_range_cols) / len(in_range_cols)).alias("validity_ratio")
        ])
    else:
        df = df.with_columns([pl.lit(1.0).alias("validity_ratio")])
        
    # 7. Derived Columns
    df = create_derived_columns(df)
    
    # Ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 8. Build Quality Report
    row_count = df.height
    pos_rate = df["dry_eye_disease"].mean() if "dry_eye_disease" in df.columns else 0.0
    bp_ok_rate = df["bp_parse_ok"].mean() if "bp_parse_ok" in df.columns else 0.0
    avg_validity = df["validity_ratio"].mean()
    
    # Missing rate
    missing_rates = {col: df[col].null_count() / row_count for col in df.columns}
    
    # Out of range counts
    oor_counts = {col: (df[f"{col}_in_range"] == 0).sum() for col in RANGE_RULES.keys() if f"{col}_in_range" in df.columns}
    
    # Histogram of validity_ratio (0.0-0.2, 0.2-0.4, etc)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    hist_counts = []
    for i in range(len(bins)-1):
        c = ((df["validity_ratio"] >= bins[i]) & (df["validity_ratio"] < bins[i+1])).sum()
        hist_counts.append({f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(c)})

    report = {
        "row_count": row_count,
        "columns": df.columns,
        "class_balance": {"positive_rate": pos_rate},
        "missing_rate_by_column": missing_rates,
        "bp_parse_ok_rate": bp_ok_rate,
        "avg_validity_ratio": avg_validity,
        "validity_ratio_histogram_bins": hist_counts,
        "out_of_range_counts_by_column": oor_counts
    }
    
    # Write outputs
    parquet_path = output_dir / "clean_assessments.parquet"
    report_path = output_dir / "data_quality_report.json"
    
    df.write_parquet(parquet_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"✅ Success! Created {parquet_path} and {report_path}")

if __name__ == "__main__":
    standardize(
        Path("data/raw/Dry_Eye_Dataset.csv"),
        Path("data/standardized")
    )
