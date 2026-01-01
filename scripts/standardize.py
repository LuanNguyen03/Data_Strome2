"""
Standardize Dry Eye dataset locally (no DB required).
Reads the raw CSV, normalizes schema, computes range flags, derived fields,
and writes clean Parquet + data quality report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl


BINARY_COLS = {
    "sleep_disorder",
    "wake_up_during_night",
    "feel_sleepy_during_day",
    "caffeine_consumption",
    "alcohol_consumption",
    "smoking",
    "medical_issue",
    "ongoing_medication",
    "smart_device_before_bed",
    "bluelight_filter",
    "discomfort_eyestrain",
    "redness_in_eye",
    "itchiness_irritation_in_eye",
    "dry_eye_disease",
}

NUMERIC_COLS = {
    "age",
    "height",
    "weight",
    "sleep_duration",
    "average_screen_time",
    "sleep_quality",
    "stress_level",
    "heart_rate",
    "daily_steps",
    "physical_activity",
}

RANGE_RULES: Dict[str, Tuple[float, float]] = {
    "age": (18, 45),
    "sleep_quality": (1, 5),
    "stress_level": (1, 5),
    "sleep_duration": (0, 24),
    "average_screen_time": (0, 24),
    "heart_rate": (40, 220),
    "daily_steps": (0, 50_000),
    "physical_activity": (0, 600),
    "height": (120, 230),
    "weight": (30, 250),
    "systolic": (70, 250),
    "diastolic": (40, 150),
}


def snake_case(name: str) -> str:
    import re

    s = name.strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    s = s.strip("_").lower()
    return s


def map_binary(df: pl.DataFrame, col: str) -> pl.Series:
    return (
        pl.when(pl.col(col).str.to_uppercase() == "Y")
        .then(1)
        .when(pl.col(col).str.to_uppercase() == "N")
        .then(0)
        .otherwise(pl.col(col).cast(pl.Int64, strict=False))
        .alias(col)
    )


def parse_bp(raw_col: str = "blood_pressure_raw") -> List[pl.Series]:
    # Extract systolic/diastolic if present; tolerate formats like "120/80" or "120 - 80"
    systolic = (
        pl.col(raw_col)
            .str.extract(r"(\d{2,3})")
            .cast(pl.Int64, strict=False)
            .alias("systolic")
    )
    diastolic = (
        pl.col(raw_col)
            .str.extract(r"\d{2,3}\D+(\d{2,3})", group_index=1)
            .cast(pl.Int64, strict=False)
            .alias("diastolic")
    )
    bp_ok = (
        pl.when((systolic.is_not_null()) & (diastolic.is_not_null()))
            .then(1)
            .otherwise(0)
            .alias("bp_parse_ok")
    )
    return [systolic, diastolic, bp_ok]


def apply_range_rules(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col, (low, high) in RANGE_RULES.items():
        if col not in out.columns:
            continue
        flag_col = f"{col}_in_range"
        out = out.with_columns(
            pl.when(pl.col(col).is_null())
            .then(0)
            .when((pl.col(col) < low) | (pl.col(col) > high))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col),
            pl.when(pl.col(col).is_null())
            .then(0)
            .when((pl.col(col) < low) | (pl.col(col) > high))
            .then(0)
            .otherwise(1)
            .alias(flag_col),
        )
    return out


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    bmi = (
        pl.when((pl.col("height") > 0) & pl.col("height").is_not_null() & pl.col("weight").is_not_null())
        .then(pl.col("weight") / ((pl.col("height") / 100) ** 2))
        .otherwise(None)
        .alias("bmi")
    )

    age_band = (
        pl.when(pl.col("age").is_null())
        .then(None)
        .when(pl.col("age") < 25)
        .then("18-24")
        .when(pl.col("age") < 30)
        .then("25-29")
        .when(pl.col("age") < 35)
        .then("30-34")
        .when(pl.col("age") < 40)
        .then("35-39")
        .otherwise("40-45")
        .alias("age_band")
    )

    screen_band = (
        pl.when(pl.col("average_screen_time").is_null())
        .then(None)
        .when(pl.col("average_screen_time") < 2)
        .then("0-2")
        .when(pl.col("average_screen_time") < 4)
        .then("2-4")
        .when(pl.col("average_screen_time") < 6)
        .then("4-6")
        .when(pl.col("average_screen_time") < 8)
        .then("6-8")
        .when(pl.col("average_screen_time") < 10)
        .then("8-10")
        .otherwise("10+")
        .alias("screen_time_band")
    )

    sleep_dur_band = (
        pl.when(pl.col("sleep_duration").is_null())
        .then(None)
        .when(pl.col("sleep_duration") < 6)
        .then("<6")
        .when(pl.col("sleep_duration") < 7)
        .then("6-7")
        .when(pl.col("sleep_duration") < 8)
        .then("7-8")
        .when(pl.col("sleep_duration") < 9)
        .then("8-9")
        .otherwise("9+")
        .alias("sleep_duration_band")
    )

    symptoms = ["discomfort_eyestrain", "redness_in_eye", "itchiness_irritation_in_eye"]
    symptom_score = (
        pl.when(pl.any_horizontal([pl.col(c).is_null() for c in symptoms]))
        .then(None)
        .otherwise(sum([pl.col(c) for c in symptoms]))
        .alias("symptom_score")
    )

    validity_cols = [c for c in df.columns if c.endswith("_in_range")]
    validity_ratio = (
        pl.when(len(validity_cols) == 0)
        .then(None)
        .otherwise(pl.mean_horizontal([pl.col(c) for c in validity_cols]))
        .alias("validity_ratio")
    )

    return df.with_columns([bmi, age_band, screen_band, sleep_dur_band, symptom_score, validity_ratio])


def build_quality_report(df: pl.DataFrame, validity_cols: List[str]) -> Dict:
    n_rows = df.height
    n_cols = len(df.columns)
    report = {
        "rows": n_rows,
        "cols": n_cols,
    }

    if "dry_eye_disease" in df.columns:
        positive = df["dry_eye_disease"].sum() if n_rows else 0
        report["class_balance_positive_rate"] = float(positive) / float(n_rows) if n_rows else None

    if "bp_parse_ok" in df.columns and n_rows:
        report["bp_parse_ok_rate"] = float(df["bp_parse_ok"].mean())

    # Missing rates
    missing_rates = []
    for col in df.columns:
        miss = df[col].null_count() / n_rows if n_rows else math.nan
        missing_rates.append((col, miss))
    missing_rates.sort(key=lambda x: x[1], reverse=True)
    report["missing_rate_top10"] = [{"col": c, "missing_rate": float(r)} for c, r in missing_rates[:10]]

    # Out-of-range counts
    oor_counts = {}
    for col, (low, high) in RANGE_RULES.items():
        if col in df.columns:
            out = df.filter((pl.col(col) < low) | (pl.col(col) > high))
            oor_counts[col] = int(out.height)
    report["out_of_range_counts"] = oor_counts

    # Validity ratio
    if validity_cols and n_rows:
        vr = df.select(pl.mean_horizontal([pl.col(c) for c in validity_cols]).alias("vr")).to_series()[0]
        report["avg_validity_ratio"] = float(vr) if vr is not None else None

    return report


def standardize(input_csv: Path, output_parquet: Path, report_path: Path) -> None:
    df = pl.read_csv(input_csv)

    # Normalize column names
    rename_map = {col: snake_case(col) for col in df.columns}
    df = df.rename(rename_map)

    # Map binary Y/N and gender
    for col in BINARY_COLS:
        if col in df.columns:
            df = df.with_columns(map_binary(df, col))

    if "gender" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("gender").str.to_uppercase() == "F")
            .then(0)
            .when(pl.col("gender").str.to_uppercase() == "M")
            .then(1)
            .otherwise(pl.col("gender").cast(pl.Int64, strict=False))
            .alias("gender")
        )

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Parse blood pressure
    if "blood_pressure_raw" in df.columns:
        df = df.with_columns(parse_bp())

    # Apply range rules and flags
    df = apply_range_rules(df)

    # Derived columns
    df = add_derived(df)

    # Persist Parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_parquet)

    # Quality report
    validity_cols = [c for c in df.columns if c.endswith("_in_range")]
    report = build_quality_report(df, validity_cols)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {output_parquet} ({df.height} rows, {len(df.columns)} cols)")
    print(f"Wrote quality report {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize Dry Eye CSV to Parquet with quality report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("DryEyeDisease/Dry_Eye_Dataset.csv"),
        help="Path to raw CSV input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/standardized/clean_assessments.parquet"),
        help="Path to output Parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/standardized/data_quality_report.json"),
        help="Path to data quality report (JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    standardize(args.input, args.output, args.report)


if __name__ == "__main__":
    main()

