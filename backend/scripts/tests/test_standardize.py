"""
Tests for the standardization pipeline.
"""
import pytest
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from backend.scripts.standardize import (
    parse_blood_pressure, 
    create_derived_columns, 
    snake_case,
    BINARY_MAP,
    RANGE_RULES
)

def test_snake_case():
    assert snake_case("Discomfort Eye-strain") == "discomfort_eye_strain"
    assert snake_case("Blood pressure") == "blood_pressure"
    assert snake_case("Gender") == "gender"

def test_bp_parsing():
    # Valid formats
    bp_data = pl.Series("bp", ["120/80", "137 / 89", "110-70", "invalid", None])
    sys, dia, ok = parse_blood_pressure(bp_data)
    
    assert sys[0] == 120
    assert dia[0] == 80
    assert ok[0] == 1
    
    assert sys[1] == 137
    assert dia[1] == 89
    assert ok[1] == 1
    
    assert sys[2] == 110
    assert dia[2] == 70
    assert ok[2] == 1
    
    assert ok[3] == 0
    assert ok[4] == 0

def test_banding():
    # Mock data for banding
    df = pl.DataFrame({
        "age": [20, 27, 32, 37, 43, 45],
        "average_screen_time": [1.5, 3.5, 5.5, 7.5, 9.5, 12.0],
        "sleep_duration": [5.0, 6.5, 7.5, 8.5, 10.0, 11.0],
        "height": [170, 170, 170, 170, 170, 170],
        "weight": [70, 70, 70, 70, 70, 70],
        "discomfort_eye_strain": [1, 0, 1, 0, 1, 1],
        "redness_in_eye": [1, 1, 0, 0, 1, 1],
        "itchiness_irritation_in_eye": [0, 1, 0, 1, 1, 1]
    })
    
    df_derived = create_derived_columns(df)
    
    # Age bands
    assert df_derived["age_band"][0] == "18-24"
    assert df_derived["age_band"][1] == "25-29"
    
    # Screen time bands
    assert df_derived["screen_time_band"][0] == "0-2"
    assert df_derived["screen_time_band"][5] == "10+"
    
    # Symptom score
    assert df_derived["symptom_score"][0] == 2 # 1+1+0
    assert df_derived["symptom_score"][4] == 3 # 1+1+1

def test_range_rules():
    # age rule is (18, 45)
    low, high = RANGE_RULES["age"]
    df = pl.DataFrame({"age": [15, 25, 50]})
    
    # Simulate logic from standardize()
    df_result = df.select([
        pl.when((pl.col("age") >= low) & (pl.col("age") <= high))
        .then(1).otherwise(0).alias("age_in_range"),
        pl.when((pl.col("age") >= low) & (pl.col("age") <= high))
        .then(pl.col("age")).otherwise(None).alias("age_validated")
    ])
    
    assert df_result["age_in_range"][0] == 0
    assert df_result["age_validated"][0] is None
    
    assert df_result["age_in_range"][1] == 1
    assert df_result["age_validated"][1] == 25
    
    assert df_result["age_in_range"][2] == 0
    assert df_result["age_validated"][2] is None

if __name__ == "__main__":
    # Simple manual test runner
    test_snake_case()
    test_bp_parsing()
    test_banding()
    test_range_rules()
    print("✅ All manual tests passed!")
