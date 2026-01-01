"""
Deep diagnostic script to analyze feature signal and label noise.
Helps determine why ROC-AUC is low and if 0.9 is feasible.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def diagnose_data(filepath: Path):
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Define features
    features = [
        "sleep_duration", "sleep_quality", "sleep_disorder", "wake_up_during_night", "feel_sleepy_during_day",
        "average_screen_time", "smart_device_before_bed", "blue_light_filter",
        "stress_level", "daily_steps", "physical_activity", "caffeine_consumption", "alcohol_consumption", "smoking",
        "age", "gender", "bmi", "medical_issue", "ongoing_medication"
    ]
    
    # Prepare data
    df_clean = df[df["dry_eye_disease"].notna()].copy()
    X = df_clean[features].copy()
    y = df_clean["dry_eye_disease"].astype(int)
    
    # Basic cleaning
    binary_map = {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, True: 1, False: 0}
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map(binary_map).fillna(0)
    
    X = X.fillna(X.median())
    
    print("\n[1] Mutual Information Analysis")
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(mi_series)
    
    print("\n[2] Label Consistency Check")
    # Group by all features and see if target varies
    duplicates = df_clean.groupby(features)["dry_eye_disease"].agg(['mean', 'count'])
    inconsistent = duplicates[((duplicates['mean'] > 0) & (duplicates['mean'] < 1))]
    print(f"Total unique feature combinations: {len(duplicates)}")
    print(f"Combinations with inconsistent labels: {len(inconsistent)}")
    if len(duplicates) > 0:
        print(f"Percentage of inconsistency: {len(inconsistent)/len(duplicates):.2%}")

    print("\n[3] Oracle Check (Stage B Features)")
    symptoms = ["discomfort_eye_strain", "redness_in_eye", "itchiness_irritation_in_eye"]
    X_b = df_clean[features + symptoms].copy()
    for col in X_b.columns:
        if X_b[col].dtype == "object":
            X_b[col] = X_b[col].map(binary_map).fillna(0)
    X_b = X_b.fillna(X_b.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    print(f"Stage B Baseline ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
    diagnose_data(Path("data/standardized/clean_assessments.parquet"))
