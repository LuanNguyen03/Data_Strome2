"""
Comprehensive data analysis script to diagnose model performance issues.
Analyzes feature importance, correlations, missing values, and data quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Stage A features (NO symptoms)
STAGE_A_FEATURES = [
    "sleep_duration", "sleep_quality", "sleep_disorder", "wake_up_during_night", "feel_sleepy_during_day",
    "average_screen_time", "smart_device_before_bed", "blue_light_filter",
    "stress_level", "daily_steps", "physical_activity", "caffeine_consumption", "alcohol_consumption", "smoking",
    "age", "gender", "bmi",
    "systolic", "diastolic", "heart_rate",
    "medical_issue", "ongoing_medication",
]

# Stage B features (includes symptoms)
STAGE_B_FEATURES = STAGE_A_FEATURES + [
    "discomfort_eye_strain",
    "redness_in_eye",
    "itchiness_irritation_in_eye",
]


def load_data(filepath: Path) -> pd.DataFrame:
    """Load standardized parquet file"""
    return pd.read_parquet(filepath)


def prepare_features_for_analysis(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for analysis"""
    df_clean = df[df["dry_eye_disease"].notna()].copy()
    
    # Select available features
    available_features = [f for f in features if f in df_clean.columns]
    X = df_clean[available_features].copy()
    y = df_clean["dry_eye_disease"].astype(int)
    
    # Convert binary columns
    binary_map = {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, True: 1, False: 0}
    for col in X.columns:
        if X[col].dtype == "object":
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) <= 2:
                try:
                    X[col] = X[col].replace(binary_map)
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                except:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Convert all to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    
    return X, y


def analyze_missing_values(X: pd.DataFrame) -> Dict:
    """Analyze missing values"""
    missing = X.isnull().sum()
    missing_pct = (missing / len(X)) * 100
    
    return {
        "total_rows": len(X),
        "columns_with_missing": int((missing > 0).sum()),
        "missing_by_column": {
            col: {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col])
            }
            for col in X.columns
            if missing[col] > 0
        },
        "columns_high_missing": [
            col for col in X.columns
            if missing_pct[col] > 50
        ]
    }


def analyze_class_distribution(y: pd.Series) -> Dict:
    """Analyze class distribution"""
    counts = y.value_counts().to_dict()
    percentages = (y.value_counts(normalize=True) * 100).to_dict()
    
    return {
        "total_samples": len(y),
        "class_0_count": int(counts.get(0, 0)),
        "class_1_count": int(counts.get(1, 0)),
        "class_0_percentage": float(percentages.get(0, 0)),
        "class_1_percentage": float(percentages.get(1, 0)),
        "imbalance_ratio": float(counts.get(0, 1) / counts.get(1, 1)) if counts.get(1, 0) > 0 else 0.0,
    }


def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, n_estimators: int = 200) -> pd.DataFrame:
    """Analyze feature importance using RandomForest"""
    # Fill missing values with median
    X_filled = X.fillna(X.median())
    
    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_filled, y)
    
    # Get importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_,
        'std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    }).sort_values('importance', ascending=False)
    
    return importance_df


def analyze_correlations(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Analyze correlations between features and target"""
    X_filled = X.fillna(X.median())
    
    correlations = {}
    for col in X_filled.columns:
        if X_filled[col].dtype in [np.float64, np.int64]:
            corr = X_filled[col].corr(y)
            correlations[col] = {
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
                "abs_correlation": float(abs(corr)) if not np.isnan(corr) else 0.0
            }
    
    # Sort by absolute correlation
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: x[1]["abs_correlation"],
        reverse=True
    )
    
    return {
        "top_10_positive": [
            {"feature": k, "correlation": v["correlation"]}
            for k, v in sorted_corr[:10]
            if v["correlation"] > 0
        ],
        "top_10_negative": [
            {"feature": k, "correlation": v["correlation"]}
            for k, v in sorted_corr[:10]
            if v["correlation"] < 0
        ],
        "all_correlations": correlations
    }


def analyze_feature_statistics(X: pd.DataFrame) -> Dict:
    """Analyze basic statistics of features"""
    stats = {}
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "unique_values": int(col_data.nunique()),
                }
    return stats


def analyze_baseline_performance(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Analyze baseline model performance"""
    X_filled = X.fillna(X.median())
    
    # Simple baseline: majority class
    majority_class = y.mode()[0]
    baseline_accuracy = (y == majority_class).mean()
    
    # RandomForest baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X_filled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    return {
        "majority_class_baseline": float(baseline_accuracy),
        "rf_baseline_roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "rf_baseline_accuracy": float(accuracy_score(y_test, y_pred)),
    }


def analyze_stage(stage: str, df: pd.DataFrame, features: List[str], output_dir: Path) -> Dict:
    """Comprehensive analysis for one stage"""
    print(f"\n{'='*60}")
    print(f"Analyzing Stage {stage}")
    print(f"{'='*60}")
    
    # Prepare data
    X, y = prepare_features_for_analysis(df, features)
    print(f"\nData shape: {X.shape}")
    print(f"Features: {len(X.columns)}")
    print(f"Available features: {list(X.columns)}")
    
    # Missing values
    print("\n[1] Analyzing missing values...")
    missing_analysis = analyze_missing_values(X)
    print(f"  Columns with missing: {missing_analysis['columns_with_missing']}")
    if missing_analysis['columns_high_missing']:
        print(f"  High missing (>50%): {missing_analysis['columns_high_missing']}")
    
    # Class distribution
    print("\n[2] Analyzing class distribution...")
    class_dist = analyze_class_distribution(y)
    print(f"  Class 0: {class_dist['class_0_count']} ({class_dist['class_0_percentage']:.1f}%)")
    print(f"  Class 1: {class_dist['class_1_count']} ({class_dist['class_1_percentage']:.1f}%)")
    print(f"  Imbalance ratio: {class_dist['imbalance_ratio']:.2f}")
    
    # Feature importance
    print("\n[3] Analyzing feature importance...")
    importance_df = analyze_feature_importance(X, y)
    print("\n  Top 10 features by importance:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f} ± {row['std']:.4f}")
    
    # Correlations
    print("\n[4] Analyzing correlations with target...")
    correlations = analyze_correlations(X, y)
    print("\n  Top 5 positive correlations:")
    for item in correlations['top_10_positive'][:5]:
        print(f"    {item['feature']}: {item['correlation']:.4f}")
    print("\n  Top 5 negative correlations:")
    for item in correlations['top_10_negative'][:5]:
        print(f"    {item['feature']}: {item['correlation']:.4f}")
    
    # Feature statistics
    print("\n[5] Analyzing feature statistics...")
    stats = analyze_feature_statistics(X)
    print(f"  Analyzed {len(stats)} numeric features")
    
    # Baseline performance
    print("\n[6] Analyzing baseline performance...")
    baseline = analyze_baseline_performance(X, y)
    print(f"  Majority class baseline: {baseline['majority_class_baseline']:.4f}")
    print(f"  RF baseline ROC-AUC: {baseline['rf_baseline_roc_auc']:.4f}")
    print(f"  RF baseline accuracy: {baseline['rf_baseline_accuracy']:.4f}")
    
    # Compile results
    results = {
        "stage": stage,
        "data_shape": {"rows": len(X), "columns": len(X.columns)},
        "features": list(X.columns),
        "missing_values": missing_analysis,
        "class_distribution": class_dist,
        "feature_importance": {
            "top_10": importance_df.head(10).to_dict('records'),
            "all": importance_df.to_dict('records')
        },
        "correlations": correlations,
        "feature_statistics": stats,
        "baseline_performance": baseline,
    }
    
    # Save results
    output_file = output_dir / f"analysis_stage_{stage}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Analysis saved to: {output_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze data quality and feature importance")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/standardized/clean_assessments.parquet"),
        help="Path to standardized parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modeling/analysis"),
        help="Output directory for analysis results"
    )
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Analyze Stage A
    results_a = analyze_stage("A", df, STAGE_A_FEATURES, args.output_dir)
    
    # Analyze Stage B
    results_b = analyze_stage("B", df, STAGE_B_FEATURES, args.output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nStage A:")
    print(f"  Baseline ROC-AUC: {results_a['baseline_performance']['rf_baseline_roc_auc']:.4f}")
    print(f"  Top feature: {results_a['feature_importance']['top_10'][0]['feature']}")
    print(f"  Imbalance ratio: {results_a['class_distribution']['imbalance_ratio']:.2f}")
    
    print(f"\nStage B:")
    print(f"  Baseline ROC-AUC: {results_b['baseline_performance']['rf_baseline_roc_auc']:.4f}")
    print(f"  Top feature: {results_b['feature_importance']['top_10'][0]['feature']}")
    print(f"  Imbalance ratio: {results_b['class_distribution']['imbalance_ratio']:.2f}")
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

