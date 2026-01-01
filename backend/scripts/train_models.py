"""
Full training pipeline for 2-stage medical modeling.
Follows docs/03_medical_modeling_plan.md and docs/threshold_notes.md.

Input: data/standardized/clean_assessments.parquet
Outputs:
  - modeling/artifacts/*.joblib, *.json
  - modeling/reports/*.json, *.md
  - modeling/registry/registry.json (append-only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost with GPU support, fallback to CPU or RandomForest
XGBOOST_AVAILABLE = False
GPU_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    # Check if GPU is available (nvidia-smi command)
    try:
        import subprocess
        import os
        # Check for nvidia-smi
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        GPU_AVAILABLE = result.returncode == 0 and 'NVIDIA' in result.stdout
        if GPU_AVAILABLE:
            gpu_info = result.stdout.split('\n')[0] if result.stdout else 'NVIDIA GPU'
            print(f"[INFO] GPU detected: {gpu_info}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        GPU_AVAILABLE = False
        print(f"[INFO] GPU check failed: {type(e).__name__}, using CPU")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    print("[INFO] XGBoost not available, using RandomForest")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()  # Use resolve() to get absolute path
sys.path.insert(0, str(project_root))


# Stage A features (NO symptoms, NO symptom_score)
STAGE_A_FEATURES = [
    # Sleep
    "sleep_duration",
    "sleep_quality",
    "sleep_disorder",
    "wake_up_during_night",
    "feel_sleepy_during_day",
    # Device/Screen
    "average_screen_time",
    "smart_device_before_bed",
    "blue_light_filter",  # Note: standardized as snake_case
    # Lifestyle
    "stress_level",
    "daily_steps",
    "physical_activity",
    "caffeine_consumption",
    "alcohol_consumption",
    "smoking",
    # Person
    "age",
    "gender",
    "bmi",
    # Vitals
    "systolic",
    "diastolic",
    "heart_rate",
    # Medical
    "medical_issue",
    "ongoing_medication",
]

# Stage B features (includes symptoms)
STAGE_B_FEATURES = STAGE_A_FEATURES + [
    "discomfort_eye_strain",
    "redness_in_eye",
    "itchiness_irritation_in_eye",
    # symptom_score is optional for Stage B
]

# Symptoms that MUST be excluded from Stage A
STAGE_A_EXCLUDED = [
    "discomfort_eyestrain",
    "discomfort_eye_strain",  # snake_case variant
    "redness_in_eye",
    "itchiness_irritation_in_eye",
    "symptom_score",
]


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file bytes"""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_python_versions() -> Dict[str, str]:
    """Get Python and key library versions"""
    import sys
    versions = {
        "python": sys.version.split()[0],
    }
    
    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except:
        pass
    
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except:
        pass
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except:
        pass
    
    try:
        import polars
        versions["polars"] = polars.__version__
    except:
        pass
    
    return versions


def load_data(input_path: Path) -> pd.DataFrame:
    """Load standardized parquet data"""
    print(f"Loading data from {input_path}...")
    df = pl.read_parquet(input_path)
    df_pandas = df.to_pandas()
    print(f"Loaded {len(df_pandas):,} rows, {len(df_pandas.columns)} columns")
    return df_pandas


def check_stage_a_leakage(df: pd.DataFrame, features: List[str]) -> None:
    """Assert that Stage A features do not include symptoms"""
    excluded_in_features = [f for f in STAGE_A_EXCLUDED if f in features]
    if excluded_in_features:
        raise ValueError(
            f"Stage A leakage detected! Features contain symptoms: {excluded_in_features}"
        )
    print("[OK] Stage A leakage check passed")


def prepare_features(
    df: pd.DataFrame, features: List[str], stage: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features and target"""
    print(f"\nPreparing {stage} features...")
    
    # Filter to rows with target available
    df_clean = df[df["dry_eye_disease"].notna()].copy()
    
    # Select features that exist in dataframe
    available_features = [f for f in features if f in df_clean.columns]
    missing_features = [f for f in features if f not in df_clean.columns]
    
    if missing_features:
        print(f"  [WARNING] Missing features: {missing_features}")
    
    # Fill missing values with median for numeric, mode for categorical
    X = df_clean[available_features].copy()
    y = df_clean["dry_eye_disease"].astype(int)
    
    # Convert binary Y/N columns to 0/1 if needed
    binary_map = {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
    X_processed = X.copy()
    
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            # Check if column contains binary values
            unique_vals = X_processed[col].dropna().unique()
            if len(unique_vals) <= 2:
                # Try to convert Y/N to 0/1
                try:
                    X_processed[col] = X_processed[col].replace(binary_map).infer_objects(copy=False)
                    X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
                except:
                    # If conversion fails, try direct numeric conversion
                    X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
            else:
                # For other object types, try to convert to numeric
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
    
    # Ensure all columns are numeric (convert any remaining objects)
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
    
    # Fill NaN with median for numeric columns (avoid chained assignment)
    fill_values = {}
    for col in X_processed.columns:
        if X_processed[col].dtype in ["int64", "float64", "float32", "int32"]:
            median_val = X_processed[col].median()
            if pd.isna(median_val):
                fill_values[col] = 0.0
            else:
                fill_values[col] = float(median_val)
        else:
            fill_values[col] = 0.0
    
    # Apply fill values
    X_processed = X_processed.fillna(fill_values)
    
    # Ensure all columns are float64 for sklearn
    X_processed = X_processed.astype(float)
    
    X = X_processed
    
    print(f"  Features: {len(available_features)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    return X, y, available_features


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train/val/test"""
    print(f"\nSplitting data (seed={seed})...")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Second split: train vs val
    # Adjust val_size to account for test already removed
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=seed + 1, stratify=y_trainval
    )
    
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X):.1%})")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X):.1%})")
    print(f"  Train positive rate: {y_train.mean():.2%}")
    print(f"  Val positive rate:   {y_val.mean():.2%}")
    print(f"  Test positive rate:  {y_test.mean():.2%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage: str,
    seed: int,
) -> Tuple[Any, StandardScaler, float]:
    """Train model and select threshold"""
    global GPU_AVAILABLE  # Declare as global to avoid UnboundLocalError
    
    print(f"\nTraining {stage} model...")
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Use XGBoost with GPU if available, otherwise CPU or RandomForest
    use_gpu = False
    if XGBOOST_AVAILABLE and GPU_AVAILABLE:
        print("  [GPU] Using XGBoost with GPU acceleration...")
        try:
            # XGBoost 3.1+ uses 'device' instead of 'gpu_id' and 'gpu_hist'
            # Use 'hist' with device='cuda' for GPU acceleration
            model = xgb.XGBClassifier(
                n_estimators=500,  # Increased for better results
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method='hist',  # 'hist' works with device='cuda' in 3.1+
                device='cuda',  # Use 'cuda' instead of gpu_id (XGBoost 3.1+)
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=1,
            )
            use_gpu = True
        except Exception as e:
            print(f"  [WARNING] GPU training failed ({e}), falling back to CPU...")
            use_gpu = False
    
    if not use_gpu:
        if XGBOOST_AVAILABLE:
            print("  [CPU] Using XGBoost with CPU (multi-threaded)...")
            model = xgb.XGBClassifier(
                n_estimators=500,  # Increased for better results
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method='hist',  # Fast CPU method
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1,  # Use all CPU cores
                verbosity=1,
            )
        else:
            print("  [CPU] Using RandomForest (XGBoost not available)...")
            model = RandomForestClassifier(
                n_estimators=500,  # Increased for better results
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=seed,
                n_jobs=-1,
            )
    
    # Train model with early stopping and best model saving
    if XGBOOST_AVAILABLE:
        try:
            # XGBoost 3.1+ uses callbacks
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],  # Save best model, stop if no improvement for 100 rounds
                verbose=False
            )
        except (TypeError, AttributeError):
            try:
                # Fallback for older versions
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=100,  # Stop if no improvement for 100 rounds
                    verbose=False
                )
            except TypeError:
                # If both fail, train without early stopping
                model.fit(X_train_scaled, y_train, verbose=False)
    else:
        # RandomForest doesn't support early stopping
        model.fit(X_train_scaled, y_train)
    
    # Get probabilities
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Select threshold based on stage
    if stage == "A":
        # Stage A: optimize for Recall (screening)
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_recall = 0.0
        
        for thresh in thresholds:
            y_pred = (y_val_proba >= thresh).astype(int)
            recall = recall_score(y_val, y_pred, zero_division=0)
            if recall > best_recall:
                best_recall = recall
                best_threshold = thresh
    else:
        # Stage B: optimize for F1 (balanced)
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for thresh in thresholds:
            y_pred = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
    
    print(f"  Selected threshold: {best_threshold:.3f}")
    
    return model, scaler, best_threshold


def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float, stage: str, split: str
) -> Dict[str, Any]:
    """Compute all metrics for a split"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        "split": split,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    
    return metrics


def save_artifacts(
    model: Any,
    scaler: StandardScaler,
    features: List[str],
    threshold: float,
    stage: str,
    artifacts_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    """Save model artifacts"""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_paths = {}
    
    # Save model
    model_path = artifacts_dir / f"model_{stage}_{'screening' if stage == 'A' else 'triage'}.joblib"
    joblib.dump(model, model_path)
    # Use absolute path and convert to relative if possible, otherwise use absolute
    try:
        artifact_paths["model"] = str(model_path.relative_to(project_root))
    except ValueError:
        artifact_paths["model"] = str(model_path)
    
    # Save scaler
    scaler_path = artifacts_dir / f"preprocessing_{stage}.joblib"
    joblib.dump(scaler, scaler_path)
    try:
        artifact_paths["preprocessing"] = str(scaler_path.relative_to(project_root))
    except ValueError:
        artifact_paths["preprocessing"] = str(scaler_path)
    
    # Save feature list
    features_path = artifacts_dir / f"feature_list_{stage}.json"
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)
    try:
        artifact_paths["features"] = str(features_path.relative_to(project_root))
    except ValueError:
        artifact_paths["features"] = str(features_path)
    
    # Save metadata (will be merged with full metadata later)
    metadata_path = artifacts_dir / "model_metadata.json"
    stage_metadata = {
        f"stage_{stage}": {
            "threshold": float(threshold),
            "features": features,
            "feature_count": len(features),
        }
    }
    
    # Load existing metadata if exists
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            full_metadata = json.load(f)
    else:
        full_metadata = {}
    
    full_metadata.update(stage_metadata)
    full_metadata.update(metadata)
    
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    try:
        artifact_paths["metadata"] = str(metadata_path.relative_to(project_root))
    except ValueError:
        artifact_paths["metadata"] = str(metadata_path)
    
    return artifact_paths


def save_reports(
    metrics_val: Dict[str, Any],
    metrics_test: Dict[str, Any],
    split_report: Dict[str, Any],
    reports_dir: Path,
    stage: str,
) -> Dict[str, str]:
    """Save all reports"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # Save metrics
    metrics_path = reports_dir / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    all_metrics[f"stage_{stage}"] = {
        "validation": metrics_val,
        "test": metrics_test,
    }
    
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    try:
        report_paths["metrics"] = str(metrics_path.relative_to(project_root))
    except ValueError:
        report_paths["metrics"] = str(metrics_path)
    
    # Save confusion matrices
    cm_path = reports_dir / "confusion_matrices.json"
    if cm_path.exists():
        with open(cm_path, "r") as f:
            all_cm = json.load(f)
    else:
        all_cm = {}
    
    all_cm[f"stage_{stage}"] = {
        "validation": metrics_val["confusion_matrix"],
        "test": metrics_test["confusion_matrix"],
    }
    
    with open(cm_path, "w") as f:
        json.dump(all_cm, f, indent=2)
    try:
        report_paths["confusion_matrices"] = str(cm_path.relative_to(project_root))
    except ValueError:
        report_paths["confusion_matrices"] = str(cm_path)
    
    # Save split report
    split_path = reports_dir / "data_split_report.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            all_splits = json.load(f)
    else:
        all_splits = {}
    
    all_splits[f"stage_{stage}"] = split_report
    
    with open(split_path, "w") as f:
        json.dump(all_splits, f, indent=2)
    try:
        report_paths["split_report"] = str(split_path.relative_to(project_root))
    except ValueError:
        report_paths["split_report"] = str(split_path)
    
    # Save calibration report (placeholder)
    cal_path = reports_dir / "calibration_report.json"
    cal_data = {
        f"stage_{stage}": {
            "note": "Calibration not implemented in this version",
            "recommendation": "Use Platt scaling or isotonic regression for production",
        }
    }
    if cal_path.exists():
        with open(cal_path, "r") as f:
            all_cal = json.load(f)
        all_cal.update(cal_data)
    else:
        all_cal = cal_data
    
    with open(cal_path, "w") as f:
        json.dump(all_cal, f, indent=2)
    try:
        report_paths["calibration"] = str(cal_path.relative_to(project_root))
    except ValueError:
        report_paths["calibration"] = str(cal_path)
    
    return report_paths


def update_registry(
    registry_path: Path,
    model_version: str,
    dataset_hash: str,
    artifact_paths: Dict[str, Dict[str, str]],
    report_paths: Dict[str, Dict[str, str]],
    metrics_summary: Dict[str, Any],
) -> None:
    """Append entry to registry.json"""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "model_version": model_version,
        "created_at": datetime.now().isoformat(),
        "dataset_hash": dataset_hash,
        "artifact_paths": artifact_paths,
        "report_paths": report_paths,
        "metrics_summary": metrics_summary,
    }
    
    # Load existing registry
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"entries": []}
    
    registry["entries"].append(entry)
    registry["latest"] = entry  # Update latest pointer
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n[OK] Registry updated: {registry_path}")


def generate_model_comparison(reports_dir: Path) -> None:
    """Generate model_comparison.md"""
    metrics_path = reports_dir / "model_metrics.json"
    if not metrics_path.exists():
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    md_content = "# Model Comparison Report\n\n"
    md_content += f"Generated: {datetime.now().isoformat()}\n\n"
    
    for stage_name, stage_metrics in metrics.items():
        stage_label = "Stage A (Screening)" if "A" in stage_name else "Stage B (Triage)"
        md_content += f"## {stage_label}\n\n"
        
        for split_name, split_metrics in stage_metrics.items():
            md_content += f"### {split_name.capitalize()} Set\n\n"
            md_content += f"- **ROC-AUC**: {split_metrics.get('roc_auc', 'N/A'):.4f}\n"
            md_content += f"- **PR-AUC**: {split_metrics.get('pr_auc', 'N/A'):.4f}\n"
            md_content += f"- **Precision**: {split_metrics.get('precision', 'N/A'):.4f}\n"
            md_content += f"- **Recall**: {split_metrics.get('recall', 'N/A'):.4f}\n"
            md_content += f"- **F1**: {split_metrics.get('f1', 'N/A'):.4f}\n"
            md_content += f"- **Threshold**: {split_metrics.get('threshold', 'N/A'):.3f}\n\n"
            
            cm = split_metrics.get("confusion_matrix", {})
            md_content += "**Confusion Matrix:**\n\n"
            md_content += f"| | Predicted 0 | Predicted 1 |\n"
            md_content += f"| --- | --- | --- |\n"
            md_content += f"| Actual 0 | {cm.get('tn', 0)} | {cm.get('fp', 0)} |\n"
            md_content += f"| Actual 1 | {cm.get('fn', 0)} | {cm.get('tp', 0)} |\n\n"
    
    comparison_path = reports_dir / "model_comparison.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"[OK] Model comparison report: {comparison_path}")


def train_stage(
    df: pd.DataFrame,
    stage: str,
    seed: int,
    artifacts_dir: Path,
    reports_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Train one stage and return artifact/report paths and metrics"""
    print(f"\n{'='*60}")
    print(f"Training Stage {stage}")
    print(f"{'='*60}")
    
    # Select features
    if stage == "A":
        features = STAGE_A_FEATURES.copy()
        check_stage_a_leakage(df, features)
    else:
        features = STAGE_B_FEATURES.copy()
    
    # Prepare data
    X, y, available_features = prepare_features(df, features, f"Stage {stage}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=seed)
    
    # Train model
    model, scaler, threshold = train_model(X_train, y_train, X_val, y_val, stage, seed)
    
    # Compute metrics
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics_val = compute_metrics(y_val, y_val_proba, threshold, stage, "validation")
    metrics_test = compute_metrics(y_test, y_test_proba, threshold, stage, "test")
    
    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC: {metrics_val['roc_auc']:.4f}")
    print(f"  PR-AUC:  {metrics_val['pr_auc']:.4f}")
    print(f"  Precision: {metrics_val['precision']:.4f}")
    print(f"  Recall:    {metrics_val['recall']:.4f}")
    print(f"  F1:        {metrics_val['f1']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  ROC-AUC: {metrics_test['roc_auc']:.4f}")
    print(f"  PR-AUC:  {metrics_test['pr_auc']:.4f}")
    print(f"  Precision: {metrics_test['precision']:.4f}")
    print(f"  Recall:    {metrics_test['recall']:.4f}")
    print(f"  F1:        {metrics_test['f1']:.4f}")
    
    # Save artifacts
    metadata = {
        "stage": stage,
        "threshold": float(threshold),
        "python_versions": get_python_versions(),
    }
    artifact_paths = save_artifacts(
        model, scaler, available_features, threshold, stage, artifacts_dir, metadata
    )
    
    # Save reports
    split_report = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    report_paths = save_reports(metrics_val, metrics_test, split_report, reports_dir, stage)
    
    # Metrics summary for registry
    metrics_summary = {
        "val_roc_auc": metrics_val["roc_auc"],
        "val_pr_auc": metrics_val["pr_auc"],
        "val_f1": metrics_val["f1"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_pr_auc": metrics_test["pr_auc"],
        "test_f1": metrics_test["f1"],
        "threshold": float(threshold),
    }
    
    return artifact_paths, report_paths, metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2-stage medical models (Screening + Triage)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/standardized/clean_assessments.parquet"),
        help="Path to standardized parquet file",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("modeling/artifacts"),
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("modeling/reports"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("modeling/registry/registry.json"),
        help="Path to registry JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print("   Run standardization first: python -m backend.scripts.standardize")
        return
    
    # Compute dataset hash
    dataset_hash = compute_file_hash(args.input)
    print(f"Dataset hash: {dataset_hash[:16]}...")
    
    # Load data
    df = load_data(args.input)
    
    # Train Stage A
    artifacts_A, reports_A, metrics_A = train_stage(
        df, "A", args.seed, args.artifacts_dir, args.reports_dir
    )
    
    # Train Stage B
    artifacts_B, reports_B, metrics_B = train_stage(
        df, "B", args.seed, args.artifacts_dir, args.reports_dir
    )
    
    # Generate model comparison
    generate_model_comparison(args.reports_dir)
    
    # Update registry
    model_version = f"v1.0.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    update_registry(
        args.registry_path,
        model_version,
        dataset_hash,
        {"stage_A": artifacts_A, "stage_B": artifacts_B},
        {"stage_A": reports_A, "stage_B": reports_B},
        {"stage_A": metrics_A, "stage_B": metrics_B},
    )
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Training completed successfully!")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to: {args.artifacts_dir}")
    print(f"Reports saved to: {args.reports_dir}")
    print(f"Registry updated: {args.registry_path}")


if __name__ == "__main__":
    main()

