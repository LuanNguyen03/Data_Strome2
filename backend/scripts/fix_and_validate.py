"""
Fix critical issues và validate properly
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

def safe_feature_engineering(X):
    """Conservative feature engineering - no leakage"""
    X = X.copy()
    
    # Only safe operations
    if 'systolic' in X.columns and 'diastolic' in X.columns:
        X['map'] = (X['systolic'] + 2 * X['diastolic']) / 3
        X['pulse_pressure'] = X['systolic'] - X['diastolic']
    
    if 'bmi' in X.columns:
        X['bmi_squared'] = X['bmi'] ** 2
    
    if 'average_screen_time' in X.columns and 'stress_level' in X.columns:
        X['screen_stress'] = X['average_screen_time'] * X['stress_level']
    
    if 'sleep_duration' in X.columns and 'sleep_quality' in X.columns:
        X['recovery'] = X['sleep_duration'] * X['sleep_quality']
    
    # Symptoms (Stage B only)
    symptom_cols = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
    if all(c in X.columns for c in symptom_cols):
        X['symptom_total'] = X[symptom_cols].sum(axis=1)
    
    return X.select_dtypes(include=['number'])

def train_fixed_model(df, stage, seed=42):
    """Train with proper validation"""
    print(f"\n{'='*70}")
    print(f"FIXED TRAINING - STAGE {stage}")
    print(f"{'='*70}")
    
    # Features
    base_features = [
        'age', 'bmi', 'systolic', 'diastolic', 'heart_rate',
        'sleep_duration', 'sleep_quality', 'stress_level',
        'average_screen_time', 'daily_steps', 'physical_activity',
        'medical_issue', 'ongoing_medication'
    ]
    
    if stage == "B":
        base_features += ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
    
    available = [f for f in base_features if f in df.columns]
    
    X = df[available].copy()
    y = df['dry_eye_disease'].astype(int)
    
    print(f"Features: {len(available)}")
    print(f"Samples: {len(X):,}, Positive: {y.mean():.2%}")
    
    # Handle categorical BEFORE split (to avoid leakage)
    for col in ['medical_issue', 'ongoing_medication']:
        if col in X.columns and X[col].dtype not in ['int64', 'int32', 'float64']:
            X[col] = pd.factorize(X[col])[0]
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())
    
    # === CRITICAL: Feature engineering BEFORE split ===
    print("\nFeature engineering (pre-split)...")
    X = safe_feature_engineering(X)
    print(f"Engineered features: {X.shape[1]}")
    
    # === PROPER SPLIT ===
    print("\nSplitting data...")
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=seed+1, stratify=y_trainval
    )
    
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X):.1%})")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X):.1%})")
    
    # === SMOTE only on TRAIN (not val/test!) ===
    if SMOTE_AVAILABLE:
        print("\nApplying SMOTE (train only)...")
        smote = SMOTETomek(random_state=seed, sampling_strategy=0.7)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  {len(X_train):,} -> {len(X_train_res):,} samples")
        X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train = pd.Series(y_train_res)
    
    # === SCALE (fit on train only) ===
    print("\nScaling...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # === TRAIN MODEL ===
    print("\nTraining XGBoost...")
    
    if not XGB_AVAILABLE:
        print("[ERROR] XGBoost not available")
        return None
    
    params = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 2.0,
        'reg_alpha': 3.0,
        'reg_lambda': 5.0,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': seed,
        'tree_method': 'hist',
        'eval_metric': 'auc'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # === CALIBRATE (on train only, use CV) ===
    print("Calibrating...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_train_scaled, y_train)
    
    # === PROPER EVALUATION ===
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Train
    y_train_proba = calibrated.predict_proba(X_train_scaled)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Val
    y_val_proba = calibrated.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    # Test (HOLDOUT - never seen before)
    y_test_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nAUC Scores:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    
    print(f"\nOverfitting Check:")
    train_test_gap = train_auc - test_auc
    print(f"  Train-Test Gap: {train_test_gap:.4f}")
    if train_test_gap > 0.15:
        print(f"  [WARNING] Significant overfitting (gap > 0.15)")
    elif train_test_gap > 0.10:
        print(f"  [CAUTION] Moderate overfitting (gap > 0.10)")
    else:
        print(f"  [OK] Acceptable generalization")
    
    # Optimal threshold on VAL
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]
    
    # Apply to test
    y_test_pred = (y_test_proba >= optimal_thresh).astype(int)
    
    print(f"\nTest Set Performance (threshold={optimal_thresh:.3f}):")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    # === DIAGNOSTIC: Check if predictions make sense ===
    print("\nDiagnostic Info:")
    print(f"  Predicted positives (test): {y_test_pred.sum()} / {len(y_test)} ({y_test_pred.mean():.2%})")
    print(f"  Actual positives (test):    {y_test.sum()} / {len(y_test)} ({y_test.mean():.2%})")
    print(f"  Prediction range: [{y_test_proba.min():.4f}, {y_test_proba.max():.4f}]")
    print(f"  Mean probability: {y_test_proba.mean():.4f}")
    
    # Save
    Path("modeling/artifacts").mkdir(parents=True, exist_ok=True)
    artifact_path = Path(f"modeling/artifacts/model_{stage}_fixed.joblib")
    joblib.dump({
        'model': calibrated,
        'scaler': scaler,
        'features': list(X.columns),
        'threshold': optimal_thresh
    }, artifact_path)
    print(f"\nSaved: {artifact_path}")
    
    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_test_gap': train_test_gap
    }

if __name__ == "__main__":
    input_path = Path("data/standardized/clean_assessments.parquet")
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        exit(1)
    
    print("Loading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")
    
    results = {}
    
    # Train both stages with FIXED pipeline
    for stage in ['A', 'B']:
        res = train_fixed_model(df, stage, seed=42)
        if res:
            results[stage] = res
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - FIXED PIPELINE")
    print(f"{'='*70}")
    
    for stage, res in results.items():
        print(f"\nStage {stage}:")
        print(f"  Train AUC: {res['train_auc']:.4f}")
        print(f"  Val AUC:   {res['val_auc']:.4f}")
        print(f"  Test AUC:  {res['test_auc']:.4f} {'[OK]' if res['test_auc'] > 0.55 else '[LOW]'}")
        print(f"  Gap:       {res['train_test_gap']:.4f} {'[OK]' if res['train_test_gap'] < 0.15 else '[OVERFIT]'}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print(f"{'='*70}")
    
    best_test_auc = max([r['test_auc'] for r in results.values()])
    
    if best_test_auc > 0.7:
        print("✅ Fixed pipeline shows improvement!")
        print(f"   Best test AUC: {best_test_auc:.4f}")
    elif best_test_auc > 0.55:
        print("⚠️ Some improvement, but still limited by data quality")
        print(f"   Best test AUC: {best_test_auc:.4f}")
        print("   Dataset has weak predictive signal")
    else:
        print("❌ Dataset fundamentally lacks predictive features")
        print(f"   Best test AUC: {best_test_auc:.4f}")
        print("   Need better quality medical data to achieve ROC_AUC > 0.9")
