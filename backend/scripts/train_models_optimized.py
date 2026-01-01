"""
Optimized Medical Training Pipeline - Fast Clinical-Grade Training
Target: ROC-AUC > 0.9 with efficient preset parameters

This version uses pre-optimized hyperparameters for fast training.
For full hyperparameter search, use train_models_advanced.py with --n-trials.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    StackingClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# Optional libraries
try:
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("[WARNING] CatBoost not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available")

# Feature definitions
STAGE_A_FEATURES = [
    "sleep_duration", "sleep_quality", "sleep_disorder", "wake_up_during_night",
    "feel_sleepy_during_day", "average_screen_time", "smart_device_before_bed",
    "blue_light_filter", "stress_level", "daily_steps", "physical_activity",
    "caffeine_consumption", "alcohol_consumption", "smoking", "age", "gender",
    "bmi", "systolic", "diastolic", "heart_rate", "medical_issue", "ongoing_medication"
]

STAGE_B_FEATURES = STAGE_A_FEATURES + [
    "discomfort_eye_strain", "redness_in_eye", "itchiness_irritation_in_eye"
]

# ==================== WRAPPERS ====================

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=2000, depth=8, learning_rate=0.02, seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, X, y):
        self.model = cb.CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_seed=self.seed,
            verbose=0, allow_writing_files=False,
            auto_class_weights='Balanced', l2_leaf_reg=5
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"iterations": self.iterations, "depth": self.depth, 
                "learning_rate": self.learning_rate, "seed": self.seed}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# ==================== FEATURE ENGINEERING ====================

def engineer_medical_features(X: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Engineer clinically-relevant features"""
    X_eng = X.copy()
    
    # Digital eye strain
    if 'average_screen_time' in X_eng.columns and 'stress_level' in X_eng.columns:
        X_eng['digital_eye_strain'] = X_eng['average_screen_time'] * X_eng['stress_level']
    
    if 'average_screen_time' in X_eng.columns and 'blue_light_filter' in X_eng.columns:
        X_eng['unprotected_screen'] = X_eng['average_screen_time'] * (1 - X_eng['blue_light_filter'])
    
    # Cardiovascular features
    if 'systolic' in X_eng.columns and 'diastolic' in X_eng.columns:
        X_eng['map'] = (X_eng['systolic'] + 2 * X_eng['diastolic']) / 3
        X_eng['pulse_pressure'] = X_eng['systolic'] - X_eng['diastolic']
        X_eng['hypertension_risk'] = ((X_eng['systolic'] >= 130) | (X_eng['diastolic'] >= 80)).astype(int)
    
    # BMI features
    if 'bmi' in X_eng.columns:
        X_eng['bmi_squared'] = X_eng['bmi'] ** 2
        X_eng['obesity_flag'] = (X_eng['bmi'] >= 30).astype(int)
    
    # Sleep quality
    if 'sleep_duration' in X_eng.columns and 'sleep_quality' in X_eng.columns:
        X_eng['recovery_score'] = X_eng['sleep_duration'] * X_eng['sleep_quality']
        X_eng['sleep_efficiency'] = X_eng['sleep_quality'] / (X_eng['sleep_duration'] + 1)
    
    if 'wake_up_during_night' in X_eng.columns and 'feel_sleepy_during_day' in X_eng.columns:
        X_eng['sleep_disruption'] = X_eng['wake_up_during_night'] + X_eng['feel_sleepy_during_day']
    
    # Lifestyle risk
    if all(c in X_eng.columns for c in ['caffeine_consumption', 'alcohol_consumption', 'smoking']):
        X_eng['substance_risk'] = (
            X_eng['caffeine_consumption'] + 
            X_eng['alcohol_consumption'] * 1.5 +
            X_eng['smoking'] * 2
        )
    
    # Activity
    if 'physical_activity' in X_eng.columns and 'daily_steps' in X_eng.columns:
        X_eng['activity_composite'] = X_eng['physical_activity'] * np.log1p(X_eng['daily_steps'])
        X_eng['sedentary_flag'] = ((X_eng['physical_activity'] < 30) | (X_eng['daily_steps'] < 5000)).astype(int)
    
    # Age interactions
    if 'age' in X_eng.columns:
        X_eng['age_squared'] = X_eng['age'] ** 2
        X_eng['age_risk_65plus'] = (X_eng['age'] >= 65).astype(int)
        if 'average_screen_time' in X_eng.columns:
            X_eng['age_screen_interaction'] = X_eng['age'] * X_eng['average_screen_time']
        if 'bmi' in X_eng.columns:
            X_eng['age_bmi_interaction'] = X_eng['age'] * X_eng['bmi']
    
    # Medical burden
    if 'medical_issue' in X_eng.columns and 'ongoing_medication' in X_eng.columns:
        X_eng['medical_burden'] = X_eng['medical_issue'] + X_eng['ongoing_medication']
    
    # Stage B symptoms
    if stage == "B":
        symptom_cols = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
        if all(c in X_eng.columns for c in symptom_cols):
            X_eng['symptom_severity'] = (
                X_eng['discomfort_eye_strain'] * 2 +
                X_eng['redness_in_eye'] +
                X_eng['itchiness_irritation_in_eye']
            )
            if 'average_screen_time' in X_eng.columns:
                X_eng['symptom_screen_interaction'] = X_eng['symptom_severity'] * X_eng['average_screen_time']
    
    # Circadian disruption
    if 'smart_device_before_bed' in X_eng.columns and 'sleep_quality' in X_eng.columns:
        X_eng['circadian_disruption'] = X_eng['smart_device_before_bed'] * (10 - X_eng['sleep_quality'])
    
    return X_eng

def prepare_data(df: pd.DataFrame, stage: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare data with feature engineering"""
    print(f"\n{'='*70}")
    print(f"PREPARING STAGE {stage} DATA")
    print(f"{'='*70}")
    
    features = STAGE_A_FEATURES.copy() if stage == "A" else STAGE_B_FEATURES.copy()
    available_features = [f for f in features if f in df.columns]
    
    print(f"  Using {len(available_features)}/{len(features)} features")
    
    X = df[available_features].copy()
    y = df["dry_eye_disease"].astype(int)
    
    # Handle categorical (target encoding)
    global_mean = y.mean()
    for col in ['medical_issue', 'ongoing_medication']:
        if col in X.columns and X[col].dtype in ['int64', 'int32', 'float64']:
            continue
        if col in X.columns:
            agg = pd.DataFrame({'val': X[col], 'target': y}).groupby('val')['target'].agg(['count', 'mean'])
            counts, means = agg['count'], agg['mean']
            smoothed = (counts * means + 10 * global_mean) / (counts + 10)
            X[col] = X[col].map(smoothed.to_dict()).fillna(global_mean)
    
    # Convert to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing
    X = X.fillna(X.median())
    
    # Feature engineering
    print("  Engineering clinical features...")
    X = engineer_medical_features(X, stage)
    X = X.select_dtypes(include=['number'])
    
    print(f"  Final features: {X.shape[1]} (engineered from {len(available_features)})")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    return X, y, list(X.columns)

# ==================== OPTIMIZED ENSEMBLE ====================

def create_optimized_ensemble(X_train, y_train, stage: str, seed: int):
    """Create efficient stacking ensemble with preset params"""
    print(f"\n  Building Optimized Ensemble for Stage {stage}...")
    
    base_models = []
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Optimized parameters based on extensive testing
    if stage == "A":
        xgb_params = {
            'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.02,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
            'gamma': 2.0, 'reg_alpha': 3.0, 'reg_lambda': 5.0
        }
        lgb_params = {
            'n_estimators': 1800, 'max_depth': 7, 'learning_rate': 0.025,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 10,
            'reg_alpha': 2.5, 'reg_lambda': 4.0
        }
    else:  # Stage B
        xgb_params = {
            'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.03,
            'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 5,
            'gamma': 1.0, 'reg_alpha': 2.0, 'reg_lambda': 3.0
        }
        lgb_params = {
            'n_estimators': 1500, 'max_depth': 9, 'learning_rate': 0.03,
            'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_samples': 5,
            'reg_alpha': 1.5, 'reg_lambda': 2.5
        }
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb_params.update({
            'scale_pos_weight': scale_pos_weight, 'random_state': seed,
            'tree_method': 'hist', 'eval_metric': 'auc', 'verbosity': 0
        })
        xgb_model = xgb.XGBClassifier(**xgb_params)
        base_models.append(('xgb', xgb_model))
        print("    [OK] XGBoost added")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb_params.update({
            'scale_pos_weight': scale_pos_weight, 'random_state': seed,
            'verbosity': -1, 'n_jobs': 4  # Limit parallelism
        })
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        base_models.append(('lgb', lgb_model))
        print("    [OK] LightGBM added")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        cat_model = CatBoostWrapper(
            iterations=1500 if stage == "B" else 2000,
            depth=8 if stage == "B" else 6,
            learning_rate=0.025,
            seed=seed
        )
        base_models.append(('cat', cat_model))
        print("    [OK] CatBoost added")
    
    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=1000, max_depth=8 if stage == "B" else 6,
        learning_rate=0.05, l2_regularization=3.0, random_state=seed
    )
    base_models.append(('hgb', hgb_model))
    print("    [OK] HistGradientBoosting added")
    
    # ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=500, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced',
        random_state=seed, n_jobs=4
    )
    base_models.append(('et', et_model))
    print("    [OK] ExtraTrees added")
    
    # RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced',
        random_state=seed, n_jobs=4
    )
    base_models.append(('rf', rf_model))
    print("    [OK] RandomForest added")
    
    print(f"  [OK] Created {len(base_models)} base learners")
    
    # Meta-learner: MLP with regularization
    meta_learner = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.01, batch_size=256,
        learning_rate='adaptive', learning_rate_init=0.001,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed, verbose=False
    )
    
    # Create stacking
    stacking = StackingClassifier(
        estimators=base_models, final_estimator=meta_learner,
        cv=5, stack_method='predict_proba', n_jobs=2, verbose=0
    )
    
    print("  Training stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    return stacking

# ==================== MAIN TRAINING ====================

def train_stage_optimized(df: pd.DataFrame, stage: str, seed: int = 42) -> Dict[str, Any]:
    """Train optimized model for one stage"""
    print(f"\n{'='*70}")
    print(f"TRAINING STAGE {stage} - OPTIMIZED CLINICAL MODEL")
    print(f"{'='*70}")
    
    # Prepare data
    X, y, feature_list = prepare_data(df, stage)
    
    # Split
    print("\n  Splitting data (stratified)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=seed+1, stratify=y_train_val
    )
    
    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # SMOTE resampling
    if SMOTE_AVAILABLE:
        print("\n  Applying SMOTE-Tomek resampling...")
        smote_tomek = SMOTETomek(random_state=seed, n_jobs=4)
        X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
        print(f"    {len(X_train):,} -> {len(X_train_res):,} samples (pos: {y_train.mean():.2%} -> {y_train_res.mean():.2%})")
        X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train = pd.Series(y_train_res)
    
    # Robust scaling
    print("\n  Scaling features (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Build ensemble
    model = create_optimized_ensemble(X_train_scaled, y_train, stage, seed)
    
    # Calibrate probabilities
    print("\n  Calibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3, n_jobs=2)
    calibrated.fit(X_train_scaled, y_train)
    
    # Cross-validation
    print("\n  Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        calibrated, X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring='roc_auc', n_jobs=2
    )
    print(f"    CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Evaluate
    print("\n  Evaluating...")
    y_val_proba = calibrated.predict_proba(X_val_scaled)[:, 1]
    y_test_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Optimal threshold
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.2, 0.7, 0.01):
        y_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    
    # Metrics
    results = {
        'stage': stage,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'threshold': best_thresh,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'n_features': len(feature_list)
    }
    
    print(f"\n  {'='*66}")
    print(f"  VALIDATION: ROC-AUC = {val_auc:.4f}")
    print(f"  TEST:       ROC-AUC = {test_auc:.4f} {'[OK] TARGET ACHIEVED!' if test_auc > 0.9 else '[WARNING] Below target'}")
    print(f"  {'='*66}")
    print(f"    Precision: {results['test_precision']:.4f}")
    print(f"    Recall:    {results['test_recall']:.4f}")
    print(f"    F1-Score:  {results['test_f1']:.4f}")
    print(f"    Threshold: {best_thresh:.3f}")
    
    # Save model
    artifacts_dir = Path("modeling/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        'model': calibrated,
        'scaler': scaler,
        'features': feature_list,
        'threshold': best_thresh
    }
    
    artifact_path = artifacts_dir / f"model_{stage}_optimized.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"\n  Model saved: {artifact_path}")
    
    results['artifact_path'] = str(artifact_path)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train Optimized Clinical Models (Fast)")
    parser.add_argument("--input", type=Path, default=Path("data/standardized/clean_assessments.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", type=str, default="A,B", help="Stages to train (A, B, or A,B)")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}")
        return
    
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    stages = [s.strip().upper() for s in args.stages.split(',')]
    results_all = {}
    
    for stage in stages:
        if stage not in ['A', 'B']:
            continue
        results = train_stage_optimized(df, stage, args.seed)
        results_all[stage] = results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    for stage, res in results_all.items():
        status = "[OK] ACHIEVED" if res['test_auc'] > 0.9 else "[WARNING] BELOW TARGET"
        print(f"\nStage {stage}: {status}")
        print(f"  Test ROC-AUC: {res['test_auc']:.4f}")
        print(f"  Test F1:      {res['test_f1']:.4f}")
        print(f"  CV AUC:       {res['cv_auc_mean']:.4f} +/- {res['cv_auc_std']:.4f}")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
