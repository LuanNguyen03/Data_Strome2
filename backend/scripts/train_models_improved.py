"""
Improved training pipeline with advanced ML techniques.
Includes: hyperparameter tuning, early stopping, class weights, calibration,
feature engineering, cross-validation, and ensemble methods.

Input: data/standardized/clean_assessments.parquet
Outputs:
  - modeling/artifacts/*.joblib, *.json (with _improved suffix)
  - modeling/reports/*.json, *.md (with _improved suffix)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Best model saving and overfitting detection
try:
    from backend.scripts.train_with_best_model import (
        check_overfitting,
        save_best_model_checkpoint,
        should_stop_training
    )
    BEST_MODEL_AVAILABLE = True
except ImportError:
    BEST_MODEL_AVAILABLE = False
    print("[WARNING] Best model utilities not available")

# Advanced ML libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] Optuna not available, using default hyperparameters")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available")

# XGBoost with GPU support
XGBOOST_AVAILABLE = False
GPU_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    # Check GPU
    try:
        import subprocess
        import os
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
    except:
        GPU_AVAILABLE = False
except ImportError:
    print("[WARNING] XGBoost not available")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
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
    "blue_light_filter",
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
]

# Symptoms that MUST be excluded from Stage A
STAGE_A_EXCLUDED = [
    "discomfort_eyestrain",
    "discomfort_eye_strain",
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
    
    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except:
        pass
    
    try:
        import lightgbm
        versions["lightgbm"] = lightgbm.__version__
    except:
        pass
    
    try:
        import optuna
        versions["optuna"] = optuna.__version__
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


def engineer_features(X: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Add interaction and polynomial features"""
    print("  Engineering features...")
    X_eng = X.copy()
    original_cols = len(X_eng.columns)
    
    # Interaction features
    if 'average_screen_time' in X_eng.columns and 'sleep_duration' in X_eng.columns:
        X_eng['screen_sleep_interaction'] = X_eng['average_screen_time'] * X_eng['sleep_duration']
    
    if 'stress_level' in X_eng.columns and 'sleep_quality' in X_eng.columns:
        X_eng['stress_sleep_quality'] = X_eng['stress_level'] * X_eng['sleep_quality']
    
    if 'bmi' in X_eng.columns and 'age' in X_eng.columns:
        X_eng['bmi_age'] = X_eng['bmi'] * X_eng['age']
    
    if 'average_screen_time' in X_eng.columns and 'sleep_duration' in X_eng.columns:
        X_eng['screen_to_sleep_ratio'] = X_eng['average_screen_time'] / (X_eng['sleep_duration'] + 1)
    
    if 'daily_steps' in X_eng.columns:
        X_eng['steps_per_hour'] = X_eng['daily_steps'] / 24
    
    # Polynomial features for important variables
    if 'average_screen_time' in X_eng.columns:
        X_eng['screen_time_squared'] = X_eng['average_screen_time'] ** 2
    
    if 'sleep_quality' in X_eng.columns:
        X_eng['sleep_quality_squared'] = X_eng['sleep_quality'] ** 2
    
    if 'stress_level' in X_eng.columns:
        X_eng['stress_level_squared'] = X_eng['stress_level'] ** 2
    
    # Age bands interaction
    if 'age' in X_eng.columns:
        X_eng['age_screen_interaction'] = X_eng['age'] * X_eng.get('average_screen_time', 0)
    
    print(f"  Added {len(X_eng.columns) - original_cols} engineered features")
    return X_eng


def prepare_features_improved(
    df: pd.DataFrame, features: List[str], stage: str, use_imputation: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features with improved imputation and engineering"""
    print(f"\nPreparing {stage} features (improved)...")
    
    # Filter to rows with target available
    df_clean = df[df["dry_eye_disease"].notna()].copy()
    
    # Select features that exist in dataframe
    available_features = [f for f in features if f in df_clean.columns]
    missing_features = [f for f in features if f not in df_clean.columns]
    
    if missing_features:
        print(f"  [WARNING] Missing features: {missing_features}")
    
    X = df_clean[available_features].copy()
    y = df_clean["dry_eye_disease"].astype(int)
    
    # Convert binary Y/N columns to 0/1
    binary_map = {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
    X_processed = X.copy()
    
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            unique_vals = X_processed[col].dropna().unique()
            if len(unique_vals) <= 2:
                try:
                    # Suppress FutureWarning by converting to numeric directly
                    # Use map instead of replace to avoid downcasting warning
                    X_processed[col] = X_processed[col].map(binary_map).fillna(X_processed[col])
                    X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
                except:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
            else:
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
    
    # Ensure all columns are numeric
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
    
    # Ensure all columns are float64 before imputation
    X_processed = X_processed.astype(float)
    
    # Improved imputation
    if use_imputation:
        print("  Using KNN imputation...")
        # Store original columns and index
        original_columns = X_processed.columns.tolist()
        original_index = X_processed.index
        
        # Check for columns with all NaN (KNNImputer will drop these)
        cols_with_all_nan = X_processed.columns[X_processed.isna().all()].tolist()
        if cols_with_all_nan:
            print(f"  [WARNING] Columns with all NaN (will fill with 0): {cols_with_all_nan}")
            X_processed[cols_with_all_nan] = 0.0
        
        # Apply KNN imputation
        try:
            imputer = KNNImputer(n_neighbors=5)
            X_imputed_array = imputer.fit_transform(X_processed)
            
            # Verify shape matches
            if X_imputed_array.shape[1] != len(original_columns):
                print(f"  [WARNING] KNN imputation changed shape: {X_processed.shape} -> {X_imputed_array.shape}")
                print(f"  Falling back to median imputation...")
                # Fallback to median
                fill_values = {}
                for col in original_columns:
                    median_val = X_processed[col].median()
                    fill_values[col] = float(median_val) if not pd.isna(median_val) else 0.0
                X_processed = X_processed.fillna(fill_values)
            else:
                X_processed = pd.DataFrame(
                    X_imputed_array, 
                    columns=original_columns, 
                    index=original_index
                )
        except Exception as e:
            print(f"  [WARNING] KNN imputation failed ({e}), falling back to median...")
            # Fallback to median
            fill_values = {}
            for col in original_columns:
                median_val = X_processed[col].median()
                fill_values[col] = float(median_val) if not pd.isna(median_val) else 0.0
            X_processed = X_processed.fillna(fill_values)
    else:
        # Fallback to median
        fill_values = {}
        for col in X_processed.columns:
            if X_processed[col].dtype in ["int64", "float64", "float32", "int32"]:
                median_val = X_processed[col].median()
                fill_values[col] = float(median_val) if not pd.isna(median_val) else 0.0
            else:
                fill_values[col] = 0.0
        X_processed = X_processed.fillna(fill_values)
    
    # Feature engineering
    X_processed = engineer_features(X_processed, stage)
    
    # Ensure all columns are float64
    X_processed = X_processed.astype(float)
    
    print(f"  Features: {len(X_processed.columns)} (original: {len(available_features)})")
    print(f"  Samples: {len(X_processed):,}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    return X_processed, y, list(X_processed.columns)


def get_hyperparameter_presets(stage: str) -> List[Dict[str, Any]]:
    """Get multiple hyperparameter presets for ensemble diversity"""
    presets = []
    
    if stage == "A":
        # Preset 1: Conservative (strong regularization, fewer trees)
        presets.append({
            'n_estimators': 300,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 15,
            'gamma': 1.0,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'max_delta_step': 3,
        })
        # Preset 2: Balanced (moderate regularization, more trees)
        presets.append({
            'n_estimators': 500,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 12,
            'gamma': 0.8,
            'reg_alpha': 1.5,
            'reg_lambda': 2.5,
            'max_delta_step': 4,
        })
        # Preset 3: Aggressive (more capacity, careful regularization)
        presets.append({
            'n_estimators': 600,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'gamma': 0.6,
            'reg_alpha': 1.2,
            'reg_lambda': 2.0,
            'max_delta_step': 5,
        })
    else:  # Stage B
        # Preset 1: Conservative
        presets.append({
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'max_delta_step': 3,
        })
        # Preset 2: Balanced
        presets.append({
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 8,
            'gamma': 0.4,
            'reg_alpha': 0.8,
            'reg_lambda': 1.5,
            'max_delta_step': 4,
        })
        # Preset 3: Aggressive
        presets.append({
            'n_estimators': 600,
            'max_depth': 7,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 6,
            'gamma': 0.3,
            'reg_alpha': 0.6,
            'reg_lambda': 1.2,
            'max_delta_step': 5,
        })
    
    return presets


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage: str,
    seed: int,
    n_trials: int = 150,
    use_cv: bool = True,
) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna with optional cross-validation and ROC-AUC optimization"""
    if not OPTUNA_AVAILABLE:
        return {}
    
    print(f"  Optimizing hyperparameters ({n_trials} trials, CV={use_cv})...")
    
    # Use cross-validation for more robust optimization
    if use_cv:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    def objective(trial):
        # Different hyperparameter ranges based on stage to prevent overfitting
        if stage == "A":
            # Stage A: Search space cực rộng để tìm tín hiệu ẩn trong nhiễu
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000), 
                'max_depth': trial.suggest_int('max_depth', 3, 10), 
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
                'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'random_state': seed,
                'eval_metric': 'logloss',
                'verbosity': 0,
            }
        else:
            # Stage B: Tập trung vào độ chính xác phân loại (F1/ROC-AUC)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 1e-8, 2.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'random_state': seed,
                'eval_metric': 'logloss',
                'verbosity': 0,
            }
        
        # Add GPU support if available
        if GPU_AVAILABLE:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['predictor'] = 'gpu_predictor'
        else:
            params['tree_method'] = 'hist'
            params['n_jobs'] = -1
        
        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        if use_cv:
            # Use cross-validation for more robust evaluation
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                try:
                    model.fit(
                        X_cv_train, y_cv_train,
                        eval_set=[(X_cv_val, y_cv_val)],
                        callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                        verbose=False
                    )
                except (TypeError, AttributeError):
                    try:
                        model.fit(
                            X_cv_train, y_cv_train,
                            eval_set=[(X_cv_val, y_cv_val)],
                            early_stopping_rounds=100,
                            verbose=False
                        )
                    except TypeError:
                        model.fit(X_cv_train, y_cv_train, verbose=False)
                
                y_cv_proba = model.predict_proba(X_cv_val)[:, 1]
                roc_auc = roc_auc_score(y_cv_val, y_cv_proba)
                cv_scores.append(roc_auc)
            
            return np.mean(cv_scores)
        else:
            # Original single validation set approach
            model = xgb.XGBClassifier(**params)
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                    verbose=False
                )
            except (TypeError, AttributeError):
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False
                    )
                except TypeError:
                    model.fit(X_train, y_train, verbose=False)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
    
    study = optuna.create_study(direction='maximize', study_name=f'stage_{stage}')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best score: {study.best_value:.4f}")
    return study.best_params


def create_ensemble_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: Dict[str, Any],
    stage: str,
    seed: int,
) -> Any:
    """Create ensemble of XGBoost, LightGBM, and RandomForest"""
    print("  Creating ensemble model...")
    
    models = []
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb_params = best_params.copy()
        if GPU_AVAILABLE:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
            xgb_params['predictor'] = 'gpu_predictor'
        else:
            xgb_params['tree_method'] = 'hist'
            xgb_params['n_jobs'] = -1
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        try:
            # XGBoost 3.1+ uses callbacks
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],  # Increased patience for best results
                verbose=False
            )
        except (TypeError, AttributeError):
            try:
                # Fallback for older versions
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,  # Increased patience for best results
                    verbose=False
                )
            except TypeError:
                # If both fail, train without early stopping
                xgb_model.fit(X_train, y_train, verbose=False)
        models.append(('xgb', xgb_model))
        print("    [OK] XGBoost added")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'n_estimators': best_params.get('n_estimators', 500),  # Increased for better results
            'max_depth': best_params.get('max_depth', 8),
            'learning_rate': best_params.get('learning_rate', 0.1),
            'subsample': best_params.get('subsample', 0.8),
            'colsample_bytree': best_params.get('colsample_bytree', 0.8),
            'min_child_samples': best_params.get('min_child_weight', 1),
            'reg_alpha': best_params.get('reg_alpha', 0),
            'reg_lambda': best_params.get('reg_lambda', 1.0),
            'random_state': seed,
            'verbosity': -1,
            'n_jobs': -1,
        }
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        lgb_params['scale_pos_weight'] = scale_pos_weight
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]  # Stop if no improvement for 100 rounds
        )
        models.append(('lgbm', lgb_model))
        print("    [OK] LightGBM added")
    
    # RandomForest
    rf_params = {
        'n_estimators': 500,  # Increased for better results
        'max_depth': best_params.get('max_depth', 10),
        'min_samples_split': best_params.get('min_child_weight', 20),
        'min_samples_leaf': 10,
        'random_state': seed,
        'n_jobs': -1,
        'class_weight': 'balanced',
    }
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    models.append(('rf', rf_model))
    print("    [OK] RandomForest added")
    
    # Create voting ensemble
    if len(models) > 1:
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        print(f"  [OK] Ensemble created with {len(models)} models")
        return ensemble
    else:
        return models[0][1]  # Return single model if only one available


def train_model_improved(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage: str,
    seed: int,
    use_ensemble: bool = True,
    n_trials: int = 150,
) -> Tuple[Any, StandardScaler, Optional[SelectKBest], float, Dict[str, Any]]:
    """Train improved model with all enhancements"""
    global GPU_AVAILABLE
    
    print(f"\nTraining {stage} model (improved)...")
    
    # Feature selection - giảm cho Stage A (đang quá aggressive làm model không học được)
    feature_selector = None
    if stage == "A" and len(X_train.columns) > 25:
        # Select top features cho Stage A nhưng giữ nhiều hơn (25 thay vì 20)
        print("  Applying feature selection...")
        n_features = min(25, len(X_train.columns))  # Tăng từ 20 lên 25
        feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        X_val_selected = feature_selector.transform(X_val)
        selected_features = X_train.columns[feature_selector.get_support()].tolist()
        print(f"  Selected {len(selected_features)} features from {len(X_train.columns)}")
        X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_val = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    # Calculate class weights
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Class balance: {y_train.mean():.2%} positive, scale_pos_weight={scale_pos_weight:.2f}")
    
    # Optimize hyperparameters with cross-validation
    if OPTUNA_AVAILABLE:
        best_params = optimize_hyperparameters(
            X_train_scaled, y_train, X_val_scaled, y_val, stage, seed, n_trials, use_cv=True
        )
    else:
        # Default parameters
        best_params = {
            'n_estimators': 500,  # Increased for better results
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        print("  Using default hyperparameters (Optuna not available)")
    
    # Add common parameters
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = seed
    best_params['eval_metric'] = 'logloss'
    # Remove use_label_encoder (not needed in XGBoost 3.1+)
    best_params.pop('use_label_encoder', None)
    best_params['verbosity'] = 1
    
    # Create model (ensemble or single)
    if use_ensemble and (XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE):
        model = create_ensemble_model(
            X_train_scaled, y_train, X_val_scaled, y_val, best_params, stage, seed
        )
    else:
        # Single XGBoost model
        if XGBOOST_AVAILABLE:
            if GPU_AVAILABLE:
                best_params['tree_method'] = 'hist'
                best_params['device'] = 'cuda'
                best_params['predictor'] = 'gpu_predictor'
            else:
                best_params['tree_method'] = 'hist'
                best_params['n_jobs'] = -1
            
            model = xgb.XGBClassifier(**best_params)
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
            # Fallback to RandomForest
            model = RandomForestClassifier(
                n_estimators=500,  # Increased for better results
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=seed,
                n_jobs=-1,
                class_weight='balanced',
            )
            model.fit(X_train_scaled, y_train)
    
    # Calibrate probabilities on TRAIN (avoid leakage on val)
    print("  Calibrating probabilities (train-based)...")
    try:
        calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=3,
            n_jobs=-1
        )
        calibrated_model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"  [WARNING] Calibration failed ({e}), using uncalibrated model")
        calibrated_model = model
    
    # Get probabilities
    y_val_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
    
    # Select threshold based on stage - ưu tiên Recall cao cho Stage A
    if stage == "A":
        # Stage A: screening → ưu tiên Recall cao (≥0.995), sau đó optimize precision/F1
        thresholds = np.arange(0.1, 0.5, 0.01)  # Range thấp hơn để giữ Recall cao
        target_recall = 0.995  # Tăng từ 0.95
        best_threshold = 0.5
        best_precision_at_target = -1.0
        best_score = -1.0
        
        for thresh in thresholds:
            y_pred = (y_val_proba >= thresh).astype(int)
            recall = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Rule 1: ưu tiên thỏa recall >= target, chọn precision cao nhất
            if recall >= target_recall:
                if precision > best_precision_at_target:
                    best_precision_at_target = precision
                    best_threshold = thresh
                    best_score = 0.5 * recall + 0.3 * precision + 0.2 * f1
                continue
            
            # Rule 2: nếu không thỏa target, dùng weighted score để chọn tốt nhất còn lại
            weighted_score = 0.5 * recall + 0.3 * precision + 0.2 * f1
            if weighted_score > best_score:
                best_score = weighted_score
                best_threshold = thresh
    else:
        # Stage B: optimize for F1 với cân bằng Recall/Precision tốt hơn
        thresholds = np.arange(0.2, 0.6, 0.01)  # Range thấp hơn để giữ Recall cao hơn
        best_threshold = 0.4
        best_f1 = 0.0
        
        for thresh in thresholds:
            y_pred = (y_val_proba >= thresh).astype(int)
            recall = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            # Weighted F1: cân bằng tốt hơn (ưu tiên Recall hơn một chút)
            weighted_f1 = 0.5 * f1 + 0.3 * recall + 0.2 * precision
            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                best_threshold = thresh
    
    print(f"  Selected threshold: {best_threshold:.3f}")
    
    return calibrated_model, scaler, feature_selector, best_threshold, best_params


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train/val/test"""
    print(f"\nSplitting data (seed={seed})...")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
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
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    
    return metrics


def save_artifacts_improved(
    model: Any,
    scaler: StandardScaler,
    features: List[str],
    threshold: float,
    stage: str,
    artifacts_dir: Path,
    metadata: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Dict[str, str]:
    """Save improved model artifacts"""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_paths = {}
    
    # Save model with _improved suffix
    model_path = artifacts_dir / f"model_{stage}_{'screening' if stage == 'A' else 'triage'}_improved.joblib"
    joblib.dump(model, model_path)
    try:
        artifact_paths["model"] = str(model_path.relative_to(project_root))
    except ValueError:
        artifact_paths["model"] = str(model_path)
    
    # Save scaler
    scaler_path = artifacts_dir / f"preprocessing_{stage}_improved.joblib"
    joblib.dump(scaler, scaler_path)
    try:
        artifact_paths["preprocessing"] = str(scaler_path.relative_to(project_root))
    except ValueError:
        artifact_paths["preprocessing"] = str(scaler_path)
    
    # Save feature list
    features_path = artifacts_dir / f"feature_list_{stage}_improved.json"
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)
    try:
        artifact_paths["features"] = str(features_path.relative_to(project_root))
    except ValueError:
        artifact_paths["features"] = str(features_path)
    
    # Save metadata
    metadata_path = artifacts_dir / "model_metadata_improved.json"
    stage_metadata = {
        f"stage_{stage}": {
            "threshold": float(threshold),
            "features": features,
            "feature_count": len(features),
            "hyperparameters": hyperparams,
            "improvements": [
                "hyperparameter_tuning",
                "early_stopping",
                "class_weights",
                "calibration",
                "feature_engineering",
                "ensemble_methods",
                "knn_imputation",
            ],
        }
    }
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            full_metadata = json.load(f)
    else:
        full_metadata = {}
    
    full_metadata.update(stage_metadata)
    for key, value in metadata.items():
        if key not in ["stage", "threshold"]:
            full_metadata[key] = value
    
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    try:
        artifact_paths["metadata"] = str(metadata_path.relative_to(project_root))
    except ValueError:
        artifact_paths["metadata"] = str(metadata_path)
    
    return artifact_paths


def save_reports_improved(
    metrics_val: Dict[str, Any],
    metrics_test: Dict[str, Any],
    split_report: Dict[str, Any],
    reports_dir: Path,
    stage: str,
) -> Dict[str, str]:
    """Save improved reports"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # Save metrics with _improved suffix
    metrics_path = reports_dir / "model_metrics_improved.json"
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
    cm_path = reports_dir / "confusion_matrices_improved.json"
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
    split_path = reports_dir / "data_split_report_improved.json"
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
    
    # Save calibration report
    cal_path = reports_dir / "calibration_report_improved.json"
    cal_data = {
        f"stage_{stage}": {
            "method": "isotonic",
            "cv_folds": 3,
            "note": "Calibration applied using isotonic regression",
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


def generate_model_comparison_improved(reports_dir: Path) -> None:
    """Generate improved model comparison report"""
    metrics_path = reports_dir / "model_metrics_improved.json"
    if not metrics_path.exists():
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    md_content = "# Improved Model Comparison Report\n\n"
    md_content += f"Generated: {datetime.now().isoformat()}\n\n"
    md_content += "## Improvements Applied\n\n"
    md_content += "- ✅ Hyperparameter tuning (Optuna)\n"
    md_content += "- ✅ Early stopping\n"
    md_content += "- ✅ Class weights (scale_pos_weight)\n"
    md_content += "- ✅ Probability calibration (isotonic)\n"
    md_content += "- ✅ Feature engineering (interactions, polynomials)\n"
    md_content += "- ✅ Ensemble methods (XGBoost + LightGBM + RandomForest)\n"
    md_content += "- ✅ KNN imputation\n\n"
    
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
    
    comparison_path = reports_dir / "model_comparison_improved.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"[OK] Improved model comparison report: {comparison_path}")


def train_stage_improved(
    df: pd.DataFrame,
    stage: str,
    seed: int,
    artifacts_dir: Path,
    reports_dir: Path,
    use_ensemble: bool = True,
    n_trials: int = 50,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Train one stage with all improvements"""
    print(f"\n{'='*60}")
    print(f"Training Stage {stage} (IMPROVED)")
    print(f"{'='*60}")
    
    # Select features
    if stage == "A":
        features = STAGE_A_FEATURES.copy()
        # Check leakage
        excluded_in_features = [f for f in STAGE_A_EXCLUDED if f in features]
        if excluded_in_features:
            raise ValueError(f"Stage A leakage detected! Features contain symptoms: {excluded_in_features}")
        print("[OK] Stage A leakage check passed")
    else:
        features = STAGE_B_FEATURES.copy()
    
    # Prepare data with improvements
    X, y, available_features = prepare_features_improved(df, features, stage, use_imputation=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=seed)
    
    # Train improved model
    model, scaler, feature_selector, threshold, hyperparams = train_model_improved(
        X_train, y_train, X_val, y_val, stage, seed, use_ensemble, n_trials
    )
    
    # Apply feature selection to test set if used
    if feature_selector is not None:
        # Get selected feature names before transforming
        selected_feature_names = X_test.columns[feature_selector.get_support()].tolist()
        X_test_selected = feature_selector.transform(X_test)
        X_test = pd.DataFrame(
            X_test_selected,
            columns=selected_feature_names,
            index=X_test.index
        )
        # Also apply to val for consistency
        X_val_selected = feature_selector.transform(X_val)
        X_val = pd.DataFrame(
            X_val_selected,
            columns=selected_feature_names,
            index=X_val.index
        )
    
    # Compute metrics
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
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
    
    # Check for overfitting
    is_overfitting, gap = check_overfitting(
        metrics_val['roc_auc'],
        metrics_test['roc_auc'],
        max_gap=0.20
    )
    print(f"\nOverfitting Check:")
    print(f"  Validation-Test Gap: {gap:.4f}")
    if is_overfitting:
        print(f"  [WARNING] Overfitting detected! Gap > 0.20")
    else:
        print(f"  [OK] No significant overfitting detected")
    
    # Save best model checkpoint
    if BEST_MODEL_AVAILABLE:
        checkpoint_dir = artifacts_dir / "checkpoints"
        save_best_model_checkpoint(
            model, scaler, feature_selector, threshold,
            metrics_val, metrics_test,
            checkpoint_dir, stage, iteration=0
        )
    
    # Save artifacts
    metadata = {
        "stage": stage,
        "threshold": float(threshold),
        "python_versions": get_python_versions(),
    }
    # Get final feature list (after feature selection if used)
    final_features = X_train.columns.tolist()
    artifact_paths = save_artifacts_improved(
        model, scaler, final_features, threshold, stage, artifacts_dir, metadata, hyperparams
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
    report_paths = save_reports_improved(metrics_val, metrics_test, split_report, reports_dir, stage)
    
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
        "improvements": "hyperparameter_tuning,early_stopping,class_weights,calibration,feature_engineering,ensemble,knn_imputation",
    }
    
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"entries": []}
    
    registry["entries"].append(entry)
    registry["latest_improved"] = entry
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n[OK] Registry updated: {registry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train improved 2-stage medical models with advanced ML techniques"
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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter tuning (increased for better results)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble methods (use single model)",
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
    artifacts_A, reports_A, metrics_A = train_stage_improved(
        df, "A", args.seed, args.artifacts_dir, args.reports_dir,
        use_ensemble=not args.no_ensemble, n_trials=args.n_trials
    )
    
    # Train Stage B
    artifacts_B, reports_B, metrics_B = train_stage_improved(
        df, "B", args.seed, args.artifacts_dir, args.reports_dir,
        use_ensemble=not args.no_ensemble, n_trials=args.n_trials
    )
    
    # Generate model comparison
    generate_model_comparison_improved(args.reports_dir)
    
    # Update registry
    model_version = f"v1.1.improved.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    update_registry(
        args.registry_path,
        model_version,
        dataset_hash,
        {"stage_A": artifacts_A, "stage_B": artifacts_B},
        {"stage_A": reports_A, "stage_B": reports_B},
        {"stage_A": metrics_A, "stage_B": metrics_B},
    )
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Improved training completed successfully!")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to: {args.artifacts_dir}")
    print(f"Reports saved to: {args.reports_dir}")
    print(f"Registry updated: {args.registry_path}")


if __name__ == "__main__":
    main()

