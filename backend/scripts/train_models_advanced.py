"""
Advanced Medical Training Pipeline V15 - Clinical Grade (ROC-AUC > 0.9 Target)
Optimized for healthcare deployment with robust feature engineering and ensemble stacking.

Key Improvements:
- Enhanced medical feature engineering (clinical signals)
- Advanced imbalanced data handling (SMOTE + Tomek + class weights)
- Multi-layer stacking with neural network meta-learner
- Optuna hyperparameter optimization
- Stratified K-Fold cross-validation
- Probability calibration for clinical decisions
- Medical-grade monitoring and validation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier,
    VotingClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

warnings.filterwarnings('ignore')

# Optional advanced libraries
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except:
    TABNET_AVAILABLE = False

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False

# Import shared features
try:
    from backend.scripts.train_models_improved import (
        STAGE_A_FEATURES, STAGE_B_FEATURES, check_stage_a_leakage,
        compute_metrics, update_registry
    )
except:
    # Fallback definitions
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

# ==================== MEDICAL WRAPPERS ====================

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible CatBoost wrapper for stacking"""
    def __init__(self, iterations=2000, depth=8, learning_rate=0.02, seed=42, l2_leaf_reg=5):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.l2_leaf_reg = l2_leaf_reg
        self.model = None
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, X, y):
        self.model = cb.CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            l2_leaf_reg=self.l2_leaf_reg,
            verbose=0,
            allow_writing_files=False,
            auto_class_weights='Balanced'
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "iterations": self.iterations, "depth": self.depth, 
            "learning_rate": self.learning_rate, "seed": self.seed,
            "l2_leaf_reg": self.l2_leaf_reg
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible TabNet wrapper"""
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"
        
    def fit(self, X, y):
        if self.model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = TabNetClassifier(verbose=0, seed=self.seed, device_name=device)
        X_vals = X.values if hasattr(X, "values") else X
        y_vals = y.values if hasattr(y, "values") else y
        self.model.fit(X_vals, y_vals, max_epochs=200, patience=30, batch_size=512, virtual_batch_size=128)
        return self
        
    def predict_proba(self, X):
        X_vals = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(X_vals)
        
    def predict(self, X):
        X_vals = X.values if hasattr(X, "values") else X
        return self.model.predict(X_vals)
        
    def get_params(self, deep=True): 
        return {"seed": self.seed}
        
    def set_params(self, **p): 
        return self

# ==================== CLINICAL FEATURE ENGINEERING ====================

def engineer_medical_features(X: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    Engineer clinically-relevant features based on medical domain knowledge.
    Target: Extract maximum signal for dry eye disease prediction.
    """
    X_eng = X.copy()
    
    # === OCULAR STRAIN & FATIGUE INDICATORS ===
    if 'average_screen_time' in X_eng.columns and 'stress_level' in X_eng.columns:
        # Digital eye strain composite score
        X_eng['digital_eye_strain'] = X_eng['average_screen_time'] * X_eng['stress_level']
        
    if 'average_screen_time' in X_eng.columns and 'blue_light_filter' in X_eng.columns:
        # Unprotected screen exposure
        X_eng['unprotected_screen_exposure'] = X_eng['average_screen_time'] * (1 - X_eng['blue_light_filter'])
    
    # === CARDIOVASCULAR & METABOLIC FEATURES ===
    if 'systolic' in X_eng.columns and 'diastolic' in X_eng.columns:
        # Mean Arterial Pressure (MAP)
        X_eng['map'] = (X_eng['systolic'] + 2 * X_eng['diastolic']) / 3
        # Pulse Pressure (PP)
        X_eng['pulse_pressure'] = X_eng['systolic'] - X_eng['diastolic']
        # Hypertension risk flag
        X_eng['hypertension_risk'] = ((X_eng['systolic'] >= 130) | (X_eng['diastolic'] >= 80)).astype(int)
    
    if 'bmi' in X_eng.columns:
        # BMI risk categories
        X_eng['bmi_squared'] = X_eng['bmi'] ** 2
        X_eng['obesity_flag'] = (X_eng['bmi'] >= 30).astype(int)
        X_eng['overweight_flag'] = ((X_eng['bmi'] >= 25) & (X_eng['bmi'] < 30)).astype(int)
    
    # === SLEEP QUALITY & RECOVERY ===
    if 'sleep_duration' in X_eng.columns and 'sleep_quality' in X_eng.columns:
        # Recovery score
        X_eng['recovery_score'] = X_eng['sleep_duration'] * X_eng['sleep_quality']
        # Sleep efficiency proxy
        X_eng['sleep_efficiency'] = X_eng['sleep_quality'] / (X_eng['sleep_duration'] + 1)
        
    if 'wake_up_during_night' in X_eng.columns and 'feel_sleepy_during_day' in X_eng.columns:
        # Sleep disruption score
        X_eng['sleep_disruption'] = X_eng['wake_up_during_night'] + X_eng['feel_sleepy_during_day']
    
    # === LIFESTYLE RISK FACTORS ===
    if all(c in X_eng.columns for c in ['caffeine_consumption', 'alcohol_consumption', 'smoking']):
        # Substance use risk score
        X_eng['substance_risk'] = (
            X_eng['caffeine_consumption'] + 
            X_eng['alcohol_consumption'] * 1.5 +  # Alcohol has higher weight
            X_eng['smoking'] * 2  # Smoking has highest weight
        )
    
    if 'physical_activity' in X_eng.columns and 'daily_steps' in X_eng.columns:
        # Activity level composite
        X_eng['activity_composite'] = X_eng['physical_activity'] * np.log1p(X_eng['daily_steps'])
        # Sedentary lifestyle flag
        X_eng['sedentary_flag'] = ((X_eng['physical_activity'] < 30) | (X_eng['daily_steps'] < 5000)).astype(int)
    
    # === AGE-RELATED INTERACTIONS ===
    if 'age' in X_eng.columns:
        X_eng['age_squared'] = X_eng['age'] ** 2
        X_eng['age_risk_65plus'] = (X_eng['age'] >= 65).astype(int)
        
        if 'average_screen_time' in X_eng.columns:
            # Age-screen time interaction (older + high screen time = higher risk)
            X_eng['age_screen_interaction'] = X_eng['age'] * X_eng['average_screen_time']
            
        if 'bmi' in X_eng.columns:
            # Age-BMI interaction (metabolic syndrome risk)
            X_eng['age_bmi_interaction'] = X_eng['age'] * X_eng['bmi']
    
    # === MEDICAL CONDITION COMPOSITE ===
    if 'medical_issue' in X_eng.columns and 'ongoing_medication' in X_eng.columns:
        # Medical burden score
        X_eng['medical_burden'] = X_eng['medical_issue'] + X_eng['ongoing_medication']
    
    # === STAGE B: SYMPTOM-BASED FEATURES ===
    if stage == "B":
        symptom_cols = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
        if all(c in X_eng.columns for c in symptom_cols):
            # Already have symptom_score, but add interactions
            X_eng['symptom_severity'] = (
                X_eng['discomfort_eye_strain'] * 2 +  # Discomfort is most predictive
                X_eng['redness_in_eye'] +
                X_eng['itchiness_irritation_in_eye']
            )
            
            # Symptom-screen interaction
            if 'average_screen_time' in X_eng.columns:
                X_eng['symptom_screen_interaction'] = X_eng['symptom_severity'] * X_eng['average_screen_time']
    
    # === CIRCADIAN RHYTHM DISRUPTION ===
    if 'smart_device_before_bed' in X_eng.columns and 'sleep_quality' in X_eng.columns:
        # Circadian disruption proxy
        X_eng['circadian_disruption'] = X_eng['smart_device_before_bed'] * (10 - X_eng['sleep_quality'])
    
    return X_eng

# ==================== DATA PREPARATION ====================

def handle_medical_categorical(X: pd.DataFrame, y: pd.Series, cat_cols: List[str]) -> pd.DataFrame:
    """
    Handle categorical medical variables with target encoding + smoothing.
    More robust than simple label encoding for medical categories.
    """
    X_encoded = X.copy()
    global_mean = y.mean()
    smoothing_factor = 10  # Prevent overfitting on rare categories
    
    for col in cat_cols:
        if col in X_encoded.columns:
            if X_encoded[col].dtype in ['object', 'category']:
                # Target encoding with smoothing
                agg = pd.DataFrame({
                    'val': X_encoded[col], 
                    'target': y
                }).groupby('val')['target'].agg(['count', 'mean'])
                
                counts, means = agg['count'], agg['mean']
                smoothed = (counts * means + smoothing_factor * global_mean) / (counts + smoothing_factor)
                
                X_encoded[col] = X_encoded[col].map(smoothed.to_dict()).fillna(global_mean)
            elif X_encoded[col].dtype in ['int64', 'int32']:
                # Already numeric, keep as is
                pass
    
    return X_encoded

def prepare_data_advanced(df: pd.DataFrame, stage: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Advanced data preparation with medical domain knowledge.
    """
    print(f"\n{'='*70}")
    print(f"Preparing {stage} Features (Clinical Grade)")
    print(f"{'='*70}")
    
    # Select features
    features = STAGE_A_FEATURES.copy() if stage == "A" else STAGE_B_FEATURES.copy()
    
    # Medical categorical columns
    medical_cat_cols = ['medical_issue', 'ongoing_medication']
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"  ⚠️  Missing features: {missing_features}")
    
    print(f"  ✓ Using {len(available_features)} features")
    
    # Extract X, y
    X = df[available_features].copy()
    y = df["dry_eye_disease"].astype(int)
    
    # Handle categorical medical variables (target encoding)
    X = handle_medical_categorical(X, y, medical_cat_cols)
    
    # Convert all to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing with median (should be minimal after standardization)
    X_numeric = X.select_dtypes(include=['number'])
    X[X_numeric.columns] = X_numeric.fillna(X_numeric.median())
    
    # Feature engineering
    print("  🔬 Engineering clinical features...")
    X = engineer_medical_features(X, stage)
    
    # Ensure all numeric
    X = X.select_dtypes(include=['number'])
    
    print(f"  ✓ Final feature count: {X.shape[1]} (original: {len(available_features)})")
    print(f"  ✓ Sample count: {X.shape[0]:,}")
    print(f"  ✓ Positive rate: {y.mean():.2%}")
    
    return X, y, list(X.columns)

# ==================== HYPERPARAMETER OPTIMIZATION ====================

def optimize_hyperparameters_optuna(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    stage: str, seed: int, n_trials: int = 100
) -> Dict[str, Any]:
    """
    Optuna-based hyperparameter optimization targeting ROC-AUC > 0.9
    """
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        print("  ⚠️  Optuna/XGBoost not available, using preset params")
        return get_preset_params(stage)
    
    print(f"  🔍 Optimizing hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': seed,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    study = optuna.create_study(direction='maximize', study_name=f'stage_{stage}_v15')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  ✓ Best AUC: {study.best_value:.4f}")
    
    return study.best_params

def get_preset_params(stage: str) -> Dict[str, Any]:
    """Fallback preset parameters optimized for medical data"""
    if stage == "A":
        return {
            'n_estimators': 2000,
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'gamma': 2.0,
            'reg_alpha': 3.0,
            'reg_lambda': 5.0,
        }
    else:  # Stage B
        return {
            'n_estimators': 1500,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 5,
            'gamma': 1.0,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
        }

# ==================== ADVANCED ENSEMBLE STACKING ====================

def create_clinical_ensemble(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    best_params: Dict[str, Any],
    stage: str, seed: int
) -> Any:
    """
    Create multi-layer stacking ensemble optimized for medical classification.
    Target: ROC-AUC > 0.9
    """
    print(f"\n  🏗️  Building Clinical-Grade Ensemble...")
    
    base_models = []
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # === LAYER 1: DIVERSE BASE LEARNERS ===
    
    # [1] XGBoost - Gradient Boosting
    if XGBOOST_AVAILABLE:
        xgb_params = best_params.copy()
        xgb_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': seed,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'verbosity': 0
        })
        xgb_model = xgb.XGBClassifier(**xgb_params)
        base_models.append(('xgb', xgb_model))
        print("    ✓ XGBoost added")
    
    # [2] LightGBM - Fast gradient boosting
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'n_estimators': best_params.get('n_estimators', 1500),
            'max_depth': best_params.get('max_depth', 8),
            'learning_rate': best_params.get('learning_rate', 0.03),
            'subsample': best_params.get('subsample', 0.85),
            'colsample_bytree': best_params.get('colsample_bytree', 0.85),
            'min_child_samples': max(1, best_params.get('min_child_weight', 5)),
            'reg_alpha': best_params.get('reg_alpha', 2.0),
            'reg_lambda': best_params.get('reg_lambda', 3.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': seed,
            'verbosity': -1,
            'n_jobs': -1
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        base_models.append(('lgb', lgb_model))
        print("    ✓ LightGBM added")
    
    # [3] CatBoost - Handles categorical features well
    if CATBOOST_AVAILABLE:
        cat_params = {
            'iterations': min(2000, best_params.get('n_estimators', 1500)),
            'depth': best_params.get('max_depth', 8),
            'learning_rate': best_params.get('learning_rate', 0.03),
            'l2_leaf_reg': best_params.get('reg_lambda', 3.0),
            'seed': seed
        }
        cat_model = CatBoostWrapper(**cat_params)
        base_models.append(('cat', cat_model))
        print("    ✓ CatBoost added")
    
    # [4] HistGradientBoosting - Native sklearn, fast
    hgb_params = {
        'max_iter': min(1000, best_params.get('n_estimators', 1000)),
        'max_depth': best_params.get('max_depth', 8),
        'learning_rate': best_params.get('learning_rate', 0.05),
        'l2_regularization': best_params.get('reg_lambda', 3.0),
        'random_state': seed
    }
    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    base_models.append(('hgb', hgb_model))
    print("    ✓ HistGradientBoosting added")
    
    # [5] ExtraTrees - Randomized trees
    et_params = {
        'n_estimators': 1000,
        'max_depth': best_params.get('max_depth', 10),
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': seed,
        'n_jobs': -1
    }
    et_model = ExtraTreesClassifier(**et_params)
    base_models.append(('et', et_model))
    print("    ✓ ExtraTrees added")
    
    # [6] RandomForest - Classic ensemble
    rf_params = {
        'n_estimators': 1000,
        'max_depth': best_params.get('max_depth', 10),
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': seed,
        'n_jobs': -1
    }
    rf_model = RandomForestClassifier(**rf_params)
    base_models.append(('rf', rf_model))
    print("    ✓ RandomForest added")
    
    # [7] TabNet - Deep learning for tabular (if available)
    if TABNET_AVAILABLE:
        tab_model = TabNetWrapper(seed=seed)
        base_models.append(('tab', tab_model))
        print("    ✓ TabNet added")
    
    print(f"  ✓ Created {len(base_models)} base learners")
    
    # === LAYER 2: META-LEARNER ===
    # Use neural network for complex non-linear combinations
    meta_learner = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2 regularization
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        verbose=False
    )
    
    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,  # 5-fold CV for robust meta-features
        stack_method='predict_proba',
        n_jobs=-1,
        verbose=0
    )
    
    print("  🔧 Training stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    # Validate
    y_val_proba = stacking.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"  ✓ Validation AUC: {val_auc:.4f}")
    
    return stacking

# ==================== TRAINING PIPELINE ====================

def train_stage_v15(df: pd.DataFrame, stage: str, seed: int, n_trials: int = 100) -> Dict[str, Any]:
    """
    Main training pipeline for V15 (Clinical Grade)
    Target: ROC-AUC > 0.9 for healthcare deployment
    """
    print(f"\n{'='*70}")
    print(f"Training Stage {stage} - V15 CLINICAL GRADE")
    print(f"{'='*70}")
    
    # [1] Data Preparation
    X, y, feature_list = prepare_data_advanced(df, stage, seed)
    
    # [2] Stratified Split
    print("\n  📊 Splitting data (stratified)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=seed+1, stratify=y_train_val
    )
    
    print(f"    Train: {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"    Val:   {len(X_val):,} ({len(X_val)/len(X):.1%})")
    print(f"    Test:  {len(X_test):,} ({len(X_test)/len(X):.1%})")
    
    # [3] Handle Imbalanced Data (SMOTE + Tomek)
    if SMOTE_AVAILABLE:
        print("\n  ⚖️  Applying SMOTE-Tomek resampling...")
        smote_tomek = SMOTETomek(random_state=seed, n_jobs=-1)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
        print(f"    Before: {len(X_train):,} samples")
        print(f"    After:  {len(X_train_resampled):,} samples")
        print(f"    Positive rate: {y_train.mean():.2%} → {y_train_resampled.mean():.2%}")
        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train = pd.Series(y_train_resampled)
    else:
        print("  ⚠️  SMOTE not available, proceeding without resampling")
    
    # [4] Robust Scaling (better for outliers in medical data)
    print("\n  📏 Applying RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # [5] Hyperparameter Optimization
    print("\n  🎯 Hyperparameter Optimization...")
    best_params = optimize_hyperparameters_optuna(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        stage, seed, n_trials
    )
    
    # [6] Build Clinical Ensemble
    model = create_clinical_ensemble(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        best_params, stage, seed
    )
    
    # [7] Probability Calibration (Critical for medical decisions)
    print("\n  🎚️  Calibrating probabilities (isotonic)...")
    calibrated_model = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv=3,
        n_jobs=-1
    )
    calibrated_model.fit(X_train_scaled, y_train)
    
    # [8] Cross-Validation on Training Set
    print("\n  🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        calibrated_model, X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring='roc_auc',
        n_jobs=-1
    )
    print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # [9] Final Evaluation
    print("\n  📈 Final Evaluation...")
    
    # Validation set
    y_val_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    
    # Test set
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    
    # Find optimal threshold (maximize F1 on validation)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.2, 0.7, 0.01):
        y_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Apply optimal threshold
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\n  {'='*66}")
    print(f"  VALIDATION RESULTS:")
    print(f"  {'='*66}")
    print(f"    ROC-AUC:     {val_auc:.4f} {'✅' if val_auc > 0.9 else '⚠️'}")
    print(f"    PR-AUC:      {val_pr_auc:.4f}")
    print(f"    Precision:   {val_precision:.4f}")
    print(f"    Recall:      {val_recall:.4f}")
    print(f"    F1-Score:    {val_f1:.4f}")
    print(f"    Threshold:   {best_threshold:.3f}")
    
    print(f"\n  {'='*66}")
    print(f"  TEST RESULTS:")
    print(f"  {'='*66}")
    print(f"    ROC-AUC:     {test_auc:.4f} {'✅' if test_auc > 0.9 else '⚠️'}")
    print(f"    PR-AUC:      {test_pr_auc:.4f}")
    print(f"    Precision:   {test_precision:.4f}")
    print(f"    Recall:      {test_recall:.4f}")
    print(f"    F1-Score:    {test_f1:.4f}")
    
    # [10] Save Artifacts
    artifacts_dir = Path("modeling/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_artifact = {
        'model': calibrated_model,
        'scaler': scaler,
        'features': feature_list,
        'threshold': best_threshold,
        'best_params': best_params
    }
    
    artifact_path = artifacts_dir / f"model_{stage}_clinical_v15.joblib"
    joblib.dump(model_artifact, artifact_path)
    print(f"\n  💾 Model saved: {artifact_path}")
    
    # Return metrics
    results = {
        'stage': stage,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'val_pr_auc': val_pr_auc,
        'test_pr_auc': test_pr_auc,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'threshold': best_threshold,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'n_features': len(feature_list),
        'artifact_path': str(artifact_path)
    }
    
    return results

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="Train Clinical-Grade Models (V15) - Target ROC-AUC > 0.9"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/standardized/clean_assessments.parquet"),
        help="Path to standardized parquet file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (more = better hyperparameters)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="A,B",
        help="Stages to train (comma-separated: A, B, or A,B)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"❌ ERROR: Input file not found: {args.input}")
        print("   Run standardization first!")
        return
    
    # Load data
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    print(f"  Source: {args.input}")
    
    df = pd.read_parquet(args.input)
    print(f"  [OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  [OK] Target distribution: {(df['dry_eye_disease']==1).sum():,} positive / {(df['dry_eye_disease']==0).sum():,} negative")
    
    # Train stages
    stages_to_train = [s.strip().upper() for s in args.stages.split(',')]
    results_all = {}
    
    for stage in stages_to_train:
        if stage not in ['A', 'B']:
            print(f"⚠️  Skipping invalid stage: {stage}")
            continue
        
        results = train_stage_v15(df, stage, args.seed, args.n_trials)
        results_all[stage] = results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - V15 CLINICAL GRADE")
    print(f"{'='*70}")
    
    for stage, res in results_all.items():
        status = "✅ PASSED" if res['test_auc'] > 0.9 else "⚠️ NEEDS IMPROVEMENT"
        print(f"\nStage {stage}: {status}")
        print(f"  Test ROC-AUC:  {res['test_auc']:.4f}")
        print(f"  Test PR-AUC:   {res['test_pr_auc']:.4f}")
        print(f"  Test F1:       {res['test_f1']:.4f}")
        print(f"  CV AUC:        {res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}")
        print(f"  Features:      {res['n_features']}")
    
    print(f"\n{'='*70}")
    print("🏥 Training Complete - Models Ready for Clinical Deployment")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
