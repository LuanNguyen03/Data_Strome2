"""
Quick Test: Simple but effective medical model
Target: ROC-AUC > 0.9 with minimal complexity
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Import available ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

try:
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

# Features
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

def engineer_features(X, stage):
    """Medical feature engineering"""
    X = X.copy()
    
    # Digital eye strain
    if 'average_screen_time' in X.columns and 'stress_level' in X.columns:
        X['digital_strain'] = X['average_screen_time'] * X['stress_level']
    
    # Cardiovascular
    if 'systolic' in X.columns and 'diastolic' in X.columns:
        X['map'] = (X['systolic'] + 2 * X['diastolic']) / 3
        X['pulse_pressure'] = X['systolic'] - X['diastolic']
    
    # Sleep quality
    if 'sleep_duration' in X.columns and 'sleep_quality' in X.columns:
        X['recovery_score'] = X['sleep_duration'] * X['sleep_quality']
    
    # BMI
    if 'bmi' in X.columns:
        X['bmi_squared'] = X['bmi'] ** 2
        X['obesity'] = (X['bmi'] >= 30).astype(int)
    
    # Age
    if 'age' in X.columns:
        X['age_squared'] = X['age'] ** 2
        if 'average_screen_time' in X.columns:
            X['age_screen'] = X['age'] * X['average_screen_time']
    
    # Stage B symptoms
    if stage == "B":
        if all(c in X.columns for c in ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']):
            X['symptom_total'] = (
                X['discomfort_eye_strain'] * 2 + 
                X['redness_in_eye'] + 
                X['itchiness_irritation_in_eye']
            )
    
    return X

def prepare_data(df, stage):
    """Prepare data"""
    features = STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES
    available = [f for f in features if f in df.columns]
    
    X = df[available].copy()
    y = df["dry_eye_disease"].astype(int)
    
    # Handle categorical
    global_mean = y.mean()
    for col in ['medical_issue', 'ongoing_medication']:
        if col in X.columns and X[col].dtype not in ['int64', 'int32', 'float64']:
            agg = pd.DataFrame({'val': X[col], 'target': y}).groupby('val')['target'].agg(['count', 'mean'])
            smoothed = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
            X[col] = X[col].map(smoothed.to_dict()).fillna(global_mean)
    
    # Convert to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())
    
    # Engineer
    X = engineer_features(X, stage)
    X = X.select_dtypes(include=['number'])
    
    return X, y

def train_simple_model(df, stage, seed=42):
    """Train simple but effective model"""
    print(f"\n{'='*70}")
    print(f"TRAINING STAGE {stage} - QUICK TEST")
    print(f"{'='*70}")
    
    # Prepare
    X, y = prepare_data(df, stage)
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]:,}, Positive: {y.mean():.2%}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed+1, stratify=y_temp)
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # SMOTE
    if SMOTE_AVAILABLE:
        print("Applying SMOTE...")
        smote = SMOTETomek(random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  Resampled to {len(X_train)} samples")
    
    # Scale
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Train XGBoost (single model for speed)
    print("\nTraining XGBoost...")
    
    if stage == "A":
        params = {
            'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.02,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
            'gamma': 2.0, 'reg_alpha': 3.0, 'reg_lambda': 5.0
        }
    else:
        params = {
            'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.03,
            'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 5,
            'gamma': 1.0, 'reg_alpha': 2.0, 'reg_lambda': 3.0
        }
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params.update({
        'scale_pos_weight': scale_pos_weight,
        'random_state': seed,
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'verbosity': 0
    })
    
    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 verbose=False)
    elif LGB_AVAILABLE:
        lgb_params = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': seed,
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    else:
        print("[ERROR] No XGBoost or LightGBM available")
        return None
    
    # Calibrate
    print("Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Evaluate
    y_val_proba = calibrated.predict_proba(X_val)[:, 1]
    y_test_proba = calibrated.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Find threshold
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.2, 0.7, 0.01):
        y_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC:       {test_auc:.4f} {'[OK]' if test_auc > 0.9 else '[WARNING]'}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Test Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"Test F1:        {f1_score(y_test, y_test_pred):.4f}")
    print(f"Threshold:      {best_thresh:.3f}")
    
    # Save
    Path("modeling/artifacts").mkdir(parents=True, exist_ok=True)
    artifact_path = Path(f"modeling/artifacts/model_{stage}_quick.joblib")
    joblib.dump({
        'model': calibrated,
        'scaler': scaler,
        'features': list(X.columns),
        'threshold': best_thresh
    }, artifact_path)
    print(f"\nSaved: {artifact_path}")
    
    return {
        'test_auc': test_auc,
        'val_auc': val_auc,
        'test_f1': f1_score(y_test, y_test_pred)
    }

if __name__ == "__main__":
    import sys
    
    input_path = Path("data/standardized/clean_assessments.parquet")
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        sys.exit(1)
    
    print("Loading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")
    
    # Train both stages
    results = {}
    for stage in ['A', 'B']:
        res = train_simple_model(df, stage)
        if res:
            results[stage] = res
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for stage, res in results.items():
        status = "TARGET ACHIEVED!" if res['test_auc'] > 0.9 else "Needs improvement"
        print(f"Stage {stage}: Test AUC = {res['test_auc']:.4f} - {status}")
