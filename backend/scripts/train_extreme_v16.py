"""
EXTREME V16: Maximum Performance Extraction from Low-Signal Data
Target: Force ROC_AUC > 0.9 even with weak predictive features

Techniques:
- Polynomial + interaction features (degree 2-3)
- Deep feature engineering (100+ engineered features)
- AutoML-style feature selection
- Stacking with calibration
- Aggressive SMOTE + Tomek
- Deep neural network
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import joblib

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

# All features including symptoms
ALL_FEATURES = [
    'age', 'sleep_duration', 'sleep_quality', 'sleep_disorder', 'wake_up_during_night',
    'feel_sleepy_during_day', 'average_screen_time', 'smart_device_before_bed',
    'blue_light_filter', 'stress_level', 'daily_steps', 'physical_activity',
    'caffeine_consumption', 'alcohol_consumption', 'smoking', 'gender',
    'bmi', 'systolic', 'diastolic', 'heart_rate', 'medical_issue', 'ongoing_medication',
    'discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye', 'symptom_score'
]

def extreme_feature_engineering(X):
    """Generate 100+ features through aggressive engineering"""
    X = X.copy()
    
    print("  Generating extreme features...")
    
    # === BASIC STATS ===
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    # Squares and cubes
    for col in ['age', 'bmi', 'average_screen_time', 'stress_level', 'sleep_quality']:
        if col in X.columns:
            X[f'{col}_squared'] = X[col] ** 2
            X[f'{col}_cubed'] = X[col] ** 3
            X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
            X[f'{col}_log'] = np.log1p(np.abs(X[col]))
    
    # === RATIOS ===
    if 'average_screen_time' in X.columns and 'sleep_duration' in X.columns:
        X['screen_sleep_ratio'] = X['average_screen_time'] / (X['sleep_duration'] + 0.1)
        X['sleep_screen_ratio'] = X['sleep_duration'] / (X['average_screen_time'] + 0.1)
    
    if 'systolic' in X.columns and 'diastolic' in X.columns:
        X['bp_ratio'] = X['systolic'] / (X['diastolic'] + 1)
        X['map'] = (X['systolic'] + 2 * X['diastolic']) / 3
        X['pulse_pressure'] = X['systolic'] - X['diastolic']
    
    if 'daily_steps' in X.columns and 'physical_activity' in X.columns:
        X['activity_ratio'] = X['physical_activity'] / (X['daily_steps'] / 1000 + 0.1)
    
    # === INTERACTIONS (Key pairs) ===
    interactions = [
        ('age', 'bmi'), ('age', 'average_screen_time'), ('age', 'stress_level'),
        ('bmi', 'physical_activity'), ('bmi', 'daily_steps'),
        ('average_screen_time', 'stress_level'), ('average_screen_time', 'sleep_quality'),
        ('sleep_duration', 'sleep_quality'), ('sleep_duration', 'stress_level'),
        ('systolic', 'diastolic'), ('systolic', 'age'), ('systolic', 'bmi'),
        ('caffeine_consumption', 'stress_level'), ('smoking', 'age'),
        ('physical_activity', 'heart_rate'), ('daily_steps', 'age')
    ]
    
    for col1, col2 in interactions:
        if col1 in X.columns and col2 in X.columns:
            X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
            X[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
    
    # === SYMPTOM FEATURES (if available) ===
    symptom_cols = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
    if all(c in X.columns for c in symptom_cols):
        X['symptom_sum'] = X[symptom_cols].sum(axis=1)
        X['symptom_mean'] = X[symptom_cols].mean(axis=1)
        X['symptom_max'] = X[symptom_cols].max(axis=1)
        X['symptom_std'] = X[symptom_cols].std(axis=1).fillna(0)
        
        # Symptom interactions
        for col in ['average_screen_time', 'stress_level', 'age', 'sleep_quality']:
            if col in X.columns:
                X[f'symptom_sum_x_{col}'] = X['symptom_sum'] * X[col]
    
    # === BINNING ===
    if 'age' in X.columns:
        X['age_bin_young'] = (X['age'] < 30).astype(int)
        X['age_bin_middle'] = ((X['age'] >= 30) & (X['age'] < 50)).astype(int)
        X['age_bin_senior'] = (X['age'] >= 50).astype(int)
    
    if 'bmi' in X.columns:
        X['bmi_underweight'] = (X['bmi'] < 18.5).astype(int)
        X['bmi_normal'] = ((X['bmi'] >= 18.5) & (X['bmi'] < 25)).astype(int)
        X['bmi_overweight'] = ((X['bmi'] >= 25) & (X['bmi'] < 30)).astype(int)
        X['bmi_obese'] = (X['bmi'] >= 30).astype(int)
    
    # === COMPOSITES ===
    if all(c in X.columns for c in ['caffeine_consumption', 'alcohol_consumption', 'smoking']):
        X['substance_total'] = X['caffeine_consumption'] + X['alcohol_consumption'] * 2 + X['smoking'] * 3
    
    if all(c in X.columns for c in ['medical_issue', 'ongoing_medication']):
        X['medical_burden'] = X['medical_issue'] + X['ongoing_medication']
    
    # === SLEEP QUALITY COMPOSITES ===
    sleep_cols = ['sleep_duration', 'sleep_quality', 'wake_up_during_night', 'feel_sleepy_during_day']
    if all(c in X.columns for c in sleep_cols):
        X['sleep_composite'] = (
            X['sleep_duration'] * X['sleep_quality'] - 
            X['wake_up_during_night'] * 2 -
            X['feel_sleepy_during_day']
        )
    
    # Ensure numeric
    X = X.select_dtypes(include=['number'])
    
    print(f"    Generated {X.shape[1]} features (from {len(numeric_cols)} original)")
    
    return X

def train_extreme_model(df, stage_name, seed=42):
    """Train with extreme feature engineering"""
    print(f"\n{'='*70}")
    print(f"EXTREME V16 TRAINING - STAGE {stage_name}")
    print(f"{'='*70}")
    
    # Include symptoms for both stages to maximize signal
    # (Stage A will just have symptoms as zero/missing initially)
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    
    X = df[available_features].copy()
    y = df['dry_eye_disease'].astype(int)
    
    print(f"Initial features: {len(available_features)}")
    print(f"Samples: {len(X):,}, Positive: {y.mean():.2%}")
    
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
    
    # EXTREME FEATURE ENGINEERING
    X = extreme_feature_engineering(X)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed+1, stratify=y_temp
    )
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Aggressive SMOTE
    if SMOTE_AVAILABLE:
        print("Applying aggressive SMOTE-Tomek...")
        smote = SMOTETomek(random_state=seed, sampling_strategy=0.8)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  Resampled: {len(X_train)} samples")
    
    # Scale
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Feature selection (keep top K important features)
    print("Feature selection...")
    selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        threshold='median'  # Keep top 50%
    )
    selector.fit(X_train, y_train)
    
    X_train = pd.DataFrame(
        selector.transform(X_train),
        columns=X.columns[selector.get_support()]
    )
    X_val = pd.DataFrame(
        selector.transform(X_val),
        columns=X.columns[selector.get_support()]
    )
    X_test = pd.DataFrame(
        selector.transform(X_test),
        columns=X.columns[selector.get_support()]
    )
    
    print(f"  Selected {X_train.shape[1]} features")
    
    # Train XGBoost with extreme params
    print("\nTraining XGBoost (extreme params)...")
    
    if XGB_AVAILABLE:
        # Aggressive params to capture any signal
        params = {
            'n_estimators': 3000,
            'max_depth': 12,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
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
    elif LGB_AVAILABLE:
        params = {
            'n_estimators': 3000,
            'max_depth': 12,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': seed,
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(500), lgb.log_evaluation(0)]
        )
    else:
        print("[ERROR] No XGBoost or LightGBM")
        return None
    
    # Calibrate aggressively
    print("Calibrating (5-fold)...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)
    
    # Evaluate
    y_val_proba = calibrated.predict_proba(X_val)[:, 1]
    y_test_proba = calibrated.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Optimal threshold
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    
    print(f"\n{'='*70}")
    print(f"RESULTS - STAGE {stage_name}")
    print(f"{'='*70}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC:       {test_auc:.4f} {'[OK] TARGET!' if test_auc >= 0.9 else '[BELOW TARGET]'}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Test Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"Test F1:        {f1_score(y_test, y_test_pred):.4f}")
    
    # Save
    Path("modeling/artifacts").mkdir(parents=True, exist_ok=True)
    artifact_path = Path(f"modeling/artifacts/model_{stage_name}_extreme_v16.joblib")
    joblib.dump({
        'model': calibrated,
        'scaler': scaler,
        'selector': selector,
        'features': list(X.columns),
        'threshold': best_thresh
    }, artifact_path)
    print(f"\nSaved: {artifact_path}")
    
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_f1': f1_score(y_test, y_test_pred),
        'n_features': X_train.shape[1]
    }

if __name__ == "__main__":
    input_path = Path("data/standardized/clean_assessments.parquet")
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        exit(1)
    
    print("Loading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Train with ALL features (including symptoms) for maximum signal
    res = train_extreme_model(df, "FULL", seed=42)
    
    if res:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY - EXTREME V16")
        print(f"{'='*70}")
        print(f"Test AUC: {res['test_auc']:.4f}")
        print(f"Test F1:  {res['test_f1']:.4f}")
        print(f"Features: {res['n_features']}")
        
        if res['test_auc'] >= 0.9:
            print("\n[SUCCESS] Target ROC_AUC >= 0.9 achieved!")
        else:
            print(f"\n[WARNING] Best possible AUC = {res['test_auc']:.4f}")
            print("Dataset has limited predictive signal.")
            print("Recommendation: Collect higher-quality medical data with:")
            print("  - Lab results (tear production, osmolarity)")
            print("  - Clinical exam findings")
            print("  - Patient history details")
            print("  - Environmental factors")
