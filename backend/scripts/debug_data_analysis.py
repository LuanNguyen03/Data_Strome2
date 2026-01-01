"""
Debug script to analyze why ROC_AUC is low
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
print("="*70)
print("DATA QUALITY ANALYSIS")
print("="*70)

df = pd.read_parquet("data/standardized/clean_assessments.parquet")
print(f"\nDataset: {df.shape}")

# Check target
print("\nTarget Distribution:")
print(df['dry_eye_disease'].value_counts())
print(f"Positive rate: {df['dry_eye_disease'].mean():.2%}")

# Check for data leakage - symptom_score should NOT be in Stage A
print("\nChecking for data leakage...")
if 'symptom_score' in df.columns:
    correlation = df['symptom_score'].corr(df['dry_eye_disease'])
    print(f"  symptom_score correlation with target: {correlation:.4f}")
    if abs(correlation) > 0.8:
        print("  [WARNING] STRONG correlation - possible data leakage!")

# Check key features correlation
print("\nTop Feature Correlations with Target:")
numeric_cols = df.select_dtypes(include=['number']).columns
correlations = df[numeric_cols].corrwith(df['dry_eye_disease']).abs().sort_values(ascending=False)
print(correlations.head(15))

# Try simple baseline model
print("\n" + "="*70)
print("BASELINE MODEL TEST (Logistic Regression)")
print("="*70)

# Use only strong predictive features
strong_features = [
    'age', 'bmi', 'systolic', 'diastolic', 'heart_rate',
    'sleep_duration', 'sleep_quality', 'stress_level',
    'average_screen_time', 'daily_steps', 'physical_activity',
    'medical_issue', 'ongoing_medication'
]

available_features = [f for f in strong_features if f in df.columns]
print(f"\nUsing features: {available_features}")

X = df[available_features].copy()
y = df['dry_eye_disease'].astype(int)

# Handle categorical
for col in ['medical_issue', 'ongoing_medication']:
    if col in X.columns and X[col].dtype not in ['int64', 'int32', 'float64']:
        X[col] = pd.factorize(X[col])[0]

# Fill missing
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression (baseline)...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
y_pred = lr.predict(X_test_scaled)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nLogistic Regression AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Try RandomForest
print("\n" + "="*70)
print("RANDOM FOREST TEST")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
print("Training RandomForest...")
rf.fit(X_train, y_train)

y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\nRandomForest AUC: {auc_rf:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature importance
print("\nTop 10 Feature Importances (RandomForest):")
feat_imp = pd.DataFrame({
    'feature': available_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp.head(10))

# Check if target is random or has signal
print("\n" + "="*70)
print("SIGNAL ANALYSIS")
print("="*70)

# Compute mutual information
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': available_features,
    'mutual_info': mi
}).sort_values('mutual_info', ascending=False)
print("\nMutual Information with Target:")
print(mi_df)

# Summary
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if auc < 0.55 and auc_rf < 0.55:
    print("\n[CRITICAL] Both simple models have AUC < 0.55")
    print("Possible causes:")
    print("  1. Target variable is random/noisy")
    print("  2. Features have no predictive power")
    print("  3. Data preprocessing destroyed signal")
    print("  4. Wrong target variable (check data_quality_report.json)")
    print("\nRecommendations:")
    print("  - Verify raw data quality")
    print("  - Check if 'dry_eye_disease' is correct target")
    print("  - Look for data collection issues")
elif auc < 0.7:
    print("\n[WARNING] AUC < 0.7 with simple models")
    print("  - Feature engineering may help")
    print("  - Try more complex models (XGBoost, LightGBM)")
    print("  - Consider adding more predictive features")
else:
    print("\n[OK] Baseline models show promise (AUC >= 0.7)")
    print("  - Complex ensemble should improve further")
    print("  - Feature engineering will boost performance")

print("\nNext steps:")
print("  1. Review data/raw/Dry_Eye_Dataset.csv")
print("  2. Check data/standardized/data_quality_report.json")
print("  3. Verify target variable definition")
