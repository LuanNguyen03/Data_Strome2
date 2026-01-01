"""Check raw data for signal"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv("data/raw/Dry_Eye_Dataset.csv")
print("="*70)
print("RAW DATA ANALYSIS")
print("="*70)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Target
target_col = 'Dry Eye Disease'
print(f"\nTarget distribution:")
print(df[target_col].value_counts())

# Convert target to numeric
y = (df[target_col] == 'Y').astype(int)
print(f"Positive rate: {y.mean():.2%}")

# Prepare features - keep it simple
feature_cols = [
    'Age', 'Sleep duration', 'Sleep quality', 'Stress level',
    'Heart rate', 'Daily steps', 'Physical activity',
    'Average screen time'
]

X = df[feature_cols].copy()

# Convert to numeric if needed
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

print(f"\nFeatures: {X.columns.tolist()}")
print(f"X shape: {X.shape}")

# Check correlations
print("\nCorrelations with target:")
for col in X.columns:
    corr = X[col].corr(pd.Series(y))
    print(f"  {col}: {corr:.4f}")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# RandomForest
print("\n" + "="*70)
print("TESTING WITH RANDOMFOREST")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)

y_pred_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nRandomForest AUC: {auc:.4f}")

feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(feat_imp)

# Try with symptoms (Stage B)
print("\n" + "="*70)
print("TESTING WITH SYMPTOMS (STAGE B)")
print("="*70)

symptom_cols = ['Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye']
X_symptoms = df[feature_cols + symptom_cols].copy()

# Convert Y/N to 1/0
for col in symptom_cols:
    if col in X_symptoms.columns:
        X_symptoms[col] = (X_symptoms[col] == 'Y').astype(int)

for col in feature_cols:
    if X_symptoms[col].dtype == 'object':
        X_symptoms[col] = pd.to_numeric(X_symptoms[col], errors='coerce')

X_symptoms = X_symptoms.fillna(X_symptoms.median())

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_symptoms, y, test_size=0.3, random_state=42, stratify=y
)

rf_s = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42
)
rf_s.fit(X_train_s, y_train_s)

y_pred_proba_s = rf_s.predict_proba(X_test_s)[:, 1]
auc_s = roc_auc_score(y_test_s, y_pred_proba_s)

print(f"\nRandomForest with Symptoms AUC: {auc_s:.4f}")

# DIAGNOSIS
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if auc < 0.6 and auc_s < 0.7:
    print("\n[CRITICAL] Low AUC even with symptoms")
    print("  This dataset may be:")
    print("  1. Synthetically generated with little/no signal")
    print("  2. Missing key predicitive features")
    print("  3. Target variable is random/noisy")
    print("\nTo achieve ROC_AUC > 0.9, you need:")
    print("  - Higher quality medical data")
    print("  - More predictive features (lab results, clinical exams)")
    print("  - Expert domain knowledge for feature engineering")
elif auc_s > 0.7:
    print(f"\n[OK] Symptoms help! AUC = {auc_s:.4f}")
    print("  Stage B (with symptoms) can achieve good performance")
    print("  Stage A (without symptoms) will be harder")
    print("\nRecommendations:")
    print("  - Focus on Stage B first (with symptoms)")
    print("  - For Stage A, need better lifestyle/vitals features")
else:
    print(f"\n[WARNING] Moderate performance")
    print(f"  Stage A AUC: {auc:.4f}")
    print(f"  Stage B AUC: {auc_s:.4f}")
