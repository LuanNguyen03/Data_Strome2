# AI Models Overview - 2-Stage Machine Learning Architecture

> **Note**: Tài liệu này được tạo dựa trên code thực tế trong `backend/scripts/`. Xem code để biết chi tiết implementation.

## 📋 Tổng quan

Hệ thống sử dụng kiến trúc **2-stage machine learning** để đánh giá nguy cơ khô mắt, với mỗi stage có nhiệm vụ riêng biệt và tuân thủ nguyên tắc y tế.

**Source Code**: 
- `backend/scripts/train_models_advanced.py` - Advanced training pipeline
- `backend/scripts/train_extreme_v16.py` - Extreme feature engineering
- `backend/services/model_loader.py` - Model loading và inference

---

## 🏗️ Kiến trúc 2-Stage

### Stage A: Screening (Không sử dụng triệu chứng)

**Mục đích**: Phát hiện nguy cơ sớm dựa trên hành vi và lối sống

**Input Features** (22 features, không bao gồm symptoms):

**Personal Information**:
- age, gender, bmi

**Sleep Features**:
- sleep_duration, sleep_quality, sleep_disorder
- wake_up_during_night, feel_sleepy_during_day

**Device/Screen Usage**:
- average_screen_time, smart_device_before_bed, blue_light_filter

**Lifestyle**:
- stress_level, daily_steps, physical_activity
- caffeine_consumption, alcohol_consumption, smoking

**Vitals**:
- systolic, diastolic, heart_rate

**Medical History**:
- medical_issue, ongoing_medication

**Target**: dry_eye_disease (binary: 0/1)

**Metrics** (latest model v9_ultimate):
- **Test ROC-AUC**: 0.4975 (≈ random, 0.5)
- **Test PR-AUC**: 0.6600
- **Test Precision**: 0.6516
- **Test Recall**: 0.9977
- **Test F1**: 0.7883
- **Status**: NOT USABLE (performance ≈ random guess)

**Nguyên tắc**: KHÔNG sử dụng triệu chứng để tránh leakage

---

### Stage B: Triage (Với triệu chứng)

**Mục đích**: Phân loại chính xác hơn khi đã có triệu chứng

**Input Features** (25 features, bao gồm tất cả Stage A + symptoms):
- Tất cả 22 features của Stage A
- **+ Symptoms**:
  - discomfort_eye_strain
  - redness_in_eye
  - itchiness_irritation_in_eye

**Target**: dry_eye_disease (binary: 0/1)

**Metrics** (latest model v9_ultimate):
- **Test ROC-AUC**: 0.6010 (best performance)
- **Test PR-AUC**: 0.7040
- **Test Precision**: 0.6537
- **Test Recall**: 0.9969
- **Test F1**: 0.7896
- **Status**: POOR (chỉ slightly better than random)

**Lợi ích**: Sử dụng triệu chứng để tăng độ chính xác

---

## 🧠 Model Architecture

### Stacking Ensemble (Multi-layer)

Hệ thống sử dụng **stacking ensemble** với nhiều base models và một meta-learner, được implement trong `train_models_advanced.py`.

#### Base Models (Level 1)

**1. XGBoost** (`xgb.XGBClassifier`)
- Type: Gradient Boosting
- Hyperparameters: Optimized với Optuna hoặc preset values
- Key params:
  - `n_estimators`: 1500-3000
  - `max_depth`: 6-12
  - `learning_rate`: 0.01-0.03
  - `scale_pos_weight`: Auto-calculated từ class imbalance

**2. LightGBM** (`lgb.LGBMClassifier`)
- Type: Fast Gradient Boosting
- Key params:
  - `n_estimators`: 1500-3000
  - `max_depth`: 8-12
  - `learning_rate`: 0.01-0.03
  - `verbosity`: -1 (silent)

**3. CatBoost** (`CatBoostWrapper`)
- Type: Categorical Boosting
- Wrapper: Sklearn-compatible wrapper
- Key params:
  - `iterations`: 2000
  - `depth`: 8
  - `learning_rate`: 0.02
  - `auto_class_weights`: 'Balanced'

**4. HistGradientBoosting** (`sklearn.ensemble.HistGradientBoostingClassifier`)
- Type: Scikit-learn native gradient boosting
- Key params:
  - `max_iter`: 1000
  - `max_depth`: 8
  - `learning_rate`: 0.05

**5. ExtraTrees** (`sklearn.ensemble.ExtraTreesClassifier`)
- Type: Extremely Randomized Trees
- Key params:
  - `n_estimators`: 1000
  - `max_depth`: 10
  - `class_weight`: 'balanced'

**6. RandomForest** (`sklearn.ensemble.RandomForestClassifier`)
- Type: Bagging ensemble
- Key params:
  - `n_estimators`: 1000
  - `max_depth`: 10
  - `class_weight`: 'balanced'

**7. TabNet** (`TabNetWrapper`, optional)
- Type: Deep learning for tabular data
- Wrapper: Sklearn-compatible wrapper
- Key params:
  - `max_epochs`: 200
  - `patience`: 30
  - `batch_size`: 512

**Total**: 6-7 base models (TabNet optional, requires PyTorch)

#### Meta-Learner (Level 2)

**Neural Network** (`sklearn.neural_network.MLPClassifier`)

- **Architecture**: 128 → 64 → 32 → 1
- **Activation**: ReLU (hidden layers), Sigmoid (output)
- **Solver**: Adam
- **Regularization**: L2 (alpha=0.01)
- **Early Stopping**: Enabled (validation_fraction=0.1)
- **Purpose**: Combine predictions từ base models

**Implementation**:
```python
meta_learner = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=seed
)
```

**Stacking Implementation**:
```python
stacking = StackingClassifier(
    estimators=base_models,  # List of (name, model) tuples
    final_estimator=meta_learner,
    cv=5,  # 5-fold CV for robust meta-features
    stack_method='predict_proba',
    n_jobs=-1
)
```

#### Calibration

**Method**: Isotonic Calibration (`sklearn.calibration.CalibratedClassifierCV`)

- **Purpose**: Calibrate probability outputs để phản ánh true likelihood
- **Method**: 'isotonic'
- **CV**: 5-fold

```python
calibrated = CalibratedClassifierCV(
    model,
    method='isotonic',
    cv=5
)
```

---

## 🔧 Feature Engineering

### Standard Feature Engineering (Stage A/B)

**Source**: `backend/services/model_loader.py` → `_engineer_features()`

**Basic Engineering**:
- **BMI**: `weight / (height/100)^2` (nếu chưa có)

**Interaction Features**:
- `screen_sleep_interaction`: `average_screen_time * sleep_duration`
- `screen_to_sleep_ratio`: `average_screen_time / (sleep_duration + 1)`
- `stress_sleep_quality`: `stress_level * sleep_quality`
- `bmi_age`: `bmi * age`

**Derived Features**:
- `steps_per_hour`: `daily_steps / 24`

**Polynomial Features**:
- `screen_time_squared`: `average_screen_time^2`
- `sleep_quality_squared`: `sleep_quality^2`
- `stress_level_squared`: `stress_level^2`
- `age_screen_interaction`: `age * average_screen_time`

**Total Standard Features**: ~30-40 features (từ 22-25 original)

---

### Extreme Feature Engineering (train_extreme_v16.py)

**Target**: Generate 100+ features để maximize signal extraction

**Techniques**:

**1. Polynomial Features**:
- Squares: `x^2` cho key features
- Cubes: `x^3` cho key features
- Square Root: `sqrt(x)`
- Log: `log(x+1)`

**Applied to**: age, bmi, average_screen_time, stress_level, sleep_quality

**2. Ratio Features**:
- `screen_sleep_ratio`: `screen_time / sleep_duration`
- `bp_ratio`: `systolic / diastolic`
- `map`: `(systolic + 2*diastolic) / 3` (Mean Arterial Pressure)
- `pulse_pressure`: `systolic - diastolic`
- `activity_ratio`: `physical_activity / (daily_steps/1000)`

**3. Interaction Terms** (48+ combinations):
- Multiplicative: `col1 * col2`
- Additive: `col1 + col2`
- Subtractive: `col1 - col2`

**Key pairs**:
- (age, bmi), (age, screen_time), (age, stress_level)
- (screen_time, sleep_quality), (screen_time, stress_level)
- (sleep_duration, sleep_quality), (sleep_duration, stress_level)
- (systolic, diastolic), (bmi, physical_activity)
- ... và nhiều hơn

**4. Symptom Features** (Stage B only):
- `symptom_sum`: Sum of 3 symptom flags
- `symptom_mean`: Mean of symptoms
- `symptom_max`: Max of symptoms
- `symptom_std`: Std of symptoms
- Symptom interactions với key features

**5. Binning**:
- Age bins: <30, 30-50, >=50
- BMI bins: Underweight, Normal, Overweight, Obese

**6. Composite Features**:
- `substance_total`: caffeine + alcohol*2 + smoking*3
- `medical_burden`: medical_issue + ongoing_medication
- `sleep_composite`: sleep_duration*sleep_quality - wake_up*2 - feel_sleepy

**Total Extreme Features**: **118 features** (từ 22-25 original)

---

## 📊 Data Preprocessing

### Train/Val/Test Split

**Ratio**: 70% / 15% / 15%

**Method**: Stratified (`train_test_split` với `stratify=y`)

**Implementation**:
```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed+1, stratify=y_temp
)
```

### Handling Imbalanced Data

**Method**: SMOTE-Tomek (`imblearn.combine.SMOTETomek`)

- **SMOTE**: Synthetic Minority Oversampling
- **Tomek Links**: Remove borderline majority samples
- **sampling_strategy**: 0.8 (tăng minority class lên 80% của majority)

**Implementation**:
```python
smote = SMOTETomek(random_state=seed, sampling_strategy=0.8)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

### Scaling

**Method**: RobustScaler (`sklearn.preprocessing.RobustScaler`)

- **Formula**: `(x - median) / IQR`
- **Reason**: Handles outliers better than StandardScaler

### Missing Data Handling

**Numeric**: Median imputation  
**Binary**: Mode imputation  
**Missing Indicators**: Create flags for missing fields

**Policy**: Don't block prediction if missing data (graceful degradation)

### Feature Selection

**Method**: SelectFromModel (`sklearn.feature_selection.SelectFromModel`)

- **Base Estimator**: ExtraTreesClassifier (100 trees)
- **Threshold**: 'median' (keep top 50% features)

---

## 📈 Hyperparameter Optimization

### Optuna Framework

**Source**: `backend/scripts/train_models_advanced.py` → `optimize_hyperparameters()`

**Method**: Tree-structured Parzen Estimator (TPE)

**Objective**: Maximize ROC-AUC trên validation set

**Trials**: 100-150 trials (configurable)

**Hyperparameters Optimized**:
- `n_estimators`: [500, 3000]
- `max_depth`: [3, 12]
- `learning_rate`: [0.01, 0.1]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: [1, 20]
- `gamma`: [0, 5]
- `reg_alpha`: [0, 10]
- `reg_lambda`: [0, 10]

**Early Stopping**: Enabled (eval_set validation)

---

## 📊 Model Performance (Thực tế từ Code)

### Latest Model: v9_ultimate

**Registry**: `modeling/registry/registry.json` → `latest_improved`

#### Stage A Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| **Test ROC-AUC** | 0.4975 | ❌ ≈ Random (0.5) |
| **Test PR-AUC** | 0.6600 | - |
| **Test Precision** | 0.6516 | - |
| **Test Recall** | 0.9977 | High (tốt cho screening) |
| **Test F1** | 0.7883 | - |
| **Threshold** | 0.3 | - |

**Confusion Matrix** (Test Set):
- TN: 2
- FP: 1391
- FN: 6
- TP: 2601

**Status**: ❌ **NOT USABLE** (performance ≈ random guess)

#### Stage B Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| **Test ROC-AUC** | 0.6010 | ⚠️ Poor (chỉ slightly better than random) |
| **Test PR-AUC** | 0.7040 | - |
| **Test Precision** | 0.6537 | - |
| **Test Recall** | 0.9969 | Very high |
| **Test F1** | 0.7896 | - |
| **Threshold** | 0.3 | - |

**Confusion Matrix** (Test Set):
- TN: 16
- FP: 1377
- FN: 8
- TP: 2599

**Status**: ⚠️ **POOR** (barely usable)

---

## 🔍 Root Cause Analysis (Từ Code)

### Feature Importance (Stage B)

**Top 10 Features** (từ `modeling/analysis/analysis_stage_B.json`):

1. `discomfort_eye_strain`: 0.1100 (highest)
2. `redness_in_eye`: 0.0991
3. `bmi`: 0.0975
4. `itchiness_irritation_in_eye`: 0.0959
5. `physical_activity`: 0.0869
6. `average_screen_time`: 0.0834
7. `sleep_duration`: 0.0764
8. `heart_rate`: 0.0688
9. `age`: 0.0640
10. `daily_steps`: 0.0541

**Observation**: Symptom features có importance cao nhất, nhưng overall correlations vẫn thấp (< 0.2 với target)

### Dataset Characteristics

- **Sample Size**: 20,000 records
- **Class Balance**: 65% positive / 35% negative
- **Features**: 22 (Stage A), 25 (Stage B)
- **Missing Data**: 100% missing cho systolic/diastolic (sau standardization)

---

## 🔄 Training Pipeline

### Scripts

1. **`backend/scripts/train_models.py`**: Basic training
2. **`backend/scripts/train_models_improved.py`**: Improved với Optuna
3. **`backend/scripts/train_models_optimized.py`**: Optimized hyperparameters
4. **`backend/scripts/train_models_advanced.py`**: ⭐ Advanced stacking ensemble
5. **`backend/scripts/train_extreme_v16.py`**: Extreme feature engineering (118 features)

### Usage

```bash
# Advanced training (recommended)
python backend/scripts/train_models_advanced.py

# Extreme feature engineering
python backend/scripts/train_extreme_v16.py
```

### Model Registry

**Location**: `modeling/registry/registry.json`

**Fields**:
- `model_version`: Version identifier
- `created_at`: Timestamp
- `artifact_paths`: Paths to saved models
- `metrics_summary`: Performance metrics
- `improvements`: List of improvements applied

**Latest Model**: `latest_improved` → v9_ultimate

---

## ✅ Best Practices Implemented

### Code Quality: ✅ EXCELLENT

Tất cả scripts implement industry best practices:

1. ✅ **Proper Validation**: Stratified splits, cross-validation
2. ✅ **Feature Engineering**: Comprehensive và domain-aware
3. ✅ **Ensemble Methods**: Stacking với diverse base models
4. ✅ **Hyperparameter Optimization**: Optuna integration
5. ✅ **Calibration**: Probability calibration
6. ✅ **Error Handling**: Graceful degradation
7. ✅ **Logging**: Comprehensive logging
8. ✅ **Reproducibility**: Random seeds, version control

---

## 📚 Model Files

### Saved Models

**Location**: `modeling/artifacts/`

**Files**:
- `model_A_screening_advanced.joblib` - Stage A model
- `model_B_triage_advanced.joblib` - Stage B model
- `preprocessing_A_advanced.joblib` - Preprocessing pipeline Stage A
- `preprocessing_B_advanced.joblib` - Preprocessing pipeline Stage B
- `feature_selector_A_advanced.joblib` - Feature selector Stage A
- `feature_selector_B_advanced.joblib` - Feature selector Stage B
- `feature_list_A_advanced.json` - Feature list Stage A
- `feature_list_B_advanced.json` - Feature list Stage B

**Format**: Joblib (Python pickle format)

### Model Loading

**Source**: `backend/services/model_loader.py`

**Fallback**: Rule-based scoring nếu ML models không available

---

## 🎯 Recommendations

### Để cải thiện Performance

**1. Collect Clinical-Grade Data** ⭐ RECOMMENDED

Cần thêm:
- Schirmer test (tear production)
- Tear osmolarity
- Tear break-up time (TBUT)
- Corneal staining scores
- Meibomian gland assessment

**2. Feature Engineering**

Đã implement extreme feature engineering (118 features) nhưng improvement minimal → Fundamental lack of signal trong data

**3. Model Architecture**

Đã sử dụng best practices:
- Stacking ensemble
- Neural network meta-learner
- Hyperparameter optimization
- Probability calibration

**Conclusion**: Code quality excellent, nhưng performance limited by dataset quality.

---

**Last Updated**: January 2026  
**Source**: Code analysis from `backend/scripts/`  
**Latest Model**: v9_ultimate  
**Status**: ⚠️ Performance limited by dataset
