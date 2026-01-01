"""
Model Loader Service - Loads trained models on startup with fallback to rule-based
Per requirements: Load latest from registry, fallback to rule-based if missing
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("modeling/registry/registry.json")
ARTIFACTS_DIR = Path("modeling/artifacts")

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages ML models with fallback to rule-based scoring"""
    
    def __init__(self):
        self.model_version: str = "fallback_rule_v1"
        self.model_A: Optional[Any] = None
        self.model_B: Optional[Any] = None
        self.scaler_A: Optional[Any] = None
        self.scaler_B: Optional[Any] = None
        self.feature_selector_A: Optional[Any] = None
        self.feature_selector_B: Optional[Any] = None
        self.features_A: list[str] = []
        self.features_B: list[str] = []
        self.threshold_A: float = 0.5
        self.threshold_B: float = 0.5
        self.metadata: Dict[str, Any] = {}
        self.use_ml_models: bool = False
        
    def load_models(self) -> None:
        """Load latest models from registry, fallback to rule-based if missing"""
        logger.info("Loading models from registry...")
        
        try:
            # Load registry
            if not REGISTRY_PATH.exists():
                logger.warning(f"Registry not found: {REGISTRY_PATH}, using fallback")
                self._setup_fallback()
                return
            
            with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # Get latest entry
            latest_entry = registry.get("latest_improved") or registry.get("latest")
            if not latest_entry:
                logger.warning("No latest entry in registry, using fallback")
                self._setup_fallback()
                return
            
            self.model_version = latest_entry.get("model_version", "fallback_rule_v1")
            logger.info(f"Loading model version: {self.model_version}")
            
            # Try to load Stage A artifacts
            artifact_paths = latest_entry.get("artifact_paths", {})
            # Handle both "stage_A"/"stage_B" and "stage_a"/"stage_b" formats
            stage_a_paths = artifact_paths.get("stage_A") or artifact_paths.get("stage_a") or {}
            stage_b_paths = artifact_paths.get("stage_B") or artifact_paths.get("stage_b") or {}
            
            # Load Stage A
            if stage_a_paths and stage_a_paths.get("model"):
                model_path_str = stage_a_paths.get("model", "")
                # Handle both absolute and relative paths
                if Path(model_path_str).is_absolute():
                    model_path = Path(model_path_str)
                else:
                    model_path = ARTIFACTS_DIR / Path(model_path_str).name
                
                if model_path.exists():
                    try:
                        artifact = joblib.load(model_path)
                        if isinstance(artifact, dict):
                            self.model_A = artifact.get("model")
                            self.scaler_A = artifact.get("scaler")
                            self.feature_selector_A = artifact.get("selector")
                            self.features_A = artifact.get("features", [])
                            self.threshold_A = artifact.get("threshold", 0.5)
                        else:
                            self.model_A = artifact
                        
                        # Try to load separate preprocessing
                        if stage_a_paths.get("preprocessing"):
                            preproc_path_str = stage_a_paths.get("preprocessing", "")
                            if Path(preproc_path_str).is_absolute():
                                preproc_path = Path(preproc_path_str)
                            else:
                                preproc_path = ARTIFACTS_DIR / Path(preproc_path_str).name
                            if preproc_path.exists() and self.scaler_A is None:
                                self.scaler_A = joblib.load(preproc_path)
                        
                        # Try to load features
                        if stage_a_paths.get("features"):
                            features_path_str = stage_a_paths.get("features", "")
                            if Path(features_path_str).is_absolute():
                                features_path = Path(features_path_str)
                            else:
                                features_path = ARTIFACTS_DIR / Path(features_path_str).name
                            if features_path.exists() and not self.features_A:
                                with open(features_path, 'r') as f:
                                    self.features_A = json.load(f)
                        
                        logger.info(f"Loaded Stage A model from {model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load Stage A model: {e}, using fallback")
                        self.model_A = None
            
            # Load Stage B
            if stage_b_paths and stage_b_paths.get("model"):
                model_path_str = stage_b_paths.get("model", "")
                if Path(model_path_str).is_absolute():
                    model_path = Path(model_path_str)
                else:
                    model_path = ARTIFACTS_DIR / Path(model_path_str).name
                
                if model_path.exists():
                    try:
                        artifact = joblib.load(model_path)
                        if isinstance(artifact, dict):
                            self.model_B = artifact.get("model")
                            self.scaler_B = artifact.get("scaler")
                            self.feature_selector_B = artifact.get("selector")
                            self.features_B = artifact.get("features", [])
                            self.threshold_B = artifact.get("threshold", 0.5)
                        else:
                            self.model_B = artifact
                        
                        # Try to load separate preprocessing
                        if stage_b_paths.get("preprocessing"):
                            preproc_path_str = stage_b_paths.get("preprocessing", "")
                            if Path(preproc_path_str).is_absolute():
                                preproc_path = Path(preproc_path_str)
                            else:
                                preproc_path = ARTIFACTS_DIR / Path(preproc_path_str).name
                            if preproc_path.exists() and self.scaler_B is None:
                                self.scaler_B = joblib.load(preproc_path)
                        
                        # Try to load features
                        if stage_b_paths.get("features"):
                            features_path_str = stage_b_paths.get("features", "")
                            if Path(features_path_str).is_absolute():
                                features_path = Path(features_path_str)
                            else:
                                features_path = ARTIFACTS_DIR / Path(features_path_str).name
                            if features_path.exists() and not self.features_B:
                                with open(features_path, 'r') as f:
                                    self.features_B = json.load(f)
                        
                        logger.info(f"Loaded Stage B model from {model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load Stage B model: {e}, using fallback")
                        self.model_B = None
            
            # Load metadata
            if stage_a_paths and stage_a_paths.get("metadata"):
                metadata_path_str = stage_a_paths.get("metadata", "")
                if Path(metadata_path_str).is_absolute():
                    metadata_path = Path(metadata_path_str)
                else:
                    metadata_path = ARTIFACTS_DIR / Path(metadata_path_str).name
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            self.metadata = json.load(f)
                    except Exception:
                        pass
            
            # Check if we have at least one model
            if self.model_A is not None or self.model_B is not None:
                self.use_ml_models = True
                logger.info("ML models loaded successfully")
            else:
                logger.warning("No ML models available, using rule-based fallback")
                self._setup_fallback()
            
            # Store metrics summary
            self.metadata["metrics_summary"] = latest_entry.get("metrics_summary", {})
            
        except Exception as e:
            logger.error(f"Error loading models: {e}, using fallback")
            self._setup_fallback()
    
    def _setup_fallback(self) -> None:
        """Setup rule-based fallback scoring"""
        self.model_version = "fallback_rule_v1"
        self.use_ml_models = False
        self.model_A = None
        self.model_B = None
        logger.info("Using rule-based fallback scoring")
    
    def predict_stage_a(self, features_dict: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict Stage A risk score
        
        Returns:
            (risk_score_0_100, probability_0_1)
        """
        if self.model_A is None or not self.use_ml_models:
            return self._rule_based_stage_a(features_dict)
        
        try:
            # Prepare features
            X = self._prepare_features(features_dict, self.features_A, stage="A")
            
            # Apply feature selector if available
            if self.feature_selector_A is not None:
                X = self.feature_selector_A.transform(X)
            
            # Scale
            if self.scaler_A is not None:
                X = self.scaler_A.transform(X)
            
            # Predict
            if hasattr(self.model_A, 'predict_proba'):
                prob = self.model_A.predict_proba(X)[0, 1]
            else:
                # Fallback if no predict_proba
                prob = 0.5
            
            # Convert to 0-100 risk score
            risk_score = prob * 100
            
            return risk_score, prob
        
        except Exception as e:
            logger.warning(f"ML model prediction failed: {e}, using rule-based")
            return self._rule_based_stage_a(features_dict)
    
    def predict_stage_b(self, features_dict: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict Stage B probability
        
        Returns:
            (risk_score_0_100, probability_0_1)
        """
        if self.model_B is None or not self.use_ml_models:
            return self._rule_based_stage_b(features_dict)
        
        try:
            # Prepare features
            X = self._prepare_features(features_dict, self.features_B, stage="B")
            
            # Apply feature selector if available
            if self.feature_selector_B is not None:
                X = self.feature_selector_B.transform(X)
            
            # Scale
            if self.scaler_B is not None:
                X = self.scaler_B.transform(X)
            
            # Predict
            if hasattr(self.model_B, 'predict_proba'):
                prob = self.model_B.predict_proba(X)[0, 1]
            else:
                prob = 0.5
            
            # Convert to 0-100 risk score
            risk_score = prob * 100
            
            return risk_score, prob
        
        except Exception as e:
            logger.warning(f"ML model prediction failed: {e}, using rule-based")
            return self._rule_based_stage_b(features_dict)
    
    def _prepare_features(
        self, features_dict: Dict[str, Any], expected_features: list[str], stage: str
    ) -> np.ndarray:
        """Prepare feature vector from dict with feature engineering"""
        # Calculate BMI if height/weight available
        if features_dict.get("height") and features_dict.get("weight"):
            height_m = features_dict["height"] / 100.0
            bmi = features_dict["weight"] / (height_m ** 2) if height_m > 0 else 0
            features_dict["bmi"] = bmi
        
        # Create DataFrame for feature engineering
        df = pd.DataFrame([features_dict])
        
        # Apply feature engineering (matching training pipeline)
        df = self._engineer_features(df, stage)
        
        # Extract values in expected order
        # If no expected_features provided, use all columns from engineered df
        if not expected_features:
            expected_features = list(df.columns)
            logger.warning(f"No expected features provided, using all {len(expected_features)} engineered features")
        
        feature_values = []
        for feat in expected_features:
            if feat in df.columns:
                value = df[feat].iloc[0]
                if pd.isna(value) or value is None:
                    value = 0.0
                try:
                    feature_values.append(float(value))
                except (ValueError, TypeError):
                    feature_values.append(0.0)
            else:
                # Feature not in engineered set, use default
                # This can happen if model expects a feature that wasn't engineered
                feature_values.append(0.0)
        
        return np.array([feature_values])
    
    def _engineer_features(self, X: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Apply feature engineering matching training pipeline"""
        X = X.copy()
        
        # BMI (if not already calculated)
        if 'bmi' not in X.columns and 'height' in X.columns and 'weight' in X.columns:
            height_m = X['height'] / 100.0
            X['bmi'] = X['weight'] / (height_m ** 2).replace(0, 1)
        
        # Interaction features
        if 'average_screen_time' in X.columns and 'sleep_duration' in X.columns:
            X['screen_sleep_interaction'] = X['average_screen_time'] * X['sleep_duration']
            X['screen_to_sleep_ratio'] = X['average_screen_time'] / (X['sleep_duration'] + 1)
        
        if 'stress_level' in X.columns and 'sleep_quality' in X.columns:
            X['stress_sleep_quality'] = X['stress_level'] * X['sleep_quality']
        
        if 'bmi' in X.columns and 'age' in X.columns:
            X['bmi_age'] = X['bmi'] * X['age']
        
        if 'daily_steps' in X.columns:
            X['steps_per_hour'] = X['daily_steps'] / 24
        
        # Polynomial features
        if 'average_screen_time' in X.columns:
            X['screen_time_squared'] = X['average_screen_time'] ** 2
        
        if 'sleep_quality' in X.columns:
            X['sleep_quality_squared'] = X['sleep_quality'] ** 2
        
        if 'stress_level' in X.columns:
            X['stress_level_squared'] = X['stress_level'] ** 2
        
        if 'age' in X.columns and 'average_screen_time' in X.columns:
            X['age_screen_interaction'] = X['age'] * X['average_screen_time']
        
        # Fill NaN with 0
        X = X.fillna(0)
        
        return X
    
    def _rule_based_stage_a(self, features_dict: Dict[str, Any]) -> Tuple[float, float]:
        """Rule-based scoring for Stage A (fallback)"""
        score = 0.0
        
        # Screen time
        screen_time = features_dict.get("average_screen_time")
        if screen_time is not None:
            if screen_time >= 8:
                score += 30
            elif screen_time >= 6:
                score += 20
            elif screen_time >= 4:
                score += 10
        
        # Sleep quality
        sleep_quality = features_dict.get("sleep_quality")
        if sleep_quality is not None:
            if sleep_quality <= 2:
                score += 25
            elif sleep_quality <= 3:
                score += 15
        
        # Sleep duration
        sleep_duration = features_dict.get("sleep_duration")
        if sleep_duration is not None:
            if sleep_duration < 6:
                score += 15
            elif sleep_duration < 7:
                score += 10
        
        # Stress
        stress = features_dict.get("stress_level")
        if stress is not None:
            if stress >= 4:
                score += 15
            elif stress >= 3:
                score += 10
        
        # Device before bed
        if features_dict.get("smart_device_before_bed") == 1:
            score += 10
        
        score = min(100, score)
        prob = score / 100.0
        
        return score, prob
    
    def _rule_based_stage_b(self, features_dict: Dict[str, Any]) -> Tuple[float, float]:
        """Rule-based scoring for Stage B (fallback)"""
        # Start with Stage A
        base_score, _ = self._rule_based_stage_a(features_dict)
        
        # Add symptoms
        symptom_count = sum([
            1 if features_dict.get("discomfort_eyestrain") == 1 else 0,
            1 if features_dict.get("redness_in_eye") == 1 else 0,
            1 if features_dict.get("itchiness_irritation_in_eye") == 1 else 0,
        ])
        
        symptom_boost = symptom_count * 15
        score = min(100, base_score + symptom_boost)
        prob = score / 100.0
        
        return score, prob


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create model loader singleton"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_models()
    return _model_loader


def reload_models() -> None:
    """Reload models (for testing/debugging)"""
    global _model_loader
    _model_loader = ModelLoader()
    _model_loader.load_models()
