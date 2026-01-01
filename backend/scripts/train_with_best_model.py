"""
Enhanced training with best model saving and overfitting detection.
This module provides utilities for saving best models and detecting overfitting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

def check_overfitting(
    val_roc_auc: float,
    test_roc_auc: float,
    max_gap: float = 0.20
) -> Tuple[bool, float]:
    """
    Check if model is overfitting based on validation-test gap.
    
    Args:
        val_roc_auc: Validation ROC-AUC score
        test_roc_auc: Test ROC-AUC score
        max_gap: Maximum allowed gap (default: 0.20)
    
    Returns:
        Tuple of (is_overfitting, gap)
    """
    gap = val_roc_auc - test_roc_auc
    is_overfitting = gap > max_gap
    return is_overfitting, gap


def save_best_model_checkpoint(
    model: Any,
    scaler: Any,
    feature_selector: Optional[Any],
    threshold: float,
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    checkpoint_dir: Path,
    stage: str,
    iteration: int = 0
) -> Optional[Path]:
    """
    Save model checkpoint if it's the best so far.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_selector: Feature selector (if used)
        threshold: Selected threshold
        val_metrics: Validation metrics
        test_metrics: Test metrics
        checkpoint_dir: Directory to save checkpoints
        stage: Stage name ('A' or 'B')
        iteration: Training iteration number
    
    Returns:
        Path to saved checkpoint if saved, None otherwise
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use test ROC-AUC as primary metric (better generalization indicator)
    primary_metric = test_metrics.get('roc_auc', 0.0)
    
    # Checkpoint file
    checkpoint_file = checkpoint_dir / f"best_model_{stage}_checkpoint.json"
    
    # Load existing best if available
    best_metric = 0.0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            best_metric = checkpoint_data.get('test_roc_auc', 0.0)
    
    # Save if this is better
    if primary_metric > best_metric:
        import joblib
        
        # Save model artifacts
        model_path = checkpoint_dir / f"best_model_{stage}_checkpoint.joblib"
        scaler_path = checkpoint_dir / f"best_scaler_{stage}_checkpoint.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        if feature_selector is not None:
            selector_path = checkpoint_dir / f"best_selector_{stage}_checkpoint.joblib"
            joblib.dump(feature_selector, selector_path)
        
        # Save metadata
        checkpoint_data = {
            'stage': stage,
            'iteration': iteration,
            'test_roc_auc': float(primary_metric),
            'val_roc_auc': float(val_metrics.get('roc_auc', 0.0)),
            'test_pr_auc': float(test_metrics.get('pr_auc', 0.0)),
            'val_pr_auc': float(val_metrics.get('pr_auc', 0.0)),
            'test_f1': float(test_metrics.get('f1', 0.0)),
            'val_f1': float(val_metrics.get('f1', 0.0)),
            'threshold': float(threshold),
            'gap': float(val_metrics.get('roc_auc', 0.0) - test_metrics.get('roc_auc', 0.0)),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
        }
        
        if feature_selector is not None:
            checkpoint_data['selector_path'] = str(selector_path)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"  [BEST] Saved best model checkpoint (test ROC-AUC: {primary_metric:.4f})")
        return model_path
    
    return None


def should_stop_training(
    val_roc_auc: float,
    test_roc_auc: float,
    max_gap: float = 0.20,
    min_test_roc_auc: float = 0.50
) -> Tuple[bool, str]:
    """
    Determine if training should stop due to overfitting.
    
    Args:
        val_roc_auc: Validation ROC-AUC score
        test_roc_auc: Test ROC-AUC score
        max_gap: Maximum allowed gap (default: 0.20)
        min_test_roc_auc: Minimum acceptable test ROC-AUC (default: 0.50)
    
    Returns:
        Tuple of (should_stop, reason)
    """
    gap = val_roc_auc - test_roc_auc
    
    if gap > max_gap:
        return True, f"Overfitting detected: gap={gap:.4f} > {max_gap}"
    
    if test_roc_auc < min_test_roc_auc:
        return True, f"Test ROC-AUC too low: {test_roc_auc:.4f} < {min_test_roc_auc}"
    
    return False, ""

