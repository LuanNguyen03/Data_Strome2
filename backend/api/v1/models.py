"""
Model metadata endpoints per contract requirements
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict, Any, Optional
import json

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.services.model_loader import get_model_loader

router = APIRouter()
model_loader = get_model_loader()

REGISTRY_PATH = Path("modeling/registry/registry.json")


@router.get("/latest")
async def get_latest_model() -> Dict[str, Any]:
    """
    GET /api/v1/models/latest
    Returns metadata summary of latest model version
    """
    try:
        # Load registry
        if not REGISTRY_PATH.exists():
            return {
                "model_version": model_loader.model_version,
                "use_ml_models": model_loader.use_ml_models,
                "status": "fallback",
                "note": "No registry found, using rule-based fallback"
            }
        
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        latest_entry = registry.get("latest_improved") or registry.get("latest")
        if not latest_entry:
            return {
                "model_version": model_loader.model_version,
                "use_ml_models": model_loader.use_ml_models,
                "status": "fallback",
                "note": "No latest entry in registry"
            }
        
        return {
            "model_version": latest_entry.get("model_version", model_loader.model_version),
            "created_at": latest_entry.get("created_at"),
            "dataset_hash": latest_entry.get("dataset_hash", "")[:16] + "...",
            "use_ml_models": model_loader.use_ml_models,
            "status": "loaded" if model_loader.use_ml_models else "fallback",
            "improvements": latest_entry.get("improvements", ""),
            "has_stage_a": model_loader.model_A is not None,
            "has_stage_b": model_loader.model_B is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model metadata: {str(e)}")


@router.get("/latest/metrics")
async def get_latest_metrics() -> Dict[str, Any]:
    """
    GET /api/v1/models/latest/metrics
    Returns performance metrics for latest model
    """
    try:
        if not REGISTRY_PATH.exists():
            return {
                "model_version": model_loader.model_version,
                "status": "fallback",
                "metrics": "Not available for rule-based fallback"
            }
        
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        latest_entry = registry.get("latest_improved") or registry.get("latest")
        if not latest_entry:
            return {
                "model_version": model_loader.model_version,
                "status": "fallback",
                "metrics": "Not available"
            }
        
        metrics_summary = latest_entry.get("metrics_summary", {})
        
        return {
            "model_version": latest_entry.get("model_version", model_loader.model_version),
            "metrics": metrics_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")


@router.get("/latest/calibration")
async def get_latest_calibration() -> Dict[str, Any]:
    """
    GET /api/v1/models/latest/calibration
    Returns calibration information for latest model
    """
    try:
        if not REGISTRY_PATH.exists():
            return {
                "model_version": model_loader.model_version,
                "status": "fallback",
                "calibration": "Not available for rule-based fallback"
            }
        
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        latest_entry = registry.get("latest_improved") or registry.get("latest")
        if not latest_entry:
            return {
                "model_version": model_loader.model_version,
                "status": "fallback",
                "calibration": "Not available"
            }
        
        # Try to load calibration report
        report_paths = latest_entry.get("report_paths", {})
        stage_a_reports = report_paths.get("stage_A") or report_paths.get("stage_a", {})
        cal_path = Path(stage_a_reports.get("calibration", ""))
        
        calibration_info = {
            "model_version": latest_entry.get("model_version", model_loader.model_version),
            "method": "isotonic",  # Default
            "cv_folds": 3,  # Default
        }
        
        if cal_path.exists():
            try:
                with open(cal_path, 'r') as f:
                    cal_data = json.load(f)
                    # Extract calibration info if available
                    if isinstance(cal_data, dict):
                        stage_cal = cal_data.get("stage_A") or cal_data.get("stage_a", {})
                        calibration_info.update(stage_cal)
            except Exception:
                pass
        
        return calibration_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading calibration: {str(e)}")
