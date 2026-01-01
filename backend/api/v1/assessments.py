"""
Assessment endpoints for Stage A (Screening) and Stage B (Triage)
Versioned API v1 - Strict contract compliance
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
import uuid
import logging

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contracts import AssessmentRequest, AssessmentResponse
from backend.services.assessment_service import AssessmentService
from backend.services.model_loader import get_model_loader
from backend.api.v1.schemas import ScreeningRequest, TriageRequest
from backend.api.v1.response_wrapper import add_model_version_to_response
from backend.utils.audit import log_assessment
from backend.utils.disclaimers import get_disclaimers

logger = logging.getLogger(__name__)

router = APIRouter()
service = AssessmentService()
model_loader = get_model_loader()


def _convert_screening_request(req: ScreeningRequest) -> AssessmentRequest:
    """Convert API v1 screening request to internal AssessmentRequest"""
    return AssessmentRequest(
        age=req.age,
        gender=req.gender,
        height=req.height,
        weight=req.weight,
        sleep_duration=req.sleep_duration,
        sleep_quality=req.sleep_quality,
        sleep_disorder=req.sleep_disorder,
        wake_up_during_night=req.wake_up_during_night,
        feel_sleepy_during_day=req.feel_sleepy_during_day,
        average_screen_time=req.average_screen_time,
        smart_device_before_bed=req.smart_device_before_bed,
        bluelight_filter=req.bluelight_filter,
        stress_level=req.stress_level,
        daily_steps=req.daily_steps,
        physical_activity=req.physical_activity,
        caffeine_consumption=req.caffeine_consumption,
        alcohol_consumption=req.alcohol_consumption,
        smoking=req.smoking,
        systolic=req.systolic,
        diastolic=req.diastolic,
        heart_rate=req.heart_rate,
        medical_issue=req.medical_issue,
        ongoing_medication=req.ongoing_medication,
        # Symptoms are optional in screening but ignored
        discomfort_eyestrain=None,  # Explicitly ignore
        redness_in_eye=None,
        itchiness_irritation_in_eye=None,
    )


def _convert_triage_request(req: TriageRequest) -> AssessmentRequest:
    """Convert API v1 triage request to internal AssessmentRequest"""
    return AssessmentRequest(
        age=req.age,
        gender=req.gender,
        height=req.height,
        weight=req.weight,
        sleep_duration=req.sleep_duration,
        sleep_quality=req.sleep_quality,
        sleep_disorder=req.sleep_disorder,
        wake_up_during_night=req.wake_up_during_night,
        feel_sleepy_during_day=req.feel_sleepy_during_day,
        average_screen_time=req.average_screen_time,
        smart_device_before_bed=req.smart_device_before_bed,
        bluelight_filter=req.bluelight_filter,
        stress_level=req.stress_level,
        daily_steps=req.daily_steps,
        physical_activity=req.physical_activity,
        caffeine_consumption=req.caffeine_consumption,
        alcohol_consumption=req.alcohol_consumption,
        smoking=req.smoking,
        systolic=req.systolic,
        diastolic=req.diastolic,
        heart_rate=req.heart_rate,
        medical_issue=req.medical_issue,
        ongoing_medication=req.ongoing_medication,
        # Symptoms are required for triage
        discomfort_eyestrain=req.discomfort_eyestrain,
        redness_in_eye=req.redness_in_eye,
        itchiness_irritation_in_eye=req.itchiness_irritation_in_eye,
    )


@router.post("/screening")
async def assess_screening(request: ScreeningRequest) -> Dict[str, Any]:
    """
    Stage A: Screening assessment (no symptoms used, no leakage)
    
    Accepts sleep/lifestyle/device/person/vitals/medical fields.
    Symptoms are optional but ignored in Stage A.
    
    Returns assessment with mode_used="A_only_screening".
    Contract: Always includes model_version and disclaimers.
    """
    try:
        # Convert to internal request (ignore symptoms)
        internal_req = _convert_screening_request(request)
        
        # Generate request_id if not provided
        request_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now()
        
        # Perform assessment
        response = await service.assess(internal_req)
        
        # Override request_id and ensure disclaimers
        response.request_id = request_id
        response.timestamp = timestamp
        response.disclaimers = get_disclaimers("screening")
        
        # Add model_version to response (per contract)
        # Note: model_version is in response metadata, we'll add it via extension
        
        # Audit log (per contract requirements)
        log_assessment(
            request_id=response.request_id,
            timestamp=timestamp,
            mode_used=response.mode_used,
            model_version=model_loader.model_version,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            confidence=response.confidence,
            missing_fields_count=len(response.missing_fields),
            trigger_symptom=response.screening.trigger_symptom if response.screening else None,
        )
        
        # Add model_version per contract
        return add_model_version_to_response(response)
    except Exception as e:
        logger.error(f"Assessment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.post("/triage")
async def assess_triage(request: TriageRequest) -> Dict[str, Any]:
    """
    Stage B: Triage assessment (with symptoms)
    
    Accepts all Stage A fields plus symptoms (3 symptom booleans).
    
    Returns assessment with mode_used="B_with_symptoms".
    Contract: Always includes model_version and disclaimers.
    """
    try:
        # Convert to internal request (include symptoms)
        internal_req = _convert_triage_request(request)
        
        # Generate request_id
        request_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now()
        
        # Perform assessment (will use Stage B since symptoms present)
        response = await service.assess(internal_req)
        
        # Override request_id and ensure disclaimers
        response.request_id = request_id
        response.timestamp = timestamp
        response.disclaimers = get_disclaimers("triage")
        
        # Audit log (per contract requirements)
        log_assessment(
            request_id=response.request_id,
            timestamp=timestamp,
            mode_used=response.mode_used,
            model_version=model_loader.model_version,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            confidence=response.confidence,
            missing_fields_count=len(response.missing_fields),
            triage_level=response.triage.triage_level if response.triage else None,
        )
        
        # Add model_version per contract
        return add_model_version_to_response(response)
    except Exception as e:
        logger.error(f"Assessment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")
