"""
Assessment endpoints for Stage A (Screening) and Stage B (Triage)
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

import sys
from pathlib import Path

# Add project root to path for contracts
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from contracts import AssessmentRequest, AssessmentResponse
from backend.services.assessment_service import AssessmentService

router = APIRouter()
service = AssessmentService()


@router.post("/assess", response_model=AssessmentResponse)
async def assess_risk(request: AssessmentRequest) -> AssessmentResponse:
    """
    Main assessment endpoint - routes to Stage A or Stage B based on symptoms
    
    Follows medical governance:
    - Stage A: Screening without symptoms (no leakage)
    - Stage B: Triage with symptoms
    - Always includes disclaimers
    """
    try:
        response = await service.assess(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.get("/assess/example")
async def get_example():
    """Get example request/response for documentation"""
    return {
        "example_request": {
            "age": 30,
            "gender": 1,
            "sleep_duration": 7.5,
            "sleep_quality": 3,
            "average_screen_time": 8.0,
            "stress_level": 4,
        },
        "note": "See /docs for full API documentation",
    }

