"""
Response wrapper to add model_version to all responses per contract
"""
from typing import Dict, Any
from contracts import AssessmentResponse
from backend.services.model_loader import get_model_loader


def add_model_version_to_response(response: AssessmentResponse) -> Dict[str, Any]:
    """
    Convert AssessmentResponse to dict and add model_version
    Per contract: Every response includes model_version
    """
    model_loader = get_model_loader()
    
    # Convert to dict
    response_dict = response.model_dump()
    
    # Add model_version
    response_dict["model_version"] = model_loader.model_version
    
    return response_dict
