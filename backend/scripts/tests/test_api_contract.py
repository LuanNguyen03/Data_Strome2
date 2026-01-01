"""
API Contract Tests - Per docs/clinical_governance_checklist.md
Tests that screening/triage endpoints always include disclaimers and model_version
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_screening_response_contract():
    """Test that screening response includes disclaimers and model_version"""
    response = client.post(
        "/api/v1/assessments/screening",
        json={
            "age": 30,
            "average_screen_time": 8.5,
            "sleep_quality": 3,
            "sleep_duration": 6.5,
            "stress_level": 4,
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Must have disclaimers
    assert "disclaimers" in data, "Response missing disclaimers"
    assert isinstance(data["disclaimers"], list), "Disclaimers must be array"
    assert len(data["disclaimers"]) > 0, "Disclaimers array must not be empty"
    
    # Must have model_version
    assert "model_version" in data, "Response missing model_version"
    assert isinstance(data["model_version"], str), "model_version must be string"
    assert len(data["model_version"]) > 0, "model_version must not be empty"
    
    # Must have mode_used
    assert data["mode_used"] == "A_only_screening", "Must be A_only_screening mode"
    
    # Must have screening object
    assert "screening" in data, "Response missing screening object"
    assert "trigger_symptom" in data["screening"], "screening missing trigger_symptom"
    
    # Must have all required contract fields
    required_fields = [
        "request_id", "timestamp", "mode_used", "risk_score", "risk_level",
        "confidence", "missing_fields", "top_factors", "next_step", "disclaimers"
    ]
    for field in required_fields:
        assert field in data, f"Response missing required field: {field}"


def test_triage_response_contract():
    """Test that triage response includes disclaimers and model_version"""
    response = client.post(
        "/api/v1/assessments/triage",
        json={
            "age": 30,
            "average_screen_time": 8.5,
            "sleep_quality": 3,
            "discomfort_eyestrain": 1,
            "redness_in_eye": 1,
            "itchiness_irritation_in_eye": 0,
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Must have disclaimers
    assert "disclaimers" in data, "Response missing disclaimers"
    assert isinstance(data["disclaimers"], list), "Disclaimers must be array"
    assert len(data["disclaimers"]) > 0, "Disclaimers array must not be empty"
    
    # Must have model_version
    assert "model_version" in data, "Response missing model_version"
    assert isinstance(data["model_version"], str), "model_version must be string"
    assert len(data["model_version"]) > 0, "model_version must not be empty"
    
    # Must have mode_used
    assert data["mode_used"] == "B_with_symptoms", "Must be B_with_symptoms mode"
    
    # Must have triage object
    assert "triage" in data, "Response missing triage object"
    assert "prob_B" in data["triage"], "triage missing prob_B"
    assert "triage_level" in data["triage"], "triage missing triage_level"
    
    # Must have all required contract fields
    required_fields = [
        "request_id", "timestamp", "mode_used", "risk_score", "risk_level",
        "confidence", "missing_fields", "top_factors", "next_step", "disclaimers"
    ]
    for field in required_fields:
        assert field in data, f"Response missing required field: {field}"


def test_disclaimers_content():
    """Test that disclaimers contain required medical language"""
    response = client.post(
        "/api/v1/assessments/screening",
        json={"average_screen_time": 8.0}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    disclaimers_text = " ".join(data["disclaimers"]).lower()
    
    # Must contain key phrases (per clinical governance)
    assert "sàng lọc" in disclaimers_text or "triage" in disclaimers_text, \
        "Disclaimers must mention screening/triage"
    assert "chẩn đoán" not in disclaimers_text or "không thay thế" in disclaimers_text, \
        "Disclaimers must clarify not replacing diagnosis"


def test_model_version_always_present():
    """Test that model_version is always present even with fallback"""
    # Test with minimal data (will use fallback)
    response = client.post(
        "/api/v1/assessments/screening",
        json={}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "model_version" in data, "model_version must be present even with fallback"
    assert data["model_version"] in ["fallback_rule_v1", "fallback_rule_v1"] or \
           data["model_version"].startswith("v") or \
           data["model_version"].startswith("advanced"), \
           f"Unexpected model_version: {data['model_version']}"


if __name__ == "__main__":
    # Manual test runner
    print("Running API Contract Tests...")
    try:
        test_screening_response_contract()
        print("✅ Screening contract test passed")
        
        test_triage_response_contract()
        print("✅ Triage contract test passed")
        
        test_disclaimers_content()
        print("✅ Disclaimers content test passed")
        
        test_model_version_always_present()
        print("✅ Model version test passed")
        
        print("\n✅ All API contract tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
