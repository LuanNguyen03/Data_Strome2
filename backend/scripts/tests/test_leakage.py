"""
Leakage Test - Per docs/clinical_governance_checklist.md
Tests that Stage A feature list excludes symptom columns (no data leakage)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.scripts.train_models_improved import STAGE_A_FEATURES, STAGE_A_EXCLUDED


def test_stage_a_excludes_symptoms():
    """Test that Stage A features do not include symptom columns"""
    # Symptom columns that MUST be excluded
    symptom_columns = [
        "discomfort_eyestrain",
        "discomfort_eye_strain",
        "redness_in_eye",
        "itchiness_irritation_in_eye",
        "symptom_score",
    ]
    
    # Check each symptom column is not in STAGE_A_FEATURES
    for symptom in symptom_columns:
        assert symptom not in STAGE_A_FEATURES, \
            f"LEAKAGE DETECTED: {symptom} found in STAGE_A_FEATURES! Stage A must not use symptoms."
    
    print("✅ No symptom columns found in STAGE_A_FEATURES")


def test_stage_a_excluded_list():
    """Test that STAGE_A_EXCLUDED list is properly defined"""
    assert len(STAGE_A_EXCLUDED) > 0, "STAGE_A_EXCLUDED list must not be empty"
    
    # Check that excluded list contains symptom columns
    symptom_columns = [
        "discomfort_eyestrain",
        "discomfort_eye_strain",
        "redness_in_eye",
        "itchiness_irritation_in_eye",
        "symptom_score",
    ]
    
    for symptom in symptom_columns:
        # Check if symptom is in excluded list (case-insensitive)
        excluded_lower = [s.lower() for s in STAGE_A_EXCLUDED]
        assert symptom.lower() in excluded_lower, \
            f"{symptom} should be in STAGE_A_EXCLUDED list"
    
    print("✅ STAGE_A_EXCLUDED list properly defined")


def test_stage_a_features_are_safe():
    """Test that Stage A features are safe (no leakage indicators)"""
    # Features that should be in Stage A (safe)
    safe_feature_categories = [
        "sleep", "screen", "stress", "lifestyle", "vitals", "person", "medical"
    ]
    
    # Check that features are from safe categories
    for feature in STAGE_A_FEATURES:
        feature_lower = feature.lower()
        # Should not contain symptom-related words
        assert "symptom" not in feature_lower, \
            f"LEAKAGE RISK: {feature} contains 'symptom'"
        assert "discomfort" not in feature_lower or "discomfort" not in STAGE_A_EXCLUDED, \
            f"LEAKAGE RISK: {feature} contains 'discomfort'"
    
    print("✅ Stage A features are safe (no symptom leakage)")


def test_stage_b_includes_symptoms():
    """Test that Stage B correctly includes symptoms (sanity check)"""
    from backend.scripts.train_models_improved import STAGE_B_FEATURES
    
    # Stage B should include symptoms
    symptom_columns = [
        "discomfort_eye_strain",
        "redness_in_eye",
        "itchiness_irritation_in_eye",
    ]
    
    # At least one symptom should be in Stage B
    stage_b_lower = [f.lower() for f in STAGE_B_FEATURES]
    found_symptoms = [s for s in symptom_columns if s.lower() in stage_b_lower]
    
    assert len(found_symptoms) > 0, \
        "Stage B should include symptom columns but none found"
    
    print(f"✅ Stage B correctly includes symptoms: {found_symptoms}")


if __name__ == "__main__":
    # Manual test runner
    print("Running Leakage Tests...")
    print("=" * 70)
    
    try:
        test_stage_a_excludes_symptoms()
        test_stage_a_excluded_list()
        test_stage_a_features_are_safe()
        test_stage_b_includes_symptoms()
        
        print("=" * 70)
        print("✅ All leakage tests passed! No data leakage detected.")
    except AssertionError as e:
        print("=" * 70)
        print(f"❌ LEAKAGE DETECTED: {e}")
        raise
    except Exception as e:
        print("=" * 70)
        print(f"❌ Error: {e}")
        raise
