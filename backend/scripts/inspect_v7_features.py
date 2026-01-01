"""
Script to inspect surviving features from Medical-Grade V7 model.
Helps identify which lifestyle factors have the strongest signal for chẩn đoán.
"""
import joblib
from pathlib import Path
import json

def inspect_features(stage="A"):
    model_path = Path(f"modeling/artifacts/model_{stage}_medical_v7.joblib")
    
    if not model_path.exists():
        print(f"❌ File not found: {model_path}")
        print("💡 Please run the training script first.")
        return

    print(f"--- Stage {stage} Surviving Features Analysis ---")
    data = joblib.load(model_path)
    
    features = data.get("features", [])
    threshold = data.get("threshold", 0.0)
    
    print(f"✅ Total surviving features: {len(features)}")
    print("\n[LIST OF FEATURES]")
    for i, f in enumerate(sorted(features), 1):
        print(f"{i:02d}. {f}")
    
    print(f"\nOptimal Medical Threshold: {threshold:.3f}")
    print("-" * 40)

if __name__ == "__main__":
    # Inspect both stages if available
    inspect_features("A")
    print()
    inspect_features("B")
