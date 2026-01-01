"""
Script to visualize feature importance and selected features from V7 Medical Model.
Helps in explaining the model logic to medical professionals.
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_stage(stage="A"):
    model_path = Path(f"modeling/artifacts/model_{stage}_medical_v7.joblib")
    if not model_path.exists():
        print(f"❌ Model for Stage {stage} not found at {model_path}")
        return

    print(f"--- Visualizing Stage {stage} ---")
    data = joblib.load(model_path)
    
    # 1. Extract feature list
    features = data.get("features", [])
    
    # 2. Extract model importance if available
    # Since we use a Stacking Ensemble, we look at the meta-learner weights
    model = data.get("model")
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    if hasattr(model, 'final_estimator_'):
        # Get coefficients from the Logistic Regression Meta-Learner
        coefs = model.final_estimator_.coef_[0]
        base_model_names = [name for name, _ in model.estimators]
        
        importance_df = pd.DataFrame({
            'Base Model': base_model_names,
            'Influence Weight': coefs
        }).sort_values(by='Influence Weight', ascending=False)
        
        sns.barplot(x='Influence Weight', y='Base Model', data=importance_df, palette='viridis')
        plt.title(f"Stage {stage}: Influence of Base Models in Ensemble", fontsize=15)
        plt.xlabel("Weight in Final Decision")
    else:
        # Fallback: Just show the list of selected features
        plt.text(0.5, 0.5, f"Selected Features: {len(features)}\n\n" + "\n".join(features[:15]) + "...", 
                 ha='center', va='center', fontsize=12)
        plt.title(f"Stage {stage}: Top Selected Features")

    output_img = f"modeling/reports/feature_importance_{stage}_v7.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"✅ Visualization saved to: {output_img}")

if __name__ == "__main__":
    # Create reports directory if not exists
    Path("modeling/reports").mkdir(parents=True, exist_ok=True)
    
    visualize_stage("A")
    visualize_stage("B")
