"""
Compare original and improved model results.
"""
import json
from pathlib import Path

def compare_models():
    """Compare metrics between original and improved models"""
    reports_dir = Path("modeling/reports")
    
    original_path = reports_dir / "model_metrics.json"
    improved_path = reports_dir / "model_metrics_improved.json"
    
    if not original_path.exists():
        print("[WARNING] Original metrics not found")
        return
    
    if not improved_path.exists():
        print("[WARNING] Improved metrics not found. Run training first:")
        print("  uv run python -m backend.scripts.train_models_improved --seed 42")
        return
    
    with open(original_path) as f:
        original = json.load(f)
    
    with open(improved_path) as f:
        improved = json.load(f)
    
    print("=" * 70)
    print("MODEL COMPARISON: Original vs Improved")
    print("=" * 70)
    
    for stage_name in ["stage_A", "stage_B"]:
        stage_label = "Stage A (Screening)" if "A" in stage_name else "Stage B (Triage)"
        print(f"\n{stage_label}")
        print("-" * 70)
        
        for split_name in ["validation", "test"]:
            print(f"\n  {split_name.upper()} Set:")
            
            orig = original[stage_name][split_name]
            impr = improved[stage_name][split_name]
            
            metrics = [
                ("ROC-AUC", "roc_auc"),
                ("PR-AUC", "pr_auc"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            ]
            
            for metric_label, metric_key in metrics:
                orig_val = orig.get(metric_key, 0)
                impr_val = impr.get(metric_key, 0)
                
                if orig_val is None or impr_val is None:
                    continue
                
                diff = impr_val - orig_val
                diff_pct = (diff / orig_val * 100) if orig_val > 0 else 0
                
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"    {metric_label:12} {orig_val:7.4f} → {impr_val:7.4f} {arrow} {diff:+.4f} ({diff_pct:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    # Calculate average improvements
    improvements = []
    for stage_name in ["stage_A", "stage_B"]:
        for split_name in ["validation", "test"]:
            orig = original[stage_name][split_name]
            impr = improved[stage_name][split_name]
            
            for metric_key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
                orig_val = orig.get(metric_key)
                impr_val = impr.get(metric_key)
                
                if orig_val is not None and impr_val is not None and orig_val > 0:
                    diff_pct = (impr_val - orig_val) / orig_val * 100
                    improvements.append(diff_pct)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        print(f"  Metrics improved: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")


if __name__ == "__main__":
    compare_models()

