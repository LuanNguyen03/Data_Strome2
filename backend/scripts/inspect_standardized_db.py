"""
Data Inspector - Deep Dive into the Standardized Database.
Extracts head, dtypes, value counts for categoricals, and high-level stats.
"""
import pandas as pd
import polars as pl
from pathlib import Path

def inspect_db(filepath: Path):
    print(f"--- Inspecting: {filepath} ---")
    df = pl.read_parquet(filepath).to_pandas()
    
    print(f"\n[1] Shape: {df.shape}")
    
    print("\n[2] Data Types & Missing Values:")
    info_df = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df)) * 100
    })
    print(info_df)
    
    print("\n[3] Sample Values (First 3 rows):")
    print(df.head(3).T)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n[4] Categorical Columns Analysis ({len(categorical_cols)}):")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        top_values = df[col].value_counts().head(5).to_dict()
        print(f"  - {col}: {unique_count} unique values. Top: {top_values}")
        
    print("\n[5] Target Correlation (Top 10):")
    # Convert object to numeric for correlation where possible
    numeric_df = df.copy()
    for col in categorical_cols:
        if numeric_df[col].nunique() < 10: # Likely categories or binary
            numeric_df[col] = pd.factorize(numeric_df[col])[0]
        else:
            numeric_df.drop(columns=[col], inplace=True)
            
    corr = numeric_df.corr()['dry_eye_disease'].abs().sort_values(ascending=False)
    print(corr.head(10))

if __name__ == "__main__":
    inspect_db(Path("data/standardized/clean_assessments.parquet"))
