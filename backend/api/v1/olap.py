"""
OLAP KPI endpoints for reading aggregate parquet files
"""
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import duckdb
import pandas as pd
import numpy as np

router = APIRouter()

# KPI file mapping
KPI_FILES = {
    "age_gender": "agg_ded_by_age_gender.parquet",
    "screen_sleep": "agg_ded_by_screen_sleep.parquet",
    "symptom_score": "agg_ded_by_symptom_score.parquet",
    "stress_sleepband": "agg_ded_by_stress_sleepband.parquet",
    "data_quality_group": "agg_data_quality_by_group.parquet",
}

AGG_DIR = Path("analytics/duckdb/agg")


@router.get("/kpis")
async def list_kpis() -> Dict[str, Any]:
    """
    List available KPI datasets with metadata
    """
    kpis = []
    
    for name, filename in KPI_FILES.items():
        file_path = AGG_DIR / filename
        exists = file_path.exists()
        
        metadata = {
            "name": name,
            "filename": filename,
            "available": exists,
        }
        
        if exists:
            # Get basic stats from parquet
            try:
                con = duckdb.connect()
                result = con.execute(
                    f"SELECT COUNT(*) as n FROM read_parquet(?)",
                    [str(file_path)]
                ).fetchone()
                metadata["row_count"] = result[0] if result else 0
                con.close()
            except Exception:
                metadata["row_count"] = None
        
        kpis.append(metadata)
    
    return {
        "kpis": kpis,
        "total": len(kpis),
    }


@router.get("/kpis/{name}")
async def get_kpi(
    name: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
) -> Dict[str, Any]:
    """
    Get KPI data with pagination
    
    Supported names:
    - age_gender
    - screen_sleep
    - symptom_score
    - stress_sleepband
    - data_quality_group
    """
    if name not in KPI_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"KPI '{name}' not found. Available: {list(KPI_FILES.keys())}"
        )
    
    filename = KPI_FILES[name]
    file_path = AGG_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"KPI file not found: {filename}. Run OLAP generation first."
        )
    
    try:
        con = duckdb.connect()
        
        # Get total count
        total_result = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(file_path)]
        ).fetchone()
        total_rows = total_result[0] if total_result else 0
        
        # Calculate pagination
        offset = (page - 1) * page_size
        total_pages = (total_rows + page_size - 1) // page_size if total_rows > 0 else 0
        
        # Get paginated data
        df = con.execute(
            """
            SELECT * FROM read_parquet(?)
            ORDER BY 1, 2
            LIMIT ? OFFSET ?
            """,
            [str(file_path), page_size, offset]
        ).fetchdf()
        
        con.close()
        
        # Convert to JSON-serializable format
        # Handle NaN and convert to native Python types
        rows = df.replace({float('nan'): None}).to_dict(orient="records")
        
        # Ensure all values are JSON-serializable
        import numpy as np
        def clean_value(v):
            if isinstance(v, (np.integer, np.int64)):
                return int(v)
            elif isinstance(v, (np.floating, np.float64)):
                return float(v) if not np.isnan(v) else None
            elif isinstance(v, np.bool_):
                return bool(v)
            elif pd.isna(v):
                return None
            return v
        
        rows = [{k: clean_value(v) for k, v in row.items()} for row in rows]
        
        return {
            "name": name,
            "filename": filename,
            "page": page,
            "page_size": page_size,
            "total_rows": int(total_rows),
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "data": rows,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading KPI data: {str(e)}"
        )

