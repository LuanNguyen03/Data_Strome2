"""
Audit logging utility for medical compliance.
Logs to append-only JSONL file per contract requirements.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from contracts import ModeUsed, RiskLevel, Confidence


AUDIT_LOG_PATH = Path("backend/logs/audit.jsonl")


def ensure_log_dir() -> None:
    """Ensure log directory exists"""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_assessment(
    request_id: str,
    timestamp: datetime,
    mode_used: ModeUsed,
    model_version: str,
    risk_score: float,
    risk_level: RiskLevel,
    confidence: Confidence,
    missing_fields_count: int,
    trigger_symptom: Optional[bool] = None,
    triage_level: Optional[RiskLevel] = None,
) -> None:
    """
    Log assessment request to audit log.
    
    Per contract requirements:
    - request_id, timestamp, mode_used, model_version
    - risk_score, risk_level, confidence
    - missing_fields_count
    - trigger_symptom (for Stage A) or triage_level (for Stage B)
    
    No personal identifiers are logged.
    """
    ensure_log_dir()
    
    log_entry: Dict[str, Any] = {
        "request_id": request_id,
        "timestamp": timestamp.isoformat(),
        "mode_used": mode_used.value,
        "model_version": model_version,
        "risk_score": float(risk_score),
        "risk_level": risk_level.value,
        "confidence": confidence.value,
        "missing_fields_count": missing_fields_count,
    }
    
    # Add mode-specific fields
    if mode_used == ModeUsed.A_ONLY_SCREENING and trigger_symptom is not None:
        log_entry["trigger_symptom"] = trigger_symptom
    elif mode_used == ModeUsed.B_WITH_SYMPTOMS and triage_level is not None:
        log_entry["triage_level"] = triage_level.value
    
    # Append to JSONL file (append-only for audit trail)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
