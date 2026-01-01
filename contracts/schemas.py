"""
Shared API contract schemas based on docs/output_contract.md
Used by both backend (FastAPI) and frontend (TypeScript types)
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ModeUsed(str, Enum):
    """Mode used for assessment"""
    A_ONLY_SCREENING = "A_only_screening"
    B_WITH_SYMPTOMS = "B_with_symptoms"


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Confidence(str, Enum):
    """Confidence level"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Direction(str, Enum):
    """Factor direction"""
    INCREASE_RISK = "increase_risk"
    DECREASE_RISK = "decrease_risk"
    UNKNOWN = "unknown"


class Strength(str, Enum):
    """Factor strength"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Urgency(str, Enum):
    """Next step urgency"""
    NONE = "none"
    MONITOR = "monitor"
    CONSIDER_VISIT = "consider_visit"
    VISIT_RECOMMENDED = "visit_recommended"


class TopFactor(BaseModel):
    """Top contributing factor"""
    feature: str
    direction: Direction
    strength: Strength
    note: str = Field(..., description="Human-readable explanation")


class NextStep(BaseModel):
    """Recommended next steps"""
    title: str
    actions: List[str] = Field(..., description="List of action items")
    ask_for_more_info: List[str] = Field(default_factory=list)
    urgency: Urgency


class ScreeningInfo(BaseModel):
    """Stage A screening information"""
    risk_A: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    trigger_symptom: bool = Field(..., description="Should ask for symptoms")


class TriageInfo(BaseModel):
    """Stage B triage information"""
    prob_B: float = Field(..., ge=0, le=100, description="Probability 0-100")
    triage_level: RiskLevel


class AssessmentResponse(BaseModel):
    """Main API response contract"""
    request_id: str
    timestamp: datetime
    mode_used: ModeUsed
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    confidence: Confidence
    missing_fields: List[str] = Field(default_factory=list)
    top_factors: List[TopFactor] = Field(default_factory=list)
    next_step: NextStep
    disclaimers: List[str] = Field(default_factory=list)
    
    # Mode-specific fields
    screening: Optional[ScreeningInfo] = None
    triage: Optional[TriageInfo] = None
    treatment_recommendations: Optional[str] = Field(None, description="Personalized treatment directions from AI")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_8f1d2c",
                "timestamp": "2025-12-26T20:10:00+07:00",
                "mode_used": "A_only_screening",
                "risk_score": 68.0,
                "risk_level": "Medium",
                "confidence": "Medium",
                "missing_fields": ["sleep_quality"],
                "top_factors": [
                    {
                        "feature": "average_screen_time",
                        "direction": "increase_risk",
                        "strength": "High",
                        "note": "Thời gian nhìn màn hình cao thường đi kèm nguy cơ tăng."
                    }
                ],
                "screening": {
                    "risk_A": 68.0,
                    "trigger_symptom": True
                },
                "next_step": {
                    "title": "Khuyến nghị bổ sung thông tin để tăng độ chắc",
                    "actions": [
                        "Điền thêm 3 triệu chứng mắt để hệ thống phân loại chính xác hơn.",
                        "Giảm screen time trước khi ngủ và nghỉ mắt theo chu kỳ."
                    ],
                    "ask_for_more_info": [
                        "discomfort_eyestrain",
                        "redness_in_eye",
                        "itchiness_irritation_in_eye"
                    ],
                    "urgency": "monitor"
                },
                "disclaimers": [
                    "Kết quả chỉ hỗ trợ sàng lọc nguy cơ, không thay thế chẩn đoán.",
                    "Nếu triệu chứng kéo dài hoặc nặng, nên tham khảo bác sĩ."
                ]
            }
        }


class AssessmentRequest(BaseModel):
    """Input request for assessment"""
    # Person
    age: Optional[int] = Field(None, ge=18, le=45)
    gender: Optional[int] = Field(None, ge=0, le=1, description="0=F, 1=M")
    height: Optional[int] = Field(None, ge=120, le=230)
    weight: Optional[int] = Field(None, ge=30, le=250)
    
    # Sleep
    sleep_duration: Optional[float] = Field(None, ge=0, le=24)
    sleep_quality: Optional[int] = Field(None, ge=1, le=5)
    sleep_disorder: Optional[int] = Field(None, ge=0, le=1)
    wake_up_during_night: Optional[int] = Field(None, ge=0, le=1)
    feel_sleepy_during_day: Optional[int] = Field(None, ge=0, le=1)
    
    # Device/Screen
    average_screen_time: Optional[float] = Field(None, ge=0, le=24)
    smart_device_before_bed: Optional[int] = Field(None, ge=0, le=1)
    bluelight_filter: Optional[int] = Field(None, ge=0, le=1)
    
    # Lifestyle
    stress_level: Optional[int] = Field(None, ge=1, le=5)
    daily_steps: Optional[int] = Field(None, ge=0, le=50000)
    physical_activity: Optional[int] = Field(None, ge=0, le=600)
    caffeine_consumption: Optional[int] = Field(None, ge=0, le=1)
    alcohol_consumption: Optional[int] = Field(None, ge=0, le=1)
    smoking: Optional[int] = Field(None, ge=0, le=1)
    
    # Vitals
    systolic: Optional[int] = Field(None, ge=70, le=250)
    diastolic: Optional[int] = Field(None, ge=40, le=150)
    heart_rate: Optional[int] = Field(None, ge=40, le=220)
    
    # Medical
    medical_issue: Optional[int] = Field(None, ge=0, le=1)
    ongoing_medication: Optional[int] = Field(None, ge=0, le=1)
    
    # Symptoms (Stage B only)
    discomfort_eyestrain: Optional[int] = Field(None, ge=0, le=1)
    redness_in_eye: Optional[int] = Field(None, ge=0, le=1)
    itchiness_irritation_in_eye: Optional[int] = Field(None, ge=0, le=1)

