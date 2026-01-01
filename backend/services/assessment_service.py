"""
Assessment service implementing 2-stage medical logic with ML models
- Stage A: Screening (no symptoms, no leakage)
- Stage B: Triage (with symptoms)
- Uses ModelLoader for predictions with fallback to rule-based
"""
from datetime import datetime
from typing import List, Set, Dict, Any
import uuid
import logging

import sys
from pathlib import Path

# Add project root to path for contracts
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from contracts import (
    AssessmentRequest,
    AssessmentResponse,
    Confidence,
    Direction,
    ModeUsed,
    NextStep,
    RiskLevel,
    ScreeningInfo,
    Strength,
    TopFactor,
    TriageInfo,
    Urgency,
)

from backend.services.model_loader import get_model_loader
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

# Thresholds from docs/threshold_notes.md
STAGE_A_THRESHOLDS = {
    "low": 40,
    "medium": 69,
    "high": 100,
}
STAGE_B_THRESHOLDS = {
    "low": 35,
    "medium": 69,
    "high": 100,
}
ROUTER_TRIGGER_THRESHOLD = 65  # Ask for symptoms if risk_A >= 65

# Critical fields for confidence calculation
CRITICAL_FIELDS = {
    "sleep_quality",
    "average_screen_time",
    "sleep_duration",
    "stress_level",
}


class AssessmentService:
    """Service for risk assessment following medical governance"""

    def __init__(self):
        self.model_loader = get_model_loader()
        self.gemini_service = GeminiService()

    async def assess(self, request: AssessmentRequest) -> AssessmentResponse:
        """
        Main assessment logic - routes to Stage A or B
        """
        request_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now()

        # Check if symptoms provided (Stage B)
        has_symptoms = self._has_symptoms(request)

        if has_symptoms:
            response = self._assess_stage_b(request, request_id, timestamp)
        else:
            response = self._assess_stage_a(request, request_id, timestamp)

        # Get personalized treatment recommendations from Gemini
        user_data = request.model_dump()
        assessment_result = {
            "risk_score": response.risk_score,
            "risk_level": response.risk_level
        }
        
        recommendations = await self.gemini_service.get_treatment_recommendations(user_data, assessment_result)
        response.treatment_recommendations = recommendations

        return response

    def _has_symptoms(self, request: AssessmentRequest) -> bool:
        """Check if any symptom fields are provided"""
        return any([
            request.discomfort_eyestrain is not None,
            request.redness_in_eye is not None,
            request.itchiness_irritation_in_eye is not None,
        ])

    def _assess_stage_a(
        self, request: AssessmentRequest, request_id: str, timestamp: datetime
    ) -> AssessmentResponse:
        """
        Stage A: Screening without symptoms
        Uses lifestyle, sleep, screen time, but NOT symptoms
        """
        # Convert request to features dict
        features_dict = self._request_to_features(request, include_symptoms=False)
        
        # Get prediction from model loader
        risk_score, prob_A = self.model_loader.predict_stage_a(features_dict)

        # Determine risk level
        risk_level = self._get_stage_a_level(risk_score)

        # Calculate confidence
        missing_fields = self._get_missing_fields(request, include_symptoms=False)
        confidence = self._calculate_confidence(missing_fields, has_symptoms=False)

        # Get top factors
        top_factors = self._get_top_factors_stage_a(request, risk_score)

        # Router logic: should we ask for symptoms?
        trigger_symptom = risk_score >= ROUTER_TRIGGER_THRESHOLD

        # Get next steps
        next_step = self._get_next_step_stage_a(risk_level, trigger_symptom, missing_fields)

        # Disclaimers will be added by API layer (per governance)
        disclaimers = []  # Will be populated by API

        return AssessmentResponse(
            request_id=request_id,
            timestamp=timestamp,
            mode_used=ModeUsed.A_ONLY_SCREENING,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            missing_fields=missing_fields,
            top_factors=top_factors,
            next_step=next_step,
            disclaimers=[],  # Will be set by API layer
            screening=ScreeningInfo(
                risk_A=prob_A * 100,  # Convert to 0-100 scale
                trigger_symptom=trigger_symptom,
            ),
        )

    def _assess_stage_b(
        self, request: AssessmentRequest, request_id: str, timestamp: datetime
    ) -> AssessmentResponse:
        """
        Stage B: Triage with symptoms
        Uses all Stage A features + symptoms
        """
        # Convert request to features dict
        features_dict = self._request_to_features(request, include_symptoms=True)
        
        # Get prediction from model loader
        risk_score, prob_B = self.model_loader.predict_stage_b(features_dict)

        # Determine triage level
        triage_level = self._get_stage_b_level(risk_score)

        # Calculate confidence
        missing_fields = self._get_missing_fields(request, include_symptoms=True)
        confidence = self._calculate_confidence(missing_fields, has_symptoms=True)

        # Get top factors (includes symptoms)
        top_factors = self._get_top_factors_stage_b(request, prob_B)

        # Get next steps
        next_step = self._get_next_step_stage_b(triage_level, missing_fields)

        # Disclaimers will be added by API layer (per governance)
        disclaimers = []  # Will be populated by API

        return AssessmentResponse(
            request_id=request_id,
            timestamp=timestamp,
            mode_used=ModeUsed.B_WITH_SYMPTOMS,
            risk_score=risk_score,
            risk_level=triage_level,  # Use triage_level as risk_level for consistency
            confidence=confidence,
            missing_fields=missing_fields,
            top_factors=top_factors,
            next_step=next_step,
            disclaimers=[],  # Will be set by API layer
            triage=TriageInfo(
                prob_B=prob_B,
                triage_level=triage_level,
            ),
        )

    def _request_to_features(self, request: AssessmentRequest, include_symptoms: bool = False) -> Dict[str, Any]:
        """Convert AssessmentRequest to features dictionary"""
        features = {
            "age": request.age,
            "gender": request.gender,
            "height": request.height,
            "weight": request.weight,
            "sleep_duration": request.sleep_duration,
            "sleep_quality": request.sleep_quality,
            "sleep_disorder": request.sleep_disorder,
            "wake_up_during_night": request.wake_up_during_night,
            "feel_sleepy_during_day": request.feel_sleepy_during_day,
            "average_screen_time": request.average_screen_time,
            "smart_device_before_bed": request.smart_device_before_bed,
            "blue_light_filter": request.bluelight_filter,
            "stress_level": request.stress_level,
            "daily_steps": request.daily_steps,
            "physical_activity": request.physical_activity,
            "caffeine_consumption": request.caffeine_consumption,
            "alcohol_consumption": request.alcohol_consumption,
            "smoking": request.smoking,
            "systolic": request.systolic,
            "diastolic": request.diastolic,
            "heart_rate": request.heart_rate,
            "medical_issue": request.medical_issue,
            "ongoing_medication": request.ongoing_medication,
        }
        
        if include_symptoms:
            features.update({
                "discomfort_eye_strain": request.discomfort_eyestrain,
                "redness_in_eye": request.redness_in_eye,
                "itchiness_irritation_in_eye": request.itchiness_irritation_in_eye,
            })
        
        return features

    def _get_stage_a_level(self, risk_score: float) -> RiskLevel:
        """Map risk score to level for Stage A"""
        if risk_score < STAGE_A_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif risk_score < STAGE_A_THRESHOLDS["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _get_stage_b_level(self, prob: float) -> RiskLevel:
        """Map probability to triage level for Stage B"""
        if prob < STAGE_B_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif prob < STAGE_B_THRESHOLDS["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _get_missing_fields(self, request: AssessmentRequest, include_symptoms: bool = False) -> List[str]:
        """Get list of missing critical fields"""
        missing = []
        if request.sleep_quality is None:
            missing.append("sleep_quality")
        if request.average_screen_time is None:
            missing.append("average_screen_time")
        if request.sleep_duration is None:
            missing.append("sleep_duration")
        if request.stress_level is None:
            missing.append("stress_level")
        
        if include_symptoms:
            if request.discomfort_eyestrain is None:
                missing.append("discomfort_eyestrain")
            if request.redness_in_eye is None:
                missing.append("redness_in_eye")
            if request.itchiness_irritation_in_eye is None:
                missing.append("itchiness_irritation_in_eye")
        
        return missing

    def _calculate_confidence(
        self, missing_fields: List[str], has_symptoms: bool
    ) -> Confidence:
        """
        Calculate confidence based on missing fields
        Per docs: High <= 10%, Medium 10-30%, Low > 30%
        """
        total_critical = len(CRITICAL_FIELDS)
        if has_symptoms:
            total_critical += 3  # Add symptom fields

        missing_count = len(missing_fields)
        missing_ratio = missing_count / total_critical if total_critical > 0 else 0

        if missing_ratio <= 0.1:
            return Confidence.HIGH
        elif missing_ratio <= 0.3:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW

    def _get_top_factors_stage_a(
        self, request: AssessmentRequest, risk_score: float
    ) -> List[TopFactor]:
        """Get top contributing factors for Stage A"""
        factors = []

        if request.average_screen_time is not None and request.average_screen_time >= 6:
            factors.append(
                TopFactor(
                    feature="average_screen_time",
                    direction=Direction.INCREASE_RISK,
                    strength=Strength.HIGH,
                    note="Thời gian nhìn màn hình cao thường đi kèm nguy cơ tăng.",
                )
            )

        if request.sleep_quality is not None and request.sleep_quality <= 3:
            factors.append(
                TopFactor(
                    feature="sleep_quality",
                    direction=Direction.INCREASE_RISK,
                    strength=Strength.HIGH,
                    note="Chất lượng ngủ thấp có thể làm tăng nguy cơ.",
                )
            )

        if request.stress_level is not None and request.stress_level >= 4:
            factors.append(
                TopFactor(
                    feature="stress_level",
                    direction=Direction.INCREASE_RISK,
                    strength=Strength.MEDIUM,
                    note="Mức độ căng thẳng cao có thể ảnh hưởng đến sức khỏe mắt.",
                )
            )

        if request.sleep_duration is not None and request.sleep_duration < 7:
            factors.append(
                TopFactor(
                    feature="sleep_duration",
                    direction=Direction.INCREASE_RISK,
                    strength=Strength.MEDIUM,
                    note="Thời lượng ngủ không đủ có thể ảnh hưởng đến sức khỏe mắt.",
                )
            )

        return factors[:5]  # Top 5

    def _get_top_factors_stage_b(
        self, request: AssessmentRequest, prob: float
    ) -> List[TopFactor]:
        """Get top contributing factors for Stage B (includes symptoms)"""
        factors = self._get_top_factors_stage_a(request, prob)

        # Add symptom factors
        symptom_count = sum([
            1 if request.discomfort_eyestrain == 1 else 0,
            1 if request.redness_in_eye == 1 else 0,
            1 if request.itchiness_irritation_in_eye == 1 else 0,
        ])

        if symptom_count > 0:
            factors.insert(
                0,
                TopFactor(
                    feature="symptom_score",
                    direction=Direction.INCREASE_RISK,
                    strength=Strength.HIGH,
                    note=f"Có {symptom_count} triệu chứng mắt được báo cáo.",
                )
            )

        return factors[:5]

    def _get_next_step_stage_a(
        self, risk_level: RiskLevel, trigger_symptom: bool, missing_fields: List[str]
    ) -> NextStep:
        """Get next step recommendations for Stage A"""
        if risk_level == RiskLevel.LOW:
            return NextStep(
                title="Nguy cơ thấp — tiếp tục theo dõi",
                actions=[
                    "Nghỉ mắt định kỳ khi dùng màn hình.",
                    "Giữ thói quen ngủ đều và đủ.",
                    "Nếu xuất hiện triệu chứng kéo dài, hãy cân nhắc khám.",
                ],
                ask_for_more_info=missing_fields[:3],
                urgency=Urgency.MONITOR,
            )
        elif risk_level == RiskLevel.MEDIUM:
            actions = [
                "Giảm thời gian nhìn màn hình liên tục, nghỉ mắt theo chu kỳ.",
                "Ưu tiên ngủ đủ và nâng chất lượng ngủ.",
            ]
            ask_for = missing_fields[:3]
            if trigger_symptom:
                actions.append("Nếu có triệu chứng, hãy trả lời thêm để phân loại rõ hơn.")
                ask_for.extend(["discomfort_eyestrain", "redness_in_eye", "itchiness_irritation_in_eye"])

            return NextStep(
                title="Nguy cơ trung bình — nên cải thiện thói quen",
                actions=actions,
                ask_for_more_info=ask_for,
                urgency=Urgency.CONSIDER_VISIT,
            )
        else:  # HIGH
            actions = [
                "Bạn nên trả lời thêm triệu chứng để hệ thống phân loại chính xác hơn.",
                "Nếu khó chịu kéo dài hoặc ảnh hưởng sinh hoạt, cân nhắc thăm khám.",
                "Trong khi chờ: hạn chế màn hình trước ngủ và nghỉ mắt định kỳ.",
            ]
            ask_for = ["discomfort_eyestrain", "redness_in_eye", "itchiness_irritation_in_eye"]
            ask_for.extend(missing_fields[:2])

            return NextStep(
                title="Nguy cơ cao — nên kiểm tra thêm",
                actions=actions,
                ask_for_more_info=ask_for,
                urgency=Urgency.CONSIDER_VISIT,
            )

    def _get_next_step_stage_b(
        self, triage_level: RiskLevel, missing_fields: List[str]
    ) -> NextStep:
        """Get next step recommendations for Stage B"""
        if triage_level == RiskLevel.LOW:
            return NextStep(
                title="Phân loại thấp — theo dõi và điều chỉnh",
                actions=[
                    "Theo dõi triệu chứng trong vài ngày.",
                    "Nghỉ mắt định kỳ, hạn chế màn hình trước ngủ.",
                    "Nếu triệu chứng tăng hoặc kéo dài, cân nhắc khám.",
                ],
                ask_for_more_info=missing_fields[:2],
                urgency=Urgency.MONITOR,
            )
        elif triage_level == RiskLevel.MEDIUM:
            return NextStep(
                title="Phân loại trung bình — theo dõi sát",
                actions=[
                    "Theo dõi triệu chứng trong 1–2 tuần.",
                    "Giảm screen time, cải thiện giấc ngủ.",
                    "Nếu không cải thiện, cân nhắc khám chuyên khoa.",
                ],
                ask_for_more_info=missing_fields[:2],
                urgency=Urgency.CONSIDER_VISIT,
            )
        else:  # HIGH
            return NextStep(
                title="Phân loại cao — khuyến nghị cân nhắc khám",
                actions=[
                    "Nếu triệu chứng kéo dài hoặc ảnh hưởng sinh hoạt, nên khám mắt.",
                    "Trong khi chờ: nghỉ mắt định kỳ, hạn chế màn hình.",
                    "Nếu có dấu hiệu bất thường nghiêm trọng, ưu tiên thăm khám sớm.",
                ],
                ask_for_more_info=[],
                urgency=Urgency.VISIT_RECOMMENDED,
            )
