"""
Standard disclaimers for medical responses.
Per clinical governance checklist.
"""
from typing import List


def get_disclaimers(mode: str = "screening") -> List[str]:
    """
    Get standard disclaimers based on mode.
    
    Per docs/clinical_governance_checklist.md:
    - Always include: "Hỗ trợ sàng lọc/triage, không thay chẩn đoán"
    - Always include: "Nếu triệu chứng kéo dài/nặng, nên tham khảo bác sĩ"
    """
    base_disclaimers = [
        "Kết quả chỉ hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.",
        "Nếu triệu chứng kéo dài hoặc nặng, nên tham khảo bác sĩ.",
    ]
    
    if mode == "screening":
        return base_disclaimers + [
            "Kết quả mang tính sàng lọc, nên bổ sung thông tin để tăng độ chắc.",
        ]
    elif mode == "triage":
        return base_disclaimers + [
            "Kết quả hỗ trợ phân loại, không thay thế chẩn đoán.",
        ]
    
    return base_disclaimers

