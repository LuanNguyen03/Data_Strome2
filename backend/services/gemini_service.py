import os
from google import genai
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
            logger.info(f"✓ Gemini AI service ENABLED (API key found: {api_key[:10]}...)")
        else:
            self.client = None
            logger.warning("✗ GEMINI_API_KEY not found in environment variables. AI treatment recommendations will be disabled.")
            logger.warning("  To enable: Set GEMINI_API_KEY in .env file or environment variable, then restart backend.")
            self.enabled = False

    async def get_treatment_recommendations(self, user_data: Dict[str, Any], assessment_result: Dict[str, Any]) -> Optional[str]:
        if not self.enabled:
            logger.info("Gemini service is disabled (no API key).")
            return "Dịch vụ tư vấn AI đang tạm tắt (thiếu API Key). Vui lòng cấu hình GEMINI_API_KEY để nhận lời khuyên cá nhân hóa."

        try:
            # Prepare the prompt
            prompt = self._build_prompt(user_data, assessment_result)
            logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")
            
            # Call Gemini using new API
            # Model: gemini-2.5-flash (Latest stable version!)
            # Alternative: gemini-2.0-flash, gemini-2.5-pro
            # See GEMINI_MODEL_OPTIONS.md for all options
            response = await self.client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            logger.info("✓ Received response from Gemini successfully.")
            return response.text
        except Exception as e:
            logger.error(f"✗ Error calling Gemini API: {e}", exc_info=True)
            return f"""❌ **Không thể lấy khuyến nghị từ AI**

Lỗi: {str(e)}

**Các khuyến nghị chung:**
- Nghỉ mắt định kỳ theo quy tắc 20-20-20 (cứ 20 phút nhìn màn hình, nhìn vật cách 20 feet trong 20 giây)
- Giảm thời gian dùng màn hình, đặc biệt trước khi ngủ
- Đảm bảo ngủ đủ 7-8 giờ mỗi ngày
- Sử dụng nước mắt nhân tạo nếu cảm thấy khô mắt
- Nếu triệu chứng kéo dài hoặc nghiêm trọng, hãy đến khám bác sĩ nhãn khoa

_Xem thêm chi tiết tại trang kết quả đánh giá._"""

    def _build_prompt(self, user_data: Dict[str, Any], assessment_result: Dict[str, Any]) -> str:
        # Extract data for prompt
        age = user_data.get('age')
        gender = "Nam" if user_data.get('gender') == 1 else "Nữ"
        height = user_data.get('height')
        weight = user_data.get('weight')
        bmi = round(weight / ((height/100)**2), 1) if height and weight else "N/A"
        
        sleep_duration = user_data.get('sleep_duration')
        sleep_quality = user_data.get('sleep_quality')
        screen_time = user_data.get('average_screen_time')
        stress_level = user_data.get('stress_level')
        
        symptoms = []
        if user_data.get('discomfort_eyestrain'): symptoms.append("Mỏi mắt, căng mắt")
        if user_data.get('redness_in_eye'): symptoms.append("Đỏ mắt")
        if user_data.get('itchiness_irritation_in_eye'): symptoms.append("Ngứa, kích ứng mắt")
        
        symptoms_str = ", ".join(symptoms) if symptoms else "Không có triệu chứng rõ rệt"
        
        risk_score = assessment_result.get('risk_score')
        risk_level = assessment_result.get('risk_level')
        
        prompt = f"""
Bạn là một chuyên gia nhãn khoa hỗ trợ tư vấn về hội chứng khô mắt và sức khỏe mắt kỹ thuật số.
Dựa trên thông tin người dùng và kết quả phân tích nguy cơ dưới đây, hãy đưa ra hướng điều trị và lời khuyên phù hợp (bằng tiếng Việt, súc tích, chuyên nghiệp).

THÔNG TIN NGƯỜI DÙNG:
- Tuổi: {age}
- Giới tính: {gender}
- BMI: {bmi}
- Thời gian ngủ: {sleep_duration} giờ/ngày (Chất lượng: {sleep_quality}/5)
- Thời gian dùng màn hình: {screen_time} giờ/ngày
- Mức độ căng thẳng: {stress_level}/5
- Triệu chứng báo cáo: {symptoms_str}

KẾT QUẢ PHÂN TÍCH:
- Điểm nguy cơ: {risk_score}/100
- Mức độ nguy cơ: {risk_level}

YÊU CẦU:
1. Đưa ra 3-5 hướng điều trị hoặc thay đổi lối sống cụ thể, cá nhân hóa dựa trên dữ liệu trên.
2. Giải thích ngắn gọn tại sao các khuyến nghị này quan trọng đối với trường hợp này.
3. Nếu nguy cơ cao, hãy nhấn mạnh việc cần đi khám bác sĩ chuyên khoa.
4. Trình bày dưới dạng danh sách gạch đầu dòng, ngôn ngữ dễ hiểu nhưng vẫn mang tính chuyên môn y khoa.
5. Không đưa ra các đơn thuốc cụ thể (thuốc kháng sinh, thuốc điều trị đặc hiệu), chỉ tập trung vào chăm sóc, thói quen và nước mắt nhân tạo nếu cần.

Hướng điều trị đề xuất:
"""
        return prompt
