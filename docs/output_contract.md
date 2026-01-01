# Output Contract — Chuẩn JSON Output cho Demo/Production

Mục tiêu: Chuẩn hoá output để:

- Frontend hiển thị nhất quán
- Backend dễ log/audit
- Sau này thay model vẫn không đổi API contract

---

## 1) Output chung (áp dụng cho cả Stage A và Stage B)

### 1.1 Trường bắt buộc

- request_id: string (UUID hoặc random)
- mode_used: "A_only_screening" | "B_with_symptoms"
- timestamp: ISO8601 string
- risk_score: number (0..100) — luôn có (dù mode B vẫn có thể map từ prob_B)
- risk_level: "Low" | "Medium" | "High"
- confidence: "High" | "Medium" | "Low"
- missing_fields: string[]
- top_factors: array
- next_step: object
- disclaimers: string[]

### 1.2 top_factors format

Mỗi item:

- feature: string
- direction: "increase_risk" | "decrease_risk" | "unknown"
- strength: "High" | "Medium" | "Low"
- note: string (1 dòng, dễ hiểu)

---

## 2) Output riêng theo mode

### 2.1 Mode A — Screening

Bắt buộc thêm:

- screening: object
  - risk_A: number (0..1) hoặc (0..100) tuỳ bạn chuẩn hoá
  - trigger_symptom: boolean

Ý nghĩa:

- trigger_symptom = true → nên hỏi symptom để chạy triage

### 2.2 Mode B — Triage

Bắt buộc thêm:

- triage: object
  - prob_B: number (0..1)
  - triage_level: "Low" | "Medium" | "High"

---

## 3) next_step (khuyến nghị hành động) — format đề xuất

next_step:

- title: string (ngắn)
- actions: string[] (2–5 gạch đầu dòng, mỗi dòng 1 ý)
- ask_for_more_info: string[] (danh sách field gợi ý nhập thêm)
- urgency: "none" | "monitor" | "consider_visit" | "visit_recommended"

---

## 4) confidence policy — mô tả cho frontend

confidence giải thích dựa trên:

- tỉ lệ field bị thiếu
- thiếu nhóm chủ lực (sleep/screen/symptom)
- validity_ratio thấp

Frontend có thể hiển thị:

- “Confidence thấp vì thiếu sleep_quality và average_screen_time”

---

## 5) Ví dụ JSON — Mode A (Screening)

```json
{
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
    },
    {
      "feature": "sleep_duration",
      "direction": "decrease_risk",
      "strength": "Low",
      "note": "Thời lượng ngủ ổn giúp giảm rủi ro, nhưng cần thêm chất lượng ngủ."
    }
  ],
  "screening": {
    "risk_A": 0.68,
    "trigger_symptom": true
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
```
