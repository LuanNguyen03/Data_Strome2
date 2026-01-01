# Risk Copywriting Library — Câu chữ UI theo chuẩn y tế (không gây hoảng)

Mục tiêu:

- Có thư viện câu chữ “người thật viết”
- Đúng chuẩn y tế: hỗ trợ sàng lọc/triage, không chẩn đoán
- Dùng được ngay trong UI: badge, CTA, next_step, disclaimer

---

## 1) Từ vựng nên dùng / nên tránh

### Nên dùng

- “nguy cơ”, “sàng lọc”, “phân loại”, “theo dõi”
- “cân nhắc khám”, “tham khảo bác sĩ”
- “độ chắc (confidence)”, “bổ sung thông tin”

### Tránh dùng

- “chẩn đoán”, “kết luận mắc bệnh”, “chắc chắn”
- “nguy hiểm”, “bệnh nặng” (trừ khi có quy trình y tế thật)

---

## 2) Badge theo mode

### Mode A (Screening)

- “Sàng lọc nguy cơ (không triệu chứng)”
- “Kết quả mang tính sàng lọc, nên bổ sung thông tin để tăng độ chắc.”

### Mode B (Triage)

- “Phân loại khi có triệu chứng”
- “Kết quả hỗ trợ phân loại, không thay thế chẩn đoán.”

---

## 3) Confidence copy (hiển thị mềm mại)

### High

- “Độ chắc: Cao — thông tin tương đối đầy đủ.”

### Medium

- “Độ chắc: Trung bình — thiếu một vài thông tin, kết quả có thể dao động.”

### Low

- “Độ chắc: Thấp — thiếu nhiều thông tin quan trọng. Bạn nên bổ sung để kết quả đáng tin hơn.”

---

## 4) CTA (nút gợi ý hành động)

### Khi Stage A trigger symptom

- “Trả lời thêm 3 triệu chứng để phân loại rõ hơn”
- “Bổ sung triệu chứng (30 giây) để tăng độ chắc”

### Khi thiếu screen/sleep

- “Bổ sung chất lượng ngủ và thời gian màn hình”
- “Điền thêm 2 mục để kết quả đáng tin hơn”

### Khi user không muốn

- “Xem kết quả hiện tại”
- “Bỏ qua và tiếp tục”

---

## 5) Next_step theo risk_level (Mode A)

### Low

Title: “Nguy cơ thấp — tiếp tục theo dõi”
Actions:

- “Nghỉ mắt định kỳ khi dùng màn hình.”
- “Giữ thói quen ngủ đều và đủ.”
- “Nếu xuất hiện triệu chứng kéo dài, hãy cân nhắc khám.”

Urgency: monitor

### Medium

Title: “Nguy cơ trung bình — nên cải thiện thói quen”
Actions:

- “Giảm thời gian nhìn màn hình liên tục, nghỉ mắt theo chu kỳ.”
- “Ưu tiên ngủ đủ và nâng chất lượng ngủ.”
- “Nếu có triệu chứng, hãy trả lời thêm để phân loại rõ hơn.”

Urgency: consider_visit

### High

Title: “Nguy cơ cao — nên kiểm tra thêm”
Actions:

- “Bạn nên trả lời thêm triệu chứng để hệ thống phân loại chính xác hơn.”
- “Nếu khó chịu kéo dài hoặc ảnh hưởng sinh hoạt, cân nhắc thăm khám.”
- “Trong khi chờ: hạn chế màn hình trước ngủ và nghỉ mắt định kỳ.”

Urgency: consider_visit

---

## 6) Next_step theo triage_level (Mode B)

### Low

Title: “Phân loại thấp — theo dõi và điều chỉnh”
Actions:

- “Theo dõi triệu chứng trong vài ngày.”
- “Nghỉ mắt định kỳ, hạn chế màn hình trước ngủ.”
- “Nếu triệu chứng tăng hoặc kéo dài, cân nhắc khám.”

Urgency: monitor

### Medium

Title: “Phân loại trung bình — theo dõi sát”
Actions:

- “Theo dõi triệu chứng trong 1–2 tuần.”
- “Giảm screen time, cải thiện giấc ngủ.”
- “Nếu không cải thiện, cân nhắc khám chuyên khoa.”

Urgency: consider_visit

### High

Title: “Phân loại cao — khuyến nghị cân nhắc khám”
Actions:

- “Nếu triệu chứng kéo dài hoặc ảnh hưởng sinh hoạt, nên khám mắt.”
- “Trong khi chờ: nghỉ mắt định kỳ, hạn chế màn hình.”
- “Nếu có dấu hiệu bất thường nghiêm trọng, ưu tiên thăm khám sớm.”

Urgency: visit_recommended

---

## 7) Disclaimers (bắt buộc luôn hiển thị)

- “Kết quả chỉ hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.”
- “Nếu triệu chứng kéo dài hoặc nặng, hãy tham khảo bác sĩ.”

(Tuỳ chọn thêm)

- “Kết quả phụ thuộc vào mức độ đầy đủ của thông tin bạn cung cấp.”

---

## 8) Microcopy cho missing_fields

- “Thiếu thông tin: {field_list}.”
- “Bạn có thể bổ sung để tăng độ chắc của kết quả.”

Gợi ý theo nhóm:

- Nếu thiếu sleep_quality:
  - “Chất lượng ngủ giúp hệ thống đánh giá chính xác hơn.”
- Nếu thiếu average_screen_time:
  - “Thời gian màn hình là yếu tố quan trọng trong sàng lọc.”

---

## 9) Definition of Done (Copywriting)

- [ ] Có copy cho mode A/B
- [ ] Có copy cho confidence
- [ ] Có CTA & next_step theo level
- [ ] Có disclaimers chuẩn y tế
