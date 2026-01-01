# OLAP Summary (Slide-ready) — Insight từ dữ liệu

Dưới đây là insight dạng “đọc lên được” khi demo.  
Lưu ý: đây là xu hướng/tương quan từ dataset demo, không kết luận nhân quả.

---

## 1) Phân tầng theo nhóm tuổi & giới

- Tỷ lệ DED có thể khác nhau theo age_band và gender.
- Dùng để gợi ý: nhóm nào nên ưu tiên screening hoặc nhắc điền thêm thông tin.

## 2) Screen time + Sleep quality là “risk domain” dễ thấy

- Khi screen_time_band cao kết hợp sleep_quality thấp, tỷ lệ DED thường có xu hướng tăng.
- Đây là lý do Stage A tập trung vào nhóm biến “sleep + screen”.

## 3) Symptom_score giúp triage mạnh hơn

- symptom_score tăng (0→1→2→3) thường đi kèm rate DED tăng.
- Đây là căn cứ để Stage B dùng symptom làm triage (không dùng cho dự báo sớm).

## 4) Stress và ngủ ít có thể tạo ra pattern rủi ro

- Stress_level cao + sleep_duration_band thấp thường tạo nhóm có rate cao hơn.
- Insight này dùng cho phần “khuyến nghị hành vi” sau khi trả kết quả.

## 5) Data quality ảnh hưởng confidence

- Một số nhóm có missing_rate cao (ví dụ thiếu screen_time hoặc sleep_quality), dẫn đến confidence thấp.
- Điều này giải thích vì sao hệ thống trả: “Bạn nên bổ sung thêm X để tăng độ chắc”.

---

## Message kết luận khi demo

- “OLAP cho mình bằng chứng số liệu để giải thích vì sao hệ thống đưa risk_score và vì sao đôi lúc cần hỏi thêm symptom.”
