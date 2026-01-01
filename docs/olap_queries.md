# OLAP Queries Map (DuckDB) — KPI → Câu hỏi → Ý nghĩa

Mục tiêu: đây là file “bản đồ OLAP” để demo: mỗi KPI trả lời 1 câu hỏi rõ, có ý nghĩa giải thích mô hình.

---

## KPI 1: DED rate theo age_band × gender

**Câu hỏi:** Nhóm tuổi nào theo giới có tỷ lệ DED cao hơn?  
**Dùng để:** phân tầng đối tượng trong screening, slide insight “ai rủi ro hơn”.

**Output:** age_band, gender, n, positives, rate  
**Ghi chú:** nếu dữ liệu thưa, gộp age_band thành 18–29, 30–39, 40–45.

---

## KPI 2: Heatmap screen_time_band × sleep_quality

**Câu hỏi:** Screen cao + ngủ kém có trend tăng DED không?  
**Dùng để:** giải thích “risk domain” Stage A.

**Output:** screen_time_band, sleep_quality, n, positives, rate  
**Ghi chú:** heatmap đẹp khi band không quá nhiều và n đủ lớn.

---

## KPI 3: DED rate theo symptom_score

**Câu hỏi:** symptom càng nhiều có tăng xác suất DED không?  
**Dùng để:** justify Stage B (triage).

**Output:** symptom_score, n, positives, rate  
**Ghi chú:** symptom_score = sum 3 symptom (0..3).

---

## KPI 4: stress_level × sleep_duration_band

**Câu hỏi:** Stress cao + ngủ ít có xu hướng tăng không?  
**Dùng để:** insight can thiệp hành vi (khuyến nghị thực tế).

**Output:** stress_level, sleep_duration_band, n, positives, rate  
**Ghi chú:** nhấn “tương quan”, không kết luận nhân quả.

---

## KPI 5: Data quality by group

**Câu hỏi:** Nhóm nào thiếu dữ liệu nhiều hoặc validity thấp?  
**Dùng để:** giải thích confidence, tránh bias do missing.

**Output gợi ý:**

- age_band, gender, n
- missing_rate_screen_time
- missing_rate_sleep_quality
- missing_rate_bp
- avg_validity_ratio

**Ghi chú:** đưa vào slide “vì sao confidence thấp ở một số trường hợp”.
