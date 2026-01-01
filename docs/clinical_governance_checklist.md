# Clinical Governance Checklist — Chuẩn y tế để lên production

Mục tiêu: Đảm bảo hệ thống AI dùng trong y tế có quản trị rủi ro, kiểm định, giám sát và truyền thông đúng.  
Lưu ý: Checklist này áp dụng cho hệ thống hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.

---

## 1) Định nghĩa bài toán & phạm vi y tế (bắt buộc chốt trước)

- [ ] Nêu rõ “Intended Use”:
  - Screening nguy cơ sớm (Stage A)
  - Triage khi có symptom (Stage B)
- [ ] Nêu rõ “Not Intended Use”:
  - Không chẩn đoán xác định
  - Không quyết định điều trị
- [ ] Xác định đối tượng sử dụng:
  - người dùng phổ thông / nhân viên y tế / phòng khám
- [ ] Xác định bối cảnh:
  - online self-assessment / kiosk / phòng khám
- [ ] Xác định hậu quả sai:
  - false negative (bỏ sót) vs false positive (báo động giả)

---

## 2) Chuẩn dữ liệu & kiểm soát chất lượng (Data Governance)

### 2.1 Data lineage & versioning

- [ ] Có version dataset (v1, v2…)
- [ ] Lưu schema chuẩn (data contract)
- [ ] Ghi lại quy tắc chuẩn hoá (standardization spec)

### 2.2 Data quality monitoring

- [ ] Missing rate theo cột
- [ ] Validity flags + validity_ratio
- [ ] Parse success rate (BP)
- [ ] Class balance drift (DED positive rate)
- [ ] Báo cáo theo nhóm (age_band, gender) để phát hiện bias/missing lệch

### 2.3 Data privacy tối thiểu

- [ ] Không lưu thông tin định danh cá nhân nếu không cần
- [ ] Nếu có lưu: giải thích rõ mục đích và thời gian lưu
- [ ] Gắn request_id thay vì thông tin nhạy cảm

---

## 3) Kiểm định mô hình (Model Validation)

### 3.1 Internal validation (trong dataset)

- [ ] ROC-AUC + PR-AUC
- [ ] Confusion matrix ở threshold dự định
- [ ] Recall cho Stage A (ưu tiên an toàn screening)
- [ ] Precision/Recall cho Stage B (triage)

### 3.2 Calibration (nếu hiển thị %)

- [ ] Reliability/Calibration check
- [ ] Nếu lệch: áp dụng calibration trước khi nói “70%”

### 3.3 Fairness & subgroup analysis

- [ ] Đánh giá theo nhóm tuổi/giới
- [ ] So sánh performance theo subgroup
- [ ] Kiểm tra missing lệch theo subgroup

---

## 4) External validation (điều kiện quan trọng trước khi “dùng thật”)

- [ ] Có dữ liệu ngoài (khác nguồn/khác địa điểm/khác thời gian)
- [ ] Đánh giá lại metrics và calibration
- [ ] So sánh với baseline y tế (rule-based hoặc chuẩn screening hiện tại)
- [ ] Có kết luận bằng văn bản: mô hình dùng được cho intended use nào

---

## 5) Ngưỡng quyết định & quản trị rủi ro (Risk Management)

### 5.1 Threshold rationale

- [ ] Stage A threshold ưu tiên recall
- [ ] Router trigger rõ ràng (khi nào hỏi symptom)
- [ ] Stage B threshold cân bằng precision/recall

### 5.2 Safe-fallback

- [ ] Nếu thiếu dữ liệu nhiều → confidence thấp → khuyến nghị bổ sung field
- [ ] Nếu hệ thống không chắc → không “kết luận mạnh” → chuyển sang “monitor/consider_visit”
- [ ] Nếu symptom nặng (nếu bạn có field) → ưu tiên khuyến nghị khám

---

## 6) Truyền thông y tế an toàn (Clinical Communication)

- [ ] Luôn có disclaimer:
  - “Hỗ trợ sàng lọc/triage, không thay chẩn đoán”
- [ ] Không dùng câu “bạn chắc chắn mắc bệnh”
- [ ] Next_step có hướng dẫn hành động:
  - theo dõi
  - thay đổi thói quen
  - cân nhắc khám / nên khám
- [ ] Tôn trọng người dùng: ngôn ngữ đơn giản, không gây hoảng loạn

---

## 7) Giám sát vận hành (Monitoring & Drift)

### 7.1 Drift monitoring (định kỳ)

- [ ] Distribution drift các feature chính (screen_time, sleep_quality…)
- [ ] Target drift (tỷ lệ positive thay đổi)
- [ ] Missing drift (thiếu dữ liệu tăng bất thường)

### 7.2 Performance monitoring (nếu có ground truth sau này)

- [ ] Theo dõi metrics theo tháng/quý
- [ ] Theo dõi calibration theo thời gian
- [ ] Alert khi performance giảm

---

## 8) Auditability & Logging (cực quan trọng trong y tế)

- [ ] Lưu request_id, timestamp, version model, version schema
- [ ] Lưu input đã chuẩn hoá (không định danh) hoặc hash
- [ ] Lưu output JSON (risk_score, confidence, next_step)
- [ ] Lưu top_factors để giải thích
- [ ] Có chính sách retention (ví dụ 30–90 ngày cho demo, lâu hơn cho production theo quy định)

---

## 9) Human-in-the-loop (nếu triển khai trong cơ sở y tế)

- [ ] Case risk cao → cho phép nhân viên y tế review
- [ ] Feedback loop: ghi nhận phản hồi bác sĩ/khách hàng
- [ ] Quy trình xử lý khi AI sai/khách khiếu nại

---

## 10) Release & Change Management

- [ ] Model registry (model version)
- [ ] A/B test hoặc rollout theo giai đoạn
- [ ] Release note thay đổi threshold/feature
- [ ] Quy trình rollback khi có sự cố

---

## 11) Definition of Done (đủ điều kiện lên production theo chuẩn y tế)

- [ ] Intended use rõ ràng, có disclaimer chuẩn
- [ ] Có internal validation + calibration
- [ ] Có external validation (ít nhất 1 nguồn ngoài)
- [ ] Có drift monitoring plan
- [ ] Có audit logging và versioning đầy đủ
- [ ] Có risk management + safe-fallback + human-in-loop (nếu cần)
