# Threshold Notes — Chọn ngưỡng (Stage A/B) theo chuẩn y tế

Mục tiêu: Giải thích cách chọn threshold để demo “đúng tư duy y tế”, tránh hiểu nhầm AI đang chẩn đoán.

---

## 1) Vì sao cần threshold?

- Model trả xác suất/risk_score liên tục (0–100).
- Demo/ứng dụng thực tế cần “ngưỡng quyết định” để:
  - phân loại mức độ (Low/Med/High)
  - kích hoạt router hỏi symptom
  - đề xuất next_step phù hợp

---

## 2) Tách 2 loại threshold (rất quan trọng)

### 2.1 Threshold cho Screening (Stage A)

- Mục tiêu: **không bỏ sót** người có nguy cơ (ưu tiên Recall).
- Chấp nhận có false positive hơn một chút (vì chỉ là “sàng lọc”, không kết luận bệnh).

**Ý nghĩa khi demo:**

- Risk cao ở Stage A = “cần hỏi thêm / cần theo dõi / nên điều chỉnh thói quen”
- Không dùng từ kiểu “bạn chắc chắn mắc bệnh”.

### 2.2 Threshold cho Triage (Stage B)

- Mục tiêu: phân loại khi đã có symptom → **cân bằng** precision/recall.
- Vì symptom gần bệnh hơn, Stage B được phép “mạnh” hơn.

**Ý nghĩa khi demo:**

- Triage level cao = “nên cân nhắc khám”
- Vẫn không kết luận chẩn đoán.

---

## 3) Bộ threshold đề xuất (dễ demo, dễ giải thích)

> Đây là default demo. Khi có metrics thật, bạn sẽ tinh chỉnh.

### 3.1 Stage A (risk_score 0–100)

- Low: < 40
- Medium: 40–69
- High: ≥ 70

### 3.2 Router trigger (hỏi symptom)

- trigger_symptom_if_risk_A ≥ 65

Lý do:

- 65 nằm trong vùng Medium-High, hợp để hỏi thêm thông tin tăng độ chắc.
- Tránh hỏi symptom cho quá nhiều người (UX tốt hơn).

### 3.3 Stage B (probability 0–100)

- Low: < 35
- Medium: 35–69
- High: ≥ 70

Gợi ý hành động theo level:

- Low: theo dõi + cải thiện thói quen
- Medium: theo dõi sát + bổ sung thông tin + cân nhắc khám nếu kéo dài
- High: khuyến nghị khám (đặc biệt nếu symptom kéo dài)

---

## 4) Cách chọn threshold khi có metrics (đúng bài)

### 4.1 Stage A: ưu tiên Recall

- Chọn threshold sao cho Recall(Y) cao (ví dụ 0.80+ nếu có thể)
- Sau đó đánh đổi với precision để UX không quá “báo động giả”

### 4.2 Stage B: cân bằng Precision/Recall

- Chọn threshold tối ưu F1 hoặc theo mục tiêu:
  - nếu muốn ít báo động giả: tăng precision
  - nếu muốn an toàn hơn: tăng recall

---

## 5) “Calibration” nếu trình bày % nguy cơ

Nếu bạn hiển thị “70%”, cần nhắc:

- % là ước lượng từ data demo
- khi production cần calibration và external validation để % đáng tin

---

## 6) Lưu ý truyền thông khi demo (tránh rủi ro y tế)

Nên nói:

- “đây là hệ thống hỗ trợ sàng lọc và phân loại”
- “điểm cao nghĩa là cần theo dõi / cần bổ sung thông tin / nên khám”

Không nên nói:

- “bạn mắc bệnh”
- “AI chẩn đoán chắc chắn”

---

## 7) Definition of Done

- [ ] Có bảng level (Low/Med/High) cho Stage A và B
- [ ] Có trigger rõ khi nào hỏi symptom
- [ ] Có mapping level → next_step (khuyến nghị hành động)
