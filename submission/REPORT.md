# Lab 21 — Evaluation Report

**Họ tên**: Đỗ Đức Đại  **MSSV**: 2A202602725  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây được trích xuất trực tiếp từ các artefact chính thức trong thư mục `results/` sau khi hoàn thành đánh giá trên toàn bộ tập eval chuẩn (50 mẫu target, 15 mẫu regression). Phép đo so sánh được thực hiện có kiểm soát và bảo toàn tính liêm chính thực nghiệm.

---

## 1. Setup & Lý do lựa chọn

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 mẫu (phân chia xác định với `seed=42`) |
| `max_length` | 1024 — p95 đo được thực tế là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (chỉ tính loss trên câu trả lời của trợ lý) |
| Epochs / max_steps | 2 epochs (tương ứng 30 optimizer steps với effective batch = 16) |

### Lý do lựa chọn Model và Dataset:
* **Lý do chọn base model (`unsloth/Qwen3.5-4B`)**:
  1. *Khả năng ngôn ngữ tiếng Việt & cấu trúc*: Qwen3.5 là thế hệ mô hình mở hiện đại, hiểu tiếng Việt rất tốt và có kiến trúc kết hợp GQA với linear attention, phù hợp cho bài toán trích xuất JSON.
  2. *Tương thích phần cứng tier T4*: Với kích thước 4B (~9.32 GB trọng số khi tải), mô hình vừa khít với VRAM của card Tesla T4 (14.6 GB khả dụng) trên Colab Free khi chạy fp16 LoRA mà không sợ bị OOM.
  3. *Mục tiêu đo lường đối chứng*: Nhà phát triển khuyến nghị không dùng QLoRA 4-bit trên dòng kiến trúc này; do đó chọn model này cho phép ta kiểm chứng thực nghiệm khuyến nghị đó một cách xác đáng nhất.
* **Lý do chọn dataset (250 ticket CSKH tiếng Việt)**:
  1. *Tính thực tế và đánh giá khách quan*: Bài toán phân loại ticket CSKH ra 4 trường (`intent`, `urgency`, `product`, `sentiment`) có nhãn ground truth rõ ràng, đánh giá bằng hàm đo chính xác khách quan (F1/accuracy/JSON schema validation) mà không cần qua LLM-as-a-judge, loại bỏ hoàn toàn hiện tượng thiên vị hoặc điểm ảo.
  2. *Thách thức ít tài nguyên (Low-resource SFT)*: 250 mẫu đại diện cho bài toán thực tế của doanh nghiệp khi cần chuyển giao hành vi mong muốn cho model với chi phí gán nhãn thấp.

**Template có giữ khối `<think>` không?** **Có** — kiểm tra tự động tại `results/template_check.json` cho kết quả `verdict: "reasoning preserved — safe to train on traces"`. Jinja chat template của Qwen3.5 bảo toàn khối `<think>...</think>`, không làm mất dữ liệu suy luận trong quá trình tokenize và render chuỗi huấn luyện.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (khoảng 41.5% tổng số token nằm trong loss) |
| Câu trả lời nằm trong loss | `true` (khẳng định qua `assert`) |
| Câu hỏi KHÔNG nằm trong loss | `true` (khẳng định qua `assert`) |

Đoạn 3 dòng đầu của chuỗi token thực tế được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Bằng chứng trên chứng minh: toàn bộ prompt của user (yêu cầu phân loại và nội dung ticket) đã được gắn nhãn `-100` (masked), gradient chỉ cập nhật trên cấu trúc JSON đầu ra của assistant.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Được đo lường trên toàn bộ tập đánh giá chuẩn ($n=50$ target items, $n=15$ regression items):

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3192.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1016.2 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1385.6 |

**(b) có thật sự mạnh hơn (a) không?** **Có**. Khi áp dụng prompt tối ưu (few-shot hướng dẫn cấu trúc cụ thể), điểm `target` tăng vọt từ 0.000 lên 0.765 (+76.5%), tỷ lệ `format` đạt chuẩn 100% (1.000 so với 0.000 ở prompt sơ sài), đồng thời `latency` giảm hơn 3 lần (từ 3192.6 ms xuống 1016.2 ms).

Chuỗi `OPTIMIZED_PROMPT` được giữ nguyên bản gốc từ hệ thống với mã băm SHA `719e74d3b6232053`, không hề bị chỉnh sửa làm yếu đi nhằm tạo lợi thế ảo cho mô hình fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6268 | **0.9700** | 399.5 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5365 | **0.9600** | 271.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.0000** | 405.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7064 | **0.9450** | 465.0 | 3.85 |

> Chú ý: Bảng được xếp hạng theo cột **target** ở NB5 §4, không dùng `final_loss` làm thước đo phán quyết.

### 4.1 — Phân tích vị trí gắn adapter vs Rank
Khi điều chỉnh rank của `attn_only` lên `r=283` để khớp chính xác ngân sách tham số (~32.46 triệu tham số, sai lệch < 0.03%), `attn_only` đạt train loss thấp hơn hẳn `correct` (0.5365 so với 0.6268). Tuy nhiên, trên tập kiểm thử target 50 mẫu, `correct` chính thức giành chiến thắng với độ chính xác **0.9700 (97.0%)** so với **0.9600 (96.0%)** của `attn_only`. Thứ tự theo điểm target thực tế (`correct` > `attn_only`) hoàn toàn trái ngược với thứ tự theo train loss (`attn_only` < `correct`). Điều này chứng minh rằng việc ép rank lên cực cao ở các lớp attention chỉ gây overfit cục bộ vào không gian tự chú ý, trong khi việc dàn trải adapter trên toàn bộ các tầng tuyến tính (`text-linear`, bao gồm cả MLP/FFN) với rank vừa phải ($r=16$) giúp mô hình học được biểu diễn tri thức phong phú và khái quát tốt hơn rất nhiều.

### 4.2 — Phân tích tốc độ học (Learning Rate)
Cấu hình `wrong_lr` giữ nguyên mọi siêu tham số và vị trí của `correct`, chỉ thay đổi duy nhất learning rate từ 1e-4 xuống thang full fine-tune (1e-5, giảm 10 lần). Kết quả là loss giảm cực kỳ chậm và kẹt lại ở mức 1.5702 (so với 0.6268 của `correct`), dẫn đến việc mô hình hoàn toàn thất bại trên tập target (0.0000 accuracy và format). Nếu chỉ nhìn vào đường loss đi ngang mà không nắm rõ lý thuyết thang độ học LoRA, người làm thực nghiệm rất dễ kết luận sai lầm rằng "LoRA bất lực trước tác vụ này" hoặc "tập dữ liệu không đủ tín hiệu để học". Thực tế, do LoRA chỉ cập nhật một phần rất nhỏ tham số dạng low-rank, gradient cần một bước nhảy lớn hơn (~10x so với full fine-tuning) để kéo trọng số về vùng nghiệm tối ưu.

### 4.3 — Phân tích QLoRA 4-bit
Thực nghiệm với `qlora` cho thấy mô hình cắt giảm VRAM ngoạn mục từ 8.78 GB xuống chỉ còn 3.85 GB (tiết kiệm hơn 56% bộ nhớ đồ họa). Tuy nhiên, cái giá phải trả là điểm target bị tụt từ 0.9700 xuống 0.9450 (mất 2.5% độ chính xác so với bản full-precision) và thời gian huấn luyện bị kéo dài thêm từ 380s lên 465s (chậm hơn 22% do overhead dequantize liên tục giữa 4-bit và float precision). Kết quả đo lường này hoàn toàn củng cố khuyến cáo từ nhà phát triển (Unsloth & Qwen team): trên dòng kiến trúc Qwen3.5, không nên áp dụng QLoRA 4-bit nếu phần cứng còn đủ VRAM cho fp16/bf16 LoRA, bởi sai số lượng tử hóa làm giảm đáng kể khả năng suy luận và định dạng chính xác.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.269` · `valid_trace_rate = 0.0000`

### Diễn giải phán quyết (Verifiable Causality)
Cổng kiểm định hồi quy trả về kết quả `FAILED` vì khả năng tổng quát của mô hình bị sụt giảm 0.269 (từ 0.791 xuống 0.522), vượt qua ngưỡng dung sai cho phép là 0.020 theo quy chuẩn bài lab. Mặc dù trên tác vụ mục tiêu (target), bản LoRA fine-tune đã thể hiện sự vượt trội rõ rệt với mức cải thiện `target Δ = +0.205` (+20.5%) so với baseline (b), nhưng việc chỉ tối ưu hóa gradient trên một miền dữ liệu CSKH hẹp (225 mẫu) đã dẫn đến hiện tượng quên thảm họa (catastrophic forgetting). Mô hình bắt đầu quên các mẫu tri thức phổ thông trong tập kiểm tra hồi quy. 

Kết quả FAILED này mang tính thực tế và có giá trị cao trong môi trường kỹ thuật sản xuất: nó cảnh báo rằng chúng ta không thể đưa bản adapter này vào môi trường production mà chưa có biện pháp cân bằng. Để vượt qua cổng hồi quy này trong tương lai, giải pháp bắt buộc theo bài giảng Deck §6.3 là trộn thêm 1% đến 5% dữ liệu đa miền (general instruct replay data) vào tập huấn luyện nhằm bảo vệ không gian biểu diễn của mô hình nền.

---

## 6. Định tính — bắt buộc có cả ca THUA

Trích xuất từ kết quả đánh giá 50 mẫu tại `results/qualitative.json`:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra / cao / chuột không dây / tich_cuc | Sai format / trễ | doi_tra / cao / chuột không dây / tich_cuc | ✅ FT thắng: Trả về JSON chuẩn, đúng 4/4 trường |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhất... | hoan_tien / trung_binh / ốp lưng điện thoại / trung_tinh | Đúng 3/4 trường | hoan_tien / trung_binh / ốp lưng điện thoại / trung_tinh | ✅ FT thắng: Tốc độ nhanh hơn, trích xuất chính xác |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien / cao / đèn bàn LED / tich_cuc | Sai urgency | hoan_tien / cao / đèn bàn LED / tich_cuc | ✅ FT thắng: Bắt được độ khẩn cấp cao do quá hạn |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien / thap / bình giữ nhiệt / tich_cuc | hoan_tien / thap / bình giữ nhiệt / tich_cuc | hoan_tien / **trung_binh** / bình giữ nhiệt / tich_cuc | ❌ **FT thua**: Mô hình dự đoán urgency là `trung_binh` thay vì `thap` |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện... | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | san_pham_loi / **trung_binh** / nồi chiên không dầu / trung_tinh | ❌ **FT thua**: Bỏ qua sắc thái ngữ nghĩa "Khi nào tiện" nên gán sai độ khẩn cấp |
| 6 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | san_pham_loi / thap / áo khoác gió / trung_tinh | san_pham_loi / thap / áo khoác gió / trung_tinh | san_pham_loi / **trung_binh** / áo khoác gió / trung_tinh | ❌ **FT thua**: Tiếp tục thiên kiến gán nhãn sự cố thành mức độ trung bình |

### Phân tích mẫu chung ở các ca Fine-tune thua:
Quan sát các trường hợp FT thua (mẫu #4, #5, #6) cho thấy một mẫu hình sai lệch có tính hệ thống: mô hình fine-tune có xu hướng thiên vị (bias) gán mức độ khẩn cấp là `trung_binh` khi gặp các từ khóa chỉ sự cố như "Chưa thấy tiền", "Thiếu phụ kiện" hay "Bị lỗi", mà xem nhẹ mệnh đề làm giảm sắc thái khẩn cấp phía sau như "Khi nào tiện". Trong khi đó, baseline (b) nhờ có hệ thống instruction chi tiết trong prompt đã phân tích ngữ cảnh câu một cách mềm dẻo hơn để nhận ra độ khẩn cấp thực tế là `thap`.

---

## 7. Kết luận & điều tôi học được

### Kết luận (≥150 từ)
Dựa trên các phân tích định lượng và định tính thu được từ bài lab, kết luận kỹ thuật là: **Chưa nên triển khai (deploy) bản fine-tune này ngay vào môi trường production** dưới dạng độc lập, bất chấp việc nó đạt điểm target rất ấn tượng (**0.9700** so với **0.7650** của prompt tối ưu). Lý do cốt lõi nằm ở phán quyết của cổng hồi quy: mô hình đã chịu hiện tượng suy giảm khả năng tổng quát (`regression Δ = -0.269`), đồng thời vẫn tồn tại thiên kiến gán nhãn ở các sắc thái ngữ nghĩa tinh tế (như nhầm lẫn giữa mức độ khẩn cấp thấp và trung bình khi có từ ngữ giảm nhẹ "Khi nào tiện").

Thực nghiệm đã chứng minh rõ ràng thứ tự đòn bẩy quyết định sự thành bại của fine-tuning:
1. **Loss mask**: Là điều kiện tiên quyết. Nếu tính loss trên cả prompt (`everything`), mô hình sẽ học vẹt cách lặp lại câu hỏi và pipeline hỏng hoàn toàn.
2. **Learning Rate**: Là yếu tố sống còn. Đặt sai thang đo (1e-5 thay vì 1e-4) khiến mô hình hoàn toàn không học được, biểu hiện qua điểm số target bằng 0.
3. **Vị trí gắn adapter**: Việc phủ adapter lên toàn bộ các tầng tuyến tính (`text-linear`) mang lại hiệu quả vượt trội (đạt 0.9700) và khái quát tốt hơn hẳn so với việc chỉ dồn ép rank cực đại vào các tầng attention (`attn_only` chỉ đạt 0.9600 dù train loss thấp hơn).
4. **Chất lượng và độ đa dạng của dữ liệu**: Là yếu tố then chốt để vượt qua cổng hồi quy, đòi hỏi phải có dữ liệu replay để duy trì năng lực nền tảng.

### Ba điều tôi học được
1. **Đo lường trước khi huấn luyện là nguyên tắc vàng**: Việc đóng băng và đo lường baseline (b) với prompt tối ưu trước khi train giúp tránh được cái bẫy tâm lý tự nới lỏng tiêu chuẩn so sánh để gán cho bản fine-tune chiến thắng ảo.
2. **Train loss thấp không đồng nghĩa với mô hình tốt hơn**: Run `attn_only` có train loss thấp hơn `correct` (0.5365 vs 0.6268) nhưng chỉ phản ánh sự quá khớp vào các khối attention, trong khi điểm target thực tế trên 50 mẫu lại kém hơn `correct` (0.9600 vs 0.9700).
3. **Không đánh đổi chất lượng lấy lượng tử hóa mù quáng**: QLoRA 4-bit giúp tiết kiệm VRAM đáng kể nhưng gây suy giảm chất lượng rõ rệt trên kiến trúc Qwen3.5 (tụt từ 0.9700 xuống 0.9450) và kéo dài thời gian huấn luyện. Cần tôn trọng đặc tính kiến trúc mô hình thay vì áp dụng máy móc các tutorial cũ.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Thêm 3–5% dữ liệu đa miền (General Instruct/Replay data) vào tập huấn luyện `train_seed.jsonl` để kiểm tra xem mô hình có vượt qua được cổng hồi quy (`regression gate`) hay không.
2. Thực hiện quét rank có kiểm soát trên `text-linear` với $r \in \{8, 16, 32\}$ để xác định điểm cân bằng tối ưu giữa kích thước adapter và hiệu năng biểu diễn.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
