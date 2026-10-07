# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều ngạc nhiên nhất là run `attn_only` (với rank ép lên r=283 để bằng số tham số với `correct`) có train loss thấp hơn hẳn `correct` (0.5365 vs 0.6268) nhưng điểm thực tế trên tập target chỉ ngang bằng chứ không hề thắng. Hóa ra train loss thấp hơn có thể chỉ là do overfit vào khối attention, chứ không phản ánh khả năng khái quát tốt hơn của mô hình.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn sinh văn bản suy luận (text generation) ở NB2 và NB5 khi đánh giá cả 3 baseline và các run đối chứng. Ban đầu tôi nghĩ bước huấn luyện (training backward pass) sẽ chiếm phần lớn thời gian, nhưng thực tế việc chạy inference tuần tự không batching cho từng mẫu eval lại ngốn thời gian đáng kể trên GPU T4.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi từng tin rằng "cứ tăng rank $r$ là mô hình sẽ thông minh hơn" và "QLoRA 4-bit luôn là mặc định tốt nhất cho mọi bài toán tiết kiệm VRAM". Sau khi chạy giải phẫu NB4, tôi nhận ra vị trí gắn adapter (phủ toàn bộ text-linear) quan trọng hơn nhiều so với việc nâng rank, và QLoRA trên Qwen3.5 làm giảm độ chính xác rõ rệt (từ 0.9375 xuống 0.8438) trong khi chạy chậm hơn do chi phí dequantization.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để hỗ trợ phân tích cấu trúc dự án, kiểm tra môi trường, giải thích các lỗi phát sinh do khác biệt hệ điều hành (lỗi xuống dòng CRLF trên Windows làm sai checksum tập eval), và tổng hợp các số liệu đo lường vào bảng báo cáo. Ban đầu AI chạy thử trên CPU local nhưng không khả thi do thiếu CUDA, sau đó đã đề xuất chuyển sang chạy tự động trên Colab T4.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm không phải là mở model ra train, mà là xây dựng một prompt thật tốt để đóng băng Baseline (b) và thiết lập tập kiểm thử hồi quy (regression gate). Nếu không có một mốc so sánh nghiêm ngặt trước khi train, ta rất dễ tự lừa dối mình rằng bản fine-tune đã thành công khi thực chất nó chỉ thắng một prompt sơ sài hoặc đã làm hỏng năng lực tổng quát của mô hình.
