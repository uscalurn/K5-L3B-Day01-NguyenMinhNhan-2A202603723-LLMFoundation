# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603723**
- **Họ và tên:Nguyễn Minh Nhân**
- **SEED** (in ở ô đầu notebook):159

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

                                precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.848      0.918        0.919   
2. BoW + MLP (deep learning)         0.990   0.911      0.949        0.974   
3a. Encoder (MiniLM) + kNN           0.868   0.938      0.901        0.974   
3b. Encoder (MiniLM) + LogReg        0.005   0.018      0.007        0.000   
3c. Encoder (MiniLM) + MLP head      0.990   0.911      0.949        0.947   
4a. LLM nhỏ zero-shot                0.909   1.000      0.952        0.952   
4b. LLM lớn zero-shot                1.000   1.000      1.000        1.000   
4c. LLM nhỏ few-shot                 1.000   1.000      1.000        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           10.107      22086.621       0.054   
2. BoW + MLP (deep learning)         5.946      38175.830       0.196   
3a. Encoder (MiniLM) + kNN          84.611         42.382      48.022   
3b. Encoder (MiniLM) + LogReg       85.007         50.685      28.897   
3c. Encoder (MiniLM) + MLP head     86.084         48.104      31.082   
4a. LLM nhỏ zero-shot                0.000          1.304     739.500   
4b. LLM lớn zero-shot                0.000          1.114     900.715   
4c. LLM nhỏ few-shot                 0.000          1.223     741.087   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.087         0.000  
2. BoW + MLP (deep learning)          0.352         0.000  
3a. Encoder (MiniLM) + kNN           57.558         0.000  
3b. Encoder (MiniLM) + LogReg        39.374         0.000  
3c. Encoder (MiniLM) + MLP head      45.613         0.000  
4a. LLM nhỏ zero-shot              1079.791         0.006  
4b. LLM lớn zero-shot              1171.413         0.059  
4c. LLM nhỏ few-shot               1120.892         0.021  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

Dự đoán LLM lớn sẽ vượt trội hơn LLM nhỏ, nhưng Luna Few-shot đạt F1 = 1.000, ngang Terra Zero-shot và chi phí thấp hơn.

Dự đoán ban đầu BoW sẽ kém hơn nhiều so với các mô hình Deep Learning, nhưng BoW + MLP đạt F1 = 0.949, ngang MiniLM + MLP head, trong khi tốc độ xử lý nhanh hơn rất nhiều.

Dự đoán kNN với k nhỏ sẽ tốt hơn, nhưng k = 25 đạt F1 = 0.976, nhỉnh hơn k = 1 và k = 5 (0.974). Điều này cho thấy cần thử nghiệm thay vì chọn tham số theo cảm tính.

## 4. Phân tích 50 case khó

BoW + LogReg và BoW + MLP chỉ đạt accuracy lần lượt 53% và 55% trên tập case khó, thấp hơn nhiều so với tập test thường. BoW gặp khó khăn nhất ở nhóm né từ khóa (0%), spam giọng tự nhiên (10%) và thao túng/lừa đảo (11–22%) vì phụ thuộc nhiều vào từ khóa đã học.

Ví dụ, tin nhắn giả làm người thân xin chuyển £200 dễ bị nhận nhầm là ham vì không có từ khóa spam quen thuộc. Tin "C0ngr4tulat10ns! Y0u h4ve w0n..." cũng dễ khiến BoW phân loại sai do các ký tự bị thay bằng số.

LLM nhỏ zero-shot đạt 98%, còn khi thêm prompt phòng thủ đạt 100%. Điều này cho thấy LLM xử lý ngữ cảnh tốt hơn trên bộ test này, nhưng vẫn cần kiểm tra trên dữ liệu thực tế lớn hơn.

## 5. Thử thách cá nhân

Em thực hiện thử thách thay đổi số neighbors K của kNN với k = 1, 5, 25.

Kết quả F1 lần lượt là 0.974, 0.974 và 0.976. K quá nhỏ dễ bị ảnh hưởng bởi nhiễu và overfitting, còn k quá lớn có thể gây underfitting. Trong thử nghiệm này, k = 25 tốt nhất nhưng chênh lệch không đáng kể.

## 6. Phương án cho sản phẩm thật

Ưu tiên BoW + MLP vì nằm trên đường Pareto, có độ trễ thấp và chất lượng tốt. Tuy nhiên, cần so sánh F1 cụ thể để xác định mô hình tốt nhất trong nhóm đạt SLA. Nếu dùng LLM xử lý toàn bộ 10 triệu SMS/ngày, tương đương 300 triệu tin/tháng, chi phí ước tính:
Luna Zero-shot: 1.800 USD/tháng.
Luna Few-shot: 6.225 USD/tháng.
Terra Zero-shot: 17.775 USD/tháng.

## 7. AI log và điều đã học

Trong quá trình làm bài, mình sử dụng ChatGPT để kiểm tra hàm tính precision, recall, F1, tìm nguyên nhân LLM trả về nhãn không hợp lệ và giải thích kết quả thử nghiệm. Mình đã sửa cách đếm FP/FN, tăng max_completion_tokens từ 3 lên 32 và chạy lại để kiểm tra.

Qua bài lab, mình hiểu rằng F1 cao chưa chắc mô hình phù hợp để triển khai thực tế. Ngoài chất lượng, cần quan tâm đến độ trễ, chi phí và khả năng xử lý các trường hợp nằm ngoài dữ liệu huấn luyện.
