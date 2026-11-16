# TIỀN ĐĂNG KÝ (khóa trước khi chạy chính thức)

Khóa lúc: **2026-11-04 16:12** · sha256 của configs/experiment.json: `672218494d83579433b35749547d8915c08bcb687309dff8027c9d4babef45fa`
Nội dung đầy đủ của thiết kế nằm ở mục 1–2 trong "Kế hoạch thực nghiệm.md". Tệp này không sửa sau thời điểm khóa.

## Câu hỏi
- RQ1: ảnh hưởng của n và π đến Bias, SD, RMSE của 9 chiến lược; phân tầng có tác dụng ở điều kiện nào?
- RQ2: độ bất ổn nội tại của thuật toán (SD của θ qua các lần rút) liên hệ thế nào với sai số ước lượng và mức lợi của K-Fold so với Holdout?
- RQ3: chiến lược rẻ hơn nào tương đương thực tế với SKF10 (|ΔRMSE| < δ)? RHO10 có tương đương SKF10 ở cùng chi phí không?
- RQ4: F1 trung bình theo fold so với F1 gộp thay đổi Bias và thứ hạng chiến lược đến mức nào?

## Giả thuyết (có thể bị bác bỏ)
- H1: RMSE giảm khi n tăng; khoảng cách Holdout − K-Fold thu hẹp theo n.
- H2: ở cùng K, RMSE(ngẫu nhiên) − RMSE(phân tầng) > 0 và tăng khi số mẫu thiểu số mỗi tập validation giảm.
- H3: thuật toán có SD(θ) lớn hơn (dự kiến DT, KNN) có RMSE lớn hơn và hưởng lợi nhiều hơn từ K-Fold.
- H4: HO80 chỉ tương đương SKF10 khi n đủ lớn; RHO10 tương đương SKF10 ở mọi điều kiện.
- H5: F1 trung bình theo fold có Bias âm hơn F1 gộp; chênh lệch tăng theo K và theo mức mất cân bằng.

## Phân tích
- Độ đo chính: F1-Macro gộp ma trận nhầm lẫn, nhãn cố định {0,1}, zero_division=0. Độ đo phụ: F1 trung bình fold, Accuracy, MCC.
- Sai số e_r = ước lượng − giá trị chuẩn của chính lần rút r (huấn luyện trên toàn D_dev, đánh giá trên Oracle có trọng số về π).
- KTC 95%: bootstrap ghép cặp theo lần rút, B = 2000. Wilcoxon ghép theo lần rút trên e², hiệu chỉnh Holm.
- Tương đương thực tế: KTC 95% của ΔRMSE nằm trong (−0,01; +0,01).
- Ma trận khuyến nghị: chiến lược rẻ nhất (theo số lượt huấn luyện) có cận trên KTC của ΔRMSE so với chiến lược tốt nhất < 0,01.
