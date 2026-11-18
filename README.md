# Mã nguồn, cấu hình và kết quả tái lập bài báo

**Đánh giá thực nghiệm độ chệch, phương sai và chi phí tính toán của các chiến lược phân chia dữ liệu trên các họ thuật toán phân lớp** (SSRC-II'2026).

Kho này chứa toàn bộ mã nguồn, cấu hình, hạt giống, tệp khóa trước (đăng ký giả thuyết), kết quả thô của từng lần rút và các bảng/hình trong bài.

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
```

Lần chạy đầu tiên tự tải dữ liệu: MAGIC Gamma Telescope (OpenML ID 1120), Mammography (OpenML ID 310); Breast Cancer Wisconsin lấy từ bản đóng gói trong scikit-learn.

## Chạy lại

| Bước | Lệnh | Kết quả |
|---|---|---|
| Kiểm thử (6 bài) | `python tests/run_checks.py` | in kết quả T1–T6 |
| E1, E2 | `bash run_main.sh` | `results/raw/e1`, `results/raw/e2` (mỗi lần rút một tệp CSV) |
| E3 – đo thời gian | `python src/run_timing.py` | `results/timing/` |
| Đối chứng cùng kích thước huấn luyện | `python src/run_control.py --exp e1 --reps 50` | `results/control/raw/e1` |
| Phân tích chính (khóa trước) | `python src/analyze.py` | `results/tables/*.csv` |
| Phân tích bổ sung theo phản biện | `python src/analyze_extra.py`, `python src/analyze_control.py`, `python src/analyze_round3.py` | `results/tables/extra_*`, `control_*`, `r3_*` |
| Hình trong bài | `python src/plots_ssrc.py` | `results/figures_ssrc/` |
| Kiểm tra tái lập từng byte | `python tests/verify_reproduction.py --rerun 24` | `results/reproduction_check.txt` (so khớp với `results/SHA256SUMS.txt`) |

## Ghi chú tái lập

- Cấu hình: `configs/experiment.json`; hạt giống gốc `20261103`. Hạt giống của mỗi lần rút sinh bằng `numpy.random.SeedSequence` từ (thí nghiệm, bộ dữ liệu, π, n, lần rút); chạy lại cho kết quả giống từng byte và có thể chạy tiếp khi bị ngắt.
- Tệp khóa trước: `configs/preregistration.md` (kèm SHA-256 của tệp cấu hình tại thời điểm khóa, 04/11/2026). Đây là khóa nội bộ, không phải đăng ký trên nền tảng bên ngoài.
- `results/SHA256SUMS.txt` liệt kê SHA-256 của toàn bộ tệp kết quả. Lần kiểm tra ngày 18/11/2026: chạy lại 24 lần rút chọn ngẫu nhiên (E1 và E2), 24/24 tệp trùng khớp từng byte trên cùng môi trường phần mềm. Trên phần cứng hoặc phiên bản thư viện khác, kết quả số có thể sai khác nhỏ.
- Phiên bản thư viện: `requirements.txt`; môi trường đã chạy: `results/env.json`.
- Phần cứng đo thời gian (E3): Intel Core i5-1135G7 (4 nhân, 8 luồng, 2,40 GHz), RAM 8 GB DDR4-3200, Windows 11 Home (build 26300); mỗi mô hình chạy đơn luồng.
