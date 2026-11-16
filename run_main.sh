#!/bin/bash
# Chạy chính thức E1 rồi E2. Chạy lại đúng lệnh này nếu bị ngắt: các lần rút đã xong được bỏ qua.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
$PY src/run_experiment.py --exp e1 --workers 3 && $PY src/run_experiment.py --exp e2 --workers 3
echo "TOÀN BỘ XONG: $?"
