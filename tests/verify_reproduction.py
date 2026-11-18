"""Kiểm tra tái lập ở mức từng byte.

  python tests/verify_reproduction.py --manifest          # ghi results/SHA256SUMS.txt cho toàn bộ tệp kết quả
  python tests/verify_reproduction.py --rerun 24          # chạy lại 24 lần rút chọn ngẫu nhiên (E1 + E2) vào thư mục tạm
                                                          # và so khớp SHA-256 với tệp đã công bố

Kết quả kiểm tra được ghi vào results/reproduction_check.txt.
"""
import argparse
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common as C  # noqa: E402
from run_experiment import cell_dir, cells, run_task  # noqa: E402

RESULT_DIRS = ["results/raw", "results/control/raw", "results/tables", "results/timing"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    lines = []
    for d in RESULT_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file():
                lines.append(f"{sha(f)}  {f.relative_to(ROOT).as_posix()}")
    out = ROOT / "results" / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(lines)} dòng vào {out.relative_to(ROOT)}")


def rerun(k, seed):
    cfg = C.load_config(str(ROOT / "configs" / "experiment.json"))
    master, data_home = cfg["master_seed"], str(ROOT / cfg["data_home"])
    pool = []
    for exp in ("e1", "e2"):
        ec = cfg["experiments"][exp]
        for cell in cells(exp, ec, master, data_home):
            for r in range(ec["reps"]):
                pool.append((exp, ec, cell, r))
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    ok, rows = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for exp, ec, cell, r in pick:
            pub = cell_dir(ROOT / "results", exp, cell) / f"rep_{r:03d}.csv"
            new = cell_dir(Path(tmp), exp, cell) / f"rep_{r:03d}.csv"
            t0 = time.time()
            run_task(exp, ec, master, data_home, cfg["models"], cfg["strategies"], cell, r, new)
            same = sha(pub) == sha(new)
            ok += same
            rows.append(f"{'KHỚP ' if same else 'KHÁC '} {pub.relative_to(ROOT).as_posix()}  ({time.time() - t0:.0f} s)")
            print(rows[-1], flush=True)
    summary = f"{ok}/{len(pick)} tệp chạy lại trùng khớp từng byte (SHA-256) với tệp đã công bố; seed chọn mẫu = {seed}."
    print(summary)
    (ROOT / "results" / "reproduction_check.txt").write_text(summary + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return ok == len(pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--rerun", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    if a.manifest:
        manifest()
    if a.rerun:
        sys.exit(0 if rerun(a.rerun, a.seed) else 1)
