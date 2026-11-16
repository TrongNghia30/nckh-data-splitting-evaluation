"""Phân tích đối chứng theo yêu cầu phản biện (E5).

Với cùng tập phát triển, cùng hạt giống và cùng các lượt chia như E1, mỗi mô hình được huấn luyện
trong từng lượt còn được đánh giá thêm trên Oracle. Từ đó có:
  * theta_tilde: hiệu năng thực (trên Oracle) của chính các mô hình mà chiến lược đã huấn luyện,
    tức mốc đối chứng có cùng kích thước tập huấn luyện với chiến lược;
  * TRUTH trên hai nửa Oracle (A, B) để kiểm tra độ nhạy của kết luận theo Oracle.
Đồng thời ghi lại ma trận nhầm lẫn trên tập kiểm định để đối chiếu từng byte với kết quả E1.

  .venv/Scripts/python.exe src/run_control.py --exp e1 --workers 6
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from run_experiment import cells, cell_dir  # noqa: E402


def oracle_halves(y_or, seed):
    """Chia Oracle thành hai nửa cố định, có phân tầng."""
    idx = np.arange(len(y_or))
    a, b = C.train_test_split(idx, test_size=0.5, stratify=y_or, random_state=seed)
    return np.sort(a), np.sort(b)


def run_task(exp_name, ec, master, data_home, models, strategies, cell, rep, out_file):
    warnings.filterwarnings("ignore")
    dataset, pi, pi_eff, n = cell
    seeds = C.task_seeds(master, exp_name, dataset, pi, n, rep)
    X, y = C.load_dataset(dataset, data_home)
    if ec["kind"] == "controlled":
        pool, oracle = C.pool_oracle_split(y, ec["oracle_frac"], master)
        rng = np.random.default_rng(seeds["sample"])
        dev = pool[C.sample_dev(y[pool], n, pi_eff, rng)]
        w = np.ones(len(oracle)) if pi == "natural" else C.oracle_weights(y[oracle], pi_eff)
    else:
        idx = np.arange(len(y))
        dev, oracle = C.train_test_split(idx, test_size=ec["oracle_frac"], stratify=y, random_state=seeds["sample"])
        w = np.ones(len(oracle))
    Xd, yd, Xo, yo = X[dev], y[dev], X[oracle], y[oracle]
    ha, hb = oracle_halves(yo, master + 7)
    t0 = time.perf_counter()
    rows = []
    for m in models:
        full = C.make_model(m, seeds["model"]).fit(Xd, yd)
        po = full.predict(Xo)
        for part, sel in [("ALL", slice(None)), ("A", ha), ("B", hb)]:
            c = C.confusion(yo[sel], po[sel], w[sel])
            rows.append(dict(model=m, strategy="TRUTH", split=0, part=part, **dict(zip(C.CM_COLS, c))))
        for st in strategies:
            for k, (tr, va) in enumerate(C.make_splits(st, yd, seeds[st])):
                est = C.make_model(m, seeds["model"]).fit(Xd[tr], yd[tr])
                cv = C.confusion(yd[va], est.predict(Xd[va]))
                co = C.confusion(yo, est.predict(Xo), w)
                rows.append(dict(model=m, strategy=st, split=k, part="VAL", **dict(zip(C.CM_COLS, cv))))
                rows.append(dict(model=m, strategy=st, split=k, part="ORACLE", **dict(zip(C.CM_COLS, co))))
    df = pd.DataFrame(rows)
    for col, val in [("rep", rep), ("n", n), ("pi", pi_eff), ("pi_label", str(pi)), ("dataset", dataset), ("exp", exp_name)]:
        df.insert(0, col, val)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, out_file)
    return cell, rep, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1", choices=["e1", "e2"])
    ap.add_argument("--reps", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = C.load_config(str(C.ROOT / "configs" / "experiment.json"))
    ec = cfg["experiments"][a.exp]
    master, data_home = cfg["master_seed"], str(C.ROOT / cfg["data_home"])
    reps = a.reps or ec["reps"]
    root = C.ROOT / "results" / "control"
    todo = []
    for cell in cells(a.exp, ec, master, data_home):
        for r in range(reps):
            f = cell_dir(root, a.exp, cell) / f"rep_{r:03d}.csv"
            if not f.exists():
                todo.append((cell, r, f))
    log = open(C.ROOT / "logs" / f"control_{a.exp}.log", "a", encoding="utf-8")

    def say(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"Bắt đầu đối chứng {a.exp}: {len(todo)} lần rút cần chạy, {a.workers} tiến trình")
    t0 = time.time()
    done = 0
    gen = Parallel(n_jobs=a.workers, return_as="generator_unordered")(
        delayed(run_task)(a.exp, ec, master, data_home, cfg["models"], cfg["strategies"], c, r, f) for c, r, f in todo)
    for _ in gen:
        done += 1
        if done % 20 == 0 or done == len(todo):
            el = (time.time() - t0) / 60
            say(f"{done}/{len(todo)} xong | {el:.1f} phút | còn khoảng {el / done * (len(todo) - done):.1f} phút")
    say(f"HOÀN TẤT đối chứng {a.exp}")


if __name__ == "__main__":
    main()
