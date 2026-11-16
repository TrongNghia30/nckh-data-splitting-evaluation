"""E3 – Đo chi phí tính toán (đơn luồng). Chạy khi máy rảnh, KHÔNG chạy song song với E1/E2.

  .venv/Scripts/python.exe src/run_timing.py
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "1"

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "experiment.json"))
    ap.add_argument("--repeats", type=int)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = C.load_config(a.config)
    tc = cfg["timing"]
    reps = a.repeats or tc["repeats"]
    master = cfg["master_seed"]
    X, y = C.load_dataset(tc["dataset"], str(C.ROOT / cfg["data_home"]))
    pool, oracle = C.pool_oracle_split(y, 0.40, master)
    rows = []
    for n in tc["n_grid"]:
        s = C.task_seeds(master, "timing", tc["dataset"], "natural", n, 0)
        n_eff = min(n, len(pool))
        dev = pool[C.sample_dev(y[pool], n_eff, float(y[pool].mean()), np.random.default_rng(s["sample"]))]
        Xd, yd = X[dev], y[dev]
        for m in cfg["models"]:
            for st in cfg["strategies"]:
                splits = C.make_splits(st, yd, s[st])
                for r in range(reps):
                    t0 = time.perf_counter()
                    for tr, va in splits:
                        est = C.make_model(m, s["model"]).fit(Xd[tr], yd[tr])
                        est.predict(Xd[va])
                    rows.append(dict(n=n_eff, model=m, strategy=st, rep=r, seconds=time.perf_counter() - t0,
                                     n_fits=len(splits)))
        print(f"n={n_eff} xong", flush=True)
    out = C.ROOT / cfg["results_dir"] / "timing"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "timing_raw.csv", index=False)
    med = df.groupby(["n", "model", "strategy"], as_index=False).seconds.median()
    base = med[med.strategy == "HO80-R"].set_index(["n", "model"]).seconds
    med["ratio_vs_HO80"] = [r.seconds / base[(r.n, r.model)] for r in med.itertuples()]
    med.to_csv(out / "timing_summary.csv", index=False)
    print(med.pivot_table(index=["n", "model"], columns="strategy", values="ratio_vs_HO80").round(1).to_string())


if __name__ == "__main__":
    main()
