"""Chạy E1/E2/E4. Mỗi lần rút lưu 1 file CSV; chạy lại sẽ bỏ qua các lần rút đã xong.

Ví dụ:
  .venv/Scripts/python.exe src/run_experiment.py --exp e1
  .venv/Scripts/python.exe src/run_experiment.py --exp e1 --reps 5 --out results/smoke
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def cells(exp_name, ec, master, data_home):
    """Liệt kê các ô điều kiện (dataset, pi_label, pi_eff, n)."""
    out = []
    if ec["kind"] == "controlled":
        X, y = C.load_dataset(ec["dataset"], data_home)
        pool, _ = C.pool_oracle_split(y, ec["oracle_frac"], master)
        nat = float(y[pool].mean())
        nat_only = set(ec.get("natural_only_n", []))
        for pi in ec["pi_grid"]:
            pi_eff = nat if pi == "natural" else float(pi)
            for n in ec["n_grid"]:
                if n in nat_only and pi != "natural":
                    continue
                if round(n * pi_eff) < ec["min_minority"]:
                    continue
                out.append((ec["dataset"], pi, pi_eff, n))
    else:
        for ds in ec["datasets"]:
            X, y = C.load_dataset(ds, data_home)
            n_dev = len(y) - int(np.ceil(ec["oracle_frac"] * len(y)))
            out.append((ds, "natural", float(y.mean()), n_dev))
    return out


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
    assert len(np.intersect1d(dev, oracle)) == 0, "Rò rỉ: D_dev giao Oracle"
    t0 = time.perf_counter()
    rows = C.evaluate_replicate(X[dev], y[dev], X[oracle], y[oracle], w, seeds, models, strategies)
    df = pd.DataFrame(rows)
    df.insert(0, "rep", rep)
    df.insert(0, "n", n)
    df.insert(0, "pi", pi_eff)
    df.insert(0, "pi_label", str(pi))
    df.insert(0, "dataset", dataset)
    df.insert(0, "exp", exp_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, out_file)
    return cell, rep, time.perf_counter() - t0


def cell_dir(root, exp_name, cell):
    dataset, pi, _, n = cell
    return root / "raw" / exp_name / f"{dataset}_pi{C.pi_code(pi)}_n{n}"


def write_env(root):
    import joblib, scipy, sklearn
    info = dict(time=datetime.now().isoformat(timespec="seconds"), python=sys.version, platform=platform.platform(),
                processor=platform.processor(), cpu_count=os.cpu_count(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__)
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["e1", "e2", "e4"])
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "experiment.json"))
    ap.add_argument("--workers", type=int)
    ap.add_argument("--reps", type=int, help="ghi đè số lần rút (chạy thử)")
    ap.add_argument("--out", help="thư mục kết quả (mặc định: results)")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    ec = cfg["experiments"][a.exp]
    master = cfg["master_seed"]
    data_home = str(C.ROOT / cfg["data_home"])
    root = Path(a.out) if a.out else C.ROOT / cfg["results_dir"]
    if not root.is_absolute():
        root = C.ROOT / root
    reps = a.reps or ec["reps"]
    workers = a.workers or cfg["workers"]
    write_env(root)

    cl = cells(a.exp, ec, master, data_home)
    tasks = [(c, r, cell_dir(root, a.exp, c) / f"rep_{r:03d}.csv") for c in cl for r in range(reps)]
    todo = [t for t in tasks if not t[2].exists()]
    log_path = C.ROOT / "logs" / f"{a.exp}{'_' + root.name if a.out else ''}.log"
    log_path.parent.mkdir(exist_ok=True)

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"BẮT ĐẦU {a.exp}: {len(cl)} ô × {reps} lần rút = {len(tasks)}; còn lại {len(todo)}; workers={workers}")
    for c in cl:
        log(f"  ô: dataset={c[0]} pi={c[1]} (hiệu dụng {c[2]:.4f}) n={c[3]}")
    if not todo:
        log("Không còn gì để chạy.")
        return
    t_start, done, per_cell = time.time(), 0, {}
    jobs = (delayed(run_task)(a.exp, ec, master, data_home, cfg["models"], cfg["strategies"], c, r, f) for c, r, f in todo)
    for cell, rep, sec in Parallel(n_jobs=workers, return_as="generator_unordered")(jobs):
        done += 1
        per_cell.setdefault(cell, []).append(sec)
        el = time.time() - t_start
        eta = el / done * (len(todo) - done)
        if done % max(1, len(todo) // 50) == 0 or done == len(todo):
            log(f"{done}/{len(todo)} xong | đã chạy {el/60:.1f} phút | còn khoảng {eta/60:.1f} phút")
    log("THỜI GIAN TRUNG BÌNH MỖI LẦN RÚT (giây, 1 tiến trình):")
    for c, v in per_cell.items():
        log(f"  {c[0]} pi={c[1]} n={c[3]}: {np.mean(v):.1f}s")
    log(f"HOÀN TẤT {a.exp} sau {(time.time()-t_start)/60:.1f} phút")


if __name__ == "__main__":
    main()
