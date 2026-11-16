"""6 bài kiểm thử trước khi chạy chính thức. Chạy: .venv/Scripts/python.exe tests/run_checks.py"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
DH = str(ROOT / "data" / "openml")
results = []


def check(name):
    def deco(fn):
        try:
            msg = fn()
            results.append((name, True, msg or ""))
        except AssertionError as e:
            results.append((name, False, str(e)))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


@check("T1 Không rò rỉ: Oracle ∩ D_dev = ∅, scaler nằm trong Pipeline")
def t1():
    X, y = C.load_dataset("magic", DH)
    pool, oracle = C.pool_oracle_split(y, 0.40, 20261103)
    assert len(np.intersect1d(pool, oracle)) == 0 and len(pool) + len(oracle) == len(y)
    for r in range(200):
        s = C.task_seeds(20261103, "test", "magic", 0.10, 500, r)
        dev = pool[C.sample_dev(y[pool], 500, 0.10, np.random.default_rng(s["sample"]))]
        assert len(np.intersect1d(dev, oracle)) == 0 and len(set(dev)) == 500
        assert y[dev].sum() == 50
    pipe = C.make_model("LR", 1)
    assert pipe.steps[0][0] == "standardscaler"
    # scaler chỉ thấy phần train: khi fit trên tập con, mean_ phải khác mean toàn bộ
    sub = np.random.default_rng(0).choice(len(y), 100, replace=False)
    pipe.fit(X[sub], y[sub])
    assert not np.allclose(pipe.steps[0][1].mean_, X.mean(0))
    return "200 lần rút đều rời nhau; scaler học riêng từng lượt"


@check("T2 F1 từ ma trận nhầm lẫn khớp sklearn (có/không trọng số)")
def t2():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(1000):
        n = int(rng.integers(5, 300))
        y = rng.integers(0, 2, n)
        p = rng.integers(0, 2, n)
        if rng.random() < 0.3:      # tạo trường hợp thiếu lớp / dự đoán một lớp
            y[:] = y[0] if rng.random() < 0.5 else y
            p[:] = p[0]
        w = rng.random(n) + 0.1 if rng.random() < 0.5 else None
        tn, fp, fn, tp = C.confusion(y, p, w)
        ours = float(C.f1_macro(tn, fp, fn, tp))
        ref = f1_score(y, p, labels=[0, 1], average="macro", zero_division=0, sample_weight=w)
        worst = max(worst, abs(ours - ref))
    assert worst < 1e-12, f"sai khác lớn nhất {worst:.2e}"
    return f"sai khác lớn nhất {worst:.1e}"


@check("T3 Oracle có trọng số: tỷ lệ hiệu dụng = π; F1 có trọng số ≈ F1 trên Oracle cắt bớt")
def t3():
    X, y = C.load_dataset("magic", DH)
    pool, oracle = C.pool_oracle_split(y, 0.40, 20261103)
    yo = y[oracle]
    for pi in [0.10, 0.03]:
        w = C.oracle_weights(yo, pi)
        assert abs(w[yo == 1].sum() / w.sum() - pi) < 1e-12
    assert np.allclose(C.oracle_weights(yo, yo.mean()), 1.0)
    # so với Oracle cắt bớt (lặp nhiều lần) ở π = 0.10
    s = C.task_seeds(20261103, "test", "magic", 0.10, 1000, 0)
    dev = pool[C.sample_dev(y[pool], 1000, 0.10, np.random.default_rng(s["sample"]))]
    Xo = X[oracle]
    model = C.make_model("DT", 1).fit(X[dev], y[dev])
    pred = model.predict(Xo)
    w = C.oracle_weights(yo, 0.10)
    f_w = float(C.f1_macro(*C.confusion(yo, pred, w)))
    pos, neg = np.flatnonzero(yo == 1), np.flatnonzero(yo == 0)
    # Oracle cắt bớt về π = 0.10: giữ toàn bộ mẫu âm, rút bớt mẫu dương
    n_keep_pos = int(round(len(neg) * 0.10 / 0.90))
    assert n_keep_pos <= len(pos)
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(300):
        keep = np.concatenate([rng.choice(pos, n_keep_pos, replace=False), neg])
        vals.append(float(C.f1_macro(*C.confusion(yo[keep], pred[keep]))))
    diff = abs(np.mean(vals) - f_w)
    assert diff < 0.005, f"chênh lệch {diff:.4f}"
    return f"F1 có trọng số {f_w:.4f}; trung bình Oracle cắt bớt {np.mean(vals):.4f}"


@check("T4 Phân tầng đúng: SKF lệch ≤ 1 mẫu thiểu số/fold; KF thì dao động")
def t4():
    rng = np.random.default_rng(5)
    kf_spread = 0
    for r in range(200):
        y = np.r_[np.ones(20, int), np.zeros(980, int)]
        rng.shuffle(y)
        for st in ["SKF5", "SKF10", "HO70-S", "HO80-S", "RHO10"]:
            counts = [int(y[va].sum()) for _, va in C.make_splits(st, y, r)]
            if st.startswith(("SKF", "RHO")):
                assert max(counts) - min(counts) <= 1, f"{st}: {counts}"
        c = [int(y[va].sum()) for _, va in C.make_splits("KF10", y, r)]
        kf_spread = max(kf_spread, max(c) - min(c))
        for st in C.STRATEGIES:   # mọi lượt phải là phân hoạch hợp lệ
            for tr, va in C.make_splits(st, y, r):
                assert len(np.intersect1d(tr, va)) == 0
    assert kf_spread >= 3, "KF ngẫu nhiên lẽ ra phải dao động"
    return f"KF10 ngẫu nhiên lệch tới {kf_spread} mẫu/fold; mọi chiến lược phân hoạch rời nhau"


@check("T5 Tái lập: cùng seed → kết quả y hệt; dừng giữa chừng rồi chạy tiếp → y hệt")
def t5():
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    script = str(ROOT / "src" / "run_experiment.py")
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        base = [py, script, "--exp", "e2", "--reps", "3", "--workers", "2"]
        for out in (a, b):
            r = subprocess.run(base + ["--out", str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert r.returncode == 0, r.stderr[-500:]
        files = sorted(a.rglob("rep_*.csv"))
        assert len(files) == 6, f"cần 6 file, có {len(files)}"
        for f in files:
            g = b / f.relative_to(a)
            assert f.read_bytes() == g.read_bytes(), f"khác nhau: {f.name}"
        # mô phỏng bị ngắt: xóa 2 file rồi chạy lại
        for f in files[:2]:
            (b / f.relative_to(a)).unlink()
        r = subprocess.run(base + ["--out", str(b)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stderr[-500:]
        for f in files:
            assert f.read_bytes() == (b / f.relative_to(a)).read_bytes(), f"sau khi chạy tiếp: {f.name}"
    return "6 file giống hệt từng byte, kể cả sau khi chạy tiếp"


@check("T6 Hợp lý chiều: DT ở n=200, Holdout 70/30 chệch âm hơn SKF10 (chỉ kiểm tra hợp lý)")
def t6():
    X, y = C.load_dataset("magic", DH)
    pool, oracle = C.pool_oracle_split(y, 0.40, 20261103)
    w = np.ones(len(oracle))
    b = {"HO70-S": [], "SKF10": []}
    for r in range(40):
        s = C.task_seeds(20261103, "test", "magic", "natural", 200, r)
        nat = float(y[pool].mean())
        dev = pool[C.sample_dev(y[pool], 200, nat, np.random.default_rng(s["sample"]))]
        rows = pd.DataFrame(C.evaluate_replicate(X[dev], y[dev], X[oracle], y[oracle], w, s, ["DT"], ["HO70-S", "SKF10"]))
        truth = float(C.f1_macro(*rows[rows.strategy == "TRUTH"][C.CM_COLS].iloc[0]))
        for st in b:
            g = rows[rows.strategy == st]
            est = float(C.f1_macro(*g[C.CM_COLS].sum()))
            b[st].append(est - truth)
    m70, m10 = np.mean(b["HO70-S"]), np.mean(b["SKF10"])
    assert m70 < m10, f"HO70 {m70:+.4f} không âm hơn SKF10 {m10:+.4f}"
    return f"Bias trung bình: HO70-S {m70:+.4f}, SKF10 {m10:+.4f}"


if __name__ == "__main__":
    ok = True
    for name, passed, msg in results:
        print(("ĐẠT  " if passed else "LỖI  ") + name + (f"\n       → {msg}" if msg else ""))
        ok &= passed
    print("\nKẾT QUẢ:", "TẤT CẢ ĐẠT" if ok else "CÓ BÀI KHÔNG ĐẠT")
    sys.exit(0 if ok else 1)
