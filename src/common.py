"""Thành phần dùng chung: dữ liệu, chiến lược chia, mô hình, độ đo, seed."""
import functools
import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]

OPENML = {"magic": "MagicTelescope", "mammography": "mammography", "miniboone": "MiniBooNE"}
DATASET_IDS = {"magic": 1, "mammography": 2, "miniboone": 3, "breast_cancer": 4}
EXP_IDS = {"e1": 1, "e2": 2, "e4": 4, "smoke": 9, "timing": 7, "test": 8}

STRATEGIES = ["HO70-R", "HO70-S", "HO80-R", "HO80-S", "KF5", "SKF5", "KF10", "SKF10", "RHO10"]
FIT_COST = {"HO70-R": 1, "HO70-S": 1, "HO80-R": 1, "HO80-S": 1, "KF5": 5, "SKF5": 5,
            "KF10": 10, "SKF10": 10, "RHO10": 10}
CM_COLS = ["tn", "fp", "fn", "tp"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- dữ liệu
@functools.lru_cache(maxsize=None)
def load_dataset(name, data_home=str(ROOT / "data" / "openml")):
    """Trả về (X float64, y int) với lớp thiểu số luôn mã hóa = 1. Lưu đệm .npz."""
    cache = Path(data_home) / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    if name == "breast_cancer":
        bc = load_breast_cancer()
        X, raw = bc.data.astype(float), bc.target
    else:
        d = fetch_openml(name=OPENML[name], version=1, as_frame=True, parser="auto", data_home=data_home)
        X, raw = d.data.to_numpy(dtype=float), np.asarray(d.target).astype(str)
    labels, counts = np.unique(raw, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"{name}: cần 2 lớp, có {labels}")
    y = (raw == labels[np.argmin(counts)]).astype(np.int64)
    if np.isnan(X).any():
        raise ValueError(f"{name}: có giá trị thiếu")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def pool_oracle_split(y, oracle_frac, seed):
    """Tách cố định Kho lấy mẫu / Oracle, có phân tầng. Trả về chỉ số."""
    idx = np.arange(len(y))
    pool, oracle = train_test_split(idx, test_size=oracle_frac, stratify=y, random_state=seed)
    return np.sort(pool), np.sort(oracle)


def sample_dev(y_pool, n, pi, rng):
    """Rút n mẫu từ Kho với đúng round(n*pi) mẫu thiểu số (không hoàn lại). Trả về chỉ số trong Kho."""
    n1 = int(round(n * pi))
    pos, neg = np.flatnonzero(y_pool == 1), np.flatnonzero(y_pool == 0)
    if n1 > len(pos) or n - n1 > len(neg):
        raise ValueError(f"Kho không đủ mẫu cho n={n}, pi={pi}")
    idx = np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(neg, n - n1, replace=False)])
    return rng.permutation(idx)


def oracle_weights(y_oracle, pi):
    """Trọng số đưa tỷ lệ thiểu số của Oracle về pi."""
    p0 = y_oracle.mean()
    return np.where(y_oracle == 1, pi / p0, (1 - pi) / (1 - p0))


def pi_code(pi):
    return 0 if pi == "natural" else int(round(float(pi) * 1000))


# ----------------------------------------------------------------- seed
def task_seeds(master, exp, dataset, pi, n, rep):
    """Seed tất định cho một lần rút: sampling, model, và 1 seed cho mỗi chiến lược."""
    ss = np.random.SeedSequence(master, spawn_key=(EXP_IDS[exp], DATASET_IDS[dataset], pi_code(pi), int(n), int(rep)))
    s = ss.generate_state(2 + len(STRATEGIES)).astype(np.int64)
    return {"sample": int(s[0]), "model": int(s[1]), **{st: int(s[2 + i]) for i, st in enumerate(STRATEGIES)}}


# ----------------------------------------------------------------- chiến lược
def make_splits(strategy, y, seed):
    """Danh sách (train_idx, val_idx) trên chỉ số 0..len(y)-1."""
    idx = np.arange(len(y))
    if strategy.startswith("HO"):
        test = 0.30 if strategy.startswith("HO70") else 0.20
        strat = y if strategy.endswith("-S") else None
        tr, va = train_test_split(idx, test_size=test, stratify=strat, random_state=seed)
        return [(tr, va)]
    if strategy == "RHO10":
        sp = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=seed)
        return list(sp.split(idx, y))
    k = 5 if strategy.endswith("5") else 10
    if strategy.startswith("SKF"):
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y))
    return list(KFold(k, shuffle=True, random_state=seed).split(idx))


# ----------------------------------------------------------------- mô hình
def make_model(name, seed):
    est = {
        "LR": lambda: LogisticRegression(max_iter=1000),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "DT": lambda: DecisionTreeClassifier(random_state=seed),
        "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    }[name]()
    return make_pipeline(StandardScaler(), est)


# ----------------------------------------------------------------- độ đo
def confusion(y_true, y_pred, w=None):
    """(tn, fp, fn, tp), có thể có trọng số."""
    c = np.bincount(2 * np.asarray(y_true) + np.asarray(y_pred), weights=w, minlength=4)
    return c.astype(float)


def _f1(tp, fp, fn, zero_division=0.0):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), zero_division)


def f1_macro(tn, fp, fn, tp, zero_division=0.0):
    """F1-Macro trên 2 lớp cố định {0,1}; nhận số hoặc mảng."""
    return (_f1(tp, fp, fn, zero_division) + _f1(tn, fn, fp, zero_division)) / 2


def accuracy(tn, fp, fn, tp):
    return (tn + tp) / (tn + fp + fn + tp)


def mcc(tn, fp, fn, tp):
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)


# ----------------------------------------------------------------- một lần rút
def evaluate_replicate(X_dev, y_dev, X_or, y_or, w_or, seeds, models, strategies):
    """Chạy mọi (mô hình, chiến lược) trên một D_dev. Trả về list dict (mỗi lượt train/val một dòng)."""
    rows = []
    for m in models:
        full = make_model(m, seeds["model"]).fit(X_dev, y_dev)
        c = confusion(y_or, full.predict(X_or), w_or)
        rows.append(dict(model=m, strategy="TRUTH", split=0, n_train=len(y_dev), n_val=float(w_or.sum()),
                         val_pos=float(w_or[y_or == 1].sum()), **dict(zip(CM_COLS, c))))
        for st in strategies:
            for k, (tr, va) in enumerate(make_splits(st, y_dev, seeds[st])):
                est = make_model(m, seeds["model"]).fit(X_dev[tr], y_dev[tr])
                c = confusion(y_dev[va], est.predict(X_dev[va]))
                rows.append(dict(model=m, strategy=st, split=k, n_train=len(tr), n_val=len(va),
                                 val_pos=int(y_dev[va].sum()), **dict(zip(CM_COLS, c))))
    return rows


def limit_threads():
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(v, "1")
