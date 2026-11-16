"""Phân tích bổ sung vòng 3 theo phiếu phản biện ngày 12/11/2026 (không chạy lại thí nghiệm).

  .venv/Scripts/python.exe src/analyze_round3.py

r3_reference.csv   NX1: mốc cùng kích thước tính bằng (a) trung bình F1 từng mô hình, (b) F1 của ma trận nhầm lẫn
                   Oracle cộng dồn qua các mô hình; RMSE và tỷ số so với SKF10 theo hai cách.
r3_folds.csv       NX3: thành phần lớp trong từng fold (tập kiểm định và tập huấn luyện) của KF và SKF.
r3_strat_levels.csv NX3: hiệu RMSE ngẫu nhiên − phân tầng của F1 gộp và F1 trung bình theo fold, theo π.
r3_empty.csv       NX3: chênh lệch sai số KF10 − SKF10 theo lần rút, tách theo có/không có fold thiếu mẫu thiểu số.
r3_margin_split.csv NX5: số tổ hợp đạt tương đương / không kém hơn, tách E1 (40), Breast Cancer (4), Mammography (4).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TAB = C.ROOT / "results" / "tables"
KEYS = ["exp", "dataset", "pi_label", "pi", "n", "rep", "model"]
CELL = ["exp", "dataset", "pi_label", "n", "model"]


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def nx1():
    files = sorted((C.ROOT / "results" / "control" / "raw").rglob("rep_*.csv"))
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["pi_label"] = raw.pi_label.astype(str)
    raw["f1"] = C.f1_macro(*raw[C.CM_COLS].to_numpy(float).T)
    truth = raw[(raw.strategy == "TRUTH") & (raw.part == "ALL")][KEYS + ["f1"]].rename(columns={"f1": "theta"})
    val = raw[raw.part == "VAL"].groupby(KEYS + ["strategy"], as_index=False)[C.CM_COLS].sum()
    val["theta_hat"] = C.f1_macro(*val[C.CM_COLS].to_numpy(float).T)
    orc = raw[raw.part == "ORACLE"]
    a = orc.groupby(KEYS + ["strategy"], as_index=False).f1.mean().rename(columns={"f1": "ref_a"})
    b = orc.groupby(KEYS + ["strategy"], as_index=False)[C.CM_COLS].sum()
    b["ref_b"] = C.f1_macro(*b[C.CM_COLS].to_numpy(float).T)
    d = val[KEYS + ["strategy", "theta_hat"]].merge(a, on=KEYS + ["strategy"]).merge(b[KEYS + ["strategy", "ref_b"]], on=KEYS + ["strategy"]).merge(truth, on=KEYS)
    d["ab_diff"] = d.ref_a - d.ref_b
    rows = []
    for key, g in d.groupby(CELL + ["strategy"]):
        rows.append(dict(zip(CELL + ["strategy"], key), rmse_main=rmse(g.theta_hat - g.theta), rmse_a=rmse(g.theta_hat - g.ref_a),
                         rmse_b=rmse(g.theta_hat - g.ref_b), gap_a=float((g.ref_a - g.theta).mean()), gap_b=float((g.ref_b - g.theta).mean()),
                         ab_mean=float(g.ab_diff.mean()), ab_absmax=float(g.ab_diff.abs().max())))
    cs = pd.DataFrame(rows)
    out = []
    for st, g in cs.groupby("strategy"):
        out.append(dict(strategy=st, rmse_main=g.rmse_main.mean(), rmse_a=g.rmse_a.mean(), rmse_b=g.rmse_b.mean(),
                        gap_a=g.gap_a.mean(), gap_b=g.gap_b.mean(), ab_mean=g.ab_mean.mean(), ab_absmax=g.ab_absmax.max()))
    res = pd.DataFrame(out).set_index("strategy")
    for col in ("rmse_main", "rmse_a", "rmse_b"):
        piv = cs.pivot_table(index=CELL, columns="strategy", values=col)
        res["ratio_" + col] = piv.div(piv["SKF10"], axis=0).mean()
    res.to_csv(TAB / "r3_reference.csv", float_format="%.6g")
    print("== NX1: mốc (a) trung bình F1 từng mô hình vs (b) F1 ma trận cộng dồn")
    print(res.round(4).to_string())
    print("Chênh (a) − (b) theo lần rút: trung bình", round(float(d.ab_diff.mean()), 5), "| |max|", round(float(d.ab_diff.abs().max()), 4),
          "| chỉ xét chiến lược nhiều lượt:", round(float(d[~d.strategy.str.startswith("HO")].ab_diff.mean()), 5))


def nx3():
    files = sorted((C.ROOT / "results" / "raw" / "e1").rglob("rep_*.csv"))
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["pi_label"] = raw.pi_label.astype(str)
    kf = raw[raw.strategy.isin(["KF5", "SKF5", "KF10", "SKF10"]) & (raw.model == "LR")].copy()   # thành phần fold không phụ thuộc mô hình
    kf["n_pos"] = (kf.pi * kf.n).round()
    kf["train_pos"] = kf.n_pos - kf.val_pos
    g = kf.groupby(["pi_label", "n", "rep", "strategy"])
    per = g.agg(val_sd=("val_pos", "std"), val_min=("val_pos", "min"), val_max=("val_pos", "max"),
                tr_sd=("train_pos", "std"), tr_mean=("train_pos", "mean"), empty=("val_pos", lambda x: int((x == 0).sum()))).reset_index()
    per["tr_cv"] = per.tr_sd / per.tr_mean
    fold = per.groupby(["pi_label", "n", "strategy"]).agg(val_sd=("val_sd", "mean"), val_range=("val_max", "mean"),
                                                         tr_cv=("tr_cv", "mean"), frac_empty=("empty", lambda x: float((x > 0).mean()))).reset_index()
    fold.to_csv(TAB / "r3_folds.csv", index=False, float_format="%.5g")
    print("\n== NX3 (i): thành phần lớp trong fold (E1, độ lệch chuẩn số mẫu thiểu số trên tập kiểm định; hệ số biến thiên trên tập huấn luyện)")
    print(fold[fold.strategy.isin(["KF10", "SKF10"])].pivot_table(index=["pi_label", "n"], columns="strategy",
                                                                    values=["val_sd", "tr_cv", "frac_empty"]).round(3).to_string())

    s = pd.read_csv(TAB / "summary.csv")
    s = s[(s.exp == "e1") & s.metric.isin(["f1_pooled", "f1_foldavg"])].copy()
    s["pi_label"] = s.pi_label.astype(str)
    rows = []
    for (m, pl, n, mod), g2 in s.groupby(["metric", "pi_label", "n", "model"]):
        r = g2.set_index("strategy").rmse
        rows.append(dict(metric=m, pi_label=pl, n=n, model=mod, d5=r["KF5"] - r["SKF5"], d10=r["KF10"] - r["SKF10"]))
    lv = pd.DataFrame(rows)
    lvs = lv.groupby(["metric", "pi_label"])[["d5", "d10"]].mean()
    lvs.to_csv(TAB / "r3_strat_levels.csv", float_format="%.6g")
    print("\n== NX3 (ii)/(iii): ΔRMSE ngẫu nhiên − phân tầng theo π")
    print(lvs.round(4).to_string())
    print(lv.groupby(["metric", "pi_label", "n"])[["d5", "d10"]].mean().round(4).to_string())

    # (iii-b) theo lần rút: KF10 có/không có fold thiếu mẫu thiểu số
    rt = pd.read_csv(TAB / "replicate_level.csv")
    rt["pi_label"] = rt.pi_label.astype(str)
    rt = rt[(rt.exp == "e1") & rt.strategy.isin(["KF10", "SKF10"])]
    idx = ["pi_label", "n", "rep", "model"]
    kf10 = rt[rt.strategy == "KF10"].set_index(idx)
    sk10 = rt[rt.strategy == "SKF10"].set_index(idx).loc[kf10.index]
    rows = []
    for metric in ("f1_pooled", "f1_foldavg"):
        ek = kf10[metric] - kf10.truth_f1
        es = sk10[metric] - sk10.truth_f1
        he = kf10.n_empty_pos_folds > 0
        for (pl, n), m in he.groupby(level=["pi_label", "n"]):
            if m.sum() < 20:
                continue
            for flag in (True, False):
                sel = m[m == flag].index
                rows.append(dict(metric=metric, pi_label=pl, n=n, has_empty=flag, rows=len(sel),
                                 rmse_kf=float(np.sqrt((ek.loc[sel] ** 2).mean())), rmse_skf=float(np.sqrt((es.loc[sel] ** 2).mean()))))
    emp = pd.DataFrame(rows)
    emp["diff"] = emp.rmse_kf - emp.rmse_skf
    emp.to_csv(TAB / "r3_empty.csv", index=False, float_format="%.6g")
    print("\n== NX3: KF10 so với SKF10 trong các lần rút có / không có fold thiếu mẫu thiểu số")
    print(emp.round(4).to_string())


def nx5():
    mg = pd.read_csv(TAB / "extra_margin.csv")
    mg["grp"] = np.where(mg.exp == "e1", "E1", mg.dataset)
    mg["delta"] = mg.delta.astype(str)
    out = mg.groupby(["A", "delta", "grp"])[["equiv95", "noninf"]].sum().unstack("grp")
    out.to_csv(TAB / "r3_margin_split.csv")
    print("\n== NX5: số tổ hợp đạt tiêu chí so với SKF10 (E1/40, breast_cancer/4, mammography/4)")
    print(out.to_string())


if __name__ == "__main__":
    nx1()
    nx3()
    nx5()
