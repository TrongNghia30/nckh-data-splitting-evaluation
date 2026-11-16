"""Phân tích kết quả thô theo đúng kế hoạch đã khóa trước (Kế hoạch thực nghiệm.md, mục 2).

  .venv/Scripts/python.exe src/analyze.py                      # phân tích results/raw
  .venv/Scripts/python.exe src/analyze.py --root results/smoke # phân tích kết quả chạy thử
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KEYS = ["exp", "dataset", "pi_label", "pi", "n", "rep", "model"]
CELL = ["exp", "dataset", "pi_label", "pi", "n", "model"]
KFOLD = ["KF5", "SKF5", "KF10", "SKF10"]


# ------------------------------------------------------------------ nạp & tính theo lần rút
def load_raw(root):
    files = sorted((root / "raw").rglob("rep_*.csv"))
    if not files:
        raise SystemExit(f"Không có dữ liệu trong {root / 'raw'}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def replicate_table(raw):
    """Một dòng cho mỗi (lần rút, mô hình, chiến lược): ước lượng và giá trị chuẩn của 4 độ đo."""
    cm = raw[C.CM_COLS].to_numpy(float)
    raw = raw.assign(f1=C.f1_macro(*cm.T), empty_pos=(raw.val_pos == 0) & (raw.strategy != "TRUTH"))
    truth = raw[raw.strategy == "TRUTH"].copy()
    est = raw[raw.strategy != "TRUTH"]
    pooled = est.groupby(KEYS + ["strategy"], as_index=False)[C.CM_COLS].sum()
    c = pooled[C.CM_COLS].to_numpy(float)
    pooled["f1_pooled"] = C.f1_macro(*c.T)
    pooled["acc_pooled"] = C.accuracy(*c.T)
    pooled["mcc_pooled"] = C.mcc(*c.T)
    extra = est.groupby(KEYS + ["strategy"], as_index=False).agg(
        f1_foldavg=("f1", "mean"), n_empty_pos_folds=("empty_pos", "sum"), n_splits=("split", "count"))
    out = pooled[KEYS + ["strategy", "f1_pooled", "acc_pooled", "mcc_pooled"]].merge(extra, on=KEYS + ["strategy"])
    t = truth[KEYS].copy()
    tc = truth[C.CM_COLS].to_numpy(float)
    t["truth_f1"], t["truth_acc"], t["truth_mcc"] = C.f1_macro(*tc.T), C.accuracy(*tc.T), C.mcc(*tc.T)
    return out.merge(t, on=KEYS)


# ------------------------------------------------------------------ thống kê
def rmse(e, axis=0):
    return np.sqrt(np.mean(np.square(e), axis=axis))


def boot_idx(R, B, rng):
    return rng.integers(0, R, size=(B, R))


def ci(x, alpha=0.05):
    return np.quantile(x, [alpha / 2, 1 - alpha / 2])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wilcoxon_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) < 10:
        return 1.0, 0.0
    p = wilcoxon(d).pvalue
    r = rankdata(np.abs(d))
    rb = (r[d > 0].sum() - r[d < 0].sum()) / r.sum()      # rank-biserial: >0 nghĩa là A sai nhiều hơn B
    return float(p), float(rb)


# ------------------------------------------------------------------ các bảng
def summary_table(rt, metric, B, rng, alpha):
    rows = []
    for key, g in rt.groupby(CELL):
        piv_e = g.pivot(index="rep", columns="strategy", values=metric)
        truth = g.drop_duplicates("rep").set_index("rep").loc[piv_e.index, "truth_" + metric.split("_")[0]]
        e = piv_e.sub(truth, axis=0)
        R = len(e)
        bi = boot_idx(R, B, rng)
        emp = g.pivot(index="rep", columns="strategy", values="n_empty_pos_folds")
        for st in e.columns:
            ev = e[st].to_numpy()
            est = piv_e[st].to_numpy()
            rb = rmse(ev[bi], axis=1)
            lo, hi = ci(rb, alpha)
            rows.append(dict(zip(CELL, key), metric=metric, strategy=st, R=R, theta_bar=est.mean(), theta_true_mean=truth.mean(),
                             theta_true_sd=truth.std(ddof=1), bias=ev.mean(), sd=est.std(ddof=1), sd_err=ev.std(ddof=1),
                             rmse=rmse(ev), rmse_lo=lo, rmse_hi=hi, fit_cost=C.FIT_COST[st],
                             frac_reps_empty_minority_fold=float((emp[st] > 0).mean()),
                             corr_est_truth=float(np.corrcoef(est, truth.to_numpy())[0, 1]) if est.std() > 0 and truth.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def pair_table(rt, metric, pairs, B, rng, alpha, delta, with_tests=True):
    """Với mỗi ô và mỗi cặp (A, ref): ΔRMSE = RMSE(A) − RMSE(ref), KTC bootstrap ghép cặp, Wilcoxon theo lần rút."""
    rows = []
    for key, g in rt.groupby(CELL):
        piv = g.pivot(index="rep", columns="strategy", values=metric)
        truth = g.drop_duplicates("rep").set_index("rep").loc[piv.index, "truth_" + metric.split("_")[0]]
        e = piv.sub(truth, axis=0)
        bi = boot_idx(len(e), B, rng)
        for a, ref in pairs:
            if a not in e or ref not in e:
                continue
            ea, er = e[a].to_numpy(), e[ref].to_numpy()
            d = rmse(ea[bi], axis=1) - rmse(er[bi], axis=1)
            lo, hi = ci(d, alpha)
            p, rbis = wilcoxon_paired(ea ** 2, er ** 2) if with_tests else (np.nan, np.nan)
            rows.append(dict(zip(CELL, key), metric=metric, A=a, ref=ref, delta_rmse=rmse(ea) - rmse(er), lo=lo, hi=hi,
                             equivalent=bool(lo > -delta and hi < delta), p_wilcoxon=p, rank_biserial=rbis))
    df = pd.DataFrame(rows)
    if with_tests and len(df):
        df["p_holm"] = holm(df.p_wilcoxon.to_numpy())
    return df


def f1def_table(rt, B, rng, alpha):
    """H5: F1 trung bình theo fold so với F1 gộp (cùng chiến lược, cùng lần rút)."""
    rows = []
    sub = rt[rt.strategy.isin(KFOLD + ["RHO10"])]
    for key, g in sub.groupby(CELL + ["strategy"]):
        diff = (g.f1_foldavg - g.f1_pooled).to_numpy()
        bi = boot_idx(len(diff), B, rng)
        lo, hi = ci(diff[bi].mean(axis=1), alpha)
        e_avg, e_pool = g.f1_foldavg - g.truth_f1, g.f1_pooled - g.truth_f1
        rows.append(dict(zip(CELL + ["strategy"], key), bias_foldavg=e_avg.mean(), bias_pooled=e_pool.mean(),
                         diff_mean=diff.mean(), diff_lo=lo, diff_hi=hi,
                         rmse_foldavg=rmse(e_avg.to_numpy()), rmse_pooled=rmse(e_pool.to_numpy())))
    return pd.DataFrame(rows)


def recommend(rt, metric, B, rng, alpha, delta):
    """Quy tắc đã khóa: chọn chiến lược RẺ NHẤT có cận trên KTC của ΔRMSE so với chiến lược tốt nhất < δ."""
    rows = []
    for key, g in rt.groupby(CELL):
        piv = g.pivot(index="rep", columns="strategy", values=metric)
        truth = g.drop_duplicates("rep").set_index("rep").loc[piv.index, "truth_" + metric.split("_")[0]]
        e = piv.sub(truth, axis=0)
        r = rmse(e.to_numpy())
        best = e.columns[int(np.argmin(r))]
        bi = boot_idx(len(e), B, rng)
        rb = rmse(e[best].to_numpy()[bi], axis=1)
        ok = []
        for st in e.columns:
            hi = ci(rmse(e[st].to_numpy()[bi], axis=1) - rb, alpha)[1] if st != best else 0.0
            if hi < delta:
                ok.append((C.FIT_COST[st], float(r[list(e.columns).index(st)]), st))
        chosen = sorted(ok)[0][2]
        rows.append(dict(zip(CELL, key), best_strategy=best, best_rmse=float(r.min()), recommended=chosen,
                         recommended_rmse=float(r[list(e.columns).index(chosen)]),
                         recommended_cost=C.FIT_COST[chosen], n_acceptable=len(ok)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "experiment.json"))
    ap.add_argument("--root", help="thư mục kết quả (mặc định: results)")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    an = cfg["analysis"]
    root = Path(a.root) if a.root else C.ROOT / cfg["results_dir"]
    if not root.is_absolute():
        root = C.ROOT / root
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["master_seed"] + 1)
    B, alpha, delta, metric, ref = an["bootstrap_B"], an["alpha"], an["delta"], an["primary_metric"], an["reference_strategy"]

    raw = load_raw(root)
    rt = replicate_table(raw)
    rt.to_csv(out / "replicate_level.csv", index=False, float_format="%.8g")
    print(f"Đã nạp {raw.exp.nunique()} thí nghiệm, {rt.groupby(CELL).ngroups} (ô × mô hình), "
          f"{rt.rep.nunique()} lần rút/ô tối đa")

    summ = pd.concat([summary_table(rt, m, B, rng, alpha) for m in [metric, "f1_foldavg", "acc_pooled", "mcc_pooled"]])
    summ.to_csv(out / "summary.csv", index=False, float_format="%.6g")

    vs_ref = pair_table(rt, metric, [(s, ref) for s in C.STRATEGIES if s != ref], B, rng, alpha, delta)
    vs_ref.to_csv(out / "pairwise_vs_ref.csv", index=False, float_format="%.6g")

    rs_pairs = [("HO70-R", "HO70-S"), ("HO80-R", "HO80-S"), ("KF5", "SKF5"), ("KF10", "SKF10")]
    rs = pair_table(rt, metric, rs_pairs, B, rng, alpha, delta)
    rs.to_csv(out / "random_vs_stratified.csv", index=False, float_format="%.6g")

    eq = pair_table(rt, metric, [("RHO10", ref), ("HO80-S", ref), ("SKF5", ref)], B, rng, alpha, delta, with_tests=False)
    eq.to_csv(out / "equivalence.csv", index=False, float_format="%.6g")

    f1d = f1def_table(rt, B, rng, alpha)
    f1d.to_csv(out / "f1_definition_effect.csv", index=False, float_format="%.6g")

    rec = recommend(rt, metric, B, rng, alpha, delta)
    rec.to_csv(out / "recommendation.csv", index=False, float_format="%.6g")

    # ---- tóm tắt nhanh ra màn hình
    s = summ[summ.metric == metric]
    print("\nRMSE trung bình qua các ô, theo chiến lược:")
    print(s.groupby("strategy").rmse.mean().reindex(C.STRATEGIES).round(4).to_string())
    print("\nSai khác RMSE ngẫu nhiên − phân tầng (dương = phân tầng tốt hơn), trung bình qua ô:")
    print(rs.groupby(["A", "ref"]).delta_rmse.mean().round(4).to_string())
    print("\nSố ô mà chiến lược tương đương SKF10 theo ngưỡng δ:")
    print(eq.groupby("A").equivalent.agg(["sum", "count"]).to_string())
    print("\nChiến lược được khuyến nghị (số ô):")
    print(rec.recommended.value_counts().to_string())
    print(f"\nĐã ghi các bảng vào {out}")


if __name__ == "__main__":
    main()
