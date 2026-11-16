"""Vẽ hình cho bài báo từ results/tables (chạy analyze.py trước).

  .venv/Scripts/python.exe src/plots.py [--root results/smoke]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STYLE = {"HO70-R": ("#999999", "--"), "HO70-S": ("#555555", "-"), "HO80-R": ("#e6a1a1", "--"), "HO80-S": ("#c0392b", "-"),
         "KF5": ("#9ec5e6", "--"), "SKF5": ("#2e75b6", "-"), "KF10": ("#a9d9a9", "--"), "SKF10": ("#2e8b57", "-"),
         "RHO10": ("#8e44ad", "-")}
MODELS = ["LR", "KNN", "DT", "RF"]
PI_NAME = {"natural": "π tự nhiên (35%)", "0.1": "π = 10%", "0.03": "π = 3%"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def fig2_rmse_vs_n(s, out):
    s = s[(s.exp == "e1") & (s.metric == "f1_pooled")]
    pis = [p for p in ["natural", "0.1", "0.03"] if p in set(s.pi_label.astype(str))]
    fig, ax = plt.subplots(len(pis), 4, figsize=(11, 2.6 * len(pis)), sharex=True, squeeze=False)
    for i, p in enumerate(pis):
        for j, m in enumerate(MODELS):
            a = ax[i, j]
            g = s[(s.pi_label.astype(str) == p) & (s.model == m)]
            for st in C.STRATEGIES:
                h = g[g.strategy == st].sort_values("n")
                if h.empty:
                    continue
                col, ls = STYLE[st]
                a.plot(h.n, h.rmse, ls, color=col, marker="o", ms=3, label=st)
                a.fill_between(h.n, h.rmse_lo, h.rmse_hi, color=col, alpha=0.12, lw=0)
            a.set_xscale("log")
            a.set_xticks([200, 500, 1000, 2000])
            a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            if i == 0:
                a.set_title(m)
            if j == 0:
                a.set_ylabel(f"{PI_NAME.get(p, p)}\nRMSE")
            if i == len(pis) - 1:
                a.set_xlabel("n (kích thước D_dev)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out / "fig2_rmse_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def fig3_pareto(s, timing, out):
    s = s[(s.exp == "e1") & (s.metric == "f1_pooled")]
    r = s.groupby("strategy").rmse.mean().reindex(C.STRATEGIES)
    if timing is not None:
        t = timing.groupby("strategy").ratio_vs_HO80.mean().reindex(C.STRATEGIES)
        xl = "Thời gian tương đối so với HO80-R (trung vị, đơn luồng)"
    else:
        t = pd.Series({k: C.FIT_COST[k] for k in C.STRATEGIES}, dtype=float)
        xl = "Số lượt huấn luyện (chưa có số đo thời gian E3)"
    fig, a = plt.subplots(figsize=(5.2, 3.8))
    for st in C.STRATEGIES:
        a.scatter(t[st], r[st], color=STYLE[st][0], s=45, zorder=3, edgecolor="k", lw=0.5)
        a.annotate(st, (t[st], r[st]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    a.set_xscale("log")
    a.set_xlabel(xl)
    a.set_ylabel("RMSE trung bình (mọi ô, mọi thuật toán)")
    fig.tight_layout()
    fig.savefig(out / "fig3_pareto.png", bbox_inches="tight")
    plt.close(fig)


def fig4_effects(rs, f1d, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    g = rs[(rs.exp == "e1") & (rs.A.isin(["KF5", "KF10"]))].copy()
    g["pi_label"] = g.pi_label.astype(str)
    h = g.groupby(["pi_label", "n"]).delta_rmse.mean().unstack("n").reindex([p for p in ["natural", "0.1", "0.03"] if p in set(g.pi_label)])
    im = ax[0].imshow(h.values, cmap="RdBu", vmin=-abs(h.values).max(), vmax=abs(h.values).max(), aspect="auto")
    ax[0].set_xticks(range(h.shape[1]), h.columns)
    ax[0].set_yticks(range(h.shape[0]), [PI_NAME.get(i, i) for i in h.index])
    for (i, j), v in np.ndenumerate(h.values):
        ax[0].text(j, i, f"{v:+.3f}" if np.isfinite(v) else "", ha="center", va="center", fontsize=8)
    ax[0].set_xlabel("n")
    ax[0].set_title("(a) ΔRMSE = ngẫu nhiên − phân tầng (K-Fold)\nxanh: phân tầng tốt hơn", fontsize=9)
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    f = f1d[f1d.exp == "e1"].copy()
    f["pi_label"] = f.pi_label.astype(str)
    w = 0.38
    xs = np.arange(4)
    for k, (st, c) in enumerate([("SKF5", "#2e75b6"), ("SKF10", "#2e8b57")]):
        v = f[f.strategy == st].groupby("pi_label").diff_mean.mean().reindex(["natural", "0.1", "0.03"]).dropna()
        ax[1].bar(np.arange(len(v)) + k * w - w / 2, v.values, w, color=c, label=st)
        ax[1].set_xticks(np.arange(len(v)), [PI_NAME.get(i, i) for i in v.index])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("F1 trung bình fold − F1 gộp")
    ax[1].set_title("(b) Ảnh hưởng của cách tính F1", fontsize=9)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_effects.png", bbox_inches="tight")
    plt.close(fig)


def fig_e2(s, out):
    g = s[(s.exp == "e2") & (s.metric == "f1_pooled")]
    if g.empty:
        return
    ds = sorted(g.dataset.unique())
    fig, ax = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 3.4), squeeze=False)
    for a, d in zip(ax[0], ds):
        h = g[g.dataset == d].pivot(index="strategy", columns="model", values="rmse").reindex(C.STRATEGIES)[MODELS]
        h.plot.bar(ax=a, width=0.8)
        a.set_title(d)
        a.set_ylabel("RMSE")
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "fig5_e2_real_datasets.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "experiment.json"))
    ap.add_argument("--root")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    root = Path(a.root) if a.root else C.ROOT / cfg["results_dir"]
    if not root.is_absolute():
        root = C.ROOT / root
    tab = root / "tables"
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(tab / "summary.csv")
    rs = pd.read_csv(tab / "random_vs_stratified.csv")
    f1d = pd.read_csv(tab / "f1_definition_effect.csv")
    tp = root / "timing" / "timing_summary.csv"
    if not tp.exists():
        tp = C.ROOT / cfg["results_dir"] / "timing" / "timing_summary.csv"
    timing = pd.read_csv(tp) if tp.exists() else None
    if (s.exp == "e1").any():
        fig2_rmse_vs_n(s, out)
        fig3_pareto(s, timing, out)
        fig4_effects(rs, f1d, out)
    fig_e2(s, out)
    print("Đã vẽ:", ", ".join(sorted(p.name for p in out.glob("*.png"))), "→", out)


if __name__ == "__main__":
    main()
