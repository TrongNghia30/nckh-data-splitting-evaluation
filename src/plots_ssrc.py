"""Hình cho bản bài báo theo template SSRC (khổ một cột ≈ 88 mm, 300 dpi, chữ Times New Roman).

  .venv/Scripts/python.exe src/plots_ssrc.py      (chạy sau analyze.py, analyze_extra.py, analyze_round3.py)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures_ssrc"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "savefig.dpi": 300, "mathtext.fontset": "stix"})
W = 3.45   # bề rộng một cột (inch)
STRATS = ["HO70-R", "HO70-S", "HO80-R", "HO80-S", "KF5", "SKF5", "KF10", "SKF10", "RHO10"]
STYLE = {"HO70-R": ("#7f7f7f", "--", "o"), "HO70-S": ("#404040", "-", "o"), "HO80-R": ("#e39a9a", "--", "s"),
         "HO80-S": ("#b22222", "-", "s"), "KF5": ("#8db8e0", "--", "^"), "SKF5": ("#1f5fa8", "-", "^"),
         "KF10": ("#98d098", "--", "D"), "SKF10": ("#1b7f3b", "-", "D"), "RHO10": ("#7b3fa0", "-", "v")}
COMMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−").replace(".", ","))
PI = [("natural", "π tự nhiên (35%)"), ("0.1", "π = 10%"), ("0.03", "π = 3%")]


def fig_rmse_n():
    c = pd.read_csv(TAB / "extra_cell_summary.csv")
    c = c[c.exp == "e1"].copy()
    c["pi_label"] = c.pi_label.astype(str)
    fig, axes = plt.subplots(3, 1, figsize=(W, 4.9), sharex=True)
    for ax, (pl, name) in zip(axes, PI):
        g = c[c.pi_label == pl]
        for st in STRATS:
            h = g[g.strategy == st].sort_values("n")
            col, ls, mk = STYLE[st]
            ax.plot(h.n, h.rmse, ls=ls, color=col, marker=mk, ms=3, label=st)
        ax.set_xscale("log")
        ax.set_xticks([200, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(170, 2350)
        ax.set_ylabel("RMSE của F1 gộp")
        ax.yaxis.set_major_formatter(COMMA)
        ax.set_title(name, pad=3)
    axes[-1].set_xlabel("Kích thước tập phát triển n (mẫu)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.53, 1.0), columnspacing=0.8, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "hinh1_rmse_theo_n.png")
    plt.close(fig)


def fig_cost():
    s = pd.read_csv(TAB / "summary.csv")
    s = s[(s.exp == "e1") & (s.metric == "f1_pooled")]
    r = s.groupby("strategy").rmse.mean()
    t = pd.read_csv(ROOT / "results" / "timing" / "timing_summary.csv").groupby("strategy").ratio_vs_HO80.median()
    off = {"HO70-R": (6, 4), "HO70-S": (6, -9), "HO80-R": (6, 3), "HO80-S": (6, -9), "KF5": (-24, 5), "SKF5": (-27, -9),
           "KF10": (7, -9), "SKF10": (7, 2), "RHO10": (-33, 4)}
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for st in STRATS:
        col, _, mk = STYLE[st]
        ax.scatter(t[st], r[st], color=col, marker=mk, s=28, edgecolor="k", linewidth=0.4, zorder=3)
        ax.annotate(st, (t[st], r[st]), textcoords="offset points", xytext=off[st], fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.7, 20)
    ax.set_ylim(0.029, 0.069)
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_xlabel("Chi phí tương đối so với HO80-R (trung vị thời gian, đơn luồng)")
    ax.set_ylabel("RMSE trung bình của F1 gộp")
    fig.tight_layout()
    fig.savefig(OUT / "hinh2_chi_phi.png")
    plt.close(fig)


def fig_strat_f1():
    lv = pd.read_csv(TAB / "r3_strat_levels.csv")
    lv["pi_label"] = lv.pi_label.astype(str)
    f1d = pd.read_csv(TAB / "f1_definition_effect.csv")
    f1d = f1d[f1d.exp == "e1"].copy()
    f1d["pi_label"] = f1d.pi_label.astype(str)
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.3))
    x = np.arange(3)
    labels = [nm for _, nm in PI]
    w = 0.2
    bars = [("f1_pooled", "d5", "F1 gộp, K = 5", "#9ec5e6"), ("f1_pooled", "d10", "F1 gộp, K = 10", "#1f5fa8"),
            ("f1_foldavg", "d5", "F1 TB fold, K = 5", "#f2b37d"), ("f1_foldavg", "d10", "F1 TB fold, K = 10", "#c55a11")]
    for i, (m, col, lab, colr) in enumerate(bars):
        v = [float(lv[(lv.metric == m) & (lv.pi_label == pl)][col].iloc[0]) for pl, _ in PI]
        a.bar(x + (i - 1.5) * w, v, w, color=colr, label=lab, edgecolor="k", linewidth=0.3)
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(x, labels)
    a.set_ylabel("ΔRMSE (ngẫu nhiên − phân tầng)")
    a.set_title("(a) Tác động của phân tầng theo cách tính F1", pad=3)
    a.set_ylim(-0.0006, 0.0042)
    a.yaxis.set_major_formatter(COMMA)
    a.legend(ncol=2, frameon=False, loc="upper left")
    strat = [("KF5", "#9ec5e6"), ("SKF5", "#1f5fa8"), ("KF10", "#98d098"), ("SKF10", "#1b7f3b")]
    for i, (st, colr) in enumerate(strat):
        v = f1d[f1d.strategy == st].groupby("pi_label").diff_mean.mean().reindex([p for p, _ in PI]).to_numpy()
        b.bar(x + (i - 1.5) * w, v, w, color=colr, label=st, edgecolor="k", linewidth=0.3)
    b.axhline(0, color="k", lw=0.6)
    b.set_xticks(x, labels)
    b.set_ylabel("F1 TB fold − F1 gộp")
    b.set_title("(b) Chênh lệch giữa hai cách tính F1", pad=3)
    b.set_ylim(-0.022, 0.001)
    b.yaxis.set_major_formatter(COMMA)
    b.legend(ncol=2, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "hinh3_phan_tang_f1.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_rmse_n()
    fig_cost()
    fig_strat_f1()
    print("Đã vẽ:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
