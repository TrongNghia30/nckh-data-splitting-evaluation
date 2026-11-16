"""Phân tích kết quả đối chứng (results/control): mốc cùng kích thước huấn luyện và độ nhạy theo Oracle.

  .venv/Scripts/python.exe src/analyze_control.py

Với mỗi lần rút, mô hình, chiến lược:
  theta_hat   ước lượng của chiến lược (F1 gộp trên tập kiểm định) — giống E1
  theta       hiệu năng trên Oracle của mô hình học toàn bộ tập phát triển (mốc chính)
  theta_tilde trung bình hiệu năng trên Oracle của các mô hình mà chính chiến lược đã huấn luyện (mốc cùng kích thước)
  theta_A/B   mốc chính tính trên hai nửa Oracle
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KEYS = ["exp", "dataset", "pi_label", "pi", "n", "rep", "model"]
CELL = ["exp", "dataset", "pi_label", "n", "model"]
OUT = C.ROOT / "results" / "tables"
B = 2000


def rmse(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def main():
    files = sorted((C.ROOT / "results" / "control" / "raw").rglob("rep_*.csv"))
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["pi_label"] = raw.pi_label.astype(str)
    cm = raw[C.CM_COLS].to_numpy(float)
    raw["f1"] = C.f1_macro(*cm.T)
    tr = raw[raw.strategy == "TRUTH"].pivot_table(index=KEYS, columns="part", values="f1").reset_index()
    tr = tr.rename(columns={"ALL": "theta", "A": "theta_A", "B": "theta_B"})
    val = raw[raw.part == "VAL"].groupby(KEYS + ["strategy"], as_index=False)[C.CM_COLS].sum()
    val["theta_hat"] = C.f1_macro(*val[C.CM_COLS].to_numpy(float).T)
    orc = raw[raw.part == "ORACLE"].groupby(KEYS + ["strategy"], as_index=False).f1.mean().rename(columns={"f1": "theta_tilde"})
    d = val[KEYS + ["strategy", "theta_hat"]].merge(orc, on=KEYS + ["strategy"]).merge(tr, on=KEYS)
    d["e"] = d.theta_hat - d.theta
    d["e_m"] = d.theta_hat - d.theta_tilde
    d["gap"] = d.theta_tilde - d.theta
    d["e_A"] = d.theta_hat - d.theta_A
    d["e_B"] = d.theta_hat - d.theta_B
    print("Số lần rút mỗi ô:", d.groupby(CELL).rep.nunique().unique())

    # Tổng hợp theo ô
    g = d.groupby(CELL + ["strategy"])
    cs = g.apply(lambda x: pd.Series(dict(
        rmse=rmse(x.e), rmse_m=rmse(x.e_m), bias=x.e.mean(), bias_m=x.e_m.mean(), gap=x.gap.mean(),
        sd_err=x.e.std(ddof=1), sd_err_m=x.e_m.std(ddof=1), rmse_A=rmse(x.e_A), rmse_B=rmse(x.e_B))), include_groups=False).reset_index()
    cs.to_csv(OUT / "control_cell.csv", index=False, float_format="%.6g")

    print("\n== Theo chiến lược (trung bình qua ô và thuật toán)")
    print(cs.groupby("strategy")[["rmse", "rmse_m", "bias", "bias_m", "gap"]].mean().round(4).to_string())
    print("\n== Khoảng cách kích thước huấn luyện (theta_tilde − theta) theo chiến lược × n")
    print(cs.groupby(["strategy", "n"]).gap.mean().unstack().round(4).to_string())
    print("\n== RMSE theo mốc cùng kích thước, theo (π, n)")
    print(cs.pivot_table(index="strategy", columns=["pi_label", "n"], values="rmse_m").round(3).to_string())

    piv = cs.pivot_table(index=["pi_label", "n", "model"], columns="strategy", values=["rmse", "rmse_m"])
    for ref in ("rmse", "rmse_m"):
        r70 = piv[ref]["HO70-S"] / piv[ref]["SKF10"]
        r80 = piv[ref]["HO80-S"] / piv[ref]["SKF10"]
        print(f"\nTỷ số HO70-S/SKF10 [{ref}]: trung vị {r70.median():.2f}, khoảng {r70.min():.2f}–{r70.max():.2f};"
              f" HO80-S/SKF10: trung vị {r80.median():.2f}, khoảng {r80.min():.2f}–{r80.max():.2f}")

    # Thứ hạng có đổi không?
    from scipy.stats import spearmanr
    rho = [spearmanr(x.rmse, x.rmse_m).statistic for _, x in cs.groupby(CELL)]
    print(f"\nTương quan hạng Spearman giữa hai mốc (mỗi ô × thuật toán): trung vị {np.median(rho):.3f}, nhỏ nhất {np.min(rho):.3f}")
    stats = dict(spearman_median=float(np.median(rho)), spearman_min=float(np.min(rho)), reps=int(d.groupby(CELL).rep.nunique().min()))

    # Bootstrap: Holdout so với SKF10 theo mốc cùng kích thước
    rng = np.random.default_rng(20261103 + 21)
    rows = []
    for key, x in d.groupby(CELL):
        pe = x.pivot(index="rep", columns="strategy", values="e_m")
        pe0 = x.pivot(index="rep", columns="strategy", values="e")
        bi = rng.integers(0, len(pe), size=(B, len(pe)))
        for a in C.STRATEGIES:
            if a == "SKF10":
                continue
            for lab, M in (("matched", pe), ("main", pe0)):
                diff = rmse(M[a].to_numpy()[bi], axis=1) - rmse(M["SKF10"].to_numpy()[bi], axis=1)
                lo, hi = np.quantile(diff, [0.025, 0.975])
                rows.append(dict(zip(CELL, key), A=a, ref=lab, delta=rmse(M[a].to_numpy()) - rmse(M["SKF10"].to_numpy()), lo=lo, hi=hi))
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "control_vs_skf10.csv", index=False, float_format="%.6g")
    bt["worse"] = bt.lo > 0
    bt["eq01"] = (bt.lo > -0.01) & (bt.hi < 0.01)
    print("\n== Tỷ lệ ô (trên 40) chiến lược A kém SKF10 có ý nghĩa (KTC 95% > 0) và tương đương δ=0,01")
    print(bt.groupby(["A", "ref"])[["worse", "eq01"]].mean().unstack("ref").round(2).to_string())

    # Độ nhạy theo Oracle
    half = d.drop_duplicates(KEYS)
    noise = (half.theta_A - half.theta_B).groupby([half.pi_label, half.n]).std(ddof=1) / 2
    print("\n== Ước lượng sai số chuẩn do Oracle hữu hạn (sd(θ_A − θ_B)/2) theo (π, n)")
    print(noise.round(4).to_string())
    pa = cs.pivot_table(index=CELL, columns="strategy", values=["rmse_A", "rmse_B"])
    dA = pa["rmse_A"].sub(pa["rmse_A"]["SKF10"], axis=0).drop(columns="SKF10")
    dB = pa["rmse_B"].sub(pa["rmse_B"]["SKF10"], axis=0).drop(columns="SKF10")
    print(f"\nΔRMSE(A − SKF10) tính trên nửa A và nửa B: tương quan {np.corrcoef(dA.to_numpy().ravel(), dB.to_numpy().ravel())[0, 1]:.4f},"
          f" chênh tuyệt đối lớn nhất {np.abs(dA - dB).to_numpy().max():.4f}, trung bình {np.abs(dA - dB).to_numpy().mean():.4f}")
    sgn = (np.sign(dA) == np.sign(dB)).to_numpy().mean()
    print(f"Cùng dấu ΔRMSE giữa hai nửa: {sgn:.3f}")
    rel = (cs.rmse_A - cs.rmse_B).abs() / cs.rmse
    print(f"Chênh RMSE tuyệt đối giữa hai nửa Oracle so với RMSE: trung vị {rel.median():.3f}, lớn nhất {rel.max():.3f}")
    import json
    stats.update(oracle_se_min=float(noise.min()), oracle_se_max=float(noise.max()), oracle_se_mean=float(noise.mean()),
                 halves_corr=float(np.corrcoef(dA.to_numpy().ravel(), dB.to_numpy().ravel())[0, 1]),
                 halves_sign=float(sgn), halves_absdiff_mean=float(np.abs(dA - dB).to_numpy().mean()))
    noise.rename("oracle_se").reset_index().to_csv(OUT / "control_oracle_se.csv", index=False)
    (OUT / "control_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
