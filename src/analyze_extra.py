"""Phân tích bổ sung theo phiếu phản biện (không thay đổi phân tích chính đã khóa trước).

  .venv/Scripts/python.exe src/analyze_extra.py

Ghi ra results/tables/extra_*.csv:
  extra_cell_summary.csv     RMSE, Bias, SD, SD_err theo từng cặp (n, π) và chiến lược (trung bình qua 4 thuật toán)
  extra_error_parts.csv      tỷ lệ Bias²/RMSE² và SD_err/SD theo chiến lược và n
  extra_margin.csv           số ô đạt tương đương (KTC 95% và 90%) và không kém hơn so với SKF10 theo δ
  extra_recommend.csv        ma trận khuyến nghị theo δ tuyệt đối và δ tương đối
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import analyze as A  # noqa: E402

ROOT = C.ROOT / "results"
TAB = ROOT / "tables"
B = 2000


def boot_diffs(rt, metric, ref, rng):
    """Với mỗi ô: phân phối bootstrap của ΔRMSE(A − ref) cho mọi chiến lược A."""
    out = {}
    for key, g in rt.groupby(A.CELL):
        piv = g.pivot(index="rep", columns="strategy", values=metric)
        truth = g.drop_duplicates("rep").set_index("rep").loc[piv.index, "truth_f1"]
        e = piv.sub(truth, axis=0)
        bi = A.boot_idx(len(e), B, rng)
        rr = A.rmse(e[ref].to_numpy()[bi], axis=1)
        out[key] = {s: (A.rmse(e[s].to_numpy()) - A.rmse(e[ref].to_numpy()), A.rmse(e[s].to_numpy()[bi], axis=1) - rr,
                        A.rmse(e[ref].to_numpy())) for s in e.columns if s != ref}
    return out


def recommend_rule(rt, metric, rng, delta=None, rel=None):
    rows = []
    for key, g in rt.groupby(A.CELL):
        piv = g.pivot(index="rep", columns="strategy", values=metric)
        truth = g.drop_duplicates("rep").set_index("rep").loc[piv.index, "truth_f1"]
        e = piv.sub(truth, axis=0)
        r = A.rmse(e.to_numpy())
        cols = list(e.columns)
        best = cols[int(np.argmin(r))]
        d = delta if delta is not None else rel * float(r.min())
        bi = A.boot_idx(len(e), B, rng)
        rb = A.rmse(e[best].to_numpy()[bi], axis=1)
        ok = []
        for st in cols:
            hi = np.quantile(A.rmse(e[st].to_numpy()[bi], axis=1) - rb, 0.975) if st != best else 0.0
            if hi < d:
                ok.append((C.FIT_COST[st], float(r[cols.index(st)]), st))
        rows.append(dict(zip(A.CELL, key), rule=f"abs{delta}" if delta is not None else f"rel{rel}", margin=d,
                         best=best, recommended=sorted(ok)[0][2], cost=sorted(ok)[0][0]))
    return pd.DataFrame(rows)


def main():
    rt = A.replicate_table(A.load_raw(ROOT))
    rng = np.random.default_rng(20261103 + 11)
    s = pd.read_csv(TAB / "summary.csv")
    s = s[s.metric == "f1_pooled"].copy()
    s["pi_label"] = s.pi_label.astype(str)

    # 1. Theo từng cặp (n, π)
    cs = (s.groupby(["exp", "dataset", "pi_label", "n", "strategy"], as_index=False)
          [["rmse", "bias", "sd", "sd_err", "theta_true_sd"]].mean())
    cs.to_csv(TAB / "extra_cell_summary.csv", index=False, float_format="%.6g")

    # 2. Thành phần sai số
    e1 = s[s.exp == "e1"].copy()
    e1["bias2_share"] = e1.bias ** 2 / e1.rmse ** 2
    e1["sderr_over_sd"] = e1.sd_err / e1.sd
    parts = e1.groupby(["strategy", "n"], as_index=False)[["bias2_share", "sderr_over_sd"]].mean()
    parts.to_csv(TAB / "extra_error_parts.csv", index=False, float_format="%.6g")
    ident = (e1.rmse ** 2 - (e1.bias ** 2 + (e1.R - 1) / e1.R * e1.sd_err ** 2)).abs().max()
    print("Đồng nhất thức RMSE², sai lệch lớn nhất:", f"{ident:.2e}")
    print("Bias²/RMSE² theo chiến lược:", e1.groupby("strategy").bias2_share.mean().round(3).to_dict())
    print("SD_err/SD: trung bình", round(e1.sderr_over_sd.mean(), 3), "min", round(e1.sderr_over_sd.min(), 3),
          "max", round(e1.sderr_over_sd.max(), 3))

    # 3. Tương đương / không kém hơn theo δ
    bd = boot_diffs(rt, "f1_pooled", "SKF10", rng)
    rows = []
    for key, d in bd.items():
        for st, (point, diffs, ref_rmse) in d.items():
            lo95, hi95 = np.quantile(diffs, [0.025, 0.975])
            lo90, hi90 = np.quantile(diffs, [0.05, 0.95])
            for delta in (0.005, 0.01, 0.02):
                rows.append(dict(zip(A.CELL, key), A=st, delta=delta, point=point,
                                 equiv95=bool(lo95 > -delta and hi95 < delta), equiv90=bool(lo90 > -delta and hi90 < delta),
                                 noninf=bool(hi95 < delta)))
            dr = 0.10 * ref_rmse
            rows.append(dict(zip(A.CELL, key), A=st, delta="rel10", point=point,
                             equiv95=bool(lo95 > -dr and hi95 < dr), equiv90=bool(lo90 > -dr and hi90 < dr), noninf=bool(hi95 < dr)))
    mg = pd.DataFrame(rows)
    mg.to_csv(TAB / "extra_margin.csv", index=False)
    print("\nSố ô (trên 48) đạt tiêu chí so với SKF10:")
    print(mg.groupby(["A", "delta"])[["equiv95", "equiv90", "noninf"]].sum().unstack("delta").to_string())

    # 4. Ma trận khuyến nghị theo các quy tắc
    rec = pd.concat([recommend_rule(rt, "f1_pooled", rng, delta=d) for d in (0.005, 0.01, 0.02)]
                    + [recommend_rule(rt, "f1_pooled", rng, rel=0.10)])
    rec.to_csv(TAB / "extra_recommend.csv", index=False)
    rec["family"] = rec.recommended.map(lambda x: "Holdout" if x.startswith("HO") else ("5-Fold" if x in ("KF5", "SKF5") else "10 lượt"))
    print("\nKhuyến nghị theo quy tắc:")
    print(rec.groupby(["rule", "exp", "dataset"]).family.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
