"""Emit the LaTeX body rows for every table in the manuscript.

Rows are printed rather than typed, so no number reaches main.tex by transcription.
check_tex.py re-runs this and diffs the result against what is actually in main.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

DISP = {
    "S_ventral": "Ventral stream", "S_lateral": "Lateral stream",
    "nsdgeneral": "nsdgeneral", "PHC": "Parahippocampal", "FFA": "FFA",
    "PPA": "PPA", "OPA": "OPA", "S_parietal": "Parietal stream", "RSC": "RSC",
    "S_midlateral": "Mid-lateral stream", "S_midventral": "Mid-ventral stream",
    "S_early": "Early stream", "hV4": "hV4", "V3": "V3", "V2": "V2", "V1": "V1",
    "ERC": "Entorhinal", "PRC": "Perirhinal", "SUB": "Subiculum",
    "CA1": "CA1", "DG_CA3": "DG/CA3",
}


def n(x, d=3):
    return f"\\num{{{x:.{d}f}}}"


def p_fmt(x):
    """p-values: three decimals down to 0.001, scientific below."""
    if pd.isna(x):
        return "---"
    if x >= 0.001:
        return f"{x:.3f}"
    mant, expo = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(expo)}}}$"


def main() -> int:
    by_roi = pd.read_csv(RESULTS / "brain_embedding_loso_by_roi.csv")
    by_cat = pd.read_csv(RESULTS / "brain_embedding_loso_by_category.csv")
    trials = pd.read_csv(RESULTS / "brain_embedding_loso.csv")
    geom = pd.read_csv(RESULTS / "intersubject_mapping_breakdown_summary.csv")

    ident = (trials.groupby(["roi", "subject_id"], observed=True)
             [["ident_transfer", "ident_ceiling"]].first().reset_index()
             .groupby("roi", observed=True)[["ident_transfer", "ident_ceiling"]]
             .mean())

    t1 = (by_roi.set_index("roi")
          .join(ident)
          .join(geom.set_index("roi")[["shared", "sem_shared",
                                       "p_shared_vs_one", "r_within"]])
          .sort_values("frac_of_ceiling", ascending=False))

    print("% ---------- TABLE 1 ----------")
    for roi, r in t1.iterrows():
        sh = "---" if pd.isna(r["shared"]) else n(r["shared"])
        shs = "---" if pd.isna(r["sem_shared"]) else n(r["sem_shared"])
        print(f"{DISP[roi]:<19s} & {n(r['frac_of_ceiling'])} & {n(r['sem'])} & "
              f"{n(r['ident_ceiling'])} & {n(r['ident_transfer'])} & "
              f"{sh} & {shs} & {p_fmt(r['p_shared_vs_one'])} \\\\")

    # ---- Table 2: paired contrast of every region against V1
    wide = (trials.groupby(["roi", "subject_id"], observed=True)
            [["transfer", "ceiling"]].mean().reset_index())
    wide["frac"] = wide["transfer"] / wide["ceiling"]
    w = wide.pivot(index="subject_id", columns="roi", values="frac")
    rows = []
    for roi in t1.index:
        if roi == "V1":
            continue
        d = w[roi] - w["V1"]
        t, p = stats.ttest_rel(w[roi], w["V1"])
        rows.append((roi, d.mean(), float(t), float(p), int((d > 0).sum())))
    raw_p = np.array([r[3] for r in rows])
    order = np.argsort(raw_p)
    m = len(raw_p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):          # Benjamini-Hochberg, step-up
        prev = min(prev, raw_p[i] * m / (m - rank))
        adj[i] = prev

    print("\n% ---------- TABLE 2 ----------")
    for (roi, d, t, p, npos), q in zip(rows, adj):
        print(f"{DISP[roi]:<19s} & {n(d)} & {t:.2f} & {p_fmt(p)} & "
              f"{p_fmt(q)} & {npos}/8 \\\\")
    print(f"% BH-FDR across {m} contrasts; all q < 0.001: "
          f"{bool((adj < 0.001).all())}")

    # ---- Table 3: category dependence
    g = (by_cat.groupby(["roi", "category"], observed=True)["frac"]
         .mean().unstack())
    print("\n% ---------- TABLE 3 ----------")
    for roi in t1.index:
        row = g.loc[roi]
        print(f"{DISP[roi]:<19s} & {n(row.min())} & {row.idxmin()} & "
              f"{n(row.max())} & {row.idxmax()} & {n(row.max() - row.min())} \\\\")

    # ---- Table 4: regions excluded for insufficient reliability
    print("\n% ---------- TABLE 4 ----------")
    excl = geom[~geom["roi"].isin(by_roi["roi"])].set_index("roi")
    for roi in ["ERC", "PRC", "SUB", "CA1", "DG_CA3"]:
        r = excl.loc[roi]
        print(f"{DISP[roi]:<19s} & {n(r['r_within'], 4)} & "
              f"{n(r['r_between'], 4)} & {p_fmt(r['p_between_vs_zero'])} \\\\")

    # ---- inline numbers quoted in the running text
    print("\n% ---------- inline ----------")
    v1, sv = t1.loc["V1"], t1.loc["S_ventral"]
    print(f"% V1 unshared     = {100 * (1 - v1['frac_of_ceiling']):.1f}%")
    print(f"% ventral unshared= {100 * (1 - sv['frac_of_ceiling']):.1f}%")
    print(f"% V1 ident gap    = {v1['ident_ceiling'] - v1['ident_transfer']:.3f}")
    rho, prho = stats.spearmanr(t1["frac_of_ceiling"].loc[~t1["shared"].isna()],
                                t1["shared"].dropna())
    rr, pr = stats.pearsonr(t1["frac_of_ceiling"].loc[~t1["shared"].isna()],
                            t1["shared"].dropna())
    print(f"% cross-method rho={rho:.3f} p={prho:.3f}  r={rr:.3f} p={pr:.3f}")
    print(f"% span ratio V1/ventral = "
          f"{(g.loc['V1'].max() - g.loc['V1'].min()) / (g.loc['S_ventral'].max() - g.loc['S_ventral'].min()):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
