"""Step 40: does a typical brain representation remember better?

PRIMARY INFERENCE
-----------------
Per-subject logistic regression, then a one-sample t-test on the 8 resulting
slopes. The participant is the unit of population inference; 5,240 trials do not
make n = 5240. Benjamini-Hochberg across the 16 ROIs.

    ISCORRECT(pres 3) ~ z_transfer_ROI + z_log_lag + z_memorability

A positive slope means: images this participant represented MORE typically
(closer to the other seven) were MORE likely to be recognised later.

SECONDARY
---------
GEE with subject clusters and robust standard errors, as a pooling check that
makes different assumptions from the primary test.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "results" / "memory_design.csv")
ROIS = sorted(c[3:] for c in D.columns if c.startswith("tr_"))
COV = ["z_log_lag", "z_memorability"]

DISP = {"S_ventral": "Ventral stream", "S_lateral": "Lateral stream",
        "nsdgeneral": "nsdgeneral", "PHC": "Parahippocampal", "FFA": "FFA",
        "PPA": "PPA", "OPA": "OPA", "S_parietal": "Parietal stream",
        "RSC": "RSC", "S_midlateral": "Mid-lateral", "S_midventral":
        "Mid-ventral", "S_early": "Early stream", "hV4": "hV4", "V3": "V3",
        "V2": "V2", "V1": "V1"}


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q


def fit_subject(sub, roi):
    """Unpenalised logistic fit for one subject. Returns the transfer slope."""
    X = sm.add_constant(sub[[f"z_tr_{roi}"] + COV].to_numpy(), has_constant="add")
    y = sub["ISCORRECT"].to_numpy()
    try:
        r = sm.Logit(y, X).fit(disp=0, maxiter=200)
        if not np.isfinite(r.params[1]) or abs(r.params[1]) > 10:
            return np.nan
        return float(r.params[1])
    except Exception:
        return np.nan


def main() -> int:
    print(f"design: {len(D)} trials | {D.subject_id.nunique()} subjects | "
          f"{D.nsd_id.nunique()} images | {int(D.ISCORRECT.sum())} hits")

    # ------------------------------------------------------------- power
    print("\n" + "=" * 74)
    print("POWER (one-sample t-test on 8 per-subject slopes, alpha = .05)")
    print("=" * 74)
    for pw in (0.80, 0.50):
        lo, hi = 0.0, 5.0
        for _ in range(60):
            dz = (lo + hi) / 2
            crit = stats.t.ppf(0.975, 7)
            got = 1 - stats.nct.cdf(crit, 7, dz * np.sqrt(8)) \
                + stats.nct.cdf(-crit, 7, dz * np.sqrt(8))
            lo, hi = (dz, hi) if got < pw else (lo, dz)
        print(f"  {pw:.0%} power needs Cohen dz = {dz:.2f}  "
              f"(a {'large' if dz > 0.8 else 'medium'} consistent effect)")
    print("  n = 8 is the binding constraint, not the 5,240 trials.")

    # ----------------------------------------------------------- primary
    print("\n" + "=" * 74)
    print("PRIMARY: per-subject slopes, one per ROI")
    print("=" * 74)
    rows, slopes = [], {}
    for roi in ROIS:
        b = np.array([fit_subject(g, roi)
                      for _, g in D.groupby("subject_id", sort=True)])
        if np.isnan(b).any():
            print(f"  [warn] {roi}: {int(np.isnan(b).sum())} subject fit(s) failed")
            b = b[~np.isnan(b)]
        slopes[roi] = b
        t, p = stats.ttest_1samp(b, 0)
        ci = stats.t.interval(0.95, len(b) - 1, loc=b.mean(),
                              scale=stats.sem(b))
        rows.append(dict(roi=roi, n=len(b), mean_beta=b.mean(),
                         sd=b.std(ddof=1), dz=b.mean() / b.std(ddof=1),
                         t=t, p=p, ci_lo=ci[0], ci_hi=ci[1],
                         n_pos=int((b > 0).sum())))
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res.p.values)
    res = res.sort_values("p")

    print(f"\n  {'Region':<18}{'beta':>8}{'95% CI':>20}{'dz':>7}"
          f"{'t(7)':>8}{'p':>9}{'q':>9}{'dir':>7}")
    print("  " + "-" * 84)
    for _, r in res.iterrows():
        star = " *" if r.q_bh < 0.05 else ""
        print(f"  {DISP[r.roi]:<18}{r.mean_beta:>+8.4f}"
              f"{f'[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]':>20}{r.dz:>+7.2f}"
              f"{r.t:>+8.2f}{r.p:>9.4f}{r.q_bh:>9.4f}"
              f"{f'{r.n_pos}/8':>7}{star}")

    sig = res[res.q_bh < 0.05]
    print(f"\n  surviving BH-FDR across {len(res)} ROIs: "
          f"{len(sig)} ({', '.join(DISP[r] for r in sig.roi) or 'none'})")

    # --------------------------------------------------------- covariates
    print("\n" + "=" * 74)
    print("COVARIATE SLOPES (sanity: both should be strong and signed correctly)")
    print("=" * 74)
    for j, nm in ((2, "log lag"), (3, "memorability")):
        bb = []
        for _, g in D.groupby("subject_id", sort=True):
            X = sm.add_constant(g[[f"z_tr_{ROIS[0]}"] + COV].to_numpy(),
                                has_constant="add")
            try:
                bb.append(sm.Logit(g.ISCORRECT.to_numpy(), X)
                          .fit(disp=0, maxiter=200).params[j])
            except Exception:
                pass
        bb = np.array(bb)
        t, p = stats.ttest_1samp(bb, 0)
        print(f"  {nm:14s} beta {bb.mean():+.4f}  t(7) = {t:+.2f}  p = {p:.4f}"
              f"   ({int((bb > 0).sum())}/8 positive)")

    # ----------------------------------------------- secondary: GEE pooled
    print("\n" + "=" * 74)
    print("SECONDARY: GEE, subject clusters, robust SE (different assumptions)")
    print("=" * 74)
    gee = []
    for roi in ROIS:
        X = sm.add_constant(D[[f"z_tr_{roi}"] + COV])
        try:
            m = sm.GEE(D.ISCORRECT, X, groups=D.subject_id,
                       family=sm.families.Binomial()).fit()
            gee.append(dict(roi=roi, beta=m.params.iloc[1],
                            p=m.pvalues.iloc[1]))
        except Exception:
            pass
    gdf = pd.DataFrame(gee)
    gdf["q_bh"] = bh(gdf.p.values)
    gdf = gdf.sort_values("p")
    for _, r in gdf.head(6).iterrows():
        print(f"  {DISP[r.roi]:<18} beta {r.beta:+.4f}   p {r.p:.4f}   "
              f"q {r.q_bh:.4f}")
    print(f"  surviving: {int((gdf.q_bh < 0.05).sum())} of {len(gdf)}")

    # ------------------------------------------- V1 vs ventral, paired
    print("\n" + "=" * 74)
    print("TARGETED CONTRAST: V1 vs ventral stream slopes, paired across subjects")
    print("=" * 74)
    a, b = slopes["V1"], slopes["S_ventral"]
    if len(a) == len(b) == 8:
        t, p = stats.ttest_rel(b, a)
        print(f"  V1      {a.mean():+.4f}")
        print(f"  Ventral {b.mean():+.4f}")
        print(f"  difference {b.mean() - a.mean():+.4f}  t(7) = {t:+.2f}  "
              f"p = {p:.4f}")

    res.to_csv(HERE / "results" / "memory_typicality_by_roi.csv", index=False)
    gdf.to_csv(HERE / "results" / "memory_typicality_gee.csv", index=False)
    pd.DataFrame(slopes).assign(
        subject_id=sorted(D.subject_id.unique())).to_csv(
        HERE / "results" / "memory_typicality_subject_slopes.csv", index=False)
    print("\nwrote results/memory_typicality_{by_roi,gee,subject_slopes}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
