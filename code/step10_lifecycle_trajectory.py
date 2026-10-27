"""
Step 10: the full Rep1 -> Rep2 -> Rep3 trajectory for the 93-stimulus lifecycle cohort.

10a  Linear mixed model with subject as a random intercept and orthogonal polynomial
     contrasts for repetition, so the linear and quadratic trends are tested
     independently rather than as correlated raw powers.
10b  Monotonicity: fraction of subjects whose trajectory is strictly decreasing.
     Under a random ordering of 3 values the chance of any one specific order is 1/6,
     so the count is tested against Binomial(8, 1/6).
10c  The Rep3 - Rep1 contrast, which step6 already computes with the full statistical
     apparatus; reproduced here alongside the trend tests for convenience.

Note on power: the lifecycle cohort is 93 stimuli, and those 93 are the images whose
entire three-presentation lifecycle happens to fall inside sessions 1-2, i.e. an extreme
short-lag tail of the NSD design (median Rep2->Rep3 lag 121 trials against 1,642
NSD-wide). Trend tests here are underpowered and the cohort is not representative.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COMPOSITES = {"EarlyVis", "SceneCtx", "MTL", "AllCortex"}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def fit_lmm(df):
    """Random-intercept LMM: acc ~ lin + quad + (1|subject), orthogonal contrasts."""
    import statsmodels.formula.api as smf
    # orthogonal polynomial contrasts for rep in {1,2,3}
    lin = {1: -1.0, 2: 0.0, 3: 1.0}
    quad = {1: 1.0, 2: -2.0, 3: 1.0}
    d = df.copy()
    d["lin"] = d.repetition.map(lin)
    d["quad"] = d.repetition.map(quad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("two_way ~ lin + quad", d, groups=d["subject_id"]).fit(reml=True)
    return fit


def main():
    src = RESULTS / "ridge_clip_text_retrieval_by_repetition.csv"
    try:
        r = pd.read_csv(src)
    except FileNotFoundError:
        print(f"[ERROR] missing {src}. Run step2 first.")
        return 1
    life = r[r.analysis == "lifecycle93"].copy()
    if life.empty:
        print("[ERROR] no lifecycle93 rows found.")
        return 1

    rows = []
    for norm in sorted(life.norm.unique()):
        for roi, g in life[life.norm == norm].groupby("roi"):
            g = g.sort_values(["subject_id", "repetition"])
            fit = fit_lmm(g)

            # 10b monotonicity
            wide = g.pivot_table(index="subject_id", columns="repetition", values="two_way")
            mono_dec = int(((wide[1] > wide[2]) & (wide[2] > wide[3])).sum())
            n = len(wide)
            p_mono = stats.binomtest(mono_dec, n, 1 / 6, alternative="greater").pvalue

            # 10c Rep3 - Rep1
            d31 = (wide[3] - wide[1]).values
            t31, p31 = stats.ttest_rel(wide[3].values, wide[1].values)

            rows.append(dict(
                roi=roi, is_composite=roi in COMPOSITES, norm=norm, n_subj=n,
                mean_rep1=wide[1].mean(), mean_rep2=wide[2].mean(),
                mean_rep3=wide[3].mean(),
                lin_beta=fit.params.get("lin", np.nan),
                lin_se=fit.bse.get("lin", np.nan),
                lin_z=fit.tvalues.get("lin", np.nan),
                lin_p=fit.pvalues.get("lin", np.nan),
                quad_beta=fit.params.get("quad", np.nan),
                quad_se=fit.bse.get("quad", np.nan),
                quad_z=fit.tvalues.get("quad", np.nan),
                quad_p=fit.pvalues.get("quad", np.nan),
                subject_var=float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else np.nan,
                n_monotonic_decreasing=mono_dec, p_monotonic_binomial=p_mono,
                delta_rep3_rep1=d31.mean(),
                sem_rep3_rep1=d31.std(ddof=1) / np.sqrt(n),
                t_rep3_rep1=t31, p_rep3_rep1=p31,
                cohens_d_rep3_rep1=d31.mean() / d31.std(ddof=1),
                n_subj_negative_rep3_rep1=int((d31 < 0).sum())))

    out = pd.DataFrame(rows)
    for col, newcol in [("lin_p", "lin_p_fdr"), ("quad_p", "quad_p_fdr"),
                        ("p_rep3_rep1", "p_rep3_rep1_fdr")]:
        out[newcol] = np.nan
        for _, idx in out.groupby("norm").groups.items():
            out.loc[idx, newcol] = bh_fdr(out.loc[idx, col].values)
    out.to_csv(RESULTS / "lifecycle_trajectory_stats.csv", index=False)
    print(f"[OK] results/lifecycle_trajectory_stats.csv  ({len(out)} rows)")

    pd.set_option("display.width", 250)
    for norm, g in out.groupby("norm"):
        print(f"\n=== lifecycle93 | norm={norm} | LMM trend tests (subject random intercept) ===")
        print(g.sort_values("lin_p")[
            ["roi", "mean_rep1", "mean_rep2", "mean_rep3", "lin_beta", "lin_z", "lin_p",
             "lin_p_fdr", "quad_beta", "quad_z", "quad_p"]].round(4).to_string(index=False))
        print(f"\n--- monotonic decrease and Rep3-Rep1 (norm={norm}) ---")
        print(g.sort_values("p_rep3_rep1")[
            ["roi", "n_monotonic_decreasing", "p_monotonic_binomial", "delta_rep3_rep1",
             "sem_rep3_rep1", "t_rep3_rep1", "p_rep3_rep1", "p_rep3_rep1_fdr",
             "cohens_d_rep3_rep1", "n_subj_negative_rep3_rep1"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
