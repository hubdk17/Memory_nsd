"""
Step 6: statistical infrastructure for every repetition contrast already computed.

Adds, for each ROI x normalisation x cohort x contrast:
  - exact sign-flip permutation test (all 2^8 = 256 sign assignments) and a
    10,000-draw random sign-flip test for comparison
  - Benjamini-Hochberg FDR across ROIs within each normalisation x cohort x contrast
  - paired Cohen's d, Hedges' g, and a 95% CI for d from the noncentral t
  - JZS Bayes factor for the paired t-test (Rouder et al. 2009), Cauchy scale r = sqrt(2)/2

The Bayes factors matter most for the hippocampal ROIs: a non-significant t-test is not
evidence of no effect, and reviewers will ask for evidence *for* the null.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 42
N_RANDOM_PERM = 10_000
CONTRASTS = [("lifecycle93", 1, 2), ("lifecycle93", 1, 3), ("repeated323", 1, 2)]
COMPOSITES = {"EarlyVis", "SceneCtx", "MTL", "AllCortex"}


# ---------------------------------------------------------------- permutation
def sign_flip_tests(d, rng):
    """Exact (all 2^n) and random sign-flip p-values, two-sided, on the mean of d."""
    n = len(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=n)))       # (2^n, n)
    exact = float((np.abs(signs @ d / n) >= obs - 1e-12).mean())
    rnd = rng.choice([1.0, -1.0], size=(N_RANDOM_PERM, n))
    rand = float((np.abs(rnd @ d / n) >= obs - 1e-12).mean())
    return exact, rand


# ---------------------------------------------------------------- effect size
def cohens_d_paired(d):
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def hedges_g(dval, n):
    """Small-sample bias correction for a paired design (df = n - 1)."""
    df = n - 1
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(dval * j)


def _nct_cdf_safe(t, df, nc):
    """nct.cdf returns NaN far out in the tails; substitute the monotone limits."""
    v = stats.nct.cdf(t, df, nc)
    if np.isnan(v):
        return 1.0 if nc < 0 else 0.0
    return float(np.clip(v, 0.0, 1.0))


def d_ci_noncentral(dval, n, conf=0.95, span=80.0, iters=200):
    """CI for paired Cohen's d by inverting the noncentral t distribution.

    cdf(t_obs; df, nc) decreases monotonically in nc, so a plain bisection is
    robust where brentq is not (scipy returns NaN for |nc| beyond ~55).
    """
    df, t_obs = n - 1, dval * np.sqrt(n)
    lo_p, hi_p = (1 - conf) / 2, 1 - (1 - conf) / 2

    def solve(target):
        lo, hi = -span, span                 # cdf(lo) ~ 1, cdf(hi) ~ 0
        if _nct_cdf_safe(t_obs, df, lo) < target or _nct_cdf_safe(t_obs, df, hi) > target:
            return np.nan                    # target unreachable
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if _nct_cdf_safe(t_obs, df, mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-8:
                break
        return 0.5 * (lo + hi) / np.sqrt(n)

    return solve(hi_p), solve(lo_p)      # cdf=0.975 -> lower bound; cdf=0.025 -> upper


# ---------------------------------------------------------------- Bayes factor
def bf10_paired(t, n, r=np.sqrt(2) / 2):
    """JZS Bayes factor for a paired t-test (Rouder et al. 2009, eq. 1)."""
    df = n - 1

    def marginal(g):
        return ((1 + n * g * r ** 2) ** -0.5
                * (1 + t ** 2 / ((1 + n * g * r ** 2) * df)) ** (-(df + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))

    num, _ = integrate.quad(marginal, 1e-10, np.inf, limit=200)
    den = (1 + t ** 2 / df) ** (-(df + 1) / 2)
    return float(num / den)


def interpret_bf(bf):
    if bf < 1 / 10:
        return "strong evidence for null"
    if bf < 1 / 3:
        return "moderate evidence for null"
    if bf < 3:
        return "anecdotal/inconclusive"
    if bf < 10:
        return "moderate evidence for alternative"
    return "strong evidence for alternative"


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def main():
    src = RESULTS / "ridge_clip_text_retrieval_by_repetition.csv"
    try:
        r = pd.read_csv(src)
    except FileNotFoundError:
        print(f"[ERROR] missing {src}. Run step2 first.")
        return 1

    rng = np.random.default_rng(SEED)
    rows = []
    for analysis, rep_a, rep_b in CONTRASTS:
        for norm in sorted(r.norm.unique()):
            sub = r[(r.analysis == analysis) & (r.norm == norm)]
            if sub.empty:
                continue
            for roi, g in sub.groupby("roi"):
                a = g[g.repetition == rep_a].set_index("subject_id").two_way.sort_index()
                b = g[g.repetition == rep_b].set_index("subject_id").two_way.reindex(a.index)
                d = (b - a).values.astype(float)
                n = len(d)
                t, p_par = stats.ttest_rel(b.values, a.values)
                p_exact, p_rand = sign_flip_tests(d, rng)
                dval = cohens_d_paired(d)
                lo, hi = d_ci_noncentral(dval, n)
                bf = bf10_paired(float(t), n)
                rows.append(dict(
                    roi=roi, norm=norm, analysis=analysis,
                    contrast=f"Rep{rep_b}-Rep{rep_a}", n_subj=n,
                    mean_rep_a=float(a.mean()), mean_rep_b=float(b.mean()),
                    delta=float(d.mean()), sem=float(d.std(ddof=1) / np.sqrt(n)),
                    t_stat=float(t), p_parametric=float(p_par),
                    p_exact_perm=p_exact, p_random_perm=p_rand,
                    cohens_d=dval, hedges_g=hedges_g(dval, n),
                    d_ci_lower=lo, d_ci_upper=hi,
                    bf10=bf, bf_interpretation=interpret_bf(bf),
                    n_subj_negative=int((d < 0).sum())))

    out = pd.DataFrame(rows)
    out["is_composite"] = out.roi.isin(COMPOSITES)

    # Two FDR families, both within normalisation x cohort x contrast:
    #  - "all15": every ROI set, as specified in the brief
    #  - "roi11": the 11 anatomical ROIs only. Composites are weighted sums of their
    #    own constituents, so including them alongside inflates the family with
    #    highly dependent tests; this is the defensible primary family.
    out["p_fdr"] = np.nan
    out["p_fdr_roi11"] = np.nan
    for _, idx in out.groupby(["norm", "analysis", "contrast"]).groups.items():
        out.loc[idx, "p_fdr"] = bh_fdr(out.loc[idx, "p_parametric"].values)
        sub = out.loc[idx]
        ind = sub.index[~sub.is_composite]
        out.loc[ind, "p_fdr_roi11"] = bh_fdr(out.loc[ind, "p_parametric"].values)
    out["fdr_significant"] = out.p_fdr < 0.05
    out["fdr_significant_roi11"] = out.p_fdr_roi11 < 0.05

    cols = ["roi", "is_composite", "norm", "analysis", "contrast", "n_subj",
            "mean_rep_a", "mean_rep_b", "delta", "sem", "t_stat", "p_parametric",
            "p_exact_perm", "p_random_perm", "p_fdr", "fdr_significant",
            "p_fdr_roi11", "fdr_significant_roi11", "cohens_d", "hedges_g",
            "d_ci_lower", "d_ci_upper", "bf10", "bf_interpretation", "n_subj_negative"]
    out[cols].to_csv(RESULTS / "statistical_infrastructure.csv", index=False)
    print(f"[OK] results/statistical_infrastructure.csv  ({len(out)} rows)")

    pd.set_option("display.width", 250)
    for (an, norm, c), g in out.groupby(["analysis", "norm", "contrast"]):
        print(f"\n=== {an} | norm={norm} | {c} ===")
        show = g.sort_values("p_parametric")[
            ["roi", "delta", "t_stat", "p_parametric", "p_exact_perm", "p_fdr",
             "p_fdr_roi11", "cohens_d", "d_ci_lower", "d_ci_upper", "bf10",
             "bf_interpretation"]]
        print(show.round(4).to_string(index=False))
        s15 = g[g.fdr_significant].roi.tolist()
        s11 = g[g.fdr_significant_roi11].roi.tolist()
        print(f"  survives FDR (all 15 ROI sets): {s15 if s15 else 'none'}")
        print(f"  survives FDR (11 anatomical ROIs): {s11 if s11 else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
