"""
Step 24: how does each region's memory trace survive the retention interval?

The measure is pattern reinstatement: the correlation between a stimulus's Rep1 and Rep2
voxel patterns, minus the correlation between that stimulus's Rep1 pattern and OTHER
stimuli's Rep2 patterns drawn from the same retention bin. The subtraction is what makes
this a memory measure rather than a similarity measure -- it removes any global shift
(session drift, arousal, head position) that moves every pattern in the same direction,
which matters here because long-interval pairs mostly cross the session boundary.

    stability_i = z[ r(B1_i, B2_i) ] - mean_j!=i z[ r(B1_i, B2_j) ]

computed on Fisher-z transformed correlations. Zero means the trace carries no
item-specific information at the second exposure.

Three tests, in order of how much weight they can bear:
  A  ROI x retention-bin interaction. Scanner drift moves all regions together, so a
     regional dissociation is the contrast that survives the session confound.
  B  stability ~ log10(retention interval) as a continuous mixed model, per ROI. Uses all
     2,584 pairs across 5.6 orders of magnitude rather than a binary split.
  C  the same continuous test restricted to the 504 cross-session pairs, where
     session-crossing is held constant and only the 1-19 day interval varies.

Every effect is also compared against an SNR-matched null: Gaussian noise is added to the
Rep1 patterns, scaled so that the short-bin stability falls to the observed long-bin
value, and the ROI profile is recomputed. If the real long-bin profile sits inside that
null, the result does not distinguish forgetting from measurement noise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROIS = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_SETS = {r: [r] for r in ROIS}
ROI_SETS.update({"EarlyVis": ["V1", "V2"], "SceneCtx": ["PPA", "RSC", "PHC"],
                 "Hippocampus": ["SUB", "CA1", "DG_CA3"],
                 "MTLcortex": ["PRC", "ERC"]})
N_BETWEEN = 60          # distractor stimuli per item for the between-item baseline
N_NULL = 20             # SNR-matched null draws
# A proportion-retained ratio divides by short-interval reinstatement. Regions whose
# reinstatement is at floor (hippocampal subfields, ~0.007) give meaningless ratios, so
# they are flagged and excluded from every ratio-based test.
MIN_BASE = 0.02
SEED = 42
RNG = np.random.default_rng(SEED)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def zscore_by(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def fisher(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise correlation between matched rows of A and B."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-12
    return num / den


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


def stability(P1: np.ndarray, P2: np.ndarray, rng) -> np.ndarray:
    """Item-specific reinstatement: within-item minus between-item, Fisher-z."""
    n = len(P1)
    within = fisher(corr_rows(P1, P2))
    k = min(N_BETWEEN, n - 1)
    if k < 5:
        return within - np.nan
    C = fisher(corr_matrix(P1, P2))
    np.fill_diagonal(C, np.nan)
    between = np.nanmean(C, axis=1)
    return within - between


def add_matched_noise(P: np.ndarray, sigma: float, rng) -> np.ndarray:
    return P + rng.normal(0.0, sigma, P.shape) * P.std()


def main() -> int:
    try:
        design = pd.read_csv(RESULTS / "retention_design.csv")
    except FileNotFoundError:
        print("[ERROR] run step23_retention_design.py first.")
        return 1
    design = design[design.retention_bin.isin(["short", "long"])]

    rows, null_rows, item_rows = [], [], []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        d = design[design.subject_id == subj]
        sess = np.repeat([1, 2], 750)

        for roi, parts in ROI_SETS.items():
            X = zscore_by(np.concatenate([z[p] for p in parts], 1).astype(np.float64),
                          sess)
            per_bin = {}
            for rbin, g in d.groupby("retention_bin"):
                i1 = g.trial_rep1.values - 1
                i2 = g.trial_rep2.values - 1
                s = stability(X[i1], X[i2], RNG)
                per_bin[rbin] = s
                rows.append(dict(subject_id=subj, roi=roi, retention_bin=rbin,
                                 n_pairs=len(s), stability=float(np.nanmean(s)),
                                 sem=float(np.nanstd(s, ddof=1) / np.sqrt(len(s))),
                                 median_lag_days=float(g.lag_days.median())))
                for nsd, lag, lgm, cs, val in zip(g.nsd_id, g.lag_days,
                                                  g.log10_lag_min, g.cross_session, s):
                    item_rows.append(dict(subject_id=subj, roi=roi, nsd_id=int(nsd),
                                          retention_bin=rbin, lag_days=float(lag),
                                          log10_lag_min=float(lgm),
                                          cross_session=int(cs),
                                          stability=float(val)))

            # SNR-matched null. The question is not whether noise CAN reproduce the
            # long-interval value -- with enough noise it always can -- but HOW MUCH
            # noise each region needs. Under uniform degradation a single noise level
            # explains every ROI; a region that needs much more or less has changed for
            # a reason other than signal loss. Each bisection step averages several
            # draws, because a single stochastic draw biases the search.
            if "short" in per_bin and "long" in per_bin:
                target = np.nanmean(per_bin["long"])
                g = d[d.retention_bin == "short"]
                i1, i2 = g.trial_rep1.values - 1, g.trial_rep2.values - 1
                base = float(np.nanmean(per_bin["short"]))

                def degraded(sig: float, k: int = 5) -> float:
                    return float(np.mean([np.nanmean(stability(
                        add_matched_noise(X[i1], sig, RNG), X[i2], RNG))
                        for _ in range(k)]))

                lo, hi = 0.0, 8.0
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    if degraded(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                sigma = 0.5 * (lo + hi)
                draws = [float(np.nanmean(stability(
                    add_matched_noise(X[i1], sigma, RNG), X[i2], RNG)))
                    for _ in range(N_NULL)]
                null_rows.append(dict(subject_id=subj, roi=roi, noise_sigma=sigma,
                                      short_stability=base, observed_long=float(target),
                                      null_mean=float(np.mean(draws)),
                                      null_sd=float(np.std(draws, ddof=1)),
                                      baseline_ok=bool(base > MIN_BASE)))
        print(f"  [OK] {subj}", flush=True)

    res = pd.DataFrame(rows)
    items = pd.DataFrame(item_rows)
    nulls = pd.DataFrame(null_rows)
    res.to_csv(RESULTS / "retention_stability_by_roi.csv", index=False)
    items.to_csv(RESULTS / "retention_stability_items.csv", index=False)
    nulls.to_csv(RESULTS / "retention_snr_matched_null.csv", index=False)

    pd.set_option("display.width", 220)

    # ---- Test A: short vs long per ROI, paired across subjects
    print("\n=== A. Reinstatement by retention interval (paired across 8 subjects) ===")
    out = []
    for roi, g in res.groupby("roi"):
        w = g.pivot_table(index="subject_id", columns="retention_bin", values="stability")
        if not {"short", "long"} <= set(w.columns):
            continue
        t, p = stats.ttest_rel(w["long"], w["short"])
        dv = (w["long"] - w["short"])
        out.append(dict(roi=roi, short=w["short"].mean(), long=w["long"].mean(),
                        delta=dv.mean(), sem=dv.std(ddof=1) / np.sqrt(len(dv)),
                        pct_retained=100 * w["long"].mean() / w["short"].mean()
                        if w["short"].mean() > 0 else np.nan,
                        t=t, p=p, d=dv.mean() / dv.std(ddof=1)))
    A = pd.DataFrame(out).sort_values("short", ascending=False)
    print(A.round(4).to_string(index=False))
    A.to_csv(RESULTS / "retention_bin_contrast.csv", index=False)

    # ---- Test A2: the regional dissociation, which drift cannot explain
    print("\n=== A2. Regional dissociation in PROPORTION retained (V1 as reference) ===")
    ok_rois = sorted(res[(res.retention_bin == "short")]
                     .groupby("roi").stability.mean().pipe(lambda s: s[s > MIN_BASE]).index)
    dropped = [r for r in res.roi.unique() if r not in ok_rois]
    print(f"  interpretable ROIs (short-interval reinstatement > {MIN_BASE}): "
          f"{len(ok_rois)}")
    print(f"  excluded, reinstatement at floor: {', '.join(sorted(dropped))}")
    w = res[res.roi.isin(ok_rois)].pivot_table(
        index=["subject_id", "roi"], columns="retention_bin",
        values="stability").reset_index()
    w["prop"] = w["long"] / w["short"].replace(0, np.nan)
    piv = w.pivot_table(index="subject_id", columns="roi", values="prop")
    diss = []
    for roi in piv.columns:
        if roi == "V1":
            continue
        dv = (piv[roi] - piv["V1"]).dropna()
        if len(dv) < 4:
            continue
        t, p = stats.ttest_1samp(dv, 0.0)
        diss.append(dict(roi=roi, mean_prop=piv[roi].mean(), v1_prop=piv["V1"].mean(),
                         diff_vs_V1=dv.mean(), t=t, p=p,
                         d=dv.mean() / dv.std(ddof=1),
                         n_subj_positive=int((dv > 0).sum())))
    D = pd.DataFrame(diss).sort_values("p")
    D["p_fdr"] = bh_fdr(D.p.values)
    print(D.round(4).to_string(index=False))
    D.to_csv(RESULTS / "retention_regional_dissociation.csv", index=False)

    # ---- Test B/C: continuous log-lag slope
    print("\n=== B/C. Stability vs log10(retention minutes), per ROI ===")
    slopes = []
    for roi, g in items.groupby("roi"):
        per = []
        for subj, gs in g.groupby("subject_id"):
            gs = gs.dropna(subset=["stability"])
            if len(gs) > 30 and gs.log10_lag_min.std() > 0:
                per.append(stats.linregress(gs.log10_lag_min, gs.stability).slope)
        cs = g[g.cross_session == 1]
        per_cs = []
        for subj, gs in cs.groupby("subject_id"):
            gs = gs.dropna(subset=["stability"])
            if len(gs) > 20 and gs.log10_lag_min.std() > 0:
                per_cs.append(stats.linregress(gs.log10_lag_min, gs.stability).slope)
        if len(per) < 4:
            continue
        t, p = stats.ttest_1samp(per, 0.0)
        rec = dict(roi=roi, slope_all=float(np.mean(per)),
                   sem_all=float(np.std(per, ddof=1) / np.sqrt(len(per))),
                   t_all=t, p_all=p, n_subj=len(per))
        if len(per_cs) >= 4:
            t2, p2 = stats.ttest_1samp(per_cs, 0.0)
            rec.update(slope_crosssession=float(np.mean(per_cs)), t_cs=t2, p_cs=p2)
        slopes.append(rec)
    B = pd.DataFrame(slopes).sort_values("p_all")
    print(B.round(5).to_string(index=False))
    B.to_csv(RESULTS / "retention_logslope_by_roi.csv", index=False)

    # ---- SNR-matched null verdict
    print("\n=== SNR-matched null: how much added noise does each region need to "
          "explain its own decay? ===")
    nn = nulls[nulls.baseline_ok].groupby("roi").agg(
        short=("short_stability", "mean"), long=("observed_long", "mean"),
        sigma=("noise_sigma", "mean"),
        sigma_sem=("noise_sigma", lambda s: s.std(ddof=1) / np.sqrt(len(s)))
    ).reset_index().sort_values("sigma", ascending=False)
    print(nn.round(4).to_string(index=False))
    nn.to_csv(RESULTS / "retention_snr_null_summary.csv", index=False)

    ref = nn[nn.roi == "V1"]
    if len(ref):
        sv = float(ref.sigma.iloc[0])
        print(f"\n  V1 requires sigma = {sv:.3f}. Under uniform degradation every region "
              f"would need the same.")
        per = nulls[nulls.baseline_ok].pivot_table(index="subject_id", columns="roi",
                                                   values="noise_sigma")
        cmp = []
        for roi in per.columns:
            if roi == "V1":
                continue
            dv = (per[roi] - per["V1"]).dropna()
            if len(dv) < 4:
                continue
            t, p = stats.ttest_1samp(dv, 0.0)
            cmp.append(dict(roi=roi, sigma=per[roi].mean(), sigma_V1=per["V1"].mean(),
                            diff_vs_V1=dv.mean(), t=t, p=p,
                            d=dv.mean() / dv.std(ddof=1)))
        C = pd.DataFrame(cmp).sort_values("p")
        C["p_fdr"] = bh_fdr(C.p.values)
        print(C.round(4).to_string(index=False))
        C.to_csv(RESULTS / "retention_snr_sigma_contrast.csv", index=False)
        print("\n  A region needing significantly LESS noise than V1 decayed less than "
              "uniform\n  signal loss predicts -- that is a regional effect, not a "
              "measurement artefact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
