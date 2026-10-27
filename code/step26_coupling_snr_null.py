"""
Step 26: the decisive test. Does uniform signal degradation reproduce the decoupling?

Step 25 found that cross-region coupling of trace strength collapses over a ~8 day
retention interval (PPA-Hippocampus 0.47 -> 0.20) while coupling within early visual
cortex is preserved (V1-V2 0.855 -> 0.837). The obvious objection is that the long
interval simply has worse signal, and correlations attenuate with noise.

The test: take the SHORT-interval data and add noise to each ROI, calibrated so that ROI
reproduces its OWN observed long-interval reinstatement. Then recompute the coupling. If
the degraded-short coupling matches the observed long coupling, attenuation explains it
and there is no decoupling to report.

The reason this is informative rather than circular: PPA reinstatement barely moves over
the interval (0.087 -> 0.084) and hippocampal reinstatement not at all (0.0071 ->
0.0069), so the calibrated noise for those regions is near zero -- yet their coupling
halves. Attenuation cannot produce a coupling drop without a reinstatement drop.

Two noise models, because the confound they represent differs:
  independent  each ROI gets its own noise. Attenuates coupling.
  shared       one noise field scaled per ROI, i.e. a global session-level fluctuation
               (motion, arousal, global signal). This is the realistic model for a
               session confound, and it INFLATES coupling -- so if the real data shows
               decoupling, a shared-noise session effect is the wrong explanation.
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
ROI_SETS = {"V1": ["V1"], "V2": ["V2"], "EarlyVis": ["V1", "V2"], "FFA": ["FFA"],
            "PPA": ["PPA"], "RSC": ["RSC"], "PHC": ["PHC"], "ERC": ["ERC"],
            "SceneCtx": ["PPA", "RSC", "PHC"], "Hippocampus": ["SUB", "CA1", "DG_CA3"]}
PAIRS = [("V1", "PPA"), ("V1", "Hippocampus"), ("V1", "SceneCtx"), ("V1", "ERC"),
         ("PPA", "Hippocampus"), ("SceneCtx", "Hippocampus"), ("PPA", "ERC"),
         ("EarlyVis", "SceneCtx"), ("EarlyVis", "Hippocampus"), ("V1", "V2")]
N_BETWEEN = 60
N_DRAWS = 40
SEED = 42


def zscore_by(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def fisher(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def corr_rows(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-12)


def corr_matrix(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def stability(P1, P2):
    within = fisher(corr_rows(P1, P2))
    C = fisher(corr_matrix(P1, P2))
    np.fill_diagonal(C, np.nan)
    return within - np.nanmean(C, axis=1)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def calibrate(P1, P2, target, rng, k=4):
    """Noise scale making degraded-short reinstatement match the long-interval mean."""
    base = float(np.nanmean(stability(P1, P2)))
    if target >= base:
        return 0.0                       # region did not lose trace strength
    sd = P1.std()

    def val(sig):
        return float(np.mean([np.nanmean(stability(
            P1 + rng.normal(0, sig, P1.shape) * sd, P2)) for _ in range(k)]))

    lo, hi = 0.0, 6.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if val(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    try:
        design = pd.read_csv(RESULTS / "retention_design.csv")
    except FileNotFoundError:
        print("[ERROR] run step23 first.")
        return 1
    design = design[design.retention_bin.isin(["short", "long"])]
    rng = np.random.default_rng(SEED)
    rows = []

    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        d = design[design.subject_id == subj]
        sh = d[d.retention_bin == "short"]
        lg = d[d.retention_bin == "long"]
        sess = np.repeat([1, 2], 750)

        P1, P2, obs_short, obs_long, sigma = {}, {}, {}, {}, {}
        for roi, parts in ROI_SETS.items():
            X = zscore_by(np.concatenate([z[p] for p in parts], 1).astype(np.float64),
                          sess)
            P1[roi] = X[sh.trial_rep1.values - 1]
            P2[roi] = X[sh.trial_rep2.values - 1]
            obs_short[roi] = stability(P1[roi], P2[roi])
            obs_long[roi] = stability(X[lg.trial_rep1.values - 1],
                                      X[lg.trial_rep2.values - 1])
            sigma[roi] = calibrate(P1[roi], P2[roi],
                                   float(np.nanmean(obs_long[roi])), rng)

        def coupling(vals, a, b):
            x, y = vals[a], vals[b]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 20 or x[m].std() == 0 or y[m].std() == 0:
                return np.nan
            return fisher(stats.pearsonr(x[m], y[m]).statistic)

        n_long = len(lg)
        for model in ("independent", "shared", "shared_nmatched"):
            draws = {p: [] for p in PAIRS}
            for _ in range(N_DRAWS):
                # n_matched: subsample the short pairs to the long-interval sample size,
                # so the null and the observation are estimated from equally many stimuli
                idx = (rng.choice(len(sh), min(n_long, len(sh)), replace=False)
                       if model == "shared_nmatched" else np.arange(len(sh)))
                deg = {}
                if model.startswith("shared"):
                    g = rng.normal(0, 1, (len(idx), 1))
                for roi in ROI_SETS:
                    s = sigma[roi]
                    A, B = P1[roi][idx], P2[roi][idx]
                    if s <= 0:
                        deg[roi] = stability(A, B)
                        continue
                    sd = A.std()
                    if model == "independent":
                        noise = rng.normal(0, s, A.shape) * sd
                    else:
                        w = rng.normal(0, 1, (1, A.shape[1]))
                        noise = (g @ w) * s * sd
                    deg[roi] = stability(A + noise, B)
                for a, b in PAIRS:
                    draws[(a, b)].append(coupling(deg, a, b))
            for a, b in PAIRS:
                rows.append(dict(
                    subject_id=subj, roi_a=a, roi_b=b, noise_model=model,
                    z_short=coupling(obs_short, a, b),
                    z_long=coupling(obs_long, a, b),
                    z_null=float(np.nanmean(draws[(a, b)])),
                    z_null_sd=float(np.nanstd(draws[(a, b)], ddof=1)),
                    sigma_a=sigma[a], sigma_b=sigma[b]))
        print(f"  [OK] {subj}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "coupling_snr_null_raw.csv", index=False)
    pd.set_option("display.width", 230)

    print("\n=== Calibrated noise required per ROI (0 = region lost no trace strength) ===")
    sg = df.drop_duplicates(["subject_id", "roi_a"])[["roi_a", "sigma_a"]]
    print(sg.groupby("roi_a").sigma_a.mean().round(3).sort_values(
        ascending=False).to_string())

    for model in ("independent", "shared", "shared_nmatched"):
        sub = df[df.noise_model == model]
        print(f"\n=== Observed long-interval coupling vs the {model}-noise null ===")
        out = []
        for (a, b), g in sub.groupby(["roi_a", "roi_b"], sort=False):
            g = g.dropna(subset=["z_short", "z_long", "z_null"])
            if len(g) < 4:
                continue
            t, p = stats.ttest_rel(g.z_long, g.z_null)
            out.append(dict(pair=f"{a}-{b}",
                            r_short=np.tanh(g.z_short.mean()),
                            r_long=np.tanh(g.z_long.mean()),
                            r_null=np.tanh(g.z_null.mean()),
                            obs_minus_null=(g.z_long - g.z_null).mean(),
                            t=t, p=p, n_neg=int((g.z_long < g.z_null).sum()),
                            n=len(g)))
        O = pd.DataFrame(out)
        O["p_fdr"] = bh_fdr(O.p.values)
        print(O.round(4).to_string(index=False))
        O.to_csv(RESULTS / f"coupling_snr_null_{model}.csv", index=False)

    print("\n  obs_minus_null < 0 with a significant p means the real long-interval")
    print("  coupling is LOWER than noise alone can account for: genuine decoupling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
