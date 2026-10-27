"""
Step 30: the inferential test the claim actually requires.

Step 29 reports coupling inside each retention bin with bootstrap intervals. That shows
a pattern but does not test it. The claim is a difference of slopes: sensory-mnemonic
coupling declines with retention interval MORE than within-sensory coupling does. A
uniform loss of signal over nine months would drag every pair's coupling down together,
so the contrast against a within-sensory reference is what separates fragmentation of
the trace from degradation of the measurement.

For each region pair, coupling is estimated in overlapping windows along log10 interval,
then regressed on window centre to give a slope. Stimulus-level bootstrap resampling
propagates uncertainty through both steps, and the same resample is used for every pair
within a draw so that the slope DIFFERENCE is properly paired.

Two reference pairs, both within early visual cortex and both expected to be flat:
V1-V2 and V1-V3. Reporting against both guards against the objection that V1 and V2
share spatially correlated noise, since V3 is further away.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _full_data as fd
from step29_full_retention_analysis import (COMPOSITES, ROIS, fisher, stability_blocked)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TEST_PAIRS = [("V1", "PPA"), ("V1", "SceneCtx"), ("V1", "ERC"), ("V1", "Hippocampus"),
              ("PPA", "Hippocampus"), ("SceneCtx", "Hippocampus"), ("PPA", "ERC"),
              ("EarlyVis", "SceneCtx"), ("EarlyVis", "Hippocampus"), ("V3", "PPA")]
REFERENCE = [("V1", "V2"), ("V1", "V3")]
N_WINDOWS = 8
MIN_PER_WINDOW = 60
N_BOOT = 1000
SEED = 42


def windowed_slope(x_lag, ya, yb, edges):
    """Coupling (Fisher-z) per lag window, then OLS slope on window centre."""
    cs, zs = [], []
    for lo, hi in edges:
        m = (x_lag >= lo) & (x_lag < hi) & np.isfinite(ya) & np.isfinite(yb)
        if m.sum() < MIN_PER_WINDOW:
            continue
        a, b = ya[m], yb[m]
        if a.std() == 0 or b.std() == 0:
            continue
        cs.append(0.5 * (lo + hi))
        zs.append(fisher(np.corrcoef(a, b)[0, 1]))
    if len(cs) < 3:
        return np.nan, np.nan
    sl = np.polyfit(cs, zs, 1)[0]
    return sl, float(np.mean(zs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["subj01"])
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()

    try:
        design = pd.read_csv(RESULTS / "retention_design_full.csv")
    except FileNotFoundError:
        print("[ERROR] run step28 first.")
        return 1
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 230)
    out_rows = []

    for subj in args.subjects:
        d = design[(design.subject_id == subj) & (design.usable == 1)
                   & (design.rep_from == args.rep)].reset_index(drop=True)
        if len(d) < 500:
            print(f"  [skip] {subj}: {len(d)} pairs")
            continue
        avail = set(fd.available_rois(subj))
        need = sorted({r for p in TEST_PAIRS + REFERENCE for r in p})
        sets = {}
        for name in need:
            parts = COMPOSITES.get(name, [name])
            if not all(p in avail for p in parts):
                continue
            sets[name] = parts
        print(f"\n=== {subj}: {len(d):,} pairs, {len(fd.available_sessions(subj))} "
              f"sessions, {len(sets)} ROI sets ===")

        S = {}
        for name, parts in sets.items():
            X, tidx, _ = fd.load_rois(subj, parts, norm="session")
            pos = {t: i for i, t in enumerate(tidx)}
            ia = np.array([pos[t] for t in d.trial_a])
            ib = np.array([pos[t] for t in d.trial_b])
            S[name] = stability_blocked(X[ia], X[ib], rng=rng)
            del X
            gc.collect()

        lag = d.log10_lag_min.values
        qs = np.linspace(0, 100, N_WINDOWS + 1)
        cuts = np.percentile(lag, qs)
        edges = [(cuts[i], cuts[i + 1]) for i in range(N_WINDOWS)]
        edges[-1] = (edges[-1][0], edges[-1][1] + 1e-6)
        print(f"   lag windows (log10 min): "
              + ", ".join(f"{a:.1f}-{b:.1f}" for a, b in edges))

        pairs = [p for p in TEST_PAIRS + REFERENCE if p[0] in S and p[1] in S]
        obs = {p: windowed_slope(lag, S[p[0]], S[p[1]], edges)[0] for p in pairs}

        n = len(d)
        boot = {p: np.empty(N_BOOT) for p in pairs}
        for k in range(N_BOOT):
            idx = rng.integers(0, n, n)          # one resample shared by all pairs
            lk = lag[idx]
            ck = np.percentile(lk, qs)
            ek = [(ck[i], ck[i + 1]) for i in range(N_WINDOWS)]
            ek[-1] = (ek[-1][0], ek[-1][1] + 1e-6)
            for p in pairs:
                boot[p][k] = windowed_slope(lk, S[p[0]][idx], S[p[1]][idx], ek)[0]
            if (k + 1) % 250 == 0:
                print(f"   bootstrap {k+1}/{N_BOOT}", flush=True)

        for p in pairs:
            bs = boot[p][np.isfinite(boot[p])]
            rec = dict(subject_id=subj, pair=f"{p[0]}-{p[1]}",
                       is_reference=p in REFERENCE, slope=obs[p],
                       ci_lo=float(np.percentile(bs, 2.5)) if len(bs) else np.nan,
                       ci_hi=float(np.percentile(bs, 97.5)) if len(bs) else np.nan,
                       p_vs_zero=float(2 * min((bs < 0).mean(), (bs > 0).mean()))
                       if len(bs) else np.nan)
            for ref in REFERENCE:
                if ref not in boot or p == ref:
                    continue
                diff = boot[p] - boot[ref]
                diff = diff[np.isfinite(diff)]
                tag = f"{ref[0]}{ref[1]}"
                rec[f"vs_{tag}_diff"] = float(np.mean(diff)) if len(diff) else np.nan
                rec[f"vs_{tag}_p"] = (float(2 * min((diff < 0).mean(), (diff > 0).mean()))
                                      if len(diff) else np.nan)
            out_rows.append(rec)
        del S
        gc.collect()

    if not out_rows:
        print("[ERROR] nothing computed.")
        return 1
    R = pd.DataFrame(out_rows)
    R.to_csv(RESULTS / f"coupling_slope_contrast_rep{args.rep}.csv", index=False)

    print("\n=== Coupling slope vs log10(retention interval) ===")
    print("  negative = coupling falls as the interval lengthens")
    cols = ["pair", "is_reference", "slope", "ci_lo", "ci_hi", "p_vs_zero"]
    print(R.sort_values("slope")[cols].round(4).to_string(index=False))

    print("\n=== Contrast against the within-early-visual reference pairs ===")
    print("  negative diff = this pair decouples MORE than the reference does")
    cc = [c for c in R.columns if c.startswith("vs_")]
    show = R[~R.is_reference][["pair"] + cc]
    print(show.round(4).to_string(index=False))
    sig = show[(show.get("vs_V1V2_p", 1) < 0.05) & (show.get("vs_V1V3_p", 1) < 0.05)]
    print(f"\n  pairs decoupling significantly more than BOTH references: "
          f"{len(sig)} of {len(show)}")
    if len(sig):
        print("   " + ", ".join(sig.pair))
    return 0


if __name__ == "__main__":
    sys.exit(main())
