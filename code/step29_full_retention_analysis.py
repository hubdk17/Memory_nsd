"""
Step 29: retention dynamics over the full experiment.

Repeats the step24/25/26 analyses on the complete session set. Two things change.

First, the inference. With two sessions each subject contributed one number per
condition and the test was a paired t across 8 subjects. With 40 sessions a single
subject contributes thousands of stimulus pairs per retention bin, so inference runs
WITHIN subject by bootstrapping over stimuli, and the across-subject test becomes a
replication check rather than the primary analysis.

Second, and more important, the confound. In the two-session data every long interval
crossed the session boundary, so retention and session-crossing were collinear and the
decay could not be attributed to elapsed time. Here every bin past one day is
near-entirely cross-session, so contrasting weeks against months against quarters holds
session-crossing constant and varies only time. That contrast is the point of the whole
download, and it is analysis E below.

  A  reinstatement per ROI per retention bin
  B  continuous decay of reinstatement against log10 interval
  C  cross-region coupling per bin -- the Phase-II finding, now with a dose-response
  D  coupling against log10 interval, bootstrapped
  E  long-bins-only contrast, session-crossing held constant
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _full_data as fd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROIS = ["V1", "V2", "V3", "hV4", "FFA", "OPA", "PPA", "RSC", "PHC",
        "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
COMPOSITES = {"EarlyVis": ["V1", "V2"], "SceneCtx": ["PPA", "RSC", "PHC"],
              "Hippocampus": ["SUB", "CA1", "DG_CA3"]}
PAIRS = [("V1", "PPA"), ("V1", "Hippocampus"), ("V1", "SceneCtx"), ("V1", "ERC"),
         ("PPA", "Hippocampus"), ("SceneCtx", "Hippocampus"), ("PPA", "ERC"),
         ("EarlyVis", "SceneCtx"), ("EarlyVis", "Hippocampus"), ("V1", "V2"),
         ("V1", "V3"), ("V3", "PPA")]
BIN_ORDER = ["min", "hours", "days", "weeks", "months", "quarters"]
N_BOOT = 2000
SEED = 42


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


def stability_blocked(P1, P2, block=1500, rng=None):
    """Within-item minus between-item reinstatement, computed in blocks.

    The between-item baseline for item i is its mean similarity to other items' second
    presentations. With 10,000 pairs the full matrix is 10k x 10k, so items are chunked
    and each chunk is compared against a fixed random reference sample.
    """
    n = len(P1)
    rng = rng or np.random.default_rng(SEED)
    ref = rng.choice(n, min(800, n), replace=False)
    Pref = P2[ref]
    out = np.empty(n, dtype=np.float64)
    within = fisher(corr_rows(P1, P2))
    for s in range(0, n, block):
        e = min(s + block, n)
        C = fisher(corr_matrix(P1[s:e], Pref))
        for k, i in enumerate(range(s, e)):
            hit = np.where(ref == i)[0]
            if len(hit):
                C[k, hit] = np.nan
        out[s:e] = within[s:e] - np.nanmean(C, axis=1)
    return out


def boot_ci(x, stat=np.mean, n=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), (n, len(x)))
    d = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_corr(x, y, n=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 30 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, np.nan, len(x)
    r = stats.pearsonr(x, y).statistic
    idx = rng.integers(0, len(x), (n, len(x)))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(1, keepdims=True)
    ys = ys - ys.mean(1, keepdims=True)
    d = (xs * ys).sum(1) / (np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1)) + 1e-12)
    return float(r), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--rep", type=int, default=1, help="repetition transition (1 = Rep1->Rep2)")
    args = ap.parse_args()

    try:
        design = pd.read_csv(RESULTS / "retention_design_full.csv")
    except FileNotFoundError:
        print("[ERROR] run step28_full_retention_design.py first.")
        return 1
    subjects = args.subjects or sorted(design.subject_id.unique())
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 230)

    stab_rows, coup_rows, cont_rows = [], [], []
    for subj in subjects:
        d = design[(design.subject_id == subj) & (design.usable == 1)
                   & (design.rep_from == args.rep)].reset_index(drop=True)
        if len(d) < 200:
            print(f"  [skip] {subj}: only {len(d)} usable pairs")
            continue
        avail = set(fd.available_rois(subj))
        rois = [r for r in ROIS if r in avail]
        comps = {k: v for k, v in COMPOSITES.items() if all(p in avail for p in v)}
        print(f"\n=== {subj}: {len(d):,} Rep{args.rep}->Rep{args.rep+1} pairs, "
              f"{len(fd.available_sessions(subj))} sessions ===")

        S = {}
        for name, parts in [(r, [r]) for r in rois] + list(comps.items()):
            X, tidx, _ = fd.load_rois(subj, parts, norm="session")
            pos = {t: i for i, t in enumerate(tidx)}
            ia = np.array([pos[t] for t in d.trial_a])
            ib = np.array([pos[t] for t in d.trial_b])
            S[name] = stability_blocked(X[ia], X[ib], rng=rng)
            del X
            gc.collect()
        print(f"   reinstatement computed for {len(S)} ROI sets")

        # ---- A: per bin
        for name, v in S.items():
            for b in BIN_ORDER:
                m = (d.bin == b).values
                if m.sum() < 30:
                    continue
                mu, lo, hi = boot_ci(v[m], rng=rng)
                stab_rows.append(dict(subject_id=subj, roi=name, bin=b,
                                      n=int(m.sum()), stability=mu,
                                      ci_lo=lo, ci_hi=hi,
                                      median_lag_days=float(d.lag_days[m].median())))
        # ---- B: continuous
        for name, v in S.items():
            ok = np.isfinite(v)
            lr = stats.linregress(d.log10_lag_min[ok], v[ok])
            cs = (d.cross_session == 1).values & ok
            lr_cs = (stats.linregress(d.log10_lag_min[cs], v[cs])
                     if cs.sum() > 200 and d.log10_lag_min[cs].std() > 0 else None)
            cont_rows.append(dict(subject_id=subj, roi=name, n=int(ok.sum()),
                                  slope=lr.slope, p=lr.pvalue, r=lr.rvalue,
                                  slope_crosssession=lr_cs.slope if lr_cs else np.nan,
                                  p_crosssession=lr_cs.pvalue if lr_cs else np.nan,
                                  n_crosssession=int(cs.sum())))
        # ---- C/D/E: coupling
        for a, b in PAIRS:
            if a not in S or b not in S:
                continue
            for scope, mask in ([(x, (d.bin == x).values) for x in BIN_ORDER]
                                + [("all", np.ones(len(d), bool)),
                                   ("long_crosssession",
                                    ((d.cross_session == 1)
                                     & d.bin.isin(["weeks", "months", "quarters"])).values)]):
                if mask.sum() < 50:
                    continue
                r, lo, hi, n = boot_corr(S[a][mask], S[b][mask], rng=rng)
                coup_rows.append(dict(subject_id=subj, roi_a=a, roi_b=b, scope=scope,
                                      n=n, r=r, ci_lo=lo, ci_hi=hi,
                                      median_lag_days=float(d.lag_days[mask].median())))
        del S
        gc.collect()

    if not stab_rows:
        print("[ERROR] nothing computed.")
        return 1
    A = pd.DataFrame(stab_rows)
    B = pd.DataFrame(cont_rows)
    C = pd.DataFrame(coup_rows)
    A.to_csv(RESULTS / f"full_reinstatement_by_bin_rep{args.rep}.csv", index=False)
    B.to_csv(RESULTS / f"full_reinstatement_slopes_rep{args.rep}.csv", index=False)
    C.to_csv(RESULTS / f"full_coupling_by_bin_rep{args.rep}.csv", index=False)

    print("\n=== A. Reinstatement by retention bin ===")
    piv = A.pivot_table(index="roi", columns="bin", values="stability")
    print(piv[[b for b in BIN_ORDER if b in piv.columns]].round(4).to_string())

    print("\n=== B. Reinstatement vs log10(interval) ===")
    print(B[["roi", "n", "slope", "p", "slope_crosssession", "p_crosssession",
             "n_crosssession"]].sort_values("p").round(5).to_string(index=False))

    print("\n=== C. Cross-region coupling by retention bin (r) ===")
    C["pair"] = C.roi_a + "-" + C.roi_b
    pc = C.pivot_table(index="pair", columns="scope", values="r")
    cols = [b for b in BIN_ORDER if b in pc.columns] + \
           [c for c in ("long_crosssession", "all") if c in pc.columns]
    print(pc[cols].round(4).to_string())

    print("\n=== E. Long bins only, session-crossing held constant ===")
    longs = [b for b in ("weeks", "months", "quarters") if b in pc.columns]
    if len(longs) >= 2:
        sub = C[C.scope.isin(longs)]
        out = []
        for pair, g in sub.groupby("pair"):
            g = g.set_index("scope").reindex(longs)
            first, last = g.r.iloc[0], g.r.iloc[-1]
            sep = (g.ci_hi.iloc[0] < g.ci_lo.iloc[-1]) or (g.ci_lo.iloc[0] > g.ci_hi.iloc[-1])
            out.append(dict(pair=pair, **{b: g.r.get(b) for b in longs},
                            change=last - first, ci_disjoint=sep))
        E = pd.DataFrame(out).sort_values("change")
        print(E.round(4).to_string(index=False))
        E.to_csv(RESULTS / f"full_coupling_longbins_rep{args.rep}.csv", index=False)
        print("\n  ci_disjoint = the bootstrap 95% CIs of the first and last long bin do")
        print("  not overlap. Every pair here crosses a session boundary, so a change")
        print("  across these bins is elapsed time, not session identity.")
    else:
        print("  need at least two long bins on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
