"""
Step 25: does the retention effect survive the session confound, and do regions decay
together?

Step 24 found that early visual reinstatement falls with log retention interval while
scene-selective and entorhinal cortex hold. Within sessions 1-2 a long interval almost
always crosses the session boundary, so that result has to clear three checks before it
means anything about memory.

  A  Within-session long pairs. Subjects 01 and 08 had their second session spread over
     about seven days, which yields 38 pairs that are long-interval WITHOUT crossing a
     session boundary. Small, but it is the only fully unconfounded cell in the data.
  B  Dose-response across subjects. The session gap varies 6.9 to 14.0 days between
     subjects. If the decay is driven by elapsed time, subjects with longer gaps should
     lose more; if it is driven by the act of re-entering the scanner, gap length should
     not matter.
  C  Pure session effect. Short pairs entirely inside session 1 versus entirely inside
     session 2 share the retention interval and differ only in which session they came
     from, so any difference between them is the session effect with retention held out.

Then the coupling question: within a stimulus, does a strong sensory trace go with a
strong mnemonic trace, and does that relationship change as the interval lengthens?
Coupling is computed across stimuli within each subject, Fisher-z averaged across
subjects, so it is a within-subject statistic throughout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAIRS = [("V1", "PPA"), ("V1", "Hippocampus"), ("V1", "SceneCtx"), ("V1", "ERC"),
         ("PPA", "Hippocampus"), ("SceneCtx", "Hippocampus"), ("PPA", "ERC"),
         ("EarlyVis", "SceneCtx"), ("EarlyVis", "Hippocampus"), ("V1", "V2")]
FOCUS = ["V1", "V2", "EarlyVis", "FFA", "PPA", "RSC", "SceneCtx", "ERC", "PHC"]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def main() -> int:
    try:
        items = pd.read_csv(RESULTS / "retention_stability_items.csv")
        gaps = pd.read_csv(RESULTS / "session_gaps.csv")
        design = pd.read_csv(RESULTS / "retention_design.csv")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Run step23 and step24 first.")
        return 1
    pd.set_option("display.width", 220)
    items = items.merge(design[["subject_id", "nsd_id", "session_rep1", "session_rep2"]],
                        on=["subject_id", "nsd_id"], how="left")

    # ---------------------------------------------------------------- A
    print("=== A. Long interval WITHOUT crossing a session boundary ===")
    wl = items[(items.retention_bin == "long") & (items.cross_session == 0)]
    sl = items[(items.retention_bin == "short") & (items.cross_session == 0)]
    subs = sorted(wl.subject_id.unique())
    print(f"  available in {subs} ({wl.groupby('roi').size().iloc[0] // len(FOCUS) if len(wl) else 0} "
          f"pairs/ROI across {len(subs)} subjects)")
    rows = []
    for roi in FOCUS:
        a = wl[wl.roi == roi].groupby("subject_id").stability.mean()
        b = sl[sl.roi == roi].groupby("subject_id").stability.mean().reindex(a.index)
        if len(a) < 2:
            continue
        rows.append(dict(roi=roi, n_subj=len(a), short_within=b.mean(),
                         long_within=a.mean(), delta=(a - b).mean(),
                         pct_retained=100 * a.mean() / b.mean() if b.mean() > 0 else np.nan,
                         n_pairs=int((wl.roi == roi).sum())))
    A = pd.DataFrame(rows)
    print(A.round(4).to_string(index=False))
    A.to_csv(RESULTS / "retention_within_session_long.csv", index=False)
    print("  (N=2 subjects -- descriptive only, no inferential claim)")

    # ---------------------------------------------------------------- B
    print("\n=== B. Does decay scale with each subject's own session gap? ===")
    per = (items[items.retention_bin.isin(["short", "long"])]
           .groupby(["subject_id", "roi", "retention_bin"]).stability.mean()
           .unstack("retention_bin").reset_index())
    per["delta"] = per["long"] - per["short"]
    per = per.merge(gaps[["subject_id", "session_gap_days"]], on="subject_id")
    rows = []
    for roi in FOCUS:
        g = per[per.roi == roi]
        if len(g) < 6:
            continue
        r = stats.pearsonr(g.session_gap_days, g.delta)
        rs = stats.spearmanr(g.session_gap_days, g.delta)
        rows.append(dict(roi=roi, n_subj=len(g), pearson_r=r.statistic, p=r.pvalue,
                         spearman_rho=rs.statistic, p_spearman=rs.pvalue))
    B = pd.DataFrame(rows).sort_values("p")
    B["p_fdr"] = bh_fdr(B.p.values)
    print(B.round(4).to_string(index=False))
    B.to_csv(RESULTS / "retention_gap_doseresponse.csv", index=False)
    print("  Negative r = subjects with longer gaps lost more (time-driven).")

    # ---------------------------------------------------------------- C
    print("\n=== C. Pure session effect: short pairs in session 1 vs session 2 ===")
    sh = items[(items.retention_bin == "short") & (items.cross_session == 0)]
    rows = []
    for roi in FOCUS:
        g = sh[sh.roi == roi]
        s1 = g[g.session_rep1 == 1].groupby("subject_id").stability.mean()
        s2 = g[g.session_rep1 == 2].groupby("subject_id").stability.mean()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 4:
            continue
        t, p = stats.ttest_rel(s2[idx], s1[idx])
        rows.append(dict(roi=roi, n_subj=len(idx), sess1=s1[idx].mean(),
                         sess2=s2[idx].mean(), delta=(s2[idx] - s1[idx]).mean(),
                         t=t, p=p))
    C = pd.DataFrame(rows).sort_values("p")
    C["p_fdr"] = bh_fdr(C.p.values)
    print(C.round(4).to_string(index=False))
    C.to_csv(RESULTS / "retention_pure_session_effect.csv", index=False)
    print("  A large effect here would mean session identity alone moves reinstatement,")
    print("  which would undercut A/B; a small one leaves the interval interpretation.")

    # ---------------------------------------------------------------- coupling
    print("\n=== D. Cross-region coupling of the memory trace, by interval ===")
    wide = items.pivot_table(index=["subject_id", "nsd_id", "retention_bin"],
                             columns="roi", values="stability").reset_index()
    rows = []
    for a, b in PAIRS:
        if a not in wide.columns or b not in wide.columns:
            continue
        rec = dict(roi_a=a, roi_b=b)
        zs = {}
        for rbin in ("short", "long"):
            per_subj = []
            for subj, g in wide[wide.retention_bin == rbin].groupby("subject_id"):
                g = g[[a, b]].dropna()
                if len(g) > 20 and g[a].std() > 0 and g[b].std() > 0:
                    per_subj.append(np.arctanh(np.clip(
                        stats.pearsonr(g[a], g[b]).statistic, -0.999, 0.999)))
            zs[rbin] = np.array(per_subj)
            if len(per_subj) >= 4:
                t, p = stats.ttest_1samp(per_subj, 0.0)
                rec[f"r_{rbin}"] = float(np.tanh(np.mean(per_subj)))
                rec[f"t_{rbin}"] = t
                rec[f"p_{rbin}"] = p
        if len(zs.get("short", [])) >= 4 and len(zs.get("long", [])) >= 4:
            n = min(len(zs["short"]), len(zs["long"]))
            t, p = stats.ttest_rel(zs["long"][:n], zs["short"][:n])
            rec["delta_z"] = float(np.mean(zs["long"][:n] - zs["short"][:n]))
            rec["t_change"] = t
            rec["p_change"] = p
        rows.append(rec)
    D = pd.DataFrame(rows)
    if "p_short" in D:
        D["p_short_fdr"] = bh_fdr(D.p_short.fillna(1).values)
    print(D.round(4).to_string(index=False))
    D.to_csv(RESULTS / "retention_cross_region_coupling.csv", index=False)
    print("\n  r_short / r_long: across-stimulus coupling of trace strength between the")
    print("  two regions. delta_z > 0 would mean regions become MORE coupled as the")
    print("  interval lengthens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
