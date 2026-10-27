"""
Step 23: build the retention-interval design.

The Phase I analyses treated "repetition" as a categorical factor and found nothing.
That framing discarded the variable the memory literature actually cares about: how much
TIME passed between the two exposures. In sessions 1-2 that interval spans three orders
of magnitude -- a median of 15 minutes within session, and 6.9 to 14 days across the
session boundary -- so retention interval can be used as a continuous regressor and as a
two-level contrast (priming vs consolidation window) within subject.

Design table columns
    subject_id, nsd_id, coco_id       stimulus identity (corrected mapping)
    trial_rep1, trial_rep2            global trial indices
    session_rep1, session_rep2        scan session of each exposure
    lag_trials                        intervening trials
    lag_days, log10_lag_min           elapsed time
    retention_bin                     'short' (<1 h) or 'long' (>1 day)
    cross_session                     1 if the pair spans the session boundary
    correct_rep2, rt_rep2             behaviour on the second exposure

Known confound, stated here because every downstream analysis inherits it: within these
two sessions a long interval NECESSARILY crosses the session boundary, so retention and
session-crossing are perfectly collinear. Three partial handles, all used downstream:
  1. ROI x retention interactions -- scanner drift should hit all ROIs alike, so a
     regional dissociation is not explained by drift.
  2. Lag varies 1-14 days WITHIN the cross-session pairs, giving a continuous test with
     session-crossing held constant.
  3. The session gap itself varies 6.9-14 days BETWEEN subjects, so the effect can be
     tested for scaling with each subject's own gap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BEHAV = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/behav")
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SHORT_MAX_DAYS = 1.0 / 24.0      # < 1 hour
LONG_MIN_DAYS = 1.0              # > 1 day


def build() -> pd.DataFrame:
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    rows = []
    for subj in SUBJECTS:
        b = pd.read_csv(BEHAV / f"{subj}_responses.tsv", sep="\t")
        b = b[b.SESSION.isin([1, 2])].reset_index(drop=True)
        b["nsd_id"] = b["73KID"] - 1
        b["global_trial"] = np.arange(1, len(b) + 1)
        m = man[man.subject_id == subj].set_index("global_trial")

        for nsd_id, g in b.groupby("nsd_id"):
            if len(g) < 2:
                continue
            g = g.sort_values("TIME")
            r1, r2 = g.iloc[0], g.iloc[1]
            lag_days = float(r2.TIME - r1.TIME)
            lag_min = max(lag_days * 24 * 60, 1e-3)
            if lag_days < SHORT_MAX_DAYS:
                rbin = "short"
            elif lag_days > LONG_MIN_DAYS:
                rbin = "long"
            else:
                rbin = "mid"
            rows.append(dict(
                subject_id=subj, nsd_id=int(nsd_id),
                coco_id=int(m.loc[int(r1.global_trial), "coco_id"]),
                trial_rep1=int(r1.global_trial), trial_rep2=int(r2.global_trial),
                session_rep1=int(r1.SESSION), session_rep2=int(r2.SESSION),
                lag_trials=int(r2.global_trial - r1.global_trial),
                lag_days=lag_days, log10_lag_min=float(np.log10(lag_min)),
                retention_bin=rbin,
                cross_session=int(r1.SESSION != r2.SESSION),
                n_presentations=int(len(g)),
                correct_rep2=float(r2.ISCORRECT) if pd.notna(r2.ISCORRECT) else np.nan,
                rt_rep2=float(r2.RT) if pd.notna(r2.RT) else np.nan))
    return pd.DataFrame(rows)


def session_gaps() -> pd.DataFrame:
    out = []
    for subj in SUBJECTS:
        b = pd.read_csv(BEHAV / f"{subj}_responses.tsv", sep="\t")
        g = b[b.SESSION.isin([1, 2])].groupby("SESSION").TIME.agg(["min", "max"])
        out.append(dict(subject_id=subj,
                        session_gap_days=float(g.loc[2, "min"] - g.loc[1, "max"]),
                        session1_span_h=float((g.loc[1, "max"] - g.loc[1, "min"]) * 24),
                        session2_span_h=float((g.loc[2, "max"] - g.loc[2, "min"]) * 24)))
    return pd.DataFrame(out)


def main() -> int:
    try:
        d = build()
        gaps = session_gaps()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    RESULTS.mkdir(exist_ok=True)
    d.to_csv(RESULTS / "retention_design.csv", index=False)
    gaps.to_csv(RESULTS / "session_gaps.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"[OK] results/retention_design.csv  ({len(d)} Rep1->Rep2 pairs)\n")

    print("=== session gap per subject (the consolidation interval each subject got) ===")
    print(gaps.round(2).to_string(index=False))
    print(f"\n  gap range: {gaps.session_gap_days.min():.1f} - "
          f"{gaps.session_gap_days.max():.1f} days  (2.0x spread, used as a "
          f"between-subject handle)")

    print("\n=== retention bins ===")
    tab = d.groupby(["subject_id", "retention_bin"]).size().unstack(fill_value=0)
    for c in ("short", "mid", "long"):
        if c not in tab:
            tab[c] = 0
    print(tab[["short", "mid", "long"]].to_string())
    print(f"\n  pooled: short={int((d.retention_bin=='short').sum())}  "
          f"mid={int((d.retention_bin=='mid').sum())}  "
          f"long={int((d.retention_bin=='long').sum())}")

    print("\n=== interval, by bin ===")
    for b, g in d.groupby("retention_bin"):
        lo, hi = g.lag_days.min(), g.lag_days.max()
        med = g.lag_days.median()
        unit = (lambda v: f"{v*24*60:.1f} min" if v < 1 / 24 else
                (f"{v*24:.1f} h" if v < 1 else f"{v:.1f} d"))
        print(f"  {b:6s} n={len(g):5d}  median {unit(med):>10s}   "
              f"range {unit(lo):>10s} - {unit(hi):>10s}")

    print("\n=== collinearity check ===")
    ct = pd.crosstab(d.retention_bin, d.cross_session)
    print(ct.to_string())
    print("  -> long intervals are entirely cross-session, as expected; handled by the")
    print("     three strategies described in the module docstring.")

    print("\n=== continuous regressor coverage (log10 minutes) ===")
    print(f"  {d.log10_lag_min.min():.2f} to {d.log10_lag_min.max():.2f}  "
          f"({d.log10_lag_min.max()-d.log10_lag_min.min():.1f} orders of magnitude)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
