"""
Step 28: retention design over the full experiment.

Rebuilds step23's design table from every completed session rather than the first two.
For subject 1 that turns 323 Rep1->Rep2 pairs with a 15-minute median into 10,000 pairs
spanning 15 minutes to 262 days, plus 10,000 Rep2->Rep3 pairs that did not exist at all
in the two-session window.

The structural gain is not just sample size. With two sessions a long interval always
crossed the session boundary, so retention and session-crossing were perfectly
collinear and no analysis could separate them. Across 40 sessions every bin past one day
crosses sessions, so contrasting 7-30 days against 3+ months isolates elapsed time with
session-crossing held constant.

Bins are logarithmic because the memory literature's timescales are: minutes (priming),
days (early consolidation), weeks to months (systems consolidation).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _full_data as fd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
BINS = [(0, 1 / 24, "min"), (1 / 24, 1, "hours"), (1, 7, "days"),
        (7, 30, "weeks"), (30, 90, "months"), (90, 400, "quarters")]


def label(lag: float) -> str:
    for lo, hi, name in BINS:
        if lo <= lag < hi:
            return name
    return "quarters"


def build(subj: str, sessions_on_disk: set[int]) -> pd.DataFrame:
    b = fd.behaviour(subj)
    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    rows = []
    for nsd_id, g in b.groupby("nsd_id"):
        g = g.sort_values("TIME")
        t = g.TIME.values
        tr = g.global_trial.values
        se = g.SESSION.values
        for k in range(len(g) - 1):
            lag = float(t[k + 1] - t[k])
            rows.append(dict(
                subject_id=subj, nsd_id=int(nsd_id), coco_id=n2c.get(int(nsd_id), -1),
                rep_from=k + 1, rep_to=k + 2,
                trial_a=int(tr[k]), trial_b=int(tr[k + 1]),
                session_a=int(se[k]), session_b=int(se[k + 1]),
                lag_days=lag, log10_lag_min=float(np.log10(max(lag * 1440, 1e-3))),
                bin=label(lag), cross_session=int(se[k] != se[k + 1]),
                n_presentations=len(g),
                usable=int(se[k] in sessions_on_disk and se[k + 1] in sessions_on_disk)))
    return pd.DataFrame(rows)


def main() -> int:
    cov = fd.coverage()
    if cov.empty:
        print("[ERROR] no extracted sessions found. Run step27 first.")
        return 1
    pd.set_option("display.width", 220)
    print("=== extraction coverage ===")
    print(cov.round(2).to_string(index=False))

    frames = []
    for subj in cov.subject_id:
        on_disk = set(fd.available_sessions(subj))
        frames.append(build(subj, on_disk))
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(RESULTS / "retention_design_full.csv", index=False)
    print(f"\n[OK] results/retention_design_full.csv  ({len(d):,} repetition pairs)")

    u = d[d.usable == 1]
    print(f"\nusable with sessions currently on disk: {len(u):,} of {len(d):,}")

    print("\n=== Rep1->Rep2 pairs by retention bin (usable only) ===")
    r12 = u[u.rep_from == 1]
    tab = r12.pivot_table(index="subject_id", columns="bin", values="nsd_id",
                          aggfunc="count", fill_value=0)
    order = [n for _, _, n in BINS if n in tab.columns]
    print(tab[order].to_string())

    print("\n=== Rep2->Rep3 pairs by retention bin (usable only) ===")
    r23 = u[u.rep_from == 2]
    if len(r23):
        tab2 = r23.pivot_table(index="subject_id", columns="bin", values="nsd_id",
                               aggfunc="count", fill_value=0)
        print(tab2[[n for _, _, n in BINS if n in tab2.columns]].to_string())
    else:
        print("  none yet")

    print("\n=== interval per bin (all subjects pooled) ===")
    for _, _, name in BINS:
        g = u[u.bin == name]
        if not len(g):
            continue
        f = (lambda v: f"{v*1440:.0f} min" if v < 1 / 24 else
             (f"{v*24:.1f} h" if v < 1 else f"{v:.1f} d"))
        print(f"  {name:9s} n={len(g):6,d}  median {f(g.lag_days.median()):>9s}   "
              f"range {f(g.lag_days.min()):>9s} - {f(g.lag_days.max()):>9s}")

    print("\n=== collinearity of retention with session-crossing ===")
    ct = pd.crosstab(u.bin, u.cross_session)
    ct = ct.reindex([n for _, _, n in BINS if n in ct.index])
    ct.columns = ["within_session", "cross_session"]
    print(ct.to_string())
    long_bins = u[u.bin.isin(["weeks", "months", "quarters"])]
    if len(long_bins):
        print(f"\n  Among long-interval pairs, {100*long_bins.cross_session.mean():.1f}% "
              f"cross a session boundary.")
        print("  Because that is near-total in EVERY long bin, contrasts BETWEEN long")
        print("  bins hold session-crossing constant and vary only elapsed time -- the")
        print("  comparison the two-session data could not support.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
