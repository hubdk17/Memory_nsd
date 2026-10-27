"""
Step 31: is the retention effect stimulus-locked? Leave-one-subject-out.

Step 29 showed the group-average time course is flat past one week, and that what had
looked like retention decay was the within-session versus across-session boundary. A
session artefact is specific to a subject and a scan day, so it cannot produce a
consistent per-IMAGE pattern across four people scanned months apart. Inter-subject
consistency therefore separates the two accounts without relying on the confounded
contrast at all.

The anchor is the shared1000: 766 of those images have at least two presentations in
every subject, and 515 have all three, so the same stimuli can be compared across brains
without any spatial alignment -- the image identity is the correspondence.

  A  Leave-one-subject-out prediction. For each ROI, predict a held-out subject's
     per-image reinstatement from the mean of the others. A positive correlation means
     the trace strength of a given image is a property of the image, not of the scan.
  B  The same for the per-image retention EFFECT (long-interval minus short-interval
     reinstatement), which is the quantity the project's claims rest on.
  C  Intraclass correlation across all available subjects, as a single reliability
     number per ROI.
  D  Noise ceiling. With k subjects the leave-one-out correlation is bounded by
     Spearman-Brown; reporting the observed value against that bound distinguishes
     "no consistency" from "not enough subjects to detect it".

Nothing here depends on retention interval being unconfounded, which is the point.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _full_data as fd
from step29_full_retention_analysis import COMPOSITES, stability_blocked

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
ROIS = ["V1", "V2", "V3", "hV4", "FFA", "OPA", "PPA", "RSC", "PHC",
        "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
SHORT_MAX_DAYS = 1.0 / 24.0
LONG_MIN_DAYS = 7.0
# For the within-image retention contrast: keep an image only if its longer interval is
# at least this many times its shorter one. At 10x, 556 of the 1,000 shared images
# qualify, against 315 under absolute sub-hour / 7-day thresholds.
MIN_LAG_RATIO = 10.0
N_PERM = 2000
SEED = 42


def shared_ids() -> set:
    si = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    return set(si.loc[si.shared1000, "nsdId"].astype(int))


def spearman_brown(r_avg_of_k: float, k: int) -> float:
    """Expected leave-one-out r when predicting 1 subject from k others."""
    if not np.isfinite(r_avg_of_k) or r_avg_of_k <= 0:
        return np.nan
    return k * r_avg_of_k / (1 + (k - 1) * r_avg_of_k)


def per_image_values(subj: str, design: pd.DataFrame, rois: list[str],
                     comps: dict, rng) -> dict:
    """Per-image reinstatement and long-minus-short retention effect, per ROI."""
    d = design[(design.subject_id == subj) & (design.usable == 1)].reset_index(drop=True)
    if len(d) < 200:
        return {}
    out = {}
    for name, parts in [(r, [r]) for r in rois] + list(comps.items()):
        X, tidx, _ = fd.load_rois(subj, parts, norm="session")
        pos = {t: i for i, t in enumerate(tidx)}
        ia = np.array([pos[t] for t in d.trial_a])
        ib = np.array([pos[t] for t in d.trial_b])
        s = stability_blocked(X[ia], X[ib], rng=rng)
        del X
        gc.collect()
        frame = pd.DataFrame(dict(nsd_id=d.nsd_id.values, lag=d.lag_days.values,
                                  rep_from=d.rep_from.values, stab=s))
        # one value per image: mean over that image's transitions
        mean_all = frame.groupby("nsd_id").stab.mean()

        # Retention effect WITHIN image. Absolute thresholds would keep only the 315
        # images that happen to have both a sub-hour and a 7-day-plus transition, since
        # an image gets at most two intervals. Contrasting each image's own longer
        # against its own shorter transition keeps every image whose two intervals
        # actually differ, and is a paired comparison by construction.
        f = frame.sort_values(["nsd_id", "lag"])
        first = f.groupby("nsd_id").first()
        last = f.groupby("nsd_id").last()
        ratio = last.lag / np.maximum(first.lag, 1e-4)
        keep = ratio >= MIN_LAG_RATIO
        eff = (last.stab - first.stab)[keep]
        out[name] = dict(mean=mean_all, effect=eff,
                         lag_ratio=ratio[keep])
    return out


def loso(matrix: pd.DataFrame, rng) -> dict:
    """matrix: images x subjects. Predict each column from the mean of the others."""
    m = matrix.dropna()
    if m.shape[0] < 50 or m.shape[1] < 3:
        return dict(n_images=m.shape[0], n_subj=m.shape[1], r=np.nan, p=np.nan,
                    icc=np.nan, ceiling=np.nan)
    rs = []
    for c in m.columns:
        others = m.drop(columns=c).mean(axis=1)
        if others.std() == 0 or m[c].std() == 0:
            continue
        rs.append(stats.pearsonr(m[c], others).statistic)
    r = float(np.mean(rs)) if rs else np.nan

    # permutation: shuffle image identity independently per subject
    null = np.empty(N_PERM)
    arr = m.values
    for i in range(N_PERM):
        perm = np.column_stack([arr[rng.permutation(arr.shape[0]), j]
                                for j in range(arr.shape[1])])
        vals = []
        for j in range(arr.shape[1]):
            oth = np.delete(perm, j, axis=1).mean(1)
            if perm[:, j].std() > 0 and oth.std() > 0:
                vals.append(np.corrcoef(perm[:, j], oth)[0, 1])
        null[i] = np.mean(vals) if vals else 0.0
    p = float((null >= r).mean()) if np.isfinite(r) else np.nan

    # mean pairwise r -> ICC-like reliability, and the Spearman-Brown ceiling
    pair = []
    for a in range(m.shape[1]):
        for b in range(a + 1, m.shape[1]):
            if m.iloc[:, a].std() > 0 and m.iloc[:, b].std() > 0:
                pair.append(stats.pearsonr(m.iloc[:, a], m.iloc[:, b]).statistic)
    mp = float(np.mean(pair)) if pair else np.nan
    return dict(n_images=m.shape[0], n_subj=m.shape[1], r=r, p=p, mean_pairwise=mp,
                ceiling=spearman_brown(mp, m.shape[1] - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()
    try:
        design = pd.read_csv(RESULTS / "retention_design_full.csv")
    except FileNotFoundError:
        print("[ERROR] run step28 first.")
        return 1

    cov = fd.coverage()
    ready = [s for s in cov.subject_id
             if cov.loc[cov.subject_id == s, "sessions_on_disk"].iloc[0] >= 20]
    subjects = args.subjects or ready
    if len(subjects) < 3:
        print(f"[ERROR] need >=3 subjects with substantial data; have {subjects}")
        return 1
    print(f"subjects: {subjects}")

    shared = shared_ids()
    rng = np.random.default_rng(SEED)
    per_subj = {}
    for subj in subjects:
        avail = set(fd.available_rois(subj))
        rois = [r for r in ROIS if r in avail]
        comps = {k: v for k, v in COMPOSITES.items() if all(p in avail for p in v)}
        print(f"  computing per-image values for {subj} ({len(rois)} ROIs)...", flush=True)
        per_subj[subj] = per_image_values(subj, design, rois, comps, rng)

    names = sorted(set.intersection(*[set(v) for v in per_subj.values()]))
    pd.set_option("display.width", 220)
    rows = []
    for quantity in ("mean", "effect"):
        for name in names:
            cols = {}
            for subj, v in per_subj.items():
                s = v[name][quantity]
                cols[subj] = s[s.index.isin(shared)]
            mat = pd.DataFrame(cols)
            res = loso(mat, rng)
            rows.append(dict(quantity=quantity, roi=name, **res))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "leave_one_subject_out.csv", index=False)

    for quantity, lab in (("mean", "per-image REINSTATEMENT"),
                          ("effect", "per-image RETENTION EFFECT (long - short)")):
        g = R[R.quantity == quantity].sort_values("r", ascending=False)
        print(f"\n=== Leave-one-subject-out: {lab} (shared1000 images) ===")
        print(g[["roi", "n_images", "n_subj", "r", "p", "mean_pairwise",
                 "ceiling"]].round(4).to_string(index=False))
        sig = g[(g.p < 0.05) & (g.r > 0)]
        print(f"  ROIs with consistent cross-subject structure: {len(sig)} of {len(g)}"
              + (f"  ({', '.join(sig.roi)})" if len(sig) else ""))

    print("\n  r  = predicting one subject's per-image values from the mean of the rest")
    print("  p  = permutation test shuffling image identity within subject")
    print("  ceiling = Spearman-Brown bound given the observed pairwise reliability;")
    print("            an r near the ceiling means the measure is as consistent as this")
    print("            many subjects can show.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
