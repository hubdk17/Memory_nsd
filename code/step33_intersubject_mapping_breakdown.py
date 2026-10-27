"""
Step 33: where does the representational mapping stop being shared between brains?

Leave-one-subject-out over representational geometry. For each region, build a
stimulus x stimulus dissimilarity matrix from the 766 shared1000 images that every
subject saw at least twice, then predict each subject's RDM from the mean of the other
seven. The RDM lives in stimulus space, which is common across brains, so no spatial
alignment or hyperalignment is needed -- image identity is the correspondence.

Why this design is worth running after the earlier nulls: inter-subject consistency is
immune to within-subject session artefacts. A scanner-drift or head-position effect in
subj01 is statistically independent of one in subj05, so nothing shared across eight
brains can come from a session confound. That is precisely what invalidated the
retention and decoupling findings, and it cannot operate here.

The trap this script is built around
------------------------------------
Transferability is bounded by reliability. Per-image reinstatement measured r = 0.55 in
V1 and r = 0.01 in CA1, so a hippocampal mapping MUST fail to transfer -- not because
the code is idiosyncratic but because there is no signal to carry. Reporting raw
transfer would make "the mapping breaks down in the hippocampus" trivially true and
scientifically empty.

So every transfer value is divided by its own noise ceiling. Each subject has repeat
presentations of all 766 images, so within-subject RDM reliability is measured directly
(odd presentations versus even), and transfer is corrected for attenuation:

    normalised = r_loo / sqrt(rel_self * rel_others)

A region whose normalised transfer is near 1 has a shared code that merely looks weak
because it is noisy. A region near 0 after correction has a genuinely individual code.
That distinction is the entire result.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

import _full_data as fd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
# ordered sensory -> mnemonic, which is the axis the result is read along
ROI_ORDER = ["V1", "V2", "V3", "hV4", "S_early", "OPA", "FFA", "PPA", "RSC",
             "S_midventral", "S_midlateral", "S_ventral", "S_lateral", "S_parietal",
             "nsdgeneral", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
MIN_RELIABILITY = 0.02      # below this the ceiling correction is unstable
N_BOOT = 1000
SEED = 42


def shared_ids() -> set:
    si = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    return set(si.loc[si.shared1000, "nsdId"].astype(int))


def rdm(P: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM as a condensed vector."""
    C = np.corrcoef(P)
    return squareform(1.0 - np.nan_to_num(C, nan=0.0), checks=False)


def subject_rdms(subj: str, images: list[int], roi: str):
    """RDMs from the first and second presentation of each image, plus their mean.

    Splitting by presentation gives an independent within-subject estimate of how
    reliable the geometry is, which becomes the noise ceiling.
    """
    b = fd.behaviour(subj)
    b = b[b.nsd_id.isin(images)].sort_values(["nsd_id", "TIME"])
    b["occ"] = b.groupby("nsd_id").cumcount()
    X, tidx, _ = fd.load_rois(subj, [roi], norm="session")
    pos = {t: i for i, t in enumerate(tidx)}

    halves = []
    for occ in (0, 1):
        sel = b[b.occ == occ].set_index("nsd_id")
        if not set(images).issubset(sel.index):
            return None
        # Row order MUST follow `images` identically for every subject, or the RDMs
        # are not comparable and every cross-subject number is meaningless. Built by
        # iterating `images` rather than by any groupby ordering, and asserted.
        ordered = [int(sel.loc[i, "global_trial"]) for i in images]
        assert len(ordered) == len(images)
        halves.append(X[[pos[t] for t in ordered]])
    del X
    gc.collect()
    r1, r2 = rdm(halves[0]), rdm(halves[1])
    n = len(images)
    assert r1.shape == r2.shape == (n * (n - 1) // 2,), "RDM length mismatch"
    return r1, r2, 0.5 * (r1 + r2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", nargs="+", default=None)
    ap.add_argument("--max-images", type=int, default=766)
    args = ap.parse_args()

    shared = shared_ids()
    per_subj_ok = []
    for s in SUBJECTS:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        per_subj_ok.append(set(n[n >= 2].index))
    images = sorted(set.intersection(*per_subj_ok))[: args.max_images]
    print(f"shared images usable in all {len(SUBJECTS)} subjects: {len(images)}")

    avail = set(fd.available_rois(SUBJECTS[0]))
    rois = args.rois or [r for r in ROI_ORDER if r in avail]
    rng = np.random.default_rng(SEED)
    rows = []

    for roi in rois:
        R = {}
        for s in SUBJECTS:
            out = subject_rdms(s, images, roi)
            if out is None:
                print(f"  [skip] {s} {roi}: incomplete image coverage")
                continue
            R[s] = out
        if len(R) < 4:
            continue

        # Within-subject reliability: presentation 1 against presentation 2.
        rel = {s: stats.pearsonr(v[0], v[1]).statistic for s, v in R.items()}

        # Noise-MATCHED cross-subject estimate. Correlating one subject's RDM against a
        # seven-subject average compares a noisy quantity with a clean one and inflates
        # the result past any sensible ceiling, so instead every cross-subject value is
        # a single presentation against a single presentation -- exactly the comparison
        # the within-subject reliability makes. The ratio is then interpretable:
        #   1.0  every reliable bit of geometry is shared between brains
        #   0.0  the reliable geometry is entirely individual
        for s in R:
            others = [o for o in R if o != s]
            if not others:
                continue
            # CROSSED presentations only. r_within is presentation 1 against
            # presentation 2, so r_between must be too. Including the matched pairings
            # (1v1, 2v2) would give r_between a boost that r_within cannot have
            # whenever presentation order carries any systematic effect -- repetition
            # suppression does -- and the ratio then exceeds 1 for purely artefactual
            # reasons.
            pairwise = []
            for o in others:
                pairwise.append(stats.pearsonr(R[s][0], R[o][1]).statistic)
                pairwise.append(stats.pearsonr(R[s][1], R[o][0]).statistic)
            r_between = float(np.mean(pairwise))
            r_between_matched = float(np.mean(
                [stats.pearsonr(R[s][i], R[o][i]).statistic
                 for o in others for i in (0, 1)]))
            r_within = rel[s]
            # leave-one-out against the group mean, kept for reference only
            r_loo = stats.pearsonr(R[s][2], np.mean([R[o][2] for o in others],
                                                    axis=0)).statistic
            ok = r_within > MIN_RELIABILITY
            rows.append(dict(roi=roi, subject_id=s, n_images=len(images),
                             r_within=r_within, r_between=r_between,
                             r_between_matched=r_between_matched, r_loo=r_loo,
                             shared_fraction=r_between / r_within if ok else np.nan,
                             presentation_bias=r_between_matched - r_between,
                             ceiling_ok=ok))
        # ---- validation: permutation null. Shuffling image identity for the held-out
        # subject only must collapse the shared fraction to ~0. If it does not, the
        # correspondence between subjects is not doing the work and the result is an
        # artefact of RDM structure rather than of shared representation.
        s0 = sorted(R)[0]
        n_img = len(images)
        null_vals = []
        for _ in range(25):
            perm = rng.permutation(n_img)
            sq = squareform(R[s0][0])
            shuffled = squareform(sq[np.ix_(perm, perm)], checks=False)
            null_vals.append(np.mean([stats.pearsonr(shuffled, R[o][0]).statistic
                                      for o in R if o != s0]))
        this = [x for x in rows if x["roi"] == roi]
        print(f"  [OK] {roi:14s} rel {np.mean(list(rel.values())):.3f}  "
              f"between {np.mean([x['r_between'] for x in this]):.3f}  "
              f"shared {np.nanmean([x['shared_fraction'] for x in this]):.3f}  "
              f"perm-null {np.mean(null_vals):+.4f}", flush=True)
        for x in this:
            x["perm_null_r"] = float(np.mean(null_vals))
        del R
        gc.collect()

    if not rows:
        print("[ERROR] nothing computed.")
        return 1
    D = pd.DataFrame(rows)
    D.to_csv(RESULTS / "intersubject_mapping_breakdown.csv", index=False)

    pd.set_option("display.width", 230)
    agg = D.groupby("roi").agg(
        n_subj=("subject_id", "count"),
        r_within=("r_within", "mean"),
        r_between=("r_between", "mean"),
        shared=("shared_fraction", "mean"),
        shared_sd=("shared_fraction", "std"),
        perm_null=("perm_null_r", "mean"),
        pres_bias=("presentation_bias", "mean"),
        ceiling_ok=("ceiling_ok", "all")).reset_index()

    stat = []
    for roi, g in D.groupby("roi"):
        t0, p0 = stats.ttest_1samp(g.r_between, 0.0)
        sf = g.shared_fraction.dropna()
        t1, p1 = stats.ttest_1samp(sf, 1.0) if len(sf) >= 4 else (np.nan, np.nan)
        stat.append(dict(roi=roi, t_between_vs_zero=t0, p_between_vs_zero=p0,
                         t_shared_vs_one=t1, p_shared_vs_one=p1,
                         sem_shared=sf.std(ddof=1) / np.sqrt(len(sf)) if len(sf) else np.nan))
    agg = agg.merge(pd.DataFrame(stat), on="roi")
    agg.to_csv(RESULTS / "intersubject_mapping_breakdown_summary.csv", index=False)

    ok = agg[agg.ceiling_ok].sort_values("shared", ascending=False)
    floor = agg[~agg.ceiling_ok]

    print("\n" + "=" * 78)
    print("LEAVE-ONE-SUBJECT-OUT: how much of each region's reliable geometry is SHARED")
    print("=" * 78)
    print("  r_within  : subject's own presentation-1 RDM vs presentation-2 RDM")
    print("  r_between : that subject vs each other subject, presentation to")
    print("              presentation -- SAME noise level as r_within, so the ratio")
    print("              is interpretable")
    print("  shared    : r_between / r_within.  1.0 = every reliable bit of geometry")
    print("              is common across brains;  0.0 = entirely subject-specific")
    print("  perm_null : image labels shuffled for one subject; must be ~0\n")
    print(ok[["roi", "r_within", "r_between", "shared", "sem_shared", "perm_null",
              "pres_bias", "p_between_vs_zero", "p_shared_vs_one"]].round(4).to_string(index=False))

    if len(ok):
        print("\n--- MOST SHARED (mapping transfers best) ---")
        for _, r in ok.head(5).iterrows():
            print(f"  {r.roi:14s} shared = {r.shared:.3f}  "
                  f"(between {r.r_between:.3f} / within {r.r_within:.3f})")
        print("\n--- MOST SUBJECT-SPECIFIC (mapping breaks down) ---")
        for _, r in ok.tail(5).iloc[::-1].iterrows():
            print(f"  {r.roi:14s} shared = {r.shared:.3f}  "
                  f"(between {r.r_between:.3f} / within {r.r_within:.3f})")

    if len(floor):
        print(f"\n--- NOT INTERPRETABLE (within-subject reliability < "
              f"{MIN_RELIABILITY}) ---")
        for _, r in floor.iterrows():
            print(f"  {r.roi:14s} r_within = {r.r_within:.4f}  "
                  f"r_between = {r.r_between:.4f}")
        print("  These regions carry too little reliable geometry to ask whether it")
        print("  transfers. Report as UNDERPOWERED, not as evidence of an individual")
        print("  code -- the two are indistinguishable at this reliability.")

    bad = agg[agg.perm_null.abs() > 0.02]
    print(f"\n[validation] permutation null within +/-0.02 for "
          f"{len(agg)-len(bad)} of {len(agg)} ROIs"
          + (f"; CHECK: {', '.join(bad.roi)}" if len(bad) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
