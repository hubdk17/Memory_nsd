"""
Step 8: bridge the decoding results to the RSA framework of ADVANCED_RESEARCH_2.

IMPORTANT DEVIATION FROM THE BRIEF, AND WHY.
The brief asks for the cross-link to be built on
ADVANCED_RESEARCH_2/results/tables/axis2_semantic_correlations.csv. That table cannot
serve as an independent validator, because it is computed from
ADVANCED_RESEARCH_2/data/multi_semantic_spaces.npz, whose stimulus annotations are
shifted by one NSD index (extract_manifests.py looks up a 1-based 73k id in the 0-based
nsdId column). Every stimulus there carries the semantics of a different, unrelated
photograph; step5 showed a decoder trained against those shifted targets performs at
chance (0.503 vs 0.674, two-way). Correlating a corrected decoding measure against RSA
values computed from randomly-paired annotations would be correlating signal with noise,
and a null would be uninformative.

So this step does both:
  8a  Recompute the RSA semantic convergence from the CORRECTED stimulus mapping,
      using the same CLIP text (and, when available, CLIP image) target spaces the
      decoder predicts. This is the valid cross-link.
  8b  Correlate corrected RSA convergence with the decoding accuracy change across
      subjects, per ROI.
  8c  Within-subject bridge: correlate each subject's RSA-convergence vector against
      their decoding-delta vector across ROIs, then test the 8 correlations against zero.
  8d  Diagnostic: repeat 8b against the legacy axis2 table, to document how far the
      published values are from the corrected ones.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
AR2_TABLES = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH_2/results/tables")
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROIS = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_CATEGORY = {"V1": "Early_Visual", "V2": "Early_Visual", "FFA": "Ventral_Stream",
                "PPA": "Scene_Selective", "RSC": "Scene_Selective",
                "PHC": "Parahippocampal", "PRC": "Medial_Temporal",
                "ERC": "Medial_Temporal", "SUB": "Hippocampus", "CA1": "Hippocampus",
                "DG_CA3": "Hippocampus"}
SEED = 42


def zs(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def rdm_corr(P):
    """Correlation-distance RDM of the rows of P, returned as a condensed vector."""
    C = np.corrcoef(P)
    C = np.nan_to_num(C, nan=0.0)
    return squareform(1.0 - C, checks=False)


def rdm_cosine(T):
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    return squareform(1.0 - Tn @ Tn.T, checks=False)


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    adj = np.full(len(p), np.nan)
    q = p[ok]
    n = len(q)
    if n == 0:
        return adj
    a, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(q)[::-1]):
        prev = min(prev, q[idx] * n / (n - rank))
        a[idx] = prev
    adj[ok] = np.clip(a, 0, 1)
    return adj


# --------------------------------------------------------------------------- 8a
def corrected_rsa():
    """Per subject x ROI x repetition: Spearman(neural RDM, semantic RDM).

    Uses the repeated323 cohort so that the stimulus set is identical at Rep1 and Rep2,
    which makes the two RDMs directly comparable.
    """
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    spaces = {}
    tt = np.load(HERE / "data" / "clip_text_targets.npz")
    spaces["S_text"] = ({int(c): i for i, c in enumerate(tt["coco_ids"])}, tt["T_text"])
    img_path = HERE / "data" / "clip_image_targets.npz"
    if img_path.exists():
        ti = np.load(img_path)
        spaces["S_image"] = ({int(c): i for i, c in enumerate(ti["coco_ids"])},
                             ti["T_image"])
        print("  semantic spaces: S_text, S_image")
    else:
        print("  semantic spaces: S_text only (run step7 for S_image)")

    rows = []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        rep = m.repeat_number.values
        mask = (m.n_presentations >= 2).values
        # stimulus order fixed by the Rep1 presentations; Rep2 reordered to match
        ids1 = m.loc[mask & (rep == 1), "nsd_id"].values
        order = {int(i): k for k, i in enumerate(ids1)}
        cocos = [int(m.loc[m.nsd_id == i, "coco_id"].iloc[0]) for i in ids1]

        sem_rdms = {}
        for name, (c2r, T) in spaces.items():
            if all(c in c2r for c in cocos):
                sem_rdms[name] = rdm_cosine(T[[c2r[c] for c in cocos]])
        if not sem_rdms:
            continue

        for roi in ROIS:
            X = zs(z[roi].astype(np.float64), rep)      # per-repetition normalisation
            neural = {}
            for r in (1, 2):
                s = mask & (rep == r)
                idx = np.array([order[int(i)] for i in m.loc[s, "nsd_id"]])
                P = np.empty((len(ids1), X.shape[1]))
                P[idx] = X[s]
                neural[r] = rdm_corr(P)
            for name, srdm in sem_rdms.items():
                rho = {r: stats.spearmanr(neural[r], srdm).statistic for r in (1, 2)}
                rows.append(dict(subject_id=subj, roi=roi,
                                 roi_category=ROI_CATEGORY[roi], semantic_space=name,
                                 rho_rep1=rho[1], rho_rep2=rho[2],
                                 convergence=rho[2] - rho[1], n_stimuli=len(ids1)))
        print(f"  [OK] {subj}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 8b/c
def crosslink(rsa, decode, label, space):
    """Across-subject correlation per ROI, plus the within-subject bridge."""
    r = rsa[rsa.semantic_space == space] if "semantic_space" in rsa else rsa
    j = r.merge(decode, on=["subject_id", "roi"], how="inner")
    if j.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for roi, g in j.groupby("roi"):
        if len(g) < 4 or g.convergence.std() == 0 or g.delta_decode.std() == 0:
            continue
        sr = stats.spearmanr(g.convergence, g.delta_decode)
        pr = stats.pearsonr(g.convergence, g.delta_decode)
        rows.append(dict(source=label, semantic_space=space, roi=roi,
                         roi_category=ROI_CATEGORY.get(roi, "?"), n_subj=len(g),
                         mean_convergence=g.convergence.mean(),
                         mean_delta_decode=g.delta_decode.mean(),
                         spearman_rho=sr.statistic, spearman_p=sr.pvalue,
                         pearson_r=pr.statistic, pearson_p=pr.pvalue))
    per_roi = pd.DataFrame(rows)
    if not per_roi.empty:
        per_roi["spearman_p_fdr"] = bh_fdr(per_roi.spearman_p.values)

    within = []
    for subj, g in j.groupby("subject_id"):
        if len(g) < 5:
            continue
        sr = stats.spearmanr(g.convergence, g.delta_decode)
        within.append(dict(source=label, semantic_space=space, subject_id=subj,
                           n_roi=len(g), spearman_rho=sr.statistic, p=sr.pvalue))
    within = pd.DataFrame(within)
    return per_roi, within


def main():
    try:
        dec = pd.read_csv(RESULTS / "ridge_clip_text_retrieval_by_repetition.csv")
    except FileNotFoundError:
        print("[ERROR] run step2 first.")
        return 1
    d = dec[(dec.analysis == "repeated323") & (dec.norm == "repgroup")
            & (dec.roi.isin(ROIS))]
    w = d.pivot_table(index=["subject_id", "roi"], columns="repetition",
                      values="two_way").reset_index().rename(columns={1: "r1", 2: "r2"})
    w["delta_decode"] = w.r2 - w.r1
    decode = w[["subject_id", "roi", "delta_decode"]]

    print("=== 8a  recomputing RSA semantic convergence from the corrected mapping ===")
    rsa = corrected_rsa()
    if rsa.empty:
        print("[ERROR] corrected RSA produced no rows.")
        return 1
    rsa.to_csv(RESULTS / "corrected_rsa_convergence.csv", index=False)
    print(f"[OK] results/corrected_rsa_convergence.csv ({len(rsa)} rows)")

    pd.set_option("display.width", 250)
    print("\n--- corrected RSA: semantic alignment by repetition (mean of 8 subjects) ---")
    for space, g in rsa.groupby("semantic_space"):
        agg = g.groupby("roi")[["rho_rep1", "rho_rep2", "convergence"]].mean()
        agg["t"] = [stats.ttest_1samp(g[g.roi == r].convergence, 0).statistic
                    for r in agg.index]
        agg["p"] = [stats.ttest_1samp(g[g.roi == r].convergence, 0).pvalue
                    for r in agg.index]
        agg["p_fdr"] = bh_fdr(agg.p.values)
        print(f"\n  {space}")
        print(agg.sort_values("convergence", ascending=False).round(5).to_string())

    all_roi, all_within = [], []
    for space in sorted(rsa.semantic_space.unique()):
        pr, wi = crosslink(rsa, decode, "corrected", space)
        all_roi.append(pr)
        all_within.append(wi)

    # --- 8d diagnostic: the legacy (shifted-annotation) axis2 table
    legacy_note = ""
    try:
        ax = pd.read_csv(AR2_TABLES / "axis2_semantic_correlations.csv")
        lg = ax[ax.semantic_space == "S_text"].pivot_table(
            index=["subject_id", "roi"], columns="repetition_stage",
            values="spearman_rho").reset_index()
        lg["convergence"] = lg["Rep2_Repeated"] - lg["Rep1_Initial"]
        lg = lg[["subject_id", "roi", "convergence"]]
        pr, wi = crosslink(lg, decode, "legacy_shifted", "S_text")
        all_roi.append(pr)
        all_within.append(wi)
        cmp = rsa[rsa.semantic_space == "S_text"][
            ["subject_id", "roi", "convergence"]].merge(
            lg, on=["subject_id", "roi"], suffixes=("_corrected", "_legacy"))
        sr = stats.spearmanr(cmp.convergence_corrected, cmp.convergence_legacy)
        legacy_note = (f"corrected vs legacy convergence agreement across "
                       f"{len(cmp)} subject x ROI cells: Spearman rho = "
                       f"{sr.statistic:.3f}, p = {sr.pvalue:.3g}")
        cmp.to_csv(RESULTS / "rsa_corrected_vs_legacy.csv", index=False)
    except Exception as e:                                      # noqa: BLE001
        legacy_note = f"legacy axis2 table unavailable ({e!r})"

    per_roi = pd.concat([x for x in all_roi if not x.empty], ignore_index=True)
    within = pd.concat([x for x in all_within if not x.empty], ignore_index=True)

    rsa.merge(decode, on=["subject_id", "roi"], how="left").to_csv(
        RESULTS / "rsa_decoding_crosslink.csv", index=False)
    per_roi.to_csv(RESULTS / "rsa_decoding_crosslink_stats.csv", index=False)
    within.to_csv(RESULTS / "rsa_decoding_within_subject.csv", index=False)

    print("\n=== 8b  across-subject correlation: RSA convergence vs decoding delta ===")
    for (src, space), g in per_roi.groupby(["source", "semantic_space"]):
        print(f"\n  source={src}  space={space}")
        print(g.sort_values("spearman_p")[
            ["roi", "roi_category", "mean_convergence", "mean_delta_decode",
             "spearman_rho", "spearman_p", "spearman_p_fdr"]].round(4).to_string(index=False))

    print("\n=== 8c  within-subject bridge (correlation across ROIs, per subject) ===")
    for (src, space), g in within.groupby(["source", "semantic_space"]):
        t, p = stats.ttest_1samp(g.spearman_rho, 0.0)
        print(f"  source={src:15s} space={space:8s} mean rho={g.spearman_rho.mean():+.3f} "
              f"n_neg={int((g.spearman_rho < 0).sum())}/{len(g)}  t={t:+.2f} p={p:.4f}")

    print(f"\n=== 8d  legacy diagnostic ===\n  {legacy_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
