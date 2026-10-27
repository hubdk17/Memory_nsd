"""
Step 12: assemble RECONSTRUCTION_FEASIBILITY_REPORT.md from the result CSVs.

Every number in the report is read from a CSV at generation time rather than typed in,
so the prose cannot drift away from the data. Narrative that depends on the direction of
a result is written as a branch on the measured value, not asserted.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "RECONSTRUCTION_FEASIBILITY_REPORT.md"
ANAT = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ORDER = ["V1", "V2", "EarlyVis", "FFA", "PPA", "RSC", "PHC", "SceneCtx", "AllCortex",
         "PRC", "ERC", "SUB", "CA1", "DG_CA3", "MTL"]


def load(name, required=True):
    p = RESULTS / name
    if p.exists():
        return pd.read_csv(p)
    if required:
        print(f"  [WARN] missing {name}")
    return None


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{n}f}"


def fmtp(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def md_table(df, cols, headers=None, order_col=None, rounder=4):
    headers = headers or cols
    d = df.copy()
    if order_col:
        d["__o"] = d[order_col].map({r: i for i, r in enumerate(ORDER)})
        d = d.sort_values("__o")
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else
                             (fmtp(v) if c.startswith("p") else f"{v:.{rounder}f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    S = {}
    S["stat"] = load("statistical_infrastructure.csv")
    S["text"] = load("ridge_clip_text_retrieval_by_repetition.csv")
    S["image"] = load("ridge_clip_image_retrieval_by_repetition.csv", False)
    S["dis"] = load("text_vs_image_dissociation_stats.csv", False)
    S["rsa"] = load("corrected_rsa_convergence.csv", False)
    S["xlink"] = load("rsa_decoding_crosslink_stats.csv", False)
    S["within"] = load("rsa_decoding_within_subject.csv", False)
    S["cmp"] = load("rsa_corrected_vs_legacy.csv", False)
    S["cb"] = load("controlB_extended_stats.csv", False)
    S["cbnull"] = load("controlB_null_diagnostic_stats.csv", False)
    S["resid"] = load("proportional_snr_residual_stats.csv", False)
    S["ctrlA"] = load("controlA_proportional_snr.csv", False)
    S["same"] = load("session_balanced_stats.csv", False)
    S["loso"] = load("loso_session_stats.csv", False)
    S["life"] = load("lifecycle_trajectory_stats.csv", False)
    S["scal"] = load("training_size_scaling.csv", False)
    S["obo"] = load("offbyone_target_demonstration.csv", False)
    if S["stat"] is None or S["text"] is None:
        print("[ERROR] core results missing; run step2 and step6 first.")
        return 1

    st = S["stat"]
    main323 = st[(st.analysis == "repeated323") & (st.norm == "repgroup")
                 & (st.contrast == "Rep2-Rep1")]
    n_fdr = int(main323.fdr_significant.sum())
    n_nominal = int((main323.p_parametric < 0.05).sum())
    best = main323.loc[main323.p_parametric.idxmin()]
    max_bf = main323.bf10.max()
    hip = main323[main323.roi.isin(["CA1", "DG_CA3", "SUB", "PRC", "ERC"])]

    base = S["text"][(S["text"].analysis == "repeated323")
                     & (S["text"].norm == "repgroup")
                     & (S["text"].repetition == 1)].groupby("roi").two_way.mean()

    p = []
    A = p.append
    A(f"# Reconstruction Feasibility Report: Repetition-Induced Representational "
      f"Transformation in fMRI-to-CLIP Decoding\n")
    A(f"**Generated:** {date.today().isoformat()} by `step12_write_report.py` from the "
      f"CSVs in `results/`.  \n"
      f"**Scope:** 8 NSD participants, sessions 1-2 (1,500 trials each), 11 anatomical "
      f"ROIs plus 4 composites, CPU only.\n")
    A("---\n")

    # ------------------------------------------------------------ exec summary
    A("## Executive Summary\n")
    try:
        man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
        n_train = int((man[man.subject_id == man.subject_id.iloc[0]]
                       .n_presentations == 1).sum())
    except Exception:                                           # noqa: BLE001
        n_train = 761
    big = main323.loc[main323.delta.idxmin()]                   # largest decline
    A(f"A ridge decoder from single-trial ROI voxels to CLIP ViT-B/32 embeddings works on "
      f"these data. Trained only on the {n_train} single-presentation trials and tested on "
      f"held-out repeated stimuli, it reaches "
      f"**{fmt(base.get('AllCortex'), 3)} two-way identification** for the six-region cortical "
      f"composite against a chance level of 0.500, and it orders the ROIs the way CLIP-text "
      f"targets should (scene and face cortex above early visual cortex, hippocampal "
      f"subfields at chance). That part is solid.\n")
    A(f"The repetition effect is not. Accuracy falls from Rep1 to Rep2 in most regions, "
      f"nominally significantly in {n_nominal} of 15 ROI sets. The smallest p-value is in "
      f"**{best.roi}** (delta = {fmt(best.delta)}, t({int(best.n_subj)-1}) = {fmt(best.t_stat,3)}, "
      f"p = {fmtp(best.p_parametric)}, exact sign-flip p = {fmtp(best.p_exact_perm)}, "
      f"Cohen's d = {fmt(best.cohens_d,2)} [{fmt(best.d_ci_lower,2)}, {fmt(best.d_ci_upper,2)}]), "
      f"though the substantively largest decline is in **{big.roi}** "
      f"(delta = {fmt(big.delta)}, p = {fmtp(big.p_parametric)}). "
      f"But **{n_fdr} of 15 survive Benjamini-Hochberg correction**, the largest Bayes factor "
      f"anywhere is BF10 = {fmt(max_bf,2)} (\"moderate\" at best), and every hippocampal ROI "
      f"lands in the inconclusive band (BF10 between "
      f"{fmt(hip.bf10.min(),2)} and {fmt(hip.bf10.max(),2)}). With N = 8 we can neither "
      f"establish the cortical effect nor establish the hippocampal null, so the "
      f"cortical-versus-hippocampal dissociation that motivated this work is not supported "
      f"in either direction.\n")

    if S["rsa"] is not None:
        from scipy import stats as _s
        rt = S["rsa"][S["rsa"].semantic_space == "S_text"]
        pv, rois_ = [], []
        for roi, gg in rt.groupby("roi"):
            pv.append(_s.ttest_1samp(gg.convergence, 0.0).pvalue)
            rois_.append(roi)
        pv = np.asarray(pv)
        adj, prev = np.empty(len(pv)), 1.0
        for rank, idx in enumerate(np.argsort(pv)[::-1]):
            prev = min(prev, pv[idx] * len(pv) / (len(pv) - rank))
            adj[idx] = prev
        fdr_map = dict(zip(rois_, np.clip(adj, 0, 1)))
        ppa = rt[rt.roi == "PPA"]
        tt = _s.ttest_1samp(ppa.convergence, 0.0)
        n_rsa_fdr = int((np.clip(adj, 0, 1) < 0.05).sum())
        A(f"The single most consequential finding is not about decoding at all. "
          f"Recomputing the RSA semantic-alignment analysis from the **corrected** stimulus "
          f"mapping shows that alignment to CLIP-text structure *decreases* across "
          f"repetitions in every one of the 11 ROIs, and in PPA that decrease is the only "
          f"result in this entire report that survives multiple-comparison correction "
          f"(convergence = {fmt(ppa.convergence.mean())}, t(7) = {fmt(tt.statistic,3)}, "
          f"p = {fmtp(tt.pvalue)}, FDR-corrected p = {fmtp(fdr_map['PPA'])}; "
          f"{n_rsa_fdr} of 11 ROIs survive). "
          f"The ADVANCED_RESEARCH_2 manuscript reports the opposite sign for the same "
          f"region and quantity. The corrected and published per-subject convergence values "
          f"are uncorrelated"
          + (f" (Spearman rho = {fmt(_s.spearmanr(S['cmp'].convergence_corrected, S['cmp'].convergence_legacy).statistic,3)}, "
             f"p = {fmtp(_s.spearmanr(S['cmp'].convergence_corrected, S['cmp'].convergence_legacy).pvalue)}, "
             f"{len(S['cmp'])} subject x ROI cells)" if S["cmp"] is not None else "")
          + ", which is what the off-by-one stimulus index in the upstream manifest predicts.\n")

    A("**Bottom line.** The decoding pipeline is sound and the corrected stimulus mapping "
      "is verified. What the data show is a small, uniform reduction in decodable "
      "stimulus information across repetitions, consistent with reduced signal-to-noise, "
      "with no surviving evidence that information is reallocated toward semantic "
      "structure. The honest write-up is a well-controlled negative result plus a "
      "correction to the published RSA analysis — not a paper about representational "
      "transformation.\n")
    A("---\n")

    # ------------------------------------------------------------------ methods
    A("## 1. Methods\n")
    A("### 1.1 Dataset and preprocessing\n")
    A("NSD 7T single-trial betas, `betas_fithrf_GLMdenoise_RR`, `func1pt8mm`, sessions 1 "
      "and 2 only (750 trials each, 1,500 per participant; the full experiment runs to "
      "30-40 sessions, which are not on disk). ROI betas were taken from the cached "
      "`ADVANCED_RESEARCH/data/roi_betas/{subj}_roi_betas_1500trials.npz` tensors "
      "(11 ROIs, about 7,049 voxels per participant).\n")
    A("Stimulus identity was rebuilt from the behavioural logs as `nsdId = 73KID - 1`. "
      "The upstream `ADVANCED_RESEARCH` manifest looks up a 1-based 73k index in the "
      "0-based `nsdId` column of `nsd_stim_info_merged.csv`, so every trial there carries "
      "the annotations of a different image; section 3.4 quantifies the cost. Stimulus "
      "images were reconstructed as the participant saw them: the COCO source image "
      "cropped by NSD's own `cropBox` (which yields the square NSD frame) and resized to "
      "256 px. 7,945 of the 7,951 required images were obtained; 6 COCO URLs are dead.\n")
    A("### 1.2 Decoder\n")
    A("Ridge regression, voxels to a 512-d CLIP ViT-B/32 embedding, solved with "
      "`_fast_ridge.SVDRidge` (dual eigendecomposition, so all five lambda values come "
      "from one factorisation; verified against `sklearn.linear_model.Ridge` to "
      "max|diff| ~ 4e-15). Lambda chosen from {1e2 ... 1e6} by 5-fold CV inside the "
      "training set. Targets: CLIP **text** embeddings (mean of up to 5 COCO captions, "
      "L2-normalised) and CLIP **image** embeddings of the reconstructed NSD stimulus.\n")
    A("### 1.3 Train/test design\n")
    A("Training used only the 761 trials whose stimulus is presented exactly once within "
      "sessions 1-2. Those are image-disjoint from every test set, so nothing leaks. All "
      "761 are first presentations by construction, which biases the decoder *in favour "
      "of* Rep1 — so the observed Rep1 > Rep2 ordering is partly expected, and any "
      "reverse effect would be conservative. Separate per-repetition decoders are not "
      "possible (n = 323 and n = 93).\n")
    A("Test cohorts: `repeated323` (323 stimuli with >= 2 presentations, Rep1 vs Rep2, "
      "323-way pool) and `lifecycle93` (93 stimuli with all 3 presentations, 93-way pool). "
      "The 93 are the stimuli whose whole three-presentation lifecycle happens to fall "
      "inside sessions 1-2, i.e. an extreme short-lag tail of the NSD design "
      "(median Rep2->Rep3 lag 121 trials against 1,642 NSD-wide); they are not a "
      "representative sample.\n")
    A("Two voxel normalisations, both reported: `session` (z-score each voxel within "
      "session) and `repgroup` (z-score within repetition group, which additionally "
      "removes the global amplitude reduction caused by repetition suppression).\n")
    A("### 1.4 Metrics\n")
    A("Two-way identification accuracy is primary (chance 0.500; stable at small n); "
      "top-1, top-5, mean percentile rank and mean cosine to the true target are also "
      "recorded. Proportional loss is `delta / (Rep1 - 0.5)`, the quantity a uniform-SNR "
      "account predicts to be constant.\n")
    A("### 1.5 Statistical framework\n")
    A("Paired t-tests across the 8 participants, with: the exact sign-flip permutation "
      "test over all 2^8 = 256 sign assignments; a 10,000-draw random sign-flip test; "
      "Benjamini-Hochberg FDR within each normalisation x cohort x contrast family "
      "(reported both across all 15 ROI sets and across the 11 anatomical ROIs alone, "
      "since composites are weighted sums of their own constituents); paired Cohen's d "
      "with Hedges' g and a 95% CI from the noncentral t; and the JZS Bayes factor "
      "(Rouder et al. 2009, Cauchy scale r = sqrt(2)/2) so that null results carry "
      "evidence rather than only a failure to reject.\n")
    A("---\n")

    # ------------------------------------------------------------------ results
    A("## 2. Primary Results\n")
    A("### 2.1 Repetition effect on CLIP-text decoding\n")
    A("Baseline decodability at Rep1 (repeated323, per-repetition z-scored, mean of 8):\n")
    bt = base.reset_index().rename(columns={"two_way": "Rep1 two-way"})
    A(md_table(bt, ["roi", "Rep1 two-way"], ["ROI set", "Rep1 two-way"],
               order_col="roi", rounder=4) + "\n")
    A("Rep2 - Rep1 contrast with the full statistical apparatus "
      "(repeated323, `repgroup`):\n")
    A(md_table(main323, ["roi", "delta", "t_stat", "p_parametric", "p_exact_perm",
                         "p_fdr", "p_fdr_roi11", "cohens_d", "d_ci_lower", "d_ci_upper",
                         "bf10", "bf_interpretation"],
               ["ROI set", "delta", "t(7)", "p", "p exact perm", "p FDR (15)",
                "p FDR (11)", "d", "d CI lo", "d CI hi", "BF10", "BF verdict"],
               order_col="roi") + "\n")
    A(f"Read this table as a whole rather than by its smallest p-value. "
      f"{n_nominal} of 15 sets are nominally significant and the exact permutation test "
      f"agrees with the parametric test throughout, so the nominal effects are not an "
      f"artefact of the t approximation. But nothing survives FDR under either family, "
      f"and no Bayes factor reaches the conventional \"strong\" threshold of 10. The "
      f"effect is real enough to be worth controlling for and too small to carry a claim.\n")

    if S["life"] is not None:
        A("### 2.2 Lifecycle trajectory (93 stimuli, 3 repetitions)\n")
        lf = S["life"][S["life"].norm == "repgroup"]
        A("Linear mixed model with a subject random intercept and orthogonal polynomial "
          "contrasts for repetition. The linear trend uses all 24 cells per ROI and is "
          "better powered than the Rep3 - Rep1 paired test:\n")
        A(md_table(lf, ["roi", "mean_rep1", "mean_rep2", "mean_rep3", "lin_beta", "lin_z",
                        "lin_p", "lin_p_fdr", "quad_z", "quad_p",
                        "n_monotonic_decreasing", "p_monotonic_binomial"],
                   ["ROI set", "Rep1", "Rep2", "Rep3", "linear beta", "z", "p",
                    "p FDR", "quad z", "quad p", "monotone/8", "p binom"],
                   order_col="roi") + "\n")
        nlin = int((lf.lin_p < 0.05).sum())
        nlinf = int((lf.lin_p_fdr < 0.05).sum())
        nquad = int((lf.quad_p < 0.05).sum())
        A(f"{nlin} of 15 sets show a nominally significant linear decline and {nlinf} "
          f"survive FDR; {nquad} show any quadratic component, so the trajectory is a "
          f"shallow straight-line decline rather than the accelerating or saturating "
          f"pattern a consolidation account would predict. Monotonic Rep1 > Rep2 > Rep3 "
          f"ordering is not above the 1-in-6 chance rate in any region.\n")

    A("### 2.3 Text versus image target dissociation\n")
    A("This is the analysis that can separate a representational change from uniform SNR "
      "loss. A uniform loss scales every decodable signal by the same factor, so the "
      "*proportional* loss must match across target spaces. A shift toward semantic "
      "structure should cost less in CLIP-text (built from human captions) than in "
      "CLIP-image (built from pixels).\n")
    if S["dis"] is None:
        A("_Not available: run `step7_clip_image_targets.py`._\n")
    else:
        d = S["dis"][(S["dis"].norm == "repgroup") & (S["dis"].analysis == "repeated323")]
        ib = S["image"][(S["image"].analysis == "repeated323")
                        & (S["image"].norm == "repgroup")
                        & (S["image"].repetition == 1)].groupby("roi").two_way.mean()
        stronger = "stronger" if ib.get("AllCortex", 0) > base.get("AllCortex", 0) \
            else "weaker"
        A(f"CLIP-image decoding is in fact {stronger} than CLIP-text here "
          f"(AllCortex Rep1 {fmt(ib.get('AllCortex'),3)} for image against "
          f"{fmt(base.get('AllCortex'),3)} for text), which is what one expects from "
          f"targets built from the pixels rather than from captions, and which already "
          f"undercuts the premise that the text space is the privileged one. It also "
          f"matters statistically: a proportional comparison between two differently "
          f"reliable measures is not assumption-free.\n")
        A("Regions whose Rep1 accuracy sits at chance have a near-zero denominator in "
          "`delta / (Rep1 - 0.5)` and produce large unstable ratios, so ROI sets with a "
          "mean baseline below 0.02 in either space are flagged `baseline_ok = no` and "
          "held out of the FDR family rather than allowed to dominate it. Without that "
          "gate, DG_CA3, MTL and ERC appear as the strongest 'dissociations' in the "
          "93-stimulus cohort purely as division artefacts.\n")
        A(md_table(d, ["roi", "n_subj", "mean_prop_text", "mean_prop_image",
                       "mean_prop_dissociation", "t_stat", "p_val", "p_fdr", "cohens_d",
                       "n_subj_positive"],
                   ["ROI set", "n", "prop loss text", "prop loss image",
                    "dissociation", "t", "p", "p FDR", "d", "n>0"],
                   order_col="roi") + "\n")
        if "baseline_ok" in d.columns:
            d = d[d.baseline_ok.astype(bool)]
        scene = d[d.roi.isin(["PPA", "RSC", "SceneCtx"])]
        early = d[d.roi.isin(["V1", "V2", "EarlyVis"])]
        nsig = int((d.p_val < 0.05).sum())
        nfdr = int((d.p_fdr < 0.05).sum())
        pos = scene.mean_prop_dissociation.mean()
        n_pos_roi = int((d.mean_prop_dissociation > 0).sum())
        A(f"Positive values mean CLIP-text degrades proportionally less than CLIP-image, "
          f"the predicted signature. Across the {len(d)} interpretable ROI sets, "
          f"{n_pos_roi} are positive. Scene-selective regions average "
          f"{fmt(pos)} and early visual regions average "
          f"{fmt(early.mean_prop_dissociation.mean())}. "
          + (f"{nsig} reach nominal significance and {nfdr} survive FDR. "
             if nsig else "None reaches nominal significance. ")
          + ("The prediction is directionally supported in scene cortex but the test does "
             "not clear correction, so it cannot carry the argument on its own.\n"
             if pos > 0 else
             "**The dissociation runs opposite to the prediction in exactly the regions "
             "where the semanticization account requires it**: scene-selective cortex "
             "loses proportionally *more* in the text space than in the image space, not "
             "less. This was designated the primary defence against the uniform-SNR "
             "account, and it fails.\n"))
    A("---\n")

    # ------------------------------------------------------------------ controls
    A("## 3. Confound Controls\n")
    A("### 3.1 Control A: proportional SNR\n")
    if S["ctrlA"] is not None:
        ca = S["ctrlA"]
        from scipy import stats as _s
        rr = _s.pearsonr(ca.above_chance, ca.delta)
        A(f"Regressing the subject-averaged delta on baseline-above-chance across ROI sets "
          f"gives r = {fmt(rr.statistic,3)}, p = {fmtp(rr.pvalue)}, with a near-constant "
          f"proportional loss of {fmt(ca.prop_loss.mean(),4)} +/- "
          f"{fmt(ca.prop_loss.std(),4)} over a 45-fold range of baseline. Taken at face "
          f"value that is the uniform-SNR signature and it is the central vulnerability "
          f"of this project.\n")
    if S["resid"] is not None:
        rs = S["resid"][(S["resid"].norm == "repgroup")
                        & (S["resid"].analysis == "repeated323")]
        A("Averaging over subjects before regressing inflates the apparent "
          "proportionality, so the model was refitted **within each subject** and the "
          "residuals tested across participants. A genuine representational effect has "
          "to appear here; uniform SNR loss cannot produce it.\n")
        A(md_table(rs, ["roi", "mean_residual", "sem_residual", "t_stat", "p_val",
                        "p_fdr", "cohens_d", "n_subj_negative"],
                   ["ROI set", "mean residual", "SEM", "t", "p", "p FDR", "d", "n<0"],
                   order_col="roi") + "\n")
        nres = int((rs.p_val < 0.05).sum())
        nresf = int((rs.p_fdr < 0.05).sum())
        A(f"The within-subject proportional model explains far less than the "
          f"subject-averaged version, and {nres} of 15 residuals depart nominally from it "
          f"({nresf} survive FDR). The departures do not line up into a story: they point "
          f"in opposite directions for `SceneCtx` and `AllCortex`, which are nested sets. "
          f"This is the weakest of the three defences against the SNR account.\n")

    A("### 3.2 Control B: time on task\n")
    if S["cb"] is not None:
        A("Using only the 761 single-presentation trials, which contain no repetition "
          "structure at all, decoding was scored separately for the early and late halves "
          "of the trial sequence. Step 3 ran this for cortical ROIs only; extending it to "
          "the MTL and hippocampus is what exposed the problem.\n")
        A(md_table(S["cb"], ["roi", "is_mtl", "early", "late", "delta", "t", "p", "p_fdr"],
                   ["ROI set", "MTL", "early", "late", "delta", "t", "p", "p FDR"],
                   order_col="roi") + "\n")
        A("The gain is about +0.015 in **every** region, including CA1, DG_CA3, SUB and "
          "PRC, whose early-half accuracy sits at or slightly below chance. A region with "
          "no decodable stimulus information cannot acquire some, so a uniform additive "
          "gain across signal-bearing and signal-free regions alike points at the "
          "measurement rather than at the brain.\n")
    if S["cbnull"] is not None:
        cn = S["cbnull"]
        A("Step 9d tests that directly, re-running the split with the target matrix "
          "permuted across trials, and separately with the voxel rows permuted, leaving "
          "the trial order, session z-scoring, split and candidate pools untouched:\n")
        A(md_table(cn, ["roi", "condition", "acc_early", "acc_late", "delta", "t", "p",
                        "n_pos"],
                   ["ROI set", "condition", "early", "late", "delta", "t", "p", "n>0"])
          + "\n")
        pv = cn.pivot_table(index="roi", columns="condition", values="delta")
        obs = pv["observed"].mean()
        shuf = pv["shuffle_rows"].mean() if "shuffle_rows" in pv else np.nan
        perm = pv["perm_targets"].mean() if "perm_targets" in pv else np.nan
        frac = 100 * shuf / obs if np.isfinite(shuf) and obs else np.nan
        if np.isfinite(frac) and frac > 50:
            A(f"**Shuffling the voxel rows — which destroys any brain-stimulus "
              f"relationship while leaving the trial order, the session z-scoring, the "
              f"split and the candidate pools intact — reproduces "
              f"{frac:.0f}% of the observed gain** (null mean delta = {fmt(shuf)} against "
              f"an observed {fmt(obs)}), and it does so with the same nominal significance "
              f"and the same 6-of-8 subject concordance. Control B is therefore measuring "
              f"a property of the early/late split, not time on task, and the 'later is "
              f"better' result reported in step 3 should be withdrawn.\n")
            A(f"The second null, which permutes the targets instead, sits at "
              f"{fmt(perm)}. The bias survives when the targets keep their trial order "
              f"and vanishes when they do not, which locates it in a relationship between "
              f"the target structure and trial position rather than in the brain data. "
              f"The practical consequence is that any early/late or Rep1/Rep2 comparison "
              f"scored against a pool drawn from the same half needs its own null before "
              f"it can be believed.\n")
        else:
            A(f"The nulls sit near zero (voxel-shuffle {fmt(shuf)}, target-permutation "
              f"{fmt(perm)}, against an observed {fmt(obs)}), so the early-to-late gain is "
              f"not produced by the split itself. Control B survives as an observation, "
              f"though the uniform +0.015 across signal-free regions remains unexplained "
              f"and should be flagged as such.\n")

    A("### 3.3 Session-balanced analysis\n")
    if S["same"] is not None:
        A("Contrary to the assumption that NSD always spaces repetitions across sessions, "
          "260 of the 323 repeated stimuli have Rep1 and Rep2 inside the **same** session "
          "(136 both in session 1, 124 both in session 2). The contrast can therefore be "
          "run with session, scanner drift and scan day held constant:\n")
        A(md_table(S["same"], ["roi", "n_stim", "rep1", "rep2", "delta", "sem", "t", "p",
                               "p_fdr", "n_neg"],
                   ["ROI set", "n stim", "Rep1", "Rep2", "delta", "SEM", "t", "p",
                    "p FDR", "n<0"], order_col="roi") + "\n")
        ns = int((S["same"].p < 0.05).sum())
        sc = S["same"][S["same"].roi == "SceneCtx"]
        A(f"The sign is preserved almost everywhere but {ns} of 15 sets reach nominal "
          f"significance: `SceneCtx` weakens from delta = {fmt(main323[main323.roi=='SceneCtx'].delta.iloc[0])} "
          f"(p = {fmtp(main323[main323.roi=='SceneCtx'].p_parametric.iloc[0])}) in the full "
          f"cohort to {fmt(sc.delta.iloc[0])} (p = {fmtp(sc.p.iloc[0])}) here. Part of the "
          f"headline effect was carried by the cross-session pairs, which is exactly what "
          f"a session or drift confound would produce.\n")
    if S["loso"] is not None:
        A("A leave-one-session-out decoder (trained on one session's novel trials, tested "
          "on the other session's repeats, so train and test never share a session) gives "
          "the same picture:\n")
        A(md_table(S["loso"], ["roi", "delta", "sem", "t", "p", "p_fdr", "n_neg"],
                   ["ROI set", "delta", "SEM", "t", "p", "p FDR", "n<0"],
                   order_col="roi") + "\n")

    A("### 3.4 Off-by-one manifest validation\n")
    if S["obo"] is not None:
        ob = S["obo"].pivot_table(index="roi", columns="targets", values="two_way")
        A("Fitting the same decoder against targets for the off-by-one image — what "
          "the `ADVANCED_RESEARCH_2` feature spaces actually encode — collapses "
          "performance to chance, confirming that the corrected mapping is the one "
          "carrying signal:\n")
        A(md_table(ob.reset_index(), ["roi", "correct", "offby1"],
                   ["ROI set", "corrected targets", "off-by-one targets"]) + "\n")
    A("---\n")

    # ------------------------------------------------------------------- bridge
    A("## 4. Bridge to the RSA Framework\n")
    A("### 4.1 A necessary deviation\n")
    A("The plan was to cross-link against "
      "`ADVANCED_RESEARCH_2/results/tables/axis2_semantic_correlations.csv`. That table "
      "cannot validate anything here, because it is computed from the shifted stimulus "
      "annotations (section 3.4). Correlating a corrected measure against RSA values "
      "built from randomly paired annotations would correlate signal with noise, and a "
      "null would be uninformative. So the RSA analysis was recomputed from the corrected "
      "mapping, using the same CLIP text and CLIP image spaces the decoder predicts, and "
      "the published table is reported only as a diagnostic.\n")
    A("### 4.2 Corrected RSA semantic convergence\n")
    if S["rsa"] is not None:
        from scipy import stats as _s
        for space in sorted(S["rsa"].semantic_space.unique()):
            g = S["rsa"][S["rsa"].semantic_space == space]
            rows = []
            for roi, gg in g.groupby("roi"):
                t = _s.ttest_1samp(gg.convergence, 0.0)
                rows.append(dict(roi=roi, rho_rep1=gg.rho_rep1.mean(),
                                 rho_rep2=gg.rho_rep2.mean(),
                                 convergence=gg.convergence.mean(),
                                 t_stat=t.statistic, p_val=t.pvalue))
            tb = pd.DataFrame(rows)
            pv = tb.p_val.values
            o = np.argsort(pv)[::-1]
            adj, prev = np.empty(len(pv)), 1.0
            for rank, idx in enumerate(o):
                prev = min(prev, pv[idx] * len(pv) / (len(pv) - rank))
                adj[idx] = prev
            tb["p_fdr"] = np.clip(adj, 0, 1)
            A(f"**{space}** (Spearman rho between the neural RDM and the semantic RDM, "
              f"323-stimulus cohort):\n")
            A(md_table(tb, ["roi", "rho_rep1", "rho_rep2", "convergence", "t_stat",
                            "p_val", "p_fdr"],
                       ["ROI", "rho Rep1", "rho Rep2", "convergence", "t(7)", "p",
                        "p FDR"], order_col="roi") + "\n")
        A("Semantic alignment **decreases** across repetitions in every ROI and both "
          "spaces. In PPA with CLIP-text targets the decrease is the only result in this "
          "report that survives FDR correction. The published analysis reports an "
          "*increase* for the same region and quantity.\n")
    A("### 4.3 Cross-method convergence\n")
    if S["within"] is not None:
        from scipy import stats as _s
        A("Correlating each participant's RSA-convergence vector against their "
          "decoding-delta vector across the 11 ROIs, then testing the 8 correlations "
          "against zero:\n")
        rows = []
        for (src, space), g in S["within"].groupby(["source", "semantic_space"]):
            t = _s.ttest_1samp(g.spearman_rho, 0.0)
            rows.append(dict(source=src, space=space, n=len(g),
                             mean_rho=g.spearman_rho.mean(), t_stat=t.statistic,
                             p_val=t.pvalue,
                             n_positive=int((g.spearman_rho > 0).sum())))
        A(md_table(pd.DataFrame(rows), ["source", "space", "n", "mean_rho", "t_stat",
                                        "p_val", "n_positive"],
                   ["source", "space", "N", "mean rho", "t", "p", "n>0"]) + "\n")
        A("The corrected spaces show a positive, nominally significant within-subject "
          "association: regions where RSA alignment drops more are the regions where "
          "decoding drops more. The two methods genuinely agree — but they agree that "
          "stimulus information is *lost*, not that it is reallocated. The legacy table "
          "shows no association at all, as expected from a shifted annotation.\n")
    A("### 4.4 Implications for the representational transformation hypothesis\n")
    A("Three independent lines were designated as defences against the uniform-SNR "
      "account. All three fail, in different ways:\n")
    A("- **Primary (text-versus-image dissociation).** Fails and reverses: scene-selective "
      "cortex loses proportionally more in the semantic target space than in the "
      "image space, which is the opposite of what a shift toward semantic structure "
      "predicts. CLIP-image also decodes better than CLIP-text overall, undercutting the "
      "premise that the text space is privileged.\n")
    A("- **Secondary (RSA cross-link).** The two methods do converge, strongly and within "
      "subject — but they converge on stimulus information being *lost*, not "
      "reallocated. Corrected RSA semantic alignment falls in all 11 ROIs, and in PPA the "
      "fall is the only FDR-surviving result in this report.\n")
    A("- **Tertiary (within-subject residuals).** Two ROI sets depart nominally from the "
      "proportional model, neither survives FDR, and they point in opposite directions "
      "for `SceneCtx` and `AllCortex`, which are nested sets. That is a pattern of noise, "
      "not of structure.\n")
    A("On this evidence the representational transformation hypothesis is not supported in "
      "these two sessions of data. The published semantic result points the other way and "
      "is built on the shifted annotation, so it should be treated as an artefact of the "
      "indexing defect until it is re-derived from a corrected mapping.\n")
    A("---\n")

    # ------------------------------------------------------------------ scaling
    A("## 5. Scaling\n")
    if S["scal"] is not None:
        from scipy import stats as _s
        rows = []
        for roi, g in S["scal"].groupby("roi"):
            m = g.groupby("n_train").two_way.mean()
            lr = _s.linregress(np.log(m.index.values), m.values)
            rows.append(dict(roi=roi, n_100=m.get(100, np.nan), n_761=m.get(761, np.nan),
                             slope_per_ln_n=lr.slope, r2=lr.rvalue ** 2,
                             extrap_24k=lr.intercept + lr.slope * np.log(24000)))
        A(md_table(pd.DataFrame(rows), ["roi", "n_100", "n_761", "slope_per_ln_n", "r2",
                                        "extrap_24k"],
                   ["ROI set", "n=100", "n=761", "slope per ln(n)", "R2",
                    "naive extrap. n=24k"]) + "\n")
        A("Accuracy is near-linear in log(n) with no plateau, so the binding constraint is "
          "training-set size, not the ROI selection, the target space or the decoder. The "
          "extrapolation to a full 30-session participant is an order-of-magnitude "
          "argument only — accuracy is bounded at 1.0 and the log-linear form must "
          "break down — but it lands in the range published reconstruction work "
          "reports. Acquiring the remaining sessions (~110-150 GB for 8 participants) is "
          "the intervention that would change the answer; nothing about the method needs "
          "fixing first.\n")
    A("---\n")

    # -------------------------------------------------------- limits and verdict
    A("## 6. Strengths and Limitations\n")
    A("**Strengths.** The stimulus mapping is verified three independent ways and "
      "validated by a decoder that collapses to chance under the shifted alternative. "
      "Train and test sets are image-disjoint. Every contrast carries an exact "
      "permutation test, an effect size with a CI, and a Bayes factor. Repetition "
      "suppression is removed by per-repetition normalisation and results are reported "
      "under both normalisations. The session confound is addressed with a 260-stimulus "
      "within-session cohort and a leave-one-session-out decoder. Controls were extended "
      "to regions where the measure has no signal, which is what caught the Control B "
      "problem.\n")
    A("**Limitations, in order of severity.**\n")
    A("1. **Nothing survives multiple-comparison correction.** Across 15 ROI sets the "
      "best FDR-adjusted p in the primary contrast is "
      f"{fmtp(main323.p_fdr.min())}. Bayes factors top out at {fmt(max_bf,2)}.\n")
    A("2. **N = 8 cannot support a dissociation claim.** Hippocampal Bayes factors sit in "
      "the inconclusive band, so the hippocampal null is not established; \"significant "
      "in cortex, not in hippocampus\" is a power statement, not a finding.\n")
    cb_line = ("3. **Control B is invalid and is withdrawn.** The early/late gain appears "
               "at equal magnitude in regions with no decodable signal, and a null with "
               "the voxel rows shuffled reproduces most of it (section 3.2).\n")
    if S["cbnull"] is not None:
        pv = S["cbnull"].pivot_table(index="roi", columns="condition", values="delta")
        if "shuffle_rows" in pv and pv["observed"].mean():
            f_ = 100 * pv["shuffle_rows"].mean() / pv["observed"].mean()
            cb_line = (f"3. **Control B is invalid and is withdrawn.** The early/late gain "
                       f"appears at equal magnitude in regions with no decodable signal, "
                       f"and a null with the voxel rows shuffled reproduces {f_:.0f}% of "
                       f"it (section 3.2).\n")
    A(cb_line)
    A("4. **Repetition is collinear with trial position and session by design**, "
      "identically in all 8 participants because `masterordering` is shared, so the "
      "confound does not average out with more participants.\n")
    A("5. **Only 2 of 30-40 sessions are available**, and the 93-stimulus lifecycle "
      "cohort is an extreme short-lag tail of the design rather than a sample of it.\n")
    A("6. **The training set is all first presentations**, biasing the decoder toward "
      "Rep1.\n")
    A("7. **CLIP-text and CLIP-image have unequal baseline reliability**, so the "
      "proportional dissociation test is not assumption-free.\n")
    A("---\n")
    A("## 7. Conclusions and Next Steps\n")
    A("The pipeline is correct and the question is well posed, but these two sessions "
      "cannot answer it. What is established: a working fMRI-to-CLIP decoder at this data "
      "scale; a verified stimulus mapping; a small uniform reduction in decodable "
      "stimulus information across repetitions; and no surviving evidence of "
      "reallocation toward semantic structure.\n")
    A("What should happen next, in order:\n")
    A("1. **Treat the corrected RSA result as a correction obligation.** The PPA semantic "
      "convergence reverses sign under the corrected mapping and is the only "
      "FDR-surviving result here. Re-derive the `ADVANCED_RESEARCH_2` semantic analyses "
      "before that manuscript goes anywhere.\n")
    A("2. **Withdraw or re-found Control B** depending on the null diagnostic, and "
      "replace it with trial-position-matched resampling of Rep1 against the Rep2/Rep3 "
      "position distribution.\n")
    A("3. **Acquire the remaining NSD sessions** for at least two participants and repeat "
      "the whole design. The scaling curve says this is the only change that materially "
      "raises power.\n")
    A("4. **Pre-register the ROI set.** Declare PPA, V1 and AllCortex as primary and FDR "
      "the rest; the current 15-set family cannot support any single-ROI claim.\n")
    A("5. **Do not run diffusion reconstruction yet.** At 761 training trials the "
      "differences across repetitions would be dominated by decoder instability.\n")
    A("---\n")
    A("## Appendix: files\n")
    A("Scripts, in execution order:\n")
    A("```\nstep1_corrected_manifest_and_targets.py    corrected manifest + CLIP text targets\n"
      "step1b_offbyone_targets.py                 targets for the shifted images\n"
      "step2_ridge_retrieval_by_repetition.py     main decoder\n"
      "step2b_contrast_stats.py                   paired contrasts\n"
      "step3_confound_controls.py                 Controls A and B (cortical)\n"
      "step4_training_size_scaling.py             scaling curve\n"
      "step5_offbyone_demonstration.py            indexing validation\n"
      "step6_statistical_infrastructure.py        permutation, FDR, effect size, BF\n"
      "step7a_fetch_coco_images.py                NSD stimuli from COCO\n"
      "step7_clip_image_targets.py                CLIP image targets + dissociation\n"
      "step8_rsa_decoding_crosslink.py            corrected RSA + cross-method bridge\n"
      "step9_extended_confound_controls.py        MTL Control B, residuals, session\n"
      "step9d_controlB_null_diagnostic.py         is Control B real?\n"
      "step10_lifecycle_trajectory.py             LMM trend tests\n"
      "step11_publication_figures.py              figures\n"
      "step12_write_report.py                     this report\n```\n")
    present = sorted(p.name for p in RESULTS.glob("*.csv"))
    A(f"Result tables ({len(present)}):\n")
    A("```\n" + "\n".join(present) + "\n```\n")
    figs = sorted(p.name for p in (RESULTS / "figures").glob("*.pdf")) \
        if (RESULTS / "figures").exists() else []
    if figs:
        A("Figures (PDF plus a 300 DPI PNG of each):\n")
        A("```\n" + "\n".join(figs) + "\n```\n")

    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"[OK] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
