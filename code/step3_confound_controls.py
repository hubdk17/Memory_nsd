"""
Step 3 (CPU-only): the two controls that decide whether a repetition effect on
decoding accuracy is representational or merely a signal-to-noise artefact.

Control A -- proportional-SNR test.
  If repetition simply lowers SNR by a constant factor, then the drop in
  identification accuracy above chance should be proportional to each ROI's
  baseline accuracy above chance, i.e. delta / (acc_Rep1 - 0.5) is constant
  across ROIs. A genuine representational reorganisation must break that
  proportionality (some ROIs gaining while others lose).

Control B -- time-on-task test.
  Rep2/Rep3 trials necessarily occur later in the 1500-trial sequence than Rep1.
  Using ONLY the 761 single-presentation trials (which have no repetition
  structure at all), we ask whether decodability declines with trial position.
  Any such slope is an alternative, repetition-free explanation for Control A.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROI_SETS = {"V1": ["V1"], "V2": ["V2"], "FFA": ["FFA"], "PPA": ["PPA"], "RSC": ["RSC"],
            "PHC": ["PHC"], "EarlyVis": ["V1", "V2"], "SceneCtx": ["PPA", "RSC", "PHC"],
            "AllCortex": ["V1", "V2", "FFA", "PPA", "RSC", "PHC"]}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]


def zscore_groups(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def two_way(P, pool, true_idx):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ pool.T
    ts = S[np.arange(len(true_idx)), true_idx]
    S[np.arange(len(true_idx)), true_idx] = -np.inf
    return float((ts[:, None] > S).sum() / (len(true_idx) * (pool.shape[0] - 1)))


def cv_alpha(X, Y):
    kf, acc = KFold(5, shuffle=True, random_state=42), np.zeros(len(ALPHAS))
    for tr, va in kf.split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(X[va], a)
            acc[j] += float(((p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)) * Y[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))]


def main():
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    tg = np.load(HERE / "data" / "clip_text_targets.npz")
    c2r = {int(c): i for i, c in enumerate(tg["coco_ids"])}
    T = tg["T_text"]
    rows_b = []

    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        Y = T[[c2r[c] for c in m.coco_id]]
        novel = (m.n_presentations == 1).values
        gt = m.global_trial.values

        for roi, parts in ROI_SETS.items():
            X = zscore_groups(np.concatenate([z[p] for p in parts], 1).astype(np.float64),
                              m.session.values)
            # Control B: 5-fold CV over the 761 novel trials, score each fold's
            # test trials, then split scored trials into early/late halves.
            Xn, Yn, gtn = X[novel], Y[novel], gt[novel]
            alpha = cv_alpha(Xn, Yn)
            kf = KFold(5, shuffle=True, random_state=7)
            pred = np.zeros_like(Yn)
            for tr, te in kf.split(Xn):
                pred[te] = SVDRidge().fit(Xn[tr], Yn[tr]).predict(Xn[te], alpha)
            med = np.median(gtn)
            early, late = gtn <= med, gtn > med
            pool_idx = np.arange(len(Yn))
            rows_b.append(dict(subject_id=subj, roi=roi,
                               acc_early=two_way(pred[early], Yn, pool_idx[early]),
                               acc_late=two_way(pred[late], Yn, pool_idx[late]),
                               n_early=int(early.sum()), n_late=int(late.sum())))
        print(f"  [OK] {subj}", flush=True)

    b = pd.DataFrame(rows_b)
    b["delta_late_minus_early"] = b.acc_late - b.acc_early
    b.to_csv(HERE / "results" / "controlB_time_on_task.csv", index=False)
    print("\n=== Control B: time-on-task, novel trials only (N=8 paired) ===")
    for roi, g in b.groupby("roi"):
        t, p = stats.ttest_rel(g.acc_late, g.acc_early)
        print(f"  {roi:10s} early={g.acc_early.mean():.4f} late={g.acc_late.mean():.4f} "
              f"delta={g.delta_late_minus_early.mean():+.5f}  t={t:+.2f} p={p:.4f}")

    # Control A, from the step-2 output
    r = pd.read_csv(HERE / "results" / "ridge_clip_text_retrieval_by_repetition.csv")
    r = r[(r.analysis == "repeated323") & (r.norm == "repgroup")]
    piv = r.pivot_table(index=["subject_id", "roi"], columns="repetition", values="two_way")
    piv = piv.reset_index().rename(columns={1: "rep1", 2: "rep2"})
    piv["above_chance"] = piv.rep1 - 0.5
    piv["delta"] = piv.rep2 - piv.rep1
    piv["prop_loss"] = piv.delta / piv.above_chance
    agg = piv.groupby("roi")[["above_chance", "delta", "prop_loss"]].mean()
    rho, pr = stats.pearsonr(agg.above_chance, agg.delta)
    print("\n=== Control A: proportional-SNR test (repeated323, Rep2-Rep1) ===")
    print(agg.round(5).sort_values("above_chance", ascending=False).to_string())
    print(f"\n  corr(baseline above chance, delta) across {len(agg)} ROIs: r={rho:.3f} p={pr:.4g}")
    print(f"  proportional loss: mean={agg.prop_loss.mean():.4f} sd={agg.prop_loss.std():.4f} "
          f"(constant => uniform SNR account)")
    agg.to_csv(HERE / "results" / "controlA_proportional_snr.csv")


if __name__ == "__main__":
    main()
