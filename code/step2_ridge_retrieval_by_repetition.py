"""
Step 2 (CPU-only): Minimal viable experiment.
Ridge regression  ROI voxels -> CLIP ViT-B/32 text embedding (512-D),
trained ONLY on single-presentation ("novel") trials, then evaluated separately
on Rep1 / Rep2 / Rep3 presentations of held-out repeated images.

Train images and test images are disjoint, so nothing leaks.

Two voxel-normalisation schemes are run:
  session  : z-score each voxel within session (removes session drift)
  repgroup : z-score each voxel within repetition group (additionally removes the
             global amplitude reduction caused by repetition suppression -- the
             main mechanical confound for this comparison)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROIS = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_SETS = {r: [r] for r in ROIS}
ROI_SETS["EarlyVis"] = ["V1", "V2"]
ROI_SETS["SceneCtx"] = ["PPA", "RSC", "PHC"]
ROI_SETS["MTL"] = ["PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_SETS["AllCortex"] = ["V1", "V2", "FFA", "PPA", "RSC", "PHC"]
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
RNG = np.random.default_rng(42)


def zscore_groups(X, groups):
    Xz = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        blk = X[m]
        Xz[m] = (blk - blk.mean(0)) / (blk.std(0) + 1e-8)
    return Xz


def pick_alpha(Xtr, Ytr):
    """Inner 5-fold CV on the training trials, scored by mean target cosine."""
    kf = KFold(5, shuffle=True, random_state=42)
    acc = np.zeros(len(ALPHAS))
    for tr, va in kf.split(Xtr):
        m = SVDRidge().fit(Xtr[tr], Ytr[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(Xtr[va], a)
            pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
            acc[j] += float((pn * Ytr[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))], (acc / 5).tolist()


def retrieval_metrics(P, Ytest_pool, true_idx):
    """P: (n,512) predictions. Ytest_pool: (K,512) L2-normalised candidates.
    true_idx: index into pool for each prediction."""
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ Ytest_pool.T                       # (n, K) cosine similarity
    K = S.shape[1]
    true_s = S[np.arange(len(true_idx)), true_idx]
    # n-way rank of the true image (1 = best)
    rank = (S > true_s[:, None]).sum(1) + 1
    top1 = float((rank == 1).mean())
    top5 = float((rank <= 5).mean())
    pct = float(((K - rank) / (K - 1)).mean())   # 1 = perfect, 0.5 = chance
    # 2-way identification accuracy against every other candidate
    others = S.copy()
    others[np.arange(len(true_idx)), true_idx] = -np.inf
    two_way = float((true_s[:, None] > np.where(np.isfinite(others), others, -np.inf)).sum()
                    / (len(true_idx) * (K - 1)))
    return dict(top1=top1, top5=top5, pct_rank=pct, two_way=two_way,
                mean_cos_true=float(true_s.mean()), n=len(true_idx), pool=K)


def run():
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    tgt = np.load(HERE / "data" / "clip_text_targets.npz")
    coco_to_row = {int(c): i for i, c in enumerate(tgt["coco_ids"])}
    T = tgt["T_text"]

    rows, comp_rows = [], []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        assert len(m) == 1500
        Y = T[[coco_to_row[c] for c in m.coco_id]]

        train = (m.n_presentations == 1).values                 # 761 novel trials
        rep = m.repeat_number.values
        npres = m.n_presentations.values
        test3 = npres == 3                                      # 93 images x 3 reps
        test2 = npres >= 2                                       # 323 images, reps 1..n

        comp_rows.append(dict(subject_id=subj, n_train=int(train.sum()),
                              n_test3=int(test3.sum()),
                              rep1_sess2_frac=float((m.session[test3 & (rep == 1)] == 2).mean()),
                              rep2_sess2_frac=float((m.session[test3 & (rep == 2)] == 2).mean()),
                              rep3_sess2_frac=float((m.session[test3 & (rep == 3)] == 2).mean()),
                              rep1_mean_trial=float(m.global_trial[test3 & (rep == 1)].mean()),
                              rep2_mean_trial=float(m.global_trial[test3 & (rep == 2)].mean()),
                              rep3_mean_trial=float(m.global_trial[test3 & (rep == 3)].mean())))

        for roi_name, parts in ROI_SETS.items():
            X_raw = np.concatenate([z[p] for p in parts], axis=1).astype(np.float64)
            for norm in ["session", "repgroup"]:
                grp = m.session.values if norm == "session" else rep.copy()
                X = zscore_groups(X_raw, grp)
                alpha, _ = pick_alpha(X[train], Y[train])
                fitted = SVDRidge().fit(X[train], Y[train])
                mdl = type("M", (), {"predict": lambda self, Z, a=alpha, f=fitted: f.predict(Z, a)})()

                # --- primary: 93 images with a complete 3-presentation lifecycle
                imgs3 = m.loc[test3, "nsd_id"].unique()
                pool3 = T[[coco_to_row[m.loc[m.nsd_id == i, "coco_id"].iloc[0]] for i in imgs3]]
                idx_of = {im: k for k, im in enumerate(imgs3)}
                for r in [1, 2, 3]:
                    sel = test3 & (rep == r)
                    met = retrieval_metrics(mdl.predict(X[sel]), pool3,
                                            np.array([idx_of[i] for i in m.loc[sel, "nsd_id"]]))
                    rows.append(dict(subject_id=subj, roi=roi_name, norm=norm, alpha=alpha,
                                     analysis="lifecycle93", repetition=r, **met))

                # --- secondary: all 323 repeated images, Rep1 vs Rep2 (more power)
                imgs2 = m.loc[test2, "nsd_id"].unique()
                pool2 = T[[coco_to_row[m.loc[m.nsd_id == i, "coco_id"].iloc[0]] for i in imgs2]]
                idx_of2 = {im: k for k, im in enumerate(imgs2)}
                for r in [1, 2]:
                    sel = test2 & (rep == r)
                    met = retrieval_metrics(mdl.predict(X[sel]), pool2,
                                            np.array([idx_of2[i] for i in m.loc[sel, "nsd_id"]]))
                    rows.append(dict(subject_id=subj, roi=roi_name, norm=norm, alpha=alpha,
                                     analysis="repeated323", repetition=r, **met))
        print(f"  [OK] {subj}", flush=True)

    out = HERE / "results"
    pd.DataFrame(rows).to_csv(out / "ridge_clip_text_retrieval_by_repetition.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(out / "repetition_group_composition.csv", index=False)
    print("[OK] wrote results/")


if __name__ == "__main__":
    run()
