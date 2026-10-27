"""
Step 9d: is the Control B "later is better" effect real signal, or a bias of the split?

Extending Control B to the hippocampal ROIs (step 9a) produced a suspicious pattern: the
early-to-late gain is about +0.015 in EVERY ROI, including CA1, DG_CA3, SUB and PRC whose
early-half accuracy sits AT or slightly BELOW chance. A region carrying no decodable
stimulus information cannot gain decodable stimulus information, so a uniform additive
gain across signal and no-signal regions alike points at the measurement rather than the
brain.

Two null controls, both leaving the pipeline otherwise untouched:
  perm_targets : the target matrix is permuted across trials, so no brain-stimulus
                 relationship survives, but the trial order, the session z-scoring, the
                 early/late split and the candidate pools are all unchanged.
  shuffle_rows : the voxel matrix rows are permuted instead, which breaks the same link
                 from the other side.

If either null reproduces a positive early-to-late difference, Control B is measuring a
property of the split and cannot be used as evidence about time on task.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
TEST_ROIS = ["V1", "PPA", "SceneCtx", "CA1", "DG_CA3"]
ROI_PARTS = {"V1": ["V1"], "PPA": ["PPA"], "SceneCtx": ["PPA", "RSC", "PHC"],
             "CA1": ["CA1"], "DG_CA3": ["DG_CA3"]}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
N_NULL = 10
SEED = 42


def zs(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def two_way(P, pool, ti):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ pool.T
    ts = S[np.arange(len(ti)), ti]
    S[np.arange(len(ti)), ti] = -np.inf
    return float((ts[:, None] > S).sum() / (len(ti) * (pool.shape[0] - 1)))


def cv_alpha(X, Y):
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(X[va], a)
            acc[j] += float(((p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8))
                             * Y[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))]


def early_late(Xn, Yn, gtn, alpha):
    pred = np.zeros_like(Yn)
    for tr, te in KFold(5, shuffle=True, random_state=7).split(Xn):
        pred[te] = SVDRidge().fit(Xn[tr], Yn[tr]).predict(Xn[te], alpha)
    med = np.median(gtn)
    e, l = gtn <= med, gtn > med
    idx = np.arange(len(Yn))
    return two_way(pred[e], Yn, idx[e]), two_way(pred[l], Yn, idx[l])


def main():
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    tg = np.load(HERE / "data" / "clip_text_targets.npz")
    c2r = {int(c): i for i, c in enumerate(tg["coco_ids"])}
    T = tg["T_text"]
    rows = []

    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        Y = T[[c2r[c] for c in m.coco_id]]
        novel = (m.n_presentations == 1).values
        sess = m.session.values
        gt = m.global_trial.values

        for roi in TEST_ROIS:
            X = zs(np.concatenate([z[p] for p in ROI_PARTS[roi]], 1).astype(np.float64),
                   sess)
            Xn, Yn, gtn = X[novel], Y[novel], gt[novel]
            alpha = cv_alpha(Xn, Yn)

            e, l = early_late(Xn, Yn, gtn, alpha)
            rows.append(dict(subject_id=subj, roi=roi, condition="observed", draw=0,
                             acc_early=e, acc_late=l, delta=l - e))

            rng = np.random.default_rng(abs(hash((subj, roi))) % 2**31)
            for k in range(N_NULL):
                p = rng.permutation(len(Yn))
                e, l = early_late(Xn, Yn[p], gtn, alpha)
                rows.append(dict(subject_id=subj, roi=roi, condition="perm_targets",
                                 draw=k, acc_early=e, acc_late=l, delta=l - e))
                q = rng.permutation(len(Xn))
                e, l = early_late(Xn[q], Yn, gtn, alpha)
                rows.append(dict(subject_id=subj, roi=roi, condition="shuffle_rows",
                                 draw=k, acc_early=e, acc_late=l, delta=l - e))
        print(f"  [OK] {subj}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "controlB_null_diagnostic.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Control B under the observed data and two nulls ===")
    print("  (null draws averaged within subject, then tested across the 8 subjects)")
    summ = []
    for (roi, cond), g in out.groupby(["roi", "condition"]):
        per_subj = g.groupby("subject_id")[["acc_early", "acc_late", "delta"]].mean()
        t, p = stats.ttest_1samp(per_subj.delta, 0.0)
        summ.append(dict(roi=roi, condition=cond, acc_early=per_subj.acc_early.mean(),
                         acc_late=per_subj.acc_late.mean(), delta=per_subj.delta.mean(),
                         sem=per_subj.delta.std(ddof=1) / np.sqrt(len(per_subj)),
                         t=t, p=p, n_pos=int((per_subj.delta > 0).sum())))
    s = pd.DataFrame(summ).sort_values(["roi", "condition"])
    print(s.round(5).to_string(index=False))
    s.to_csv(RESULTS / "controlB_null_diagnostic_stats.csv", index=False)

    print("\n  Verdict: if the nulls show delta ~ 0, Control B reflects the data; if the "
          "nulls\n  reproduce a positive delta, the early/late comparison is biased.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
