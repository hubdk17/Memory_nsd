"""
Step 5 (CPU-only): quantify the cost of the stimulus-indexing defect.

ADVANCED_RESEARCH/extract_manifests.py looks up the 1-based 73k stimulus ID in the
0-based `nsdId` column of nsd_stim_info_merged.csv, so every trial is annotated with
the COCO image whose nsdId is one greater than the image actually shown. All
downstream semantic feature spaces inherit that shift.

This script fits the same ridge decoder twice -- once against CLIP text targets for
the image actually presented, once against targets for the off-by-one image -- and
compares identification accuracy. If the shifted targets decode at chance, every
semantic result computed from that manifest is uninterpretable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SETS = {"AllCortex": ["V1", "V2", "FFA", "PPA", "RSC", "PHC"], "PPA": ["PPA"], "V1": ["V1"]}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]


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
    for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(X[va], a)
            acc[j] += float(((p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)) * Y[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))]


def main():
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    # merge the correct-image and shifted-image target banks
    a = np.load(HERE / "data" / "clip_text_targets.npz")
    b = np.load(HERE / "data" / "clip_text_targets_shifted.npz")
    ids = np.concatenate([a["coco_ids"], b["coco_ids"]])
    T = np.concatenate([a["T_text"], b["T_text"]])
    c2r = {}
    for i, c in enumerate(ids):
        c2r.setdefault(int(c), i)

    # off-by-one COCO id: the image the buggy manifest assigned to each trial
    man["coco_offby1"] = man.nsd_id.map(lambda n: n2c.get(n + 1, -1))
    have = man.coco_offby1.map(lambda c: c in c2r)
    print(f"  off-by-one COCO ids resolvable in the local target set: "
          f"{int(have.sum())}/{len(man)} trials")

    rows = []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        ok = m.coco_offby1.map(lambda c: c in c2r).values
        novel = ((m.n_presentations == 1).values) & ok
        test = ((m.n_presentations >= 2) & (m.repeat_number == 1)).values & ok
        for label, col in [("correct", "coco_id"), ("offby1", "coco_offby1")]:
            Y = T[[c2r[c] if c in c2r else 0 for c in m[col]]]
            pool = T[[c2r[c] for c in m.loc[test, col]]]
            ti = np.arange(int(test.sum()))
            for name, parts in SETS.items():
                X = zs(np.concatenate([z[p] for p in parts], 1).astype(np.float64), m.session.values)
                a = cv_alpha(X[novel], Y[novel])
                p = SVDRidge().fit(X[novel], Y[novel]).predict(X[test], a)
                rows.append(dict(subject_id=subj, roi=name, targets=label,
                                 two_way=two_way(p, pool, ti), n_train=int(novel.sum()),
                                 n_test=int(test.sum())))
        print(f"  [OK] {subj}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "offbyone_target_demonstration.csv", index=False)
    print("\n=== 2-way identification accuracy, correct vs off-by-one targets ===")
    piv = d.pivot_table(index="roi", columns="targets", values="two_way")
    for roi in piv.index:
        a = d[(d.roi == roi) & (d.targets == "correct")].two_way.values
        b = d[(d.roi == roi) & (d.targets == "offby1")].two_way.values
        t, p = stats.ttest_rel(a, b)
        print(f"  {roi:10s} correct={a.mean():.4f}  offby1={b.mean():.4f}  "
              f"t={t:+.2f} p={p:.2g}")


if __name__ == "__main__":
    main()
