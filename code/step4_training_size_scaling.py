"""
Step 4 (CPU-only): how does fMRI->CLIP-text decoding scale with training-set size?
Answers the practical question of whether downloading more NSD sessions would move
this from "retrieval works" to "reconstruction works".

Training pool = the 761 single-presentation trials. Test = Rep1 of the 323
repeated images (disjoint images). Subsample the training pool and refit.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SETS = {"AllCortex": ["V1", "V2", "FFA", "PPA", "RSC", "PHC"], "PPA": ["PPA"], "V1": ["V1"]}
SIZES = [100, 200, 300, 450, 600, 761]
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
    n = min(5, max(2, len(X) // 30))
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(n, shuffle=True, random_state=42).split(X):
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
    rows = []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        Y = T[[c2r[c] for c in m.coco_id]]
        novel = (m.n_presentations == 1).values
        test = ((m.n_presentations >= 2) & (m.repeat_number == 1)).values
        imgs = m.loc[test, "nsd_id"].values
        pool = T[[c2r[c] for c in m.loc[test, "coco_id"]]]
        ti = np.arange(len(imgs))
        for name, parts in SETS.items():
            X = zs(np.concatenate([z[p] for p in parts], 1).astype(np.float64), m.session.values)
            Xn, Yn = X[novel], Y[novel]
            for size in SIZES:
                accs = []
                for seed in range(3 if size < 761 else 1):
                    idx = np.random.default_rng(seed).choice(len(Xn), size, replace=False)
                    a = cv_alpha(Xn[idx], Yn[idx])
                    p = SVDRidge().fit(Xn[idx], Yn[idx]).predict(X[test], a)
                    accs.append(two_way(p, pool, ti))
                rows.append(dict(subject_id=subj, roi=name, n_train=size,
                                 two_way=float(np.mean(accs))))
        print(f"  [OK] {subj}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "training_size_scaling.csv", index=False)
    print("\n=== 2-way identification vs training-set size (mean of 8 subjects) ===")
    print(d.pivot_table(index="n_train", columns="roi", values="two_way").round(4).to_string())


if __name__ == "__main__":
    main()
