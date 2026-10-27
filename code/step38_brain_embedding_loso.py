"""
Step 38: where does individual information processing live?

Every subject's voxels are mapped into one common space (CLIP ViT-L/14 image
embeddings), so eight different brains produce comparable 768-d vectors for the same
picture. Then seven subjects are used to predict the eighth, per region, and the regions
where that prediction fails are the ones doing something individual.

    train   each subject's own ~9,000 non-shared images (~24,000 trials)
            ridge: voxels -> 768-d CLIP embedding
    test    the 766 shared1000 images every subject saw
    predict subject s's embedding for image X from the MEAN of the other seven
    score   cosine similarity, per image, per region

The ceiling is what makes this interpretable. A region can fail to transfer because its
code is individual or because it carries no decodable signal at all, and those look
identical without a bound. Here the bound is the subject predicting THEMSELVES: decode
presentation 1 and presentation 2 separately and correlate. Everything is reported as a
fraction of that.

    transfer / ceiling ~ 1   the representation of this image is shared
    transfer / ceiling ~ 0   with a usable ceiling: genuinely individual

Voxels are reduced by PCA fitted on the training images only before the ridge -- standard
practice for this many voxels, and it keeps the largest regions tractable. Fitting PCA on
training images alone matters: fitting it on all images would leak test structure into
the basis.

Finally the 766 test images are split by COCO supercategory, so transfer can be read per
kind of scene rather than only in aggregate.
"""
import argparse
import gc
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

import _full_data as fd
from _fast_ridge import SVDRidge
from step33_intersubject_mapping_breakdown import MIN_RELIABILITY, ROI_ORDER, shared_ids

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
N_PCA = 1000
SEED = 42


def _unit(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def cos(A, B):
    """Cosine AFTER removing the across-image mean from each set.

    Ridge shrinks every prediction toward the training centroid, so raw predictions
    for completely different images already sit at cosine 0.93-0.996. A raw cosine
    between two such sets is a comparison of two near-identical mean vectors and
    carries no image-specific information at all. Centring removes that shared
    offset so what is left is the part that actually varies with the stimulus.
    """
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    return (_unit(A) * _unit(B)).sum(1)


def identification(P, T):
    """Two-way identification: is P[i] closer to T[i] than to a random T[j]?

    Chance 0.5, immune to any shared offset, and the standard metric in the
    decoding literature. Used for model selection because raw cosine saturates.
    """
    Pn, Tn = _unit(P - P.mean(0, keepdims=True)), _unit(T - T.mean(0, keepdims=True))
    S = Pn @ Tn.T
    true = np.diag(S).copy()
    np.fill_diagonal(S, -np.inf)
    n = len(P)
    return float((true[:, None] > S).sum() / (n * (n - 1)))


def cv_alpha(X, Y, rng, folds=5):
    """Select alpha by identification accuracy, not by raw cosine.

    Scoring on raw cosine picks the most over-regularised model, because collapsing
    every prediction onto the centroid maximises that metric while destroying all
    stimulus-specific signal.
    """
    from sklearn.model_selection import KFold
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            acc[j] += identification(m.predict(X[va], a), Y[va])
    return ALPHAS[int(np.argmax(acc))]


def supercategories(coco_ids):
    """COCO supercategory per image; the most frequent annotated one."""
    want = set(int(c) for c in coco_ids)
    cat2super, per_img = {}, {}
    with zipfile.ZipFile(NSD_RAW / "annotations_trainval2017.zip") as z:
        for nm in ("annotations/instances_train2017.json",
                   "annotations/instances_val2017.json"):
            j = json.loads(z.read(nm))
            for c in j["categories"]:
                cat2super[c["id"]] = c["supercategory"]
            for a in j["annotations"]:
                i = int(a["image_id"])
                if i in want:
                    per_img.setdefault(i, []).append(cat2super[a["category_id"]])
    out = {}
    for i in want:
        s = per_img.get(i, [])
        out[i] = pd.Series(s).mode().iloc[0] if s else "none"
    return out


def decode_subject(subj, roi, target, nsd_to_row, shared, rng):
    """Train on the subject's own non-shared images; predict the shared ones.

    Returns per-shared-image embeddings from presentation 1 and presentation 2
    separately, so the within-subject ceiling can be measured.
    """
    b = fd.behaviour(subj)
    X, tidx, _ = fd.load_rois(subj, [roi], norm="session")
    pos = {t: i for i, t in enumerate(tidx)}
    b = b[b.global_trial.isin(pos)]
    b = b[b.nsd_id.isin(nsd_to_row)]
    rows = np.array([pos[t] for t in b.global_trial])
    y = target[[nsd_to_row[i] for i in b.nsd_id]]
    is_shared = b.nsd_id.isin(shared).values

    Xtr, Ytr = X[rows[~is_shared]], y[~is_shared]
    if len(Xtr) < 2000:
        del X
        gc.collect()
        return None

    k = min(N_PCA, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=k, random_state=SEED).fit(Xtr)   # training images only
    Ztr = pca.transform(Xtr)
    alpha = cv_alpha(Ztr, Ytr, rng)
    model = SVDRidge().fit(Ztr, Ytr)

    sh = b[is_shared].copy()
    sh["occ"] = sh.sort_values("TIME").groupby("nsd_id").cumcount()
    preds = {}
    for occ in (0, 1):
        sel = sh[sh.occ == occ].set_index("nsd_id")
        ids = [i for i in nsd_to_row if i in sel.index]
        if not ids:
            continue
        r = np.array([pos[int(sel.loc[i, "global_trial"])] for i in ids])
        preds[occ] = (ids, model.predict(pca.transform(X[r]), alpha))
    del X, Xtr, Ztr
    gc.collect()
    return preds, alpha, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", nargs="+", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    try:
        E = np.load(HERE / "data" / "clip_vitL_image_embeddings.npz")
    except FileNotFoundError:
        print("[ERROR] run step37_embed_all_stimuli.py first.")
        return 1
    emb_ids = list(E["nsd_ids"])
    target = E["embeddings"]
    nsd_to_row = {int(v): i for i, v in enumerate(emb_ids)}
    print(f"embedding space: {target.shape}")

    shared = shared_ids()
    sets = []
    for s in SUBJECTS:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        sets.append(set(n[n >= 2].index))
    test_ids = sorted(set.intersection(*sets) & set(nsd_to_row))[:766]
    print(f"test stimuli shared by all 8: {len(test_ids)}")

    rel = pd.read_csv(RESULTS / "intersubject_mapping_breakdown.csv"
                      ).groupby("roi").r_within.mean()
    avail = set(fd.available_rois(SUBJECTS[0]))
    rois = args.rois or [r for r in ROI_ORDER
                         if r in avail and rel.get(r, 0) > MIN_RELIABILITY]
    print(f"regions: {len(rois)}  ({', '.join(rois)})")

    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    sup = supercategories([n2c[i] for i in test_ids])
    cat = {i: sup[n2c[i]] for i in test_ids}
    print("categories: " + ", ".join(f"{k}={v}" for k, v in
                                     pd.Series(list(cat.values())).value_counts()
                                     .head(8).items()))

    rows = []
    for roi in rois:
        P = {}
        for s in SUBJECTS:
            out = decode_subject(s, roi, target, nsd_to_row, shared, rng)
            if out is None:
                continue
            preds, alpha, k = out
            if 0 not in preds or 1 not in preds:
                continue
            ids0, e0 = preds[0]
            ids1, e1 = preds[1]
            common = [i for i in ids0 if i in set(ids1)]
            i0 = {v: j for j, v in enumerate(ids0)}
            i1 = {v: j for j, v in enumerate(ids1)}
            P[s] = (common, e0[[i0[i] for i in common]], e1[[i1[i] for i in common]])
        if len(P) < 4:
            print(f"  [skip] {roi}: only {len(P)} subjects decoded")
            continue

        ids = sorted(set.intersection(*[set(v[0]) for v in P.values()]))
        idx = {s: {v: j for j, v in enumerate(P[s][0])} for s in P}
        mean_emb = {s: 0.5 * (P[s][1][[idx[s][i] for i in ids]]
                              + P[s][2][[idx[s][i] for i in ids]]) for s in P}
        ceil = {s: cos(P[s][1][[idx[s][i] for i in ids]],
                       P[s][2][[idx[s][i] for i in ids]]) for s in P}

        # NOISE-MATCHED transfer. The ceiling compares one presentation against one
        # other presentation, so the transfer must too. Comparing a subject's
        # 2-presentation mean against a 14-presentation group mean puts a clean
        # quantity against a clean quantity while the ceiling puts a noisy one
        # against a noisy one, and the ratio then exceeds 1 for purely arithmetic
        # reasons. Only CROSSED pairings are used (s's first vs o's second, and
        # s's second vs o's first), matching the ceiling's structure exactly and
        # avoiding any presentation-order effect.
        pres = {s: (P[s][1][[idx[s][i] for i in ids]],
                    P[s][2][[idx[s][i] for i in ids]]) for s in P}
        for s in P:
            others = [o for o in P if o != s]
            tr = np.mean([np.vstack([cos(pres[s][0], pres[o][1]),
                                     cos(pres[s][1], pres[o][0])]).mean(0)
                          for o in others], axis=0)
            c = ceil[s]
            ident_tr = float(np.mean([0.5 * (identification(pres[s][0], pres[o][1])
                                             + identification(pres[s][1], pres[o][0]))
                                      for o in others]))
            ident_ce = identification(pres[s][0], pres[s][1])
            for j, i in enumerate(ids):
                rows.append(dict(roi=roi, subject_id=s, nsd_id=i, category=cat.get(i),
                                 transfer=float(tr[j]), ceiling=float(c[j]),
                                 ident_transfer=ident_tr, ident_ceiling=ident_ce))
        g = pd.DataFrame([r for r in rows if r["roi"] == roi])
        print(f"  [OK] {roi:14s} n={len(ids)} subj={len(P)}  "
              f"transfer {g.transfer.mean():.3f}  ceiling {g.ceiling.mean():.3f}  "
              f"frac {g.transfer.mean()/max(g.ceiling.mean(),1e-9):.3f}  "
              f"ident {g.ident_transfer.mean():.3f}/{g.ident_ceiling.mean():.3f} "
              f"(chance .500)", flush=True)
        del P, mean_emb
        gc.collect()

    if not rows:
        print("[ERROR] nothing decoded.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "brain_embedding_loso.csv", index=False)
    pd.set_option("display.width", 230)

    print("\n=== Transfer by region (mean of 7 -> held-out subject) ===")
    agg = df.groupby(["roi", "subject_id"]).agg(
        transfer=("transfer", "mean"), ceiling=("ceiling", "mean")).reset_index()
    out = []
    for roi, g in agg.groupby("roi"):
        frac = (g.transfer / g.ceiling.replace(0, np.nan)).dropna()
        t, p = stats.ttest_1samp(g.transfer, 0.0)
        out.append(dict(roi=roi, transfer=g.transfer.mean(), ceiling=g.ceiling.mean(),
                        frac_of_ceiling=frac.mean(), sem=frac.std(ddof=1) / np.sqrt(len(frac)),
                        t_vs_zero=t, p_vs_zero=p))
    S = pd.DataFrame(out).sort_values("frac_of_ceiling", ascending=False)
    print(S.round(4).to_string(index=False))
    S.to_csv(RESULTS / "brain_embedding_loso_by_roi.csv", index=False)

    print("\n--- MOST SHARED ---")
    for _, r in S.head(4).iterrows():
        print(f"  {r.roi:14s} {100*r.frac_of_ceiling:.0f}% of ceiling "
              f"(transfer {r.transfer:.3f} / ceiling {r.ceiling:.3f})")
    print("--- MOST INDIVIDUAL ---")
    for _, r in S.tail(4).iloc[::-1].iterrows():
        print(f"  {r.roi:14s} {100*r.frac_of_ceiling:.0f}% of ceiling "
              f"(transfer {r.transfer:.3f} / ceiling {r.ceiling:.3f})")

    print("\n=== Transfer by image category (fraction of ceiling) ===")
    cc = df.groupby(["roi", "subject_id", "category"]).agg(
        transfer=("transfer", "mean"), ceiling=("ceiling", "mean"),
        n=("nsd_id", "count")).reset_index()
    cc = cc[cc.n >= 10]
    cc["frac"] = cc.transfer / cc.ceiling.replace(0, np.nan)
    piv = cc.pivot_table(index="roi", columns="category", values="frac")
    keep = [c for c in piv.columns if cc[cc.category == c].n.sum() >= 80]
    print(piv[keep].round(3).to_string())
    cc.to_csv(RESULTS / "brain_embedding_loso_by_category.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
