"""Step 42: three controls that decide whether the hierarchy effect is real.

THE PROBLEM THIS SOLVES
=======================
step38 measured, for each region, how much of a subject's decodable representation
transfers to the other seven, expressed as a fraction of a within-subject ceiling.
The ceiling was

    cos( F_s(presentation 1) , F_s(presentation 2) )          [step38]

where F_s is subject s's ONE decoder. Both sides of that comparison run through the
SAME ridge model, the SAME PCA basis and the SAME training images. Any error that
model makes which depends on the image -- from its particular training set, its
particular basis, its particular alpha -- is therefore IDENTICAL on both sides and
adds to the ceiling. Transfer compares two DIFFERENT subjects' decoders, so that
shared error cancels instead.

The ceiling is thus inflated by exactly the amount of decoder idiosyncrasy, and the
reported fraction is deflated by it. Worse, the inflation should be largest where the
voxels-to-CLIP mapping is weakest -- early visual cortex, against a semantic target --
which is precisely the pattern the paper reports. V1 below ventral is predicted by
this artefact alone, with no individuality of brains required.

CONTROL 1: a decoder-independent ceiling
----------------------------------------
Split each subject's training IMAGES into two disjoint halves and fit two decoders,
A_s and B_s. Then

    ceiling  = mean[ cos(A_s(p1), B_s(p2)) , cos(B_s(p1), A_s(p2)) ]
    transfer = mean over o of
               mean[ cos(A_s(p1), B_o(p2)) , cos(B_s(p1), A_o(p2)) ]

Now BOTH sides use two independent decoders, each trained on half the images, each
compared across one presentation pair. Numerator and denominator are matched in
training-set size, in decoder independence and in presentation count. The ONLY
remaining difference is same brain versus different brain -- which is the question.

The split is by image, not by trial. NSD shows each non-shared image about three
times; splitting by trial would put the same image in both halves and leave A and B
sharing image-specific information, under-correcting the very thing being corrected.

CONTROL 2: a low-level target
-----------------------------
The same pipeline with GIST (step41) instead of CLIP. A semantic target can make
high-level regions look shared by construction. If the gradient only exists for
CLIP, the finding is about CLIP and not about cortex.

CONTROL 3: matched voxel count
------------------------------
Regions differ in size by a factor of 30 (hV4 about 500 voxels, nsdgeneral about
14,500). Every region is subsampled to the smallest region's voxel count for that
subject and the split-half analysis repeated, so region size cannot drive the ranking.

All three controls are reported whether or not they support the paper's claim.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import _full_data as fd
from _fast_ridge import SVDRidge
from step33_intersubject_mapping_breakdown import MIN_RELIABILITY, ROI_ORDER, shared_ids

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
N_PCA = 1000
SEED = 42
N_VOX_REPEATS = 3


# ----------------------------------------------------------------- metrics
def _unit(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def cos(A, B):
    """Per-image cosine after removing the across-image mean from each set.

    Identical to step38. Ridge shrinks every prediction toward the training
    centroid; without centring, two sets of predictions for completely different
    images already sit at cosine about 0.95 and carry no stimulus-specific
    information.
    """
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    return (_unit(A) * _unit(B)).sum(1)


def identification(P, T):
    """Two-way identification accuracy; chance 0.5. Identical to step38."""
    Pn, Tn = _unit(P - P.mean(0, keepdims=True)), _unit(T - T.mean(0, keepdims=True))
    S = Pn @ Tn.T
    true = np.diag(S).copy()
    np.fill_diagonal(S, -np.inf)
    n = len(P)
    return float((true[:, None] > S).sum() / (n * (n - 1)))


def cv_alpha(X, Y, folds=5):
    """Select alpha by identification accuracy, as step38 did."""
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            acc[j] += identification(m.predict(X[va], a), Y[va])
    return ALPHAS[int(np.argmax(acc))]


# ----------------------------------------------------------------- decoding
def _basis(Xtr, Xte_list):
    """PCA fitted on the training rows ONLY, applied to train and test.

    Fitting on the training images alone matters: fitting on all images would leak
    test structure into the basis. The basis depends only on X, so it is computed
    once per row-split and reused for every target -- the PCA is by far the most
    expensive step, and CLIP and GIST share it.
    """
    k = min(N_PCA, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
    out = (pca.transform(Xtr), [pca.transform(Xte) for Xte in Xte_list])
    del pca
    return out


def _ridge(Ztr, Ytr, Zte_list, alpha=None):
    a = cv_alpha(Ztr, Ytr) if alpha is None else alpha
    model = SVDRidge().fit(Ztr, Ytr)
    out = [model.predict(Z, a) for Z in Zte_list]
    del model
    return out, a


def decode_subject(subj, roi, targets, shared, test_ids, rng):
    """Every decoder variant for one subject and one region.

    Returns (ids, predictions, alphas, n_voxels) where predictions is keyed by
    (target, variant, decoder, presentation) and aligned to ids.
    """
    b = fd.behaviour(subj)
    X, tidx, _ = fd.load_rois(subj, [roi], norm="session")
    pos = {t: i for i, t in enumerate(tidx)}
    b = b[b.global_trial.isin(pos)]
    nsd_to_row = targets["clip"][1]
    b = b[b.nsd_id.isin(nsd_to_row)]
    rows = np.array([pos[t] for t in b.global_trial])
    is_shared = b.nsd_id.isin(shared).values

    tr_rows = rows[~is_shared]
    tr_ids = b.nsd_id.values[~is_shared]
    if len(tr_rows) < 2000:
        del X
        gc.collect()
        return None

    # test blocks: presentation 1 and 2 of each shared test image
    sh = b[is_shared].copy()
    sh["occ"] = sh.sort_values("TIME").groupby("nsd_id").cumcount()
    have0 = set(sh[sh.occ == 0].nsd_id)
    have1 = set(sh[sh.occ == 1].nsd_id)
    ids = [i for i in test_ids if i in have0 and i in have1]
    if not ids:
        del X
        gc.collect()
        return None
    te_rows = []
    for occ in (0, 1):
        sel = sh[sh.occ == occ].set_index("nsd_id")
        te_rows.append(np.array([pos[int(sel.loc[i, "global_trial"])] for i in ids]))

    # split TRAINING IMAGES (not trials) into two disjoint halves
    uniq = np.unique(tr_ids)
    perm = rng.permutation(len(uniq))
    halfA = set(uniq[perm[:len(uniq) // 2]])
    mA = np.array([i in halfA for i in tr_ids])
    n_vox = X.shape[1]

    Xte = [X[r] for r in te_rows]
    P = {}
    alphas = {}
    # three row-splits of the training set, each with its own independent basis.
    # A and B are fitted on DISJOINT image sets, which is what makes the two
    # decoders independent; sharing a basis between targets does not touch that.
    bases = {"full": _basis(X[tr_rows], Xte),
             "A": _basis(X[tr_rows[mA]], Xte),
             "B": _basis(X[tr_rows[~mA]], Xte)}
    for tname in ("clip", "gist"):
        T, n2r = targets[tname]
        ytr = T[[n2r[i] for i in b.nsd_id]][~is_shared]
        # ---- full decoder (reproduces step38) ----
        Ztr, Zte = bases["full"]
        (f0, f1), a_full = _ridge(Ztr, ytr, Zte)
        P[(tname, "full", "F", 0)], P[(tname, "full", "F", 1)] = f0, f1
        alphas[(tname, "full")] = a_full
        # ---- split-half decoders; alpha selected once on half A, reused on B,
        #      so A and B differ only in which images trained them ----
        ZtrA, ZteA = bases["A"]
        (a0, a1), a_half = _ridge(ZtrA, ytr[mA], ZteA)
        ZtrB, ZteB = bases["B"]
        (b0, b1), _ = _ridge(ZtrB, ytr[~mA], ZteB, alpha=a_half)
        P[(tname, "half", "A", 0)], P[(tname, "half", "A", 1)] = a0, a1
        P[(tname, "half", "B", 0)], P[(tname, "half", "B", 1)] = b0, b1
        alphas[(tname, "half")] = a_half
        del ytr
        gc.collect()
    del bases
    gc.collect()

    # ---- control 3: matched voxel count, split-half, CLIP only ----
    n_target = targets["_vox_target"]
    T, n2r = targets["clip"]
    ytr = T[[n2r[i] for i in b.nsd_id]][~is_shared]
    if n_vox > n_target:
        a_vox = None
        for r in range(N_VOX_REPEATS):
            vsel = rng.choice(n_vox, n_target, replace=False)
            Xs = np.ascontiguousarray(X[:, vsel])
            Xte_s = [Xs[q] for q in te_rows]
            ZA, ZteA = _basis(Xs[tr_rows[mA]], Xte_s)
            (a0, a1), a_vox = _ridge(ZA, ytr[mA], ZteA, alpha=a_vox)
            ZB, ZteB = _basis(Xs[tr_rows[~mA]], Xte_s)
            (b0, b1), _ = _ridge(ZB, ytr[~mA], ZteB, alpha=a_vox)
            del ZA, ZteA, ZB, ZteB
            P[("clip", f"vox{r}", "A", 0)], P[("clip", f"vox{r}", "A", 1)] = a0, a1
            P[("clip", f"vox{r}", "B", 0)], P[("clip", f"vox{r}", "B", 1)] = b0, b1
            del Xs, Xte_s
            gc.collect()
    else:
        # region already at or below the matched size: its split-half fit already
        # uses every voxel it has, so reuse it rather than resampling nothing.
        for r in range(N_VOX_REPEATS):
            for d in ("A", "B"):
                for o in (0, 1):
                    P[("clip", f"vox{r}", d, o)] = P[("clip", "half", d, o)]

    del X, Xte, ytr
    gc.collect()
    return ids, P, alphas, n_vox


# ----------------------------------------------------------------- per-ROI stats
def roi_rows(roi, store, ids_common):
    """Per-subject scores for every variant, on the images all subjects share."""
    out = []
    subs = sorted(store)
    idx = {s: {v: j for j, v in enumerate(store[s][0])} for s in subs}

    def take(s, key):
        return store[s][1][key][[idx[s][i] for i in ids_common]]

    def crossed(s_key_a, s_key_b, s, o):
        """Crossed presentation pairing, matching the step38 ceiling structure."""
        return np.vstack([cos(take(s, s_key_a), take(o, s_key_b)),
                          cos(take(s, s_key_b), take(o, s_key_a))]).mean(0)

    for s in subs:
        others = [o for o in subs if o != s]
        rec = dict(roi=roi, subject_id=s, n_images=len(ids_common),
                   n_voxels=store[s][3])

        # ---- naive (step38): one decoder drives both sides of the ceiling ----
        for tname, pre in (("clip", "naive"), ("gist", "gist_naive")):
            k0, k1 = (tname, "full", "F", 0), (tname, "full", "F", 1)
            ce = cos(take(s, k0), take(s, k1))
            tr = np.mean([crossed(k0, k1, s, o) for o in others], axis=0)
            rec.update(**{f"{pre}_ceiling": ce.mean(), f"{pre}_transfer": tr.mean(),
                          f"{pre}_frac": tr.mean() / ce.mean()})

        # ---- corrected: independent half decoders on BOTH sides ----
        for tname in ("clip", "gist"):
            kA0, kA1 = (tname, "half", "A", 0), (tname, "half", "A", 1)
            kB0, kB1 = (tname, "half", "B", 0), (tname, "half", "B", 1)
            ce = np.vstack([cos(take(s, kA0), take(s, kB1)),
                            cos(take(s, kB0), take(s, kA1))]).mean(0)
            tr = np.mean([np.vstack([cos(take(s, kA0), take(o, kB1)),
                                     cos(take(s, kB0), take(o, kA1))]).mean(0)
                          for o in others], axis=0)
            ice = 0.5 * (identification(take(s, kA0), take(s, kB1))
                         + identification(take(s, kB0), take(s, kA1)))
            itr = float(np.mean([0.5 * (identification(take(s, kA0), take(o, kB1))
                                        + identification(take(s, kB0), take(o, kA1)))
                                 for o in others]))
            rec.update(**{f"{tname}_ceiling": ce.mean(), f"{tname}_transfer": tr.mean(),
                          f"{tname}_frac": tr.mean() / ce.mean(),
                          f"{tname}_ident_ceiling": ice,
                          f"{tname}_ident_transfer": itr})

        # ---- voxel-matched, averaged over repeats ----
        fr = []
        for r in range(N_VOX_REPEATS):
            kA0, kA1 = ("clip", f"vox{r}", "A", 0), ("clip", f"vox{r}", "A", 1)
            kB0, kB1 = ("clip", f"vox{r}", "B", 0), ("clip", f"vox{r}", "B", 1)
            ce = np.vstack([cos(take(s, kA0), take(s, kB1)),
                            cos(take(s, kB0), take(s, kA1))]).mean(0)
            tr = np.mean([np.vstack([cos(take(s, kA0), take(o, kB1)),
                                     cos(take(s, kB0), take(o, kA1))]).mean(0)
                          for o in others], axis=0)
            fr.append(tr.mean() / ce.mean())
        rec.update(voxmatched_frac=float(np.mean(fr)),
                   voxmatched_sd_over_repeats=float(np.std(fr)))
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", nargs="+", default=None)
    args = ap.parse_args()

    E = np.load(HERE / "data" / "clip_vitL_image_embeddings.npz")
    G = np.load(HERE / "data" / "gist_features.npz")
    clip_t = (E["embeddings"], {int(v): i for i, v in enumerate(E["nsd_ids"])})
    gist_t = (G["features"], {int(v): i for i, v in enumerate(G["nsd_ids"])})
    print(f"CLIP target {clip_t[0].shape} | GIST target {gist_t[0].shape}")

    shared = shared_ids()
    sets = []
    for s in SUBJECTS:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        sets.append(set(n[n >= 2].index))
    test_ids = sorted(set.intersection(*sets) & set(clip_t[1]) & set(gist_t[1]))[:766]
    print(f"test stimuli shared by all 8: {len(test_ids)}")

    rel = pd.read_csv(RESULTS / "intersubject_mapping_breakdown.csv"
                      ).groupby("roi").r_within.mean()
    avail = set(fd.available_rois(SUBJECTS[0]))
    rois = args.rois or [r for r in ROI_ORDER
                         if r in avail and rel.get(r, 0) > MIN_RELIABILITY]
    print(f"regions: {len(rois)}  ({', '.join(rois)})")

    # matched voxel count = smallest of these regions, per subject
    vox_target = {}
    for s in SUBJECTS:
        ss = fd.available_sessions(s)[0]
        with np.load(fd.STORE / s / f"session{ss:02d}.npz") as z:
            vox_target[s] = min(z[r].shape[1] for r in rois if r in z.files)
    print("matched voxel count per subject: "
          + ", ".join(f"{s}={v}" for s, v in vox_target.items()))

    # The decisive comparison is V1 against the ventral stream, so those regions run
    # first and the answer is on disk early; the rest follow cheapest-first.
    size = {}
    for r in rois:
        size[r] = np.mean([pd.read_csv(RESULTS / "roi_voxel_counts.csv")[r].mean()]) \
            if (RESULTS / "roi_voxel_counts.csv").exists() else 0
    first = [r for r in ("V1", "V2", "hV4", "S_midventral", "S_ventral") if r in rois]
    rois = first + sorted([r for r in rois if r not in first], key=lambda r: size[r])
    print(f"run order: {', '.join(rois)}")

    out_rows = []
    part = RESULTS / "decoder_controls_by_subject.csv"
    done = set()
    if part.exists():
        prev = pd.read_csv(part)
        out_rows = prev.to_dict("records")
        done = set(prev.roi.unique())
        print(f"resuming: {len(done)} region(s) already done ({', '.join(sorted(done))})")
    for roi in rois:
        if roi in done:
            continue
        store = {}
        for s in SUBJECTS:
            rng = np.random.default_rng(
                int.from_bytes(f"{roi}:{s}".encode(), "little") % (2 ** 32))
            targets = {"clip": clip_t, "gist": gist_t, "_vox_target": vox_target[s]}
            r = decode_subject(s, roi, targets, shared, test_ids, rng)
            if r is not None:
                store[s] = r
            print(f"    {roi:14s} {s} {'ok' if r else 'SKIP'}", flush=True)
        if len(store) < 4:
            print(f"  [skip] {roi}: only {len(store)} subjects decoded")
            continue
        ids_common = sorted(set.intersection(*[set(v[0]) for v in store.values()]))
        rows = roi_rows(roi, store, ids_common)
        out_rows.extend(rows)
        d = pd.DataFrame(rows)
        print(f"  [OK] {roi:14s} n={len(ids_common)} subj={len(store)}  "
              f"naive {d.naive_frac.mean():.3f}  corrected {d.clip_frac.mean():.3f}  "
              f"gist {d.gist_frac.mean():.3f}  voxmatched {d.voxmatched_frac.mean():.3f}",
              flush=True)
        pd.DataFrame(out_rows).to_csv(part, index=False)   # partial, so a crash keeps work
        del store
        gc.collect()

    df = pd.DataFrame(out_rows)
    df.to_csv(part, index=False)

    # ------------------------------------------------------------- summary
    agg = []
    for roi, g in df.groupby("roi"):
        rec = dict(roi=roi, n_subj=len(g), n_voxels=g.n_voxels.mean())
        for col in ("naive_frac", "clip_frac", "gist_naive_frac", "gist_frac",
                    "voxmatched_frac"):
            rec[col] = g[col].mean()
            rec[col + "_sem"] = g[col].std(ddof=1) / np.sqrt(len(g))
        for col in ("clip_ident_ceiling", "clip_ident_transfer",
                    "gist_ident_ceiling", "gist_ident_transfer"):
            rec[col] = g[col].mean()
        agg.append(rec)
    S = pd.DataFrame(agg).sort_values("clip_frac", ascending=False)
    S.to_csv(RESULTS / "decoder_controls_by_roi.csv", index=False)
    print("\n=== fraction of ceiling, by control ===")
    print(S[["roi", "n_voxels", "naive_frac", "clip_frac", "gist_naive_frac",
             "gist_frac", "voxmatched_frac"]].round(3).to_string(index=False))

    # paired V1 vs ventral under every control
    print("\n=== V1 vs ventral stream, paired across subjects ===")
    w = df.pivot(index="subject_id", columns="roi")
    for col in ("naive_frac", "clip_frac", "gist_naive_frac", "gist_frac",
                "voxmatched_frac"):
        if ("V1" in w[col].columns) and ("S_ventral" in w[col].columns):
            a, b2 = w[col]["V1"], w[col]["S_ventral"]
            t, p = stats.ttest_rel(b2, a)
            dz = (b2 - a).mean() / (b2 - a).std(ddof=1)
            print(f"  {col:18s} V1 {a.mean():.3f}  ventral {b2.mean():.3f}  "
                  f"diff {(b2 - a).mean():+.3f}  t(7)={t:+.2f}  p={p:.2e}  "
                  f"dz={dz:+.2f}  {(b2 > a).sum()}/{len(a)} subj")
    print(f"\nwrote {part.name} and decoder_controls_by_roi.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
