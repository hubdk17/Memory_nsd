"""
Step 35: what does the shared geometry encode? Brain-DNN alignment, leave-one-subject-out.

Step 33 established that scene-selective and ventral-stream cortex share nearly all of
their reliable representational geometry across brains (S_ventral 0.97, PPA 0.91) while
early and mid visual cortex keep a third of theirs private (hV4 0.61, V1 0.73). It
cannot say what either component represents.

This step decomposes every subject's RDM into the part the other seven predict and the
part they do not, then asks a DNN to explain each separately:

    shared_s    = fitted value of subject s's RDM regressed on the mean of the other 7
    residual_s  = what is left -- the individual component

  A  DNN vs the subject's own RDM          (conventional brain-DNN RSA)
  B  DNN vs the SHARED component           (does the model capture what brains agree on?)
  C  DNN vs the INDIVIDUAL residual        (is the private part stimulus-driven at all?)

C is the informative one. If a DNN explains the residual, the "individual" geometry is
still driven by image properties and merely weighted differently per person. If nothing
explains it, that component is genuinely idiosyncratic -- retinotopic layout, say --
and no image-computable model will ever reach it.

Two networks dissociate visual from semantic:
  CLIP ViT-L/14    language-supervised; deep layers carry caption semantics
  DINOv2 ViT-B/14  self-supervised on images alone, no language in training

Everything is reported as a FRACTION OF THE NOISE CEILING. The ceiling is each region's
own within-subject RDM reliability, already measured in step33 (V1 0.199, PPA 0.154,
hV4 0.132, CA1 0.0007). A raw correlation of 0.08 means one thing against a ceiling of
0.13 and something else entirely against 0.23, and regions whose ceiling is at floor are
excluded rather than reported as model failures.
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
from step33_intersubject_mapping_breakdown import (MIN_RELIABILITY, ROI_ORDER,
                                                   shared_ids, subject_rdms)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
N_PERM = 500
SEED = 42


def norm(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else x


def decompose(own: np.ndarray, group: np.ndarray):
    """Split a subject's RDM into the part the group predicts and the remainder."""
    o, g = norm(own), norm(group)
    beta = float(np.dot(o, g) / np.dot(g, g))
    fitted = beta * g
    return fitted, o - fitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", nargs="+", default=None)
    ap.add_argument("--max-images", type=int, default=766)
    args = ap.parse_args()

    try:
        D = np.load(HERE / "data" / "dnn_layer_rdms_shared.npz")
    except FileNotFoundError:
        print("[ERROR] run step34_dnn_layer_features.py first.")
        return 1
    layers = [k for k in D.files if k != "nsd_ids"]
    dnn_ids = list(D["nsd_ids"])
    print(f"DNN layers: {len(layers)}  ({', '.join(sorted(layers)[:4])} ...)")

    shared = shared_ids()
    sets = []
    for s in SUBJECTS:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        sets.append(set(n[n >= 2].index))
    images = sorted(set.intersection(*sets))[: args.max_images]
    # Both sides must be in the SAME stimulus order or every number below is noise.
    assert list(dnn_ids) == images, (
        f"stimulus order mismatch: DNN has {len(dnn_ids)}, brain has {len(images)}")
    print(f"stimuli: {len(images)} (order verified identical to step34)")

    try:
        prev = pd.read_csv(RESULTS / "intersubject_mapping_breakdown.csv")
        r1 = prev.groupby("roi").r_within.mean()
        # Spearman-Brown: r_own is measured on the mean of two presentations, which is
        # more reliable than either alone. Dividing by the single-presentation value
        # inflates the ratio past 1.0, which is impossible for a noise ceiling.
        ceiling = 2 * r1 / (1 + r1)
        ceiling_1pres = r1
    except FileNotFoundError:
        print("[ERROR] run step33 first; its r_within is the noise ceiling.")
        return 1

    avail = set(fd.available_rois(SUBJECTS[0]))
    rois = args.rois or [r for r in ROI_ORDER
                         if r in avail and ceiling.get(r, 0) > MIN_RELIABILITY]
    skipped = [r for r in ROI_ORDER if r in avail and r not in rois]
    print(f"regions with a usable ceiling: {len(rois)}")
    print(f"excluded at floor: {', '.join(skipped) if skipped else 'none'}")

    rng = np.random.default_rng(SEED)
    rows = []
    for roi in rois:
        R = {}
        for s in SUBJECTS:
            out = subject_rdms(s, images, roi)
            if out is not None:
                R[s] = out[2]                     # mean of the two presentations
        if len(R) < 4:
            continue
        cei = float(ceiling[roi])

        for s in R:
            others = [R[o] for o in R if o != s]
            group = np.mean(others, axis=0)
            fitted, resid = decompose(R[s], group)
            own = norm(R[s])
            g = norm(group)
            for lay in layers:
                d = norm(D[lay])
                # PARTIAL correlation for the residual. Removing the group mean from
                # the brain RDM alone is not enough: the DNN RDM still contains the
                # group-shared component, so a plain correlation with the residual
                # picks up shared variance that the noisy group mean failed to
                # absorb. Both sides must be orthogonalised against the group.
                d_resid = d - float(np.dot(d, g) / np.dot(g, g)) * g
                rows.append(dict(
                    roi=roi, subject_id=s, layer=lay, ceiling=cei,
                    r_own=float(np.corrcoef(own, d)[0, 1]),
                    r_shared=float(np.corrcoef(fitted, d)[0, 1]),
                    r_residual_naive=float(np.corrcoef(resid, d)[0, 1]),
                    r_residual=float(np.corrcoef(resid, d_resid)[0, 1])))

        # validation: shuffling stimulus order must destroy the alignment
        lay0 = sorted(layers)[0]
        sq = squareform(D[lay0])
        nulls = []
        for _ in range(20):
            p = rng.permutation(len(images))
            sh = squareform(sq[np.ix_(p, p)], checks=False)
            nulls.append(np.corrcoef(norm(R[sorted(R)[0]]), norm(sh))[0, 1])
        best = max(np.mean([x["r_own"] for x in rows
                            if x["roi"] == roi and x["layer"] == l]) for l in layers)
        print(f"  [OK] {roi:14s} ceiling {cei:.3f}  best layer r {best:.3f} "
              f"({100*best/cei:.0f}% of ceiling)  perm-null {np.mean(nulls):+.4f}",
              flush=True)
        del R
        gc.collect()

    if not rows:
        print("[ERROR] nothing computed.")
        return 1
    df = pd.DataFrame(rows)
    df["frac_ceiling_own"] = df.r_own / df.ceiling
    df["model"] = df.layer.str.split("_").str[0]
    df["depth"] = df.layer.str.extract(r"L(\d+)").astype(int)
    df.to_csv(RESULTS / "brain_dnn_loso.csv", index=False)
    pd.set_option("display.width", 230)

    print("\n=== A. Best-fitting layer per region (fraction of noise ceiling) ===")
    out = []
    for (roi, model), g in df.groupby(["roi", "model"]):
        m = g.groupby("depth").agg(r_own=("r_own", "mean"),
                                   r_shared=("r_shared", "mean"),
                                   r_resid=("r_residual", "mean"))
        best = m.r_own.idxmax()
        out.append(dict(roi=roi, model=model, best_depth=best,
                        r_own=m.loc[best, "r_own"],
                        frac_ceiling=m.loc[best, "r_own"] / g.ceiling.iloc[0],
                        r_shared=m.loc[best, "r_shared"],
                        r_residual=m.loc[best, "r_resid"]))
    A = pd.DataFrame(out)
    A["order"] = A.roi.map({r: i for i, r in enumerate(ROI_ORDER)})
    A = A.sort_values(["model", "order"]).drop(columns="order")
    print(A.round(4).to_string(index=False))
    A.to_csv(RESULTS / "brain_dnn_best_layer.csv", index=False)

    print("\n=== C. Is the INDIVIDUAL residual explained by any DNN layer? ===")
    res = []
    for roi, g in df.groupby("roi"):
        best_lay = g.groupby("layer").r_residual.mean().abs().idxmax()
        gg = g[g.layer == best_lay]
        t, p = stats.ttest_1samp(gg.r_residual, 0.0)
        res.append(dict(roi=roi, best_layer=best_lay,
                        r_residual=gg.r_residual.mean(),
                        r_residual_naive=gg.r_residual_naive.mean(),
                        r_shared=gg.r_shared.mean(), t=t, p=p))
    C = pd.DataFrame(res)
    C["order"] = C.roi.map({r: i for i, r in enumerate(ROI_ORDER)})
    C = C.sort_values("order").drop(columns="order")
    print(C.round(4).to_string(index=False))
    C.to_csv(RESULTS / "brain_dnn_residual.csv", index=False)
    unstable = A[A.frac_ceiling > 1.0].roi.unique()
    if len(unstable):
        print(f"\n  [warn] frac_ceiling > 1 in: {', '.join(unstable)} -- these have the")
        print("  lowest ceilings, where the ratio is numerically unstable; treat as")
        print("  'at ceiling' rather than as a point estimate.")
    n_sig = int((C.p < 0.05).sum())
    print(f"\n  regions whose individual component any DNN explains: {n_sig} of {len(C)}")
    print("  If ~0: the private geometry is not image-computable -- consistent with")
    print("  idiosyncratic retinotopy rather than a differently-weighted visual code.")

    print("\n=== CLIP vs DINOv2 at the deepest layer (semantic dissociation) ===")
    deep = df[df.depth == df.groupby("model").depth.transform("max")]
    cmp = deep.pivot_table(index="roi", columns="model", values="r_own")
    if {"clip", "dino"} <= set(cmp.columns):
        cmp["clip_minus_dino"] = cmp["clip"] - cmp["dino"]
        cmp["order"] = cmp.index.map({r: i for i, r in enumerate(ROI_ORDER)})
        print(cmp.sort_values("order").drop(columns="order").round(4).to_string())
        print("\n  positive = deep language-supervised features fit better than")
        print("  deep self-supervised ones, i.e. the geometry is semantic rather")
        print("  than merely visual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
