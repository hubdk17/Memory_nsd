"""
Step 7: CLIP image targets, and the text-vs-image dissociation.

This is the analysis that can break the proportional-SNR account. A uniform SNR loss
scales every decodable signal by the same factor, so the PROPORTIONAL loss must be the
same whichever target space the decoder predicts. A representational transformation
toward semantic structure does not: it should cost less in a target space that encodes
semantics (CLIP text, derived from human captions) than in one that stays closer to
image appearance (CLIP image, derived from the pixels).

7a  Encode every stimulus with the CLIP ViT-B/32 IMAGE encoder. Inputs are the NSD
    stimuli reconstructed by step7a_fetch_coco_images.py, i.e. the COCO image cropped
    with NSD's own cropBox, which is what the participant actually saw.
7b  Re-run the step2 retrieval pipeline unchanged except for the target matrix.
7c  Compare the proportional loss between the two target spaces, within subject.

Note the asymmetry that limits interpretation: CLIP text and CLIP image targets do not
have equal baseline decodability, and a proportional comparison between two differently
reliable measures is not assumption-free. The dissociation test is therefore run on the
proportional loss (delta divided by baseline-above-chance), which is the quantity the
uniform-SNR account predicts to be equal.
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
IMG_DIR = HERE / "data" / "nsd_stimuli_256"
CLIP_DIR = Path("D:/Desktop/FMRI_NSD/data/raw/clip_vit_base_patch32")
CLIP_FALLBACK = "openai/clip-vit-base-patch32"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROIS = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_SETS = {r: [r] for r in ROIS}
ROI_SETS.update({"EarlyVis": ["V1", "V2"], "SceneCtx": ["PPA", "RSC", "PHC"],
                 "MTL": ["PRC", "ERC", "SUB", "CA1", "DG_CA3"],
                 "AllCortex": ["V1", "V2", "FFA", "PPA", "RSC", "PHC"]})
COMPOSITES = {"EarlyVis", "SceneCtx", "MTL", "AllCortex"}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
SEED = 42
BATCH = 64


# --------------------------------------------------------------------------- 7a
def build_image_targets():
    out = HERE / "data" / "clip_image_targets.npz"
    if out.exists():
        d = np.load(out)
        print(f"[cached] clip_image_targets.npz  T_image={d['T_image'].shape}")
        return d["coco_ids"], d["T_image"]

    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel
    torch.set_num_threads(8)

    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    need = sorted(man.coco_id.unique())
    paths = {c: IMG_DIR / f"coco_{c:012d}.jpg" for c in need}
    missing = [c for c, p in paths.items() if not p.exists()]
    if missing:
        print(f"[WARN] {len(missing)} of {len(need)} stimulus images are missing; "
              f"run step7a_fetch_coco_images.py first.")
        if len(missing) > 0.05 * len(need):
            raise FileNotFoundError(
                f"{len(missing)} missing images -- too many to proceed. "
                f"Run step7a_fetch_coco_images.py.")
    ids = [c for c in need if paths[c].exists()]

    try:
        model = CLIPModel.from_pretrained(str(CLIP_DIR)).eval()
        proc = CLIPImageProcessor.from_pretrained(str(CLIP_DIR))
        print(f"  loaded local CLIP from {CLIP_DIR.name}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  local CLIP failed ({e!r}); falling back to {CLIP_FALLBACK}")
        model = CLIPModel.from_pretrained(CLIP_FALLBACK).eval()
        proc = CLIPImageProcessor.from_pretrained(CLIP_FALLBACK)

    print(f"  encoding {len(ids)} images on CPU...")
    vecs = []
    with torch.no_grad():
        for i in range(0, len(ids), BATCH):
            chunk = [Image.open(paths[c]).convert("RGB") for c in ids[i:i + BATCH]]
            inp = proc(images=chunk, return_tensors="pt")
            v = model.get_image_features(**inp)
            vecs.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
            if (i // BATCH) % 20 == 0:
                print(f"    {i}/{len(ids)}", flush=True)
    T = np.concatenate(vecs)
    np.savez_compressed(out, coco_ids=np.array(ids), T_image=T)
    print(f"[OK] data/clip_image_targets.npz  T_image={T.shape}")
    return np.array(ids), T


# --------------------------------------------------------------------------- 7b
def zs(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def retrieval_metrics(P, pool, ti):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ pool.T
    K = S.shape[1]
    ts = S[np.arange(len(ti)), ti]
    rank = (S > ts[:, None]).sum(1) + 1
    others = S.copy()
    others[np.arange(len(ti)), ti] = -np.inf
    return dict(top1=float((rank == 1).mean()), top5=float((rank <= 5).mean()),
                pct_rank=float(((K - rank) / (K - 1)).mean()),
                two_way=float((ts[:, None] > others).sum() / (len(ti) * (K - 1))),
                mean_cos_true=float(ts.mean()), n=len(ti), pool=K)


def cv_alpha(X, Y):
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(X[va], a)
            acc[j] += float(((p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8))
                             * Y[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))]


def run_retrieval(coco_ids, T_image):
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    c2r = {int(c): i for i, c in enumerate(coco_ids)}
    rows = []
    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        keep = m.coco_id.map(lambda c: c in c2r).values
        if not keep.all():
            print(f"  [{subj}] dropping {int((~keep).sum())} trials with no image target")
        Y = np.zeros((len(m), T_image.shape[1]), np.float32)
        Y[keep] = T_image[[c2r[c] for c in m.coco_id[keep]]]

        train = ((m.n_presentations == 1).values) & keep
        rep = m.repeat_number.values
        npres = m.n_presentations.values
        test3 = (npres == 3) & keep
        test2 = (npres >= 2) & keep

        for roi, parts in ROI_SETS.items():
            Xraw = np.concatenate([z[p] for p in parts], 1).astype(np.float64)
            for norm in ("session", "repgroup"):
                grp = m.session.values if norm == "session" else rep
                X = zs(Xraw, grp)
                alpha = cv_alpha(X[train], Y[train])
                mdl = SVDRidge().fit(X[train], Y[train])

                for label, mask, reps in (("lifecycle93", test3, (1, 2, 3)),
                                          ("repeated323", test2, (1, 2))):
                    imgs = m.loc[mask & (rep == 1), "nsd_id"].values
                    pool = T_image[[c2r[m.loc[m.nsd_id == i, "coco_id"].iloc[0]]
                                    for i in imgs]]
                    idx_of = {int(im): k for k, im in enumerate(imgs)}
                    for r in reps:
                        s = mask & (rep == r)
                        ti = np.array([idx_of[int(i)] for i in m.loc[s, "nsd_id"]])
                        met = retrieval_metrics(mdl.predict(X[s], alpha), pool, ti)
                        rows.append(dict(subject_id=subj, roi=roi, norm=norm,
                                         alpha=alpha, analysis=label, repetition=r,
                                         **met))
        print(f"  [OK] {subj}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ridge_clip_image_retrieval_by_repetition.csv", index=False)
    print(f"[OK] results/ridge_clip_image_retrieval_by_repetition.csv ({len(out)} rows)")
    return out


# --------------------------------------------------------------------------- 7c
def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def dissociation(text_df, img_df):
    def wide(df, tag):
        w = df.pivot_table(index=["subject_id", "roi", "norm", "analysis"],
                           columns="repetition", values="two_way").reset_index()
        w = w.rename(columns={1: f"{tag}_rep1", 2: f"{tag}_rep2", 3: f"{tag}_rep3"})
        return w

    t = wide(text_df, "text")
    i = wide(img_df, "image")
    j = t.merge(i, on=["subject_id", "roi", "norm", "analysis"])
    j["delta_text"] = j.text_rep2 - j.text_rep1
    j["delta_image"] = j.image_rep2 - j.image_rep1
    j["base_text"] = j.text_rep1 - 0.5
    j["base_image"] = j.image_rep1 - 0.5
    j["prop_loss_text"] = j.delta_text / j.base_text
    j["prop_loss_image"] = j.delta_image / j.base_image
    j["dissociation"] = j.delta_text - j.delta_image
    j["prop_dissociation"] = j.prop_loss_text - j.prop_loss_image

    cols = ["subject_id", "roi", "norm", "analysis", "text_rep1", "image_rep1",
            "delta_text", "delta_image", "base_text", "base_image", "prop_loss_text",
            "prop_loss_image", "dissociation", "prop_dissociation"]
    j[cols].to_csv(RESULTS / "text_vs_image_dissociation.csv", index=False)

    # A proportional loss is only interpretable where the denominator is meaningfully
    # above zero. Regions decoding at chance (hippocampal subfields, baseline ~0.005)
    # produce huge unstable ratios, so they are flagged and kept out of the FDR family
    # rather than being allowed to dominate it.
    MIN_BASE = 0.02          # Rep1 two-way >= 0.52 in BOTH target spaces
    rows = []
    for (roi, norm, an), g in j.groupby(["roi", "norm", "analysis"]):
        base_ok = bool(g.base_text.mean() >= MIN_BASE and g.base_image.mean() >= MIN_BASE)
        ok = g[(g.base_text > 0.002) & (g.base_image > 0.002)]
        if len(ok) < 4:
            rows.append(dict(roi=roi, norm=norm, analysis=an, n_subj=len(ok),
                             baseline_ok=base_ok,
                             mean_base_text=g.base_text.mean(),
                             mean_base_image=g.base_image.mean(),
                             mean_prop_text=np.nan, mean_prop_image=np.nan,
                             mean_dissociation=np.nan, mean_prop_dissociation=np.nan,
                             t_stat=np.nan, p_val=np.nan, cohens_d=np.nan,
                             n_subj_positive=0, excluded_subj=len(g) - len(ok)))
            continue
        d = ok.prop_dissociation.values
        t, p = stats.ttest_1samp(d, 0.0)
        rows.append(dict(roi=roi, norm=norm, analysis=an, n_subj=len(ok),
                         baseline_ok=base_ok,
                         mean_base_text=g.base_text.mean(),
                         mean_base_image=g.base_image.mean(),
                         mean_prop_text=ok.prop_loss_text.mean(),
                         mean_prop_image=ok.prop_loss_image.mean(),
                         mean_dissociation=ok.dissociation.mean(),
                         mean_prop_dissociation=d.mean(),
                         t_stat=t, p_val=p,
                         cohens_d=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0,
                         n_subj_positive=int((d > 0).sum()),
                         excluded_subj=len(g) - len(ok)))
    st = pd.DataFrame(rows)
    st["is_composite"] = st.roi.isin(COMPOSITES)
    st["p_fdr"] = np.nan
    for _, idx in st.groupby(["norm", "analysis"]).groups.items():
        sub = st.loc[idx]
        valid = sub.index[sub.p_val.notna() & sub.baseline_ok]
        if len(valid):
            st.loc[valid, "p_fdr"] = bh_fdr(st.loc[valid, "p_val"].values)
    st.to_csv(RESULTS / "text_vs_image_dissociation_stats.csv", index=False)

    pd.set_option("display.width", 250)
    for (norm, an), g in st.groupby(["norm", "analysis"]):
        print(f"\n=== text-vs-image dissociation | norm={norm} | {an} ===")
        print("  positive prop_dissociation = CLIP-text degrades proportionally LESS "
              "than CLIP-image")
        print(g.sort_values(["baseline_ok", "p_val"], ascending=[False, True])[
            ["roi", "baseline_ok", "mean_base_text", "mean_base_image",
             "mean_prop_text", "mean_prop_image", "mean_prop_dissociation", "t_stat",
             "p_val", "p_fdr", "cohens_d", "n_subj_positive"]].round(4).to_string(index=False))
    return j, st


def main():
    try:
        coco_ids, T_image = build_image_targets()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    img_df = run_retrieval(coco_ids, T_image)
    try:
        text_df = pd.read_csv(RESULTS / "ridge_clip_text_retrieval_by_repetition.csv")
    except FileNotFoundError:
        print("[ERROR] text retrieval results missing; run step2 first.")
        return 1

    print("\n--- CLIP image baseline (Rep1, repeated323, repgroup) ---")
    b = img_df[(img_df.analysis == "repeated323") & (img_df.norm == "repgroup")
               & (img_df.repetition == 1)]
    print(b.groupby("roi")[["two_way", "top1", "top5"]].mean().round(4)
          .sort_values("two_way", ascending=False).to_string())

    dissociation(text_df, img_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
