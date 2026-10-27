"""
Step 36 (GPU): visual versus semantic, tested WITHIN one model.

Step 35 compared CLIP ViT-L/14 against DINOv2 ViT-B/14 and found CLIP fitting better
everywhere, with the gap widening up the hierarchy. That looked like a semantic effect
but it is not interpretable as one: the two networks differ in width (1024 vs 768) and
depth (24 vs 12 blocks), so model capacity is confounded with training signal. Asking
"is the brain's geometry semantic?" by swapping architectures answers a different
question.

The clean test uses ONE model and contrasts its two encoders:

    CLIP ViT-L/14 image encoder  ->  embedding of the picture
    CLIP ViT-L/14 text encoder   ->  embedding of its COCO captions

Both are projected into the same 768-d space that CLIP's contrastive objective aligns,
so they are directly comparable: identical parameters count, identical training run,
identical output geometry. Any difference between them is exactly the visual-versus-
semantic axis and nothing else.

    brain matches TEXT better  ->  geometry reflects what the scene means
    brain matches IMAGE better ->  geometry reflects how the scene looks

Because the encoders are trained to agree, their RDMs are correlated; the informative
quantity is therefore the PARTIAL correlation of each against the brain with the other
held constant, which is what the unique-variance columns report.
"""
import ast
import gc
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from scipy.spatial.distance import squareform

import _full_data as fd
from step33_intersubject_mapping_breakdown import (MIN_RELIABILITY, ROI_ORDER,
                                                   shared_ids, subject_rdms)
from step34_dnn_layer_features import CLIP_MODEL, load_images

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
BATCH = 16
SEED = 42


def rdm(F):
    return squareform(1.0 - np.nan_to_num(np.corrcoef(F), nan=0.0), checks=False)


def norm(x):
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else x


def captions_for(coco_ids):
    want = set(int(c) for c in coco_ids)
    caps = {}
    with zipfile.ZipFile(NSD_RAW / "annotations_trainval2017.zip") as z:
        for nm in ("annotations/captions_train2017.json",
                   "annotations/captions_val2017.json"):
            for a in json.loads(z.read(nm))["annotations"]:
                c = int(a["image_id"])
                if c in want:
                    caps.setdefault(c, []).append(a["caption"].strip())
    return caps


@torch.no_grad()
def encode(images, caps, coco_ids, device):
    """Projected image and text embeddings from the SAME CLIP model."""
    from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
    model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.float16).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)

    img = []
    for i in range(0, len(images), BATCH):
        px = proc(images=images[i:i + BATCH], return_tensors="pt")["pixel_values"]
        v = model.get_image_features(pixel_values=px.to(device, torch.float16)).float()
        img.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        if (i // BATCH) % 15 == 0:
            print(f"    image {i}/{len(images)}", flush=True)
    img = np.concatenate(img)

    txt = []
    for cid in coco_ids:
        cs = caps.get(int(cid), ["a photograph of a natural scene."])[:5]
        enc = tok(cs, padding=True, truncation=True, max_length=77, return_tensors="pt")
        v = model.get_text_features(**{k: t.to(device) for k, t in enc.items()}).float()
        v = torch.nn.functional.normalize(v, dim=-1).mean(0)
        txt.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    txt = np.stack(txt)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return img, txt


def partial(a, b, c):
    """Correlation of a with b, holding c constant."""
    a, b, c = norm(a), norm(b), norm(c)
    ra = a - np.dot(a, c) / np.dot(c, c) * c
    rb = b - np.dot(b, c) / np.dot(c, c) * c
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        print("[ERROR] no CUDA device.")
        return 1
    device = torch.device("cuda")

    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    shared = shared_ids()
    sets = []
    for s in SUBJECTS:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        sets.append(set(n[n >= 2].index))
    images_ids = sorted(set.intersection(*sets))[:766]
    prev = np.load(HERE / "data" / "dnn_layer_rdms_shared.npz")
    assert list(prev["nsd_ids"]) == images_ids, "stimulus order differs from step34"
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    coco_ids = [n2c[i] for i in images_ids]
    print(f"stimuli: {len(images_ids)} (order verified)")

    pics, missing = load_images(coco_ids, stim)
    if missing:
        print(f"[ERROR] {len(missing)} images unavailable")
        return 1
    caps = captions_for(coco_ids)
    print(f"captions: {len(caps)}/{len(coco_ids)} images")

    img_f, txt_f = encode(pics, caps, coco_ids, device)
    np.savez_compressed(HERE / "data" / "clip_vitL_image_text.npz",
                        nsd_ids=np.array(images_ids), coco_ids=np.array(coco_ids),
                        image=img_f, text=txt_f)
    r_img, r_txt = rdm(img_f), rdm(txt_f)
    cross = float(np.corrcoef(r_img, r_txt)[0, 1])
    print(f"\nimage-RDM vs text-RDM r = {cross:.3f}  "
          f"(shared space, so correlation is expected; partials below)")

    prev_tab = pd.read_csv(RESULTS / "intersubject_mapping_breakdown.csv")
    r1 = prev_tab.groupby("roi").r_within.mean()
    ceiling = 2 * r1 / (1 + r1)
    avail = set(fd.available_rois(SUBJECTS[0]))
    rois = [r for r in ROI_ORDER if r in avail and r1.get(r, 0) > MIN_RELIABILITY]

    rows = []
    for roi in rois:
        for s in SUBJECTS:
            out = subject_rdms(s, images_ids, roi)
            if out is None:
                continue
            brain = norm(out[2])
            rows.append(dict(
                roi=roi, subject_id=s, ceiling=float(ceiling[roi]),
                r_image=float(np.corrcoef(brain, norm(r_img))[0, 1]),
                r_text=float(np.corrcoef(brain, norm(r_txt))[0, 1]),
                unique_image=partial(brain, r_img, r_txt),
                unique_text=partial(brain, r_txt, r_img)))
        gc.collect()
        print(f"  [OK] {roi}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "clip_image_vs_text.csv", index=False)
    pd.set_option("display.width", 230)

    out = []
    for roi, g in df.groupby("roi"):
        t, p = stats.ttest_rel(g.unique_text, g.unique_image)
        out.append(dict(roi=roi, ceiling=g.ceiling.iloc[0],
                        r_image=g.r_image.mean(), r_text=g.r_text.mean(),
                        uniq_image=g.unique_image.mean(),
                        uniq_text=g.unique_text.mean(),
                        text_minus_image=(g.unique_text - g.unique_image).mean(),
                        t=t, p=p, n_pos=int((g.unique_text > g.unique_image).sum())))
    S = pd.DataFrame(out)
    S["order"] = S.roi.map({r: i for i, r in enumerate(ROI_ORDER)})
    S = S.sort_values("order").drop(columns="order")
    S.to_csv(RESULTS / "clip_image_vs_text_stats.csv", index=False)

    print("\n=== Visual vs semantic, within ONE model (CLIP ViT-L/14) ===")
    print("  uniq_image / uniq_text: partial correlation with the brain, each")
    print("  holding the other encoder constant. Same parameters, same space.\n")
    print(S[["roi", "ceiling", "r_image", "r_text", "uniq_image", "uniq_text",
             "text_minus_image", "p", "n_pos"]].round(4).to_string(index=False))
    sig_t = S[(S.p < 0.05) & (S.text_minus_image > 0)]
    sig_i = S[(S.p < 0.05) & (S.text_minus_image < 0)]
    print(f"\n  semantic (text wins): {', '.join(sig_t.roi) if len(sig_t) else 'none'}")
    print(f"  visual  (image wins): {', '.join(sig_i.roi) if len(sig_i) else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
