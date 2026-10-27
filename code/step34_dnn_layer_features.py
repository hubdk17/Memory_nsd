"""
Step 34 (GPU): layer-wise DNN features for the shared1000 stimuli.

Step 33 measures how much of each region's representational geometry is shared across
brains. It cannot say WHAT that shared geometry encodes. Comparing the brain RDMs
against layer-wise DNN RDMs localises the shared code along a processing hierarchy:
a region whose shared geometry matches early convolutional layers is sharing low-level
image structure, one matching late layers is sharing semantics.

Two networks, chosen to dissociate the two accounts:
  CLIP ViT-L/14   language-supervised, so its deep layers carry caption semantics
  DINOv2 ViT-B/14 self-supervised on images alone, no language anywhere in training

If the shared geometry is semantic it should align better with deep CLIP than with
deep DINOv2. If it is visual, the two should behave alike. CLIP alone cannot separate
these, which is why the self-supervised control is here.

Runs in fp16 on GPU, one model resident at a time (8 GB card). Outputs L2-normalised
[CLS] features per layer plus the corresponding RDMs, in the same condensed-vector
format and the same image order as step33, so the two are directly comparable.
"""
import argparse
import ast
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
SHARED_IMG = Path("D:/Desktop/FMRI_NSD/nsd_shared1000_multitrial/stimulus_images")
LOCAL_256 = HERE / "data" / "nsd_stimuli_256"
OUT = HERE / "data"
CLIP_MODEL = "openai/clip-vit-large-patch14"
DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
CLIP_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]
DINO_LAYERS = [0, 2, 4, 6, 8, 10, 11]
BATCH = 16
SEED = 42


def nsd_crop(img: Image.Image, crop_box) -> Image.Image:
    """NSD cropBox is (top, bottom, left, right) as fractions to remove."""
    t, b, l, r = crop_box
    W, H = img.size
    return img.crop((int(round(l * W)), int(round(t * H)),
                     W - int(round(r * W)), H - int(round(b * H))))


def load_images(coco_ids, stim):
    """Stimuli as the participant saw them, in the exact order given."""
    by_coco = {int(c): (n, cb) for c, n, cb in
               zip(stim.cocoId, stim.nsdId, stim.cropBox)}
    out, missing = [], []
    for cid in coco_ids:
        nsd_id, cb = by_coco[int(cid)]
        p256 = LOCAL_256 / f"coco_{int(cid):012d}.jpg"
        if p256.exists():
            out.append(Image.open(p256).convert("RGB"))       # already NSD-cropped
            continue
        raw = SHARED_IMG / f"nsd{int(nsd_id):05d}_coco{int(cid):012d}.jpg"
        if raw.exists():
            out.append(nsd_crop(Image.open(raw).convert("RGB"),
                                ast.literal_eval(str(cb))).resize((256, 256),
                                                                  Image.LANCZOS))
            continue
        missing.append(int(cid))
        out.append(None)
    return out, missing


@torch.no_grad()
def clip_features(images, device):
    from transformers import CLIPImageProcessor, CLIPVisionModel
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPVisionModel.from_pretrained(
        CLIP_MODEL, torch_dtype=torch.float16).to(device).eval()
    feats = {f"clip_L{i}": [] for i in CLIP_LAYERS}
    for i in range(0, len(images), BATCH):
        inp = proc(images=images[i:i + BATCH], return_tensors="pt")
        px = inp["pixel_values"].to(device, torch.float16)
        out = model(pixel_values=px, output_hidden_states=True)
        for L in CLIP_LAYERS:
            # hidden_states[0] is the INPUT embedding, where the [CLS] token is a
            # fixed learned parameter that has not yet seen the image -- identical
            # for every stimulus, so its RDM is degenerate and every correlation
            # against it is NaN. hidden_states[L+1] is the output of block L.
            cls = out.hidden_states[L + 1][:, 0, :].float()
            feats[f"clip_L{L}"].append(
                torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"    CLIP {i}/{len(images)}  "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in feats.items()}


@torch.no_grad()
def dino_features(images, device):
    import timm
    model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0)
    model = model.to(device, torch.float16).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    feats = {f"dino_L{i}": [] for i in DINO_LAYERS}
    for i in range(0, len(images), BATCH):
        px = torch.stack([tf(im) for im in images[i:i + BATCH]]).to(
            device, torch.float16)
        x = model.patch_embed(px)
        x = model._pos_embed(x)
        x = model.norm_pre(x)
        for idx, blk in enumerate(model.blocks):
            x = blk(x)
            if idx in DINO_LAYERS:
                cls = x[:, 0, :].float()
                feats[f"dino_L{idx}"].append(
                    torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"    DINOv2 {i}/{len(images)}  "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in feats.items()}


def rdm(F: np.ndarray) -> np.ndarray:
    C = np.corrcoef(F)
    return squareform(1.0 - np.nan_to_num(C, nan=0.0), checks=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=766)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        print("[ERROR] no CUDA device.")
        return 1
    device = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}, {p.total_memory/1e9:.1f} GB")

    # Same image set and ORDER as step33, so brain and DNN RDMs are comparable.
    import _full_data as fd
    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    shared = set(stim.loc[stim.shared1000, "nsdId"].astype(int))
    sets = []
    for s in [f"subj{i:02d}" for i in range(1, 9)]:
        b = fd.behaviour(s)
        n = b[b.nsd_id.isin(shared)].groupby("nsd_id").size()
        sets.append(set(n[n >= 2].index))
    nsd_ids = sorted(set.intersection(*sets))[: args.max_images]
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    coco_ids = [n2c[i] for i in nsd_ids]
    print(f"stimuli: {len(nsd_ids)} (same order as step33)")

    images, missing = load_images(coco_ids, stim)
    if missing:
        print(f"[ERROR] {len(missing)} stimulus images unavailable: {missing[:5]}")
        return 1
    print(f"loaded {len(images)} images, first size {images[0].size}")

    allf = {}
    print("\nCLIP ViT-L/14 (language-supervised)")
    allf.update(clip_features(images, device))
    print("\nDINOv2 ViT-B/14 (self-supervised, no language)")
    allf.update(dino_features(images, device))

    # A layer whose features are constant across stimuli yields a degenerate RDM and
    # NaN everywhere downstream. Catch it here rather than letting it propagate.
    degenerate = [k for k, v in allf.items() if v.std(0).mean() < 1e-8]
    if degenerate:
        print(f"\n[ERROR] constant features in: {degenerate}")
        print("  These layers carry no stimulus information; fix the layer indexing.")
        return 1

    np.savez_compressed(OUT / "dnn_layer_features_shared.npz",
                        nsd_ids=np.array(nsd_ids), coco_ids=np.array(coco_ids), **allf)
    rdms = {k: rdm(v) for k, v in allf.items()}
    bad_rdm = [k for k, v in rdms.items() if not np.isfinite(v).all()]
    if bad_rdm:
        print(f"\n[ERROR] non-finite RDM values in: {bad_rdm}")
        return 1
    np.savez_compressed(OUT / "dnn_layer_rdms_shared.npz",
                        nsd_ids=np.array(nsd_ids), **rdms)

    print("\n=== extracted ===")
    for k in sorted(allf):
        print(f"  {k:12s} features {allf[k].shape}  rdm {rdms[k].shape}")
    sz = (OUT / "dnn_layer_features_shared.npz").stat().st_size / 1e6
    print(f"\nsaved dnn_layer_features_shared.npz ({sz:.0f} MB) and "
          f"dnn_layer_rdms_shared.npz")

    # sanity: adjacent layers should be more similar than distant ones
    ks = [f"clip_L{L}" for L in CLIP_LAYERS if f"clip_L{L}" in rdms]
    if len(ks) >= 3:
        adj = np.mean([np.corrcoef(rdms[ks[i]], rdms[ks[i + 1]])[0, 1]
                       for i in range(len(ks) - 1)])
        far = np.corrcoef(rdms[ks[0]], rdms[ks[-1]])[0, 1]
        print(f"\n[validation] CLIP adjacent-layer RDM r = {adj:.3f}, "
              f"first-to-last r = {far:.3f}"
              + ("  OK" if adj > far else "  CHECK: hierarchy not monotone"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
