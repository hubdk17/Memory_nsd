"""
Step 37 (GPU): CLIP ViT-L/14 image embeddings for every stimulus in the experiment.

This is the common space. Each subject's decoder will be trained to map their own voxels
into it, after which all eight brains produce comparable 768-dimensional vectors for the
same picture and the voxel-space mismatch between subjects stops mattering.

The projected image embedding is used (get_image_features), not a hidden state, because
that is the space CLIP's contrastive objective actually shapes and the one a text
embedding would share if we later need captions for categorisation.

~70,566 stimuli, fp16, batched. Images are the NSD stimulus as presented -- COCO source
cropped by NSD's own cropBox and resized to 256 -- already prepared by step7a.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import _full_data as fd

HERE = Path(__file__).resolve().parent
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
IMG_DIR = HERE / "data" / "nsd_stimuli_256"
OUT = HERE / "data" / "clip_vitL_image_embeddings.npz"
CLIP_MODEL = "openai/clip-vit-large-patch14"
BATCH = 64
SEED = 42


@torch.no_grad()
def main() -> int:
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        print("[ERROR] no CUDA device.")
        return 1
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_properties(0).name}")

    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    seen = set()
    for subj in [f"subj{i:02d}" for i in range(1, 9)]:
        seen |= set(fd.behaviour(subj).nsd_id.unique())
    nsd_ids = sorted(int(i) for i in seen if int(i) in n2c)
    coco_ids = [n2c[i] for i in nsd_ids]

    paths = [IMG_DIR / f"coco_{c:012d}.jpg" for c in coco_ids]
    present = [i for i, p in enumerate(paths) if p.exists()]
    if len(present) < len(paths):
        print(f"[WARN] {len(paths)-len(present)} stimulus images missing; "
              f"embedding the {len(present)} available")
    nsd_ids = [nsd_ids[i] for i in present]
    coco_ids = [coco_ids[i] for i in present]
    paths = [paths[i] for i in present]
    print(f"stimuli to embed: {len(paths):,}")

    from transformers import CLIPImageProcessor, CLIPModel
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.float16).to(device).eval()

    feats, t0 = [], time.time()
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"]
        v = model.get_image_features(pixel_values=px.to(device, torch.float16)).float()
        feats.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        if (i // BATCH) % 50 == 0:
            done = i + len(imgs)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done:,}/{len(paths):,}  {rate:.0f} img/s  "
                  f"eta {(len(paths)-done)/max(rate,1e-9)/60:.1f} min  "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    E = np.concatenate(feats).astype(np.float32)

    bad = int((~np.isfinite(E)).any(1).sum())
    if bad:
        print(f"[ERROR] {bad} embeddings contain non-finite values")
        return 1
    sd = float(E.std(0).mean())
    if sd < 1e-6:
        print(f"[ERROR] embeddings are constant across stimuli (sd={sd:.2e})")
        return 1

    np.savez_compressed(OUT, nsd_ids=np.array(nsd_ids),
                        coco_ids=np.array(coco_ids), embeddings=E)
    print(f"\n[OK] {OUT.name}  {E.shape}  ({OUT.stat().st_size/1e6:.0f} MB)")
    print(f"  mean per-dim sd {sd:.4f}, L2 norm {np.linalg.norm(E, axis=1).mean():.4f}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
