"""Step 1b: CLIP text targets for the off-by-one images, so step5 can compare
correct vs shifted annotations on the same footing. No downloads."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer

NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)

man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
shifted = sorted({n2c[n + 1] for n in man.nsd_id.unique() if (n + 1) in n2c})
have = set(int(c) for c in np.load(HERE / "data" / "clip_text_targets.npz")["coco_ids"])
need = [c for c in shifted if c not in have]
print(f"shifted images: {len(shifted)}  already encoded: {len(shifted)-len(need)}  to encode: {len(need)}")

caps = {}
with zipfile.ZipFile(NSD_RAW / "annotations_trainval2017.zip") as z:
    for nm in ["annotations/captions_train2017.json", "annotations/captions_val2017.json"]:
        for a in json.loads(z.read(nm))["annotations"]:
            cid = int(a["image_id"])
            if cid in set(need):
                caps.setdefault(cid, []).append(a["caption"].strip())
print(f"  with captions: {len(caps)}")

model = CLIPModel.from_pretrained(str(NSD_RAW / "clip_vit_base_patch32")).eval()
tok = CLIPTokenizer.from_pretrained(str(NSD_RAW / "clip_vit_base_patch32"))
ids = [c for c in need if c in caps]
flat, owner = [], []
for c in ids:
    for s in caps[c][:5]:
        flat.append(s); owner.append(c)
print(f"  encoding {len(flat)} captions...")
V = []
with torch.no_grad():
    for i in range(0, len(flat), 256):
        e = tok(flat[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt")
        v = model.get_text_features(**e)
        V.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        if (i // 256) % 20 == 0:
            print(f"    {i}/{len(flat)}", flush=True)
V = np.concatenate(V); owner = np.array(owner)
T = np.zeros((len(ids), 512), np.float32)
for j, c in enumerate(ids):
    m = V[owner == c].mean(0)
    T[j] = m / (np.linalg.norm(m) + 1e-8)
np.savez_compressed(HERE / "data" / "clip_text_targets_shifted.npz",
                    coco_ids=np.array(ids), T_text=T)
print(f"[OK] clip_text_targets_shifted.npz  {T.shape}")
