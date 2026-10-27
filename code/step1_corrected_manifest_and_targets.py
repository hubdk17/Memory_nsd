"""
Step 1 (CPU-only): Build a CORRECTED trial->stimulus manifest for all 8 NSD subjects
(sessions 1-2, 1500 trials each) and derive CLIP text-embedding decoding targets.

Fixes the indexing defect present in ADVANCED_RESEARCH/extract_manifests.py, which
looked up the 1-based 73k stimulus ID in the 0-based `nsdId` column of
nsd_stim_info_merged.csv. Correct relation (verified three ways):
    nsdId (0-based, stim_info row) == behav 73KID - 1 == subjectim[s, 10KID-1] - 1

No downloads. Captions come from the local annotations_trainval2017.zip.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPTokenizer

NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
BEHAV_DIR = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/behav")
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
CLIP_DIR = NSD_RAW / "clip_vit_base_patch32"
torch.set_num_threads(8)


def build_manifest():
    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    nsd_to_coco = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    shared = set(stim.loc[stim.shared1000, "nsdId"].astype(int))

    rows = []
    for subj in SUBJECTS:
        b = pd.read_csv(BEHAV_DIR / f"{subj}_responses.tsv", sep="\t")
        b = b[b.SESSION.isin([1, 2])].reset_index(drop=True)
        assert len(b) == 1500, (subj, len(b))
        seen = {}
        for i, r in b.iterrows():
            nsd_id = int(r["73KID"]) - 1          # <-- the fix
            seen[nsd_id] = seen.get(nsd_id, 0) + 1
            rows.append(dict(
                subject_id=subj, global_trial=i + 1,
                session=int(r.SESSION), run=int(r.RUN),
                nsd_id=nsd_id, coco_id=nsd_to_coco[nsd_id],
                repeat_number=seen[nsd_id],
                is_shared1000=int(nsd_id in shared),
                correct=float(r.ISCORRECT) if pd.notna(r.ISCORRECT) else np.nan,
                rt=float(r.RT) if pd.notna(r.RT) else np.nan,
            ))
    df = pd.DataFrame(rows)
    # total presentations of each image for this subject within sessions 1-2
    tot = df.groupby(["subject_id", "nsd_id"]).size().rename("n_presentations")
    df = df.merge(tot, on=["subject_id", "nsd_id"])
    # trial lag from previous presentation
    df = df.sort_values(["subject_id", "nsd_id", "repeat_number"])
    df["trial_lag"] = df.groupby(["subject_id", "nsd_id"]).global_trial.diff()
    df = df.sort_values(["subject_id", "global_trial"]).reset_index(drop=True)
    df.to_csv(OUT / "trial_manifest_corrected.csv", index=False)
    print(f"[OK] trial_manifest_corrected.csv  {df.shape}")
    return df


def load_captions(coco_ids):
    want = set(int(c) for c in coco_ids)
    caps = {}
    with zipfile.ZipFile(NSD_RAW / "annotations_trainval2017.zip") as z:
        for name in ["annotations/captions_train2017.json",
                     "annotations/captions_val2017.json"]:
            for a in json.loads(z.read(name))["annotations"]:
                cid = int(a["image_id"])
                if cid in want:
                    caps.setdefault(cid, []).append(a["caption"].strip())
    return caps


def clip_text_targets(df):
    coco_ids = sorted(df.coco_id.unique())
    caps = load_captions(coco_ids)
    missing = [c for c in coco_ids if c not in caps]
    print(f"  unique images={len(coco_ids)}  with captions={len(caps)}  missing={len(missing)}")

    model = CLIPModel.from_pretrained(str(CLIP_DIR)).eval()
    tok = CLIPTokenizer.from_pretrained(str(CLIP_DIR))

    flat, owner = [], []
    for cid in coco_ids:
        for s in caps.get(cid, ["a photograph of a natural scene."])[:5]:
            flat.append(s); owner.append(cid)
    print(f"  encoding {len(flat)} captions on CPU...")
    vecs = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            enc = tok(flat[i:i + 256], padding=True, truncation=True,
                      max_length=77, return_tensors="pt")
            v = model.get_text_features(**enc)
            vecs.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
            if (i // 256) % 20 == 0:
                print(f"    {i}/{len(flat)}", flush=True)
    vecs = np.concatenate(vecs)
    owner = np.array(owner)

    # mean over an image's captions, renormalised
    T = np.zeros((len(coco_ids), 512), dtype=np.float32)
    for j, cid in enumerate(coco_ids):
        m = vecs[owner == cid].mean(0)
        T[j] = m / (np.linalg.norm(m) + 1e-8)
    np.savez_compressed(OUT / "clip_text_targets.npz",
                        coco_ids=np.array(coco_ids), T_text=T)
    print(f"[OK] clip_text_targets.npz  T_text={T.shape}")


if __name__ == "__main__":
    df = build_manifest()
    clip_text_targets(df)
