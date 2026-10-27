"""
Step 7a: fetch the COCO source images for every stimulus in the corrected manifest and
store them as the NSD stimulus the participant actually saw.

Only 145 of the 7,951 required images exist locally, and only 42 of the 2,374
repeated-cohort test images, so the CLIP-image analysis in step 7 cannot run without
this. Images come from images.cocodataset.org (public), keyed on the CORRECTED cocoId.

Each image is cropped with the NSD `cropBox` from nsd_stim_info_merged.csv --
(top, bottom, left, right) as fractions to remove, which yields the square NSD
stimulus -- then resized to 256x256 (CLIP ViT-B/32 preprocesses to 224 anyway).

Resumable: already-present files are skipped.
"""
import ast
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from PIL import Image

HERE = Path(__file__).resolve().parent
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
OUT_DIR = HERE / "data" / "nsd_stimuli_256"
LOCAL_DIRS = [Path("D:/Desktop/FMRI_NSD/nsd_shared1000_multitrial/stimulus_images"),
              Path("D:/Desktop/Memory_retention/results/stimulus_images_323")]
URL = "http://images.cocodataset.org/{split}/{cid:012d}.jpg"
SIZE = 256
WORKERS = 64
RETRIES = 3

SESSION = requests.Session()
SESSION.mount("http://", HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS,
                                     max_retries=0))


def index_local():
    """coco_id -> local path, for images already on disk."""
    found = {}
    for d in LOCAL_DIRS:
        if not d.exists():
            continue
        for f in os.listdir(d):
            m = re.match(r"nsd(\d+)_coco(\d+)\.jpg$", f) or re.match(r"coco_(\d+)\.jpg$", f)
            if not m:
                continue
            cid = int(m.group(2) if m.lastindex == 2 else m.group(1))
            found.setdefault(cid, d / f)
    return found


def nsd_crop(img, crop_box):
    t, b, l, r = crop_box
    W, H = img.size
    box = (int(round(l * W)), int(round(t * H)),
           W - int(round(r * W)), H - int(round(b * H)))
    return img.crop(box)


def fetch_one(cid, split, crop_box, local_path):
    dest = OUT_DIR / f"coco_{cid:012d}.jpg"
    if dest.exists() and dest.stat().st_size > 1000:
        return "cached"
    try:
        if local_path is not None:
            img = Image.open(local_path).convert("RGB")
        else:
            last = None
            for split_try in [split, "val2017" if split == "train2017" else "train2017"]:
                for attempt in range(RETRIES):
                    try:
                        resp = SESSION.get(URL.format(split=split_try, cid=cid), timeout=25)
                        if resp.status_code == 200:
                            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                            last = None
                            break
                        last = f"HTTP {resp.status_code}"
                    except Exception as e:                      # transient network errors
                        last = repr(e)
                        time.sleep(1.5 * (attempt + 1))
                else:
                    continue
                break
            if last is not None:
                return f"fail:{last}"
        img = nsd_crop(img, crop_box).resize((SIZE, SIZE), Image.LANCZOS)
        img.save(dest, "JPEG", quality=95)
        return "fetched"
    except Exception as e:
        return f"fail:{e!r}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every image any subject saw across the FULL experiment, not just sessions 1-2.
    import _full_data as fd
    stim_all = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim_all.nsdId.astype(int), stim_all.cocoId.astype(int)))
    seen = set()
    for subj in [f"subj{i:02d}" for i in range(1, 9)]:
        seen |= set(fd.behaviour(subj).nsd_id.unique())
    stim = stim_all.set_index("cocoId")
    local = index_local()
    need = sorted({n2c[int(i)] for i in seen if int(i) in n2c})
    print(f"unique stimuli required: {len(need)}")
    print(f"  usable local originals: {sum(1 for c in need if c in local)}")

    tasks = []
    for cid in need:
        row = stim.loc[cid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tasks.append((int(cid), str(row["cocoSplit"]),
                      ast.literal_eval(str(row["cropBox"])), local.get(int(cid))))

    counts, fails, t0 = {}, [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, *t): t[0] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            key = r.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "fail":
                fails.append((futs[f], r))
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {counts}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n[OK] {counts}")
    if fails:
        pd.DataFrame(fails, columns=["coco_id", "reason"]).to_csv(
            HERE / "results" / "step7a_fetch_failures.csv", index=False)
        print(f"  {len(fails)} failures written to results/step7a_fetch_failures.csv")
    n = len(list(OUT_DIR.glob('*.jpg')))
    mb = sum(p.stat().st_size for p in OUT_DIR.glob('*.jpg')) / 1e6
    print(f"  {n} images on disk in {OUT_DIR} ({mb:.0f} MB)")
    return 0 if n >= 0.98 * len(need) else 1


if __name__ == "__main__":
    sys.exit(main())
