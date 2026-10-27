"""
Integrity check on the stimulus set before it feeds any analysis.

A truncated or half-written JPEG does not always raise on open -- PIL is lenient, and a
partially downloaded file can decode to a grey block or a half image. That would enter
the CLIP embeddings silently and corrupt every downstream number without ever throwing.
So each file is fully decoded (not just opened), checked for the expected geometry, and
screened for degenerate content.

Checks per image:
  decodes fully        img.load() forces the whole JPEG through the decoder
  correct size         256x256, the size step7a writes
  three channels       RGB
  not blank            pixel standard deviation above a floor
  not truncated        PIL's truncation warning promoted to an error
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import _full_data as fd

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "data" / "nsd_stimuli_256"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
EXPECT = (256, 256)
MIN_SD = 1.0
ImageFile.LOAD_TRUNCATED_IMAGES = False      # truncation must raise, not be patched


def check(path_str: str):
    p = Path(path_str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # DecompressionBomb / truncation -> error
            img = Image.open(p)
            img.load()                       # full decode, not a lazy header read
            if img.size != EXPECT:
                return (p.name, f"size {img.size}")
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return (p.name, f"shape {arr.shape}")
            if arr.std() < MIN_SD:
                return (p.name, f"blank (sd={arr.std():.3f})")
    except Exception as e:                                   # noqa: BLE001
        return (p.name, repr(e)[:80])
    return None


def main() -> int:
    stim = pd.read_csv(NSD_RAW / "nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    seen = set()
    for subj in [f"subj{i:02d}" for i in range(1, 9)]:
        seen |= set(fd.behaviour(subj).nsd_id.unique())
    required = {n2c[int(i)] for i in seen if int(i) in n2c}

    on_disk = {int(p.stem.split("_")[1]) for p in IMG_DIR.glob("coco_*.jpg")}
    missing = required - on_disk
    extra = on_disk - required
    print(f"required by the experiment : {len(required):,}")
    print(f"present on disk            : {len(on_disk):,}")
    print(f"missing                    : {len(missing):,}")
    print(f"present but unused         : {len(extra):,}")

    paths = [str(IMG_DIR / f"coco_{c:012d}.jpg") for c in sorted(required & on_disk)]
    print(f"\ndecoding {len(paths):,} files ...")
    bad = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for k, r in enumerate(ex.map(check, paths, chunksize=200), 1):
            if r is not None:
                bad.append(r)
            if k % 10000 == 0:
                print(f"  {k:,}/{len(paths):,}  bad so far {len(bad)}", flush=True)

    print(f"\ncorrupt or malformed: {len(bad)}")
    for name, why in bad[:15]:
        print(f"    {name}  {why}")
    if len(bad) > 15:
        print(f"    ... and {len(bad)-15} more")
    if bad:
        pd.DataFrame(bad, columns=["file", "problem"]).to_csv(
            HERE / "results" / "stimulus_integrity_failures.csv", index=False)
        print("  written to results/stimulus_integrity_failures.csv")
        print("  delete those files and re-run step7a; it will refetch only the gaps.")

    ok = len(paths) - len(bad)
    print(f"\n[{'OK' if not bad and not missing else 'INCOMPLETE'}] "
          f"{ok:,}/{len(required):,} stimuli verified usable "
          f"({100*ok/len(required):.2f}%)")
    return 0 if (not bad and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
