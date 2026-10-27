"""
Step 27: stream NSD preprocessed session betas, extract ROI voxels, discard the volume.

Written fresh rather than patched onto the FMRI_NSD downloader, which carries two
defects this pipeline must not inherit:
  1. it maps trials with subjectim[subj, trial_idx], which indexes the subject's IMAGE
     list rather than the trial sequence (1 of 1500 trials correctly paired);
  2. it labels prf-visualrois 1/2/3/4 as V1/V2/V3/hV4, when NSD's labels are
     1=V1v 2=V1d 3=V2v 4=V2d 5=V3v 6=V3d 7=hV4 -- so its "V3" is really V2v and true
     V3 and hV4 were never extracted at all.

Here stimulus identity comes from the behavioural log's 73KID (nsdId = 73KID - 1,
verified three ways in Phase I) and ROI labels follow the NSD documentation. V3, hV4 and
OPA are included for the first time, plus nsdgeneral where a mask is available.

Data are stored as int16 exactly as NSD ships them (percent signal change x 300), one
compressed file per session, so the run is resumable and memory stays bounded. The
analysis layer concatenates and normalises.

Downloads use parallel range requests; a single stream measured 4.6 MB/s.
"""
import argparse
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
BEHAV = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/behav")
OUT_ROOT = HERE / "data" / "roi_betas_full"
TMP = HERE / "data" / "_tmp_sessions"
S3 = ("https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/"
      "{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{sess:02d}.nii.gz")
NSDGENERAL = [Path("D:/Downloads/lh.nsdgeneral.nii.gz"),
              Path("D:/Downloads/rh.nsdgeneral.nii.gz")]
ROI_S3 = ("https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/"
          "{subj}/func1pt8mm/roi/{name}")
# Fetched once per subject if absent. nsdgeneral is the mask the reconstruction
# literature uses, so having it for every subject keeps voxel sets comparable;
# streams gives the coarse visual-hierarchy parcellation. Both are ~65 KB.
EXTRA_MASKS = ["nsdgeneral.nii.gz", "streams.nii.gz"]
# streams labels (nsddata): 1 early, 2 midventral, 3 midlateral, 4 midparietal,
# 5 ventral, 6 lateral, 7 parietal
STREAMS = {"S_early": [1], "S_midventral": [2], "S_midlateral": [3],
           "S_ventral": [5], "S_lateral": [6], "S_parietal": [7]}
CHUNK_MB = 16
WORKERS = 12
TRIALS_PER_SESSION = 750

# NSD ROI label definitions (nsddata/freesurfer/*/label/*.mgz.ctab)
PRF = {"V1": [1, 2], "V2": [3, 4], "V3": [5, 6], "hV4": [7]}
FACES = {"FFA": [2, 3]}
PLACES = {"OPA": [1], "PPA": [2], "RSC": [3]}
MTL = {"PRC": [1], "ERC": [2], "PHC": [3], "SUB": [4], "CA1": [5], "DG_CA3": [7, 8]}


def ensure_masks(subj: str) -> None:
    """Fetch the small ROI masks this subject is missing (a few tens of KB each)."""
    roi = NSD_RAW / subj / "roi"
    roi.mkdir(parents=True, exist_ok=True)
    for name in EXTRA_MASKS:
        dest = roi / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            r = requests.get(ROI_S3.format(subj=subj, name=name), timeout=60)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                print(f"    fetched {name} ({len(r.content)/1024:.0f} KB)")
            else:
                print(f"    [WARN] {name}: HTTP {r.status_code}")
        except Exception as e:                                   # noqa: BLE001
            print(f"    [WARN] {name}: {e!r}")


def load_masks(subj: str) -> dict:
    ensure_masks(subj)
    roi = NSD_RAW / subj / "roi"
    out = {}
    for fname, spec in (("prf-visualrois.nii.gz", PRF), ("floc-faces.nii.gz", FACES),
                        ("floc-places.nii.gz", PLACES), ("MTL.nii.gz", MTL)):
        path = roi / fname
        if not path.exists():
            print(f"    [WARN] missing {fname}; skipping {list(spec)}")
            continue
        d = nib.load(path).get_fdata().astype(int)
        for name, labels in spec.items():
            m = np.isin(d, labels)
            if m.sum():
                out[name] = m
    ref = next(iter(out.values())).shape if out else None

    gen = roi / "nsdgeneral.nii.gz"
    if gen.exists():
        g = nib.load(gen).get_fdata() > 0
        if ref is None or g.shape == ref:
            out["nsdgeneral"] = g
    elif all(p.exists() for p in NSDGENERAL):          # subj01 fallback, local copies
        g = np.zeros(ref, dtype=bool)
        for p in NSDGENERAL:
            g |= nib.load(p).get_fdata() > 0
        out["nsdgeneral"] = g

    st = roi / "streams.nii.gz"
    if st.exists():
        sd = nib.load(st).get_fdata().astype(int)
        if ref is None or sd.shape == ref:
            for name, labels in STREAMS.items():
                m = np.isin(sd, labels)
                if m.sum():
                    out[name] = m
    return out


def fetch(url: str, dest: Path) -> bool:
    """Parallel range download. Falls back to a single stream if ranges are refused."""
    try:
        head = requests.head(url, timeout=30)
        if head.status_code != 200:
            return False
        total = int(head.headers["Content-Length"])
    except Exception:
        return False
    size = CHUNK_MB * 1024 * 1024
    spans = [(i, min(i + size - 1, total - 1)) for i in range(0, total, size)]
    buf = [None] * len(spans)

    def grab(k):
        if buf[k] is not None:
            return True
        lo, hi = spans[k]
        for attempt in range(6):
            try:
                r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                if r.status_code in (200, 206) and len(r.content) == hi - lo + 1:
                    buf[k] = r.content
                    return True
            except Exception:
                pass
            time.sleep(min(2.0 * (2 ** attempt), 20.0))      # exponential backoff
        return False

    # Two sweeps: a transient chunk failure should not discard a whole session's
    # worth of successful transfers, so the second pass retries only what is missing.
    for sweep in range(2):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            ok = all(ex.map(grab, range(len(spans))))
        if ok:
            break
        missing = sum(1 for b in buf if b is None)
        print(f"      retry sweep {sweep+1}: {missing} chunk(s) missing", flush=True)
    if any(b is None for b in buf):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(buf))
    return dest.stat().st_size == total


def extract(subj: str, sess: int, masks: dict, keep_volume: bool) -> dict | None:
    out_dir = OUT_ROOT / subj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"session{sess:02d}.npz"
    if out_file.exists() and out_file.stat().st_size > 10_000:
        # Only reuse an extraction that contains every ROI we currently ask for.
        # Adding a mask (nsdgeneral, streams) must invalidate earlier partial runs,
        # or subjects end up with mismatched voxel sets.
        try:
            with np.load(out_file) as z:
                if not set(masks).difference(z.files):
                    return {"status": "cached"}
                missing = sorted(set(masks).difference(z.files))
        except Exception:                                        # noqa: BLE001
            missing = ["unreadable"]
        print(f"    session {sess:02d}: re-extracting, missing {missing}")

    local = NSD_RAW / subj / f"betas_session{sess:02d}.nii.gz"
    tmp = TMP / subj / f"betas_session{sess:02d}.nii.gz"
    src, transient = (local, False) if local.exists() else (tmp, True)
    t0 = time.time()
    if transient and not src.exists():
        if not fetch(S3.format(subj=subj, sess=sess), src):
            return None
    dl = time.time() - t0

    try:
        img = nib.load(src)
        if img.shape[-1] != TRIALS_PER_SESSION:
            print(f"    [WARN] session {sess} has {img.shape[-1]} trials")
        # Forces the read, so a truncated or corrupt gzip fails here (CRC32) rather
        # than producing silently wrong voxels.
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
        arrays = {}
        for name, m in masks.items():
            if m.shape != vol.shape[:3]:
                continue
            arrays[name] = np.round(vol[m, :].T).astype(np.int16)  # (trials, voxels)
        del vol, img
        gc.collect()
        if not arrays:
            raise ValueError("no ROI matched the volume geometry")

        # Atomic write: a crash mid-save must not leave a truncated npz behind while
        # the source volume has already been discarded.
        # The staging name MUST end in .npz: np.savez_compressed silently appends
        # ".npz" to any other extension, so "session05.npz.part" would be written as
        # "session05.npz.part.npz" and the rename below would not find it.
        stage = out_file.with_name(out_file.stem + ".part.npz")
        np.savez_compressed(stage, **arrays)
        with np.load(stage) as chk:                    # verify before committing
            if set(chk.files) != set(arrays):
                raise ValueError("staged file is missing ROIs")
        stage.replace(out_file)
    except Exception as e:                             # noqa: BLE001
        print(f"    [ERROR] session {sess:02d} extraction failed: {e!r}")
        for junk in (out_file.with_name(out_file.stem + ".part.npz"),
                     out_file.with_suffix(".npz.part.npz")):
            if junk.exists():
                junk.unlink()
        if transient and src.exists():
            src.unlink()                               # force a clean re-download
        gc.collect()
        return None

    if transient and not keep_volume and src.exists():
        src.unlink()
        gc.collect()
    return {"status": "done", "dl_s": dl, "total_s": time.time() - t0,
            "mb": out_file.stat().st_size / 1e6,
            "voxels": int(sum(a.shape[1] for a in arrays.values()))}


def n_sessions(subj: str) -> int:
    b = pd.read_csv(BEHAV / f"{subj}_responses.tsv", sep="\t")
    return int(b.SESSION.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["subj01"])
    ap.add_argument("--sessions", type=int, default=None,
                    help="limit to the first N sessions (default: all completed)")
    ap.add_argument("--keep-volume", action="store_true",
                    help="keep the downloaded session volume instead of discarding it")
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)

    for subj in args.subjects:
        total = n_sessions(subj)
        if args.sessions:
            total = min(total, args.sessions)
        masks = load_masks(subj)
        print(f"\n=== {subj}: {total} sessions, {len(masks)} ROIs, "
              f"{sum(int(m.sum()) for m in masks.values()):,} voxels ===")
        print("   " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()))
        t0, done, failed = time.time(), 0, []
        for sess in range(1, total + 1):
            r = extract(subj, sess, masks, args.keep_volume)
            if r is None:
                failed.append(sess)
                print(f"  [FAIL] session {sess:02d}", flush=True)
                continue
            done += 1
            if r["status"] == "done":
                el = time.time() - t0
                rate = done / el * 3600
                print(f"  [{sess:02d}/{total}] {r['mb']:5.0f} MB  "
                      f"dl {r['dl_s']:5.1f}s  tot {r['total_s']:5.1f}s  "
                      f"~{rate:.0f} sess/h  eta {(total-sess)/max(rate,1e-9):.1f}h",
                      flush=True)
        print(f"  {subj}: {done}/{total} sessions ready"
              + (f", failed {failed}" if failed else ""))
        sz = sum(p.stat().st_size for p in (OUT_ROOT / subj).glob("*.npz")) / 1e9
        print(f"  stored {sz:.2f} GB in {OUT_ROOT / subj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
