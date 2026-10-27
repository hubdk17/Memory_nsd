"""Step 41: GIST descriptors for every NSD stimulus -- a LOW-LEVEL decoding target.

WHY THIS EXISTS
---------------
step38 decoded voxels into CLIP ViT-L/14, a semantic embedding. A semantic target
can make high-level regions look shared by construction: if two people agree about
what a picture means but not about its retinal details, a semantic read-out will
find agreement in ventral cortex and disagreement in V1 whatever the brains are
doing. The gradient reported in the paper would then be a fact about CLIP.

GIST is the control. It is an oriented spatial-frequency energy descriptor -- a V1-like
model, with no semantic content at all. Running the identical pipeline with GIST as
the target asks whether the hierarchy effect is a property of cortex or of the target
space.

IMPLEMENTATION
--------------
The standard Oliva & Torralba descriptor: 128x128 grayscale, 4 scales x 8 orientations
of Gabor energy in the Fourier domain, averaged over a 4x4 spatial grid.
  4 scales * 8 orientations * 16 cells = 512 dimensions.
Filter transfer functions follow Torralba's createGabor so the bank is reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

HERE = Path(__file__).resolve().parent
IMG = HERE / "data" / "nsd_stimuli_256"
OUT = HERE / "data" / "gist_features.npz"
N = 128
N_SCALES = 4
N_ORI = 8
GRID = 4


def gabor_bank(n: int = N) -> np.ndarray:
    """(n_filters, n, n) transfer functions, following Torralba's createGabor."""
    fx, fy = np.meshgrid(np.arange(-n / 2, n / 2), np.arange(-n / 2, n / 2))
    fr = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
    t = np.fft.fftshift(np.angle(fx + 1j * fy))
    filters = []
    for i in range(N_SCALES):
        for j in range(N_ORI):
            p = (0.35, 0.3 / (1.85 ** i), 16 * N_ORI ** 2 / 32 ** 2,
                 np.pi / N_ORI * j)
            tr = t + p[3]
            tr = tr + 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
            G = np.exp(-10 * p[0] * (fr / n / p[1] - 1) ** 2
                       - 2 * p[2] * np.pi * tr ** 2)
            filters.append(G.astype(np.float32))
    return np.stack(filters)


BANK = gabor_bank()


def gist_one(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((N, N), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-8)          # per-image contrast normalisation
    F = np.fft.fft2(a)
    step = N // GRID
    out = np.empty(len(BANK) * GRID * GRID, dtype=np.float32)
    for k, G in enumerate(BANK):
        e = np.abs(np.fft.ifft2(F * G))
        # mean energy per cell of the GRID x GRID map
        cells = e.reshape(GRID, step, GRID, step).mean(axis=(1, 3))
        out[k * GRID * GRID:(k + 1) * GRID * GRID] = cells.ravel()
    return out


def main() -> int:
    stim = pd.read_csv("D:/Desktop/FMRI_NSD/data/raw/nsd_stim_info_merged.csv")
    n2c = dict(zip(stim.nsdId.astype(int), stim.cocoId.astype(int)))
    files, nsd_ids, coco_ids = [], [], []
    for nsd_id, coco in sorted(n2c.items()):
        p = IMG / f"coco_{coco:012d}.jpg"
        if p.exists():
            files.append(p)
            nsd_ids.append(nsd_id)
            coco_ids.append(coco)
    print(f"stimuli on disk: {len(files)} of {len(n2c)}")
    print(f"descriptor: {N_SCALES} scales x {N_ORI} orientations x {GRID}x{GRID} "
          f"cells = {len(BANK) * GRID * GRID} dims")

    feats = Parallel(n_jobs=14, batch_size=64, verbose=5)(
        delayed(gist_one)(p) for p in files)
    X = np.stack(feats).astype(np.float32)
    print(f"features: {X.shape}  finite: {np.isfinite(X).all()}")
    # guard: a descriptor that is constant across images would silently produce
    # a null result downstream rather than an error.
    sd = X.std(0)
    print(f"per-dimension sd: min {sd.min():.4g}  median {np.median(sd):.4g}")
    if sd.min() <= 0:
        print("[FAIL] a GIST dimension is constant across stimuli")
        return 1

    np.savez_compressed(OUT, nsd_ids=np.array(nsd_ids), coco_ids=np.array(coco_ids),
                        features=X)
    print(f"wrote {OUT.relative_to(HERE)}  ({OUT.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
