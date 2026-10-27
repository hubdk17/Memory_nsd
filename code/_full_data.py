"""Loader for the full-experiment ROI beta store written by step27.

Betas are stored one compressed file per session as int16, exactly as NSD ships them
(percent signal change x 300). This module concatenates the sessions a subject has on
disk and applies normalisation at load time, so the store stays raw and the analysis
layer decides how to standardise.

Normalisation matters more here than it did with two sessions: a 40-session experiment
spans nine months of scanner drift, coil changes and head-position differences, so
per-session z-scoring is the minimum. It is the default.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STORE = HERE / "data" / "roi_betas_full"
BEHAV = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/behav")
TRIALS_PER_SESSION = 750


_SESSION_RE = re.compile(r"^session(\d+)\.npz$")


def available_sessions(subj: str) -> list[int]:
    """Session numbers on disk, ignoring staging files left by an interrupted run."""
    d = STORE / subj
    if not d.exists():
        return []
    out = []
    for p in d.glob("session*.npz"):
        m = _SESSION_RE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def available_rois(subj: str) -> list[str]:
    s = available_sessions(subj)
    if not s:
        return []
    with np.load(STORE / subj / f"session{s[0]:02d}.npz") as z:
        return sorted(z.files)


def load_rois(subj: str, rois: list[str], sessions: list[int] | None = None,
              norm: str = "session") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, trial_index, session_index).

    X            (n_trials, n_voxels) float32, ROIs concatenated in the given order
    trial_index  1-based global trial number within the subject's experiment
    session_index session each trial came from

    norm : 'session' z-scores each voxel within session (default, recommended)
           'none'    returns the stored values unchanged
    """
    sess = sessions if sessions is not None else available_sessions(subj)
    if not sess:
        raise FileNotFoundError(f"no extracted sessions for {subj} in {STORE}")
    blocks, tidx, sidx = [], [], []
    for s in sess:
        with np.load(STORE / subj / f"session{s:02d}.npz") as z:
            missing = [r for r in rois if r not in z.files]
            if missing:
                raise KeyError(f"{subj} session{s:02d} lacks ROIs: {missing}")
            block = np.concatenate([z[r] for r in rois], axis=1).astype(np.float32)
        n = block.shape[0]
        if norm == "session":
            block = (block - block.mean(0)) / (block.std(0) + 1e-8)
        blocks.append(block)
        tidx.append(np.arange(1, n + 1) + (s - 1) * TRIALS_PER_SESSION)
        sidx.append(np.full(n, s))
    return (np.concatenate(blocks), np.concatenate(tidx), np.concatenate(sidx))


def behaviour(subj: str) -> pd.DataFrame:
    """Full behavioural log with corrected stimulus identity and a global trial index."""
    b = pd.read_csv(BEHAV / f"{subj}_responses.tsv", sep="\t")
    b = b.sort_values(["SESSION", "RUN", "TRIAL"]).reset_index(drop=True)
    b["nsd_id"] = b["73KID"] - 1                    # verified: nsdId = 73KID - 1
    b["global_trial"] = (b.SESSION - 1) * TRIALS_PER_SESSION + \
                        b.groupby("SESSION").cumcount() + 1
    return b


def coverage() -> pd.DataFrame:
    rows = []
    for d in sorted(STORE.glob("subj*")):
        s = available_sessions(d.name)
        b = behaviour(d.name)
        rows.append(dict(subject_id=d.name, sessions_on_disk=len(s),
                         sessions_total=int(b.SESSION.max()),
                         trials_on_disk=len(s) * TRIALS_PER_SESSION,
                         rois=len(available_rois(d.name)),
                         gb=sum(p.stat().st_size for p in d.glob("*.npz")) / 1e9))
    return pd.DataFrame(rows)
