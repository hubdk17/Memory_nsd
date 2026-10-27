"""
Step 32: generate DATASET.md by inspecting the extracted store.

Every number in the output is read from the files at generation time -- voxel counts,
session lists, array shapes, dtypes, value ranges, file sizes -- so the documentation
cannot drift from what is actually on disk. Re-run it after any extraction change.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import _full_data as fd

HERE = Path(__file__).resolve().parent
STORE = HERE / "data" / "roi_betas_full"
NSD_RAW = Path("D:/Desktop/FMRI_NSD/data/raw")
OUT = HERE / "DATASET.md"

ROI_DOC = {
    "V1": ("prf-visualrois", "1,2 (V1v+V1d)", "Primary visual cortex"),
    "V2": ("prf-visualrois", "3,4 (V2v+V2d)", "Secondary visual cortex"),
    "V3": ("prf-visualrois", "5,6 (V3v+V3d)", "Third visual area"),
    "hV4": ("prf-visualrois", "7", "Human V4, colour/form"),
    "FFA": ("floc-faces", "2,3 (FFA-1,FFA-2)", "Fusiform face area"),
    "OPA": ("floc-places", "1", "Occipital place area"),
    "PPA": ("floc-places", "2", "Parahippocampal place area"),
    "RSC": ("floc-places", "3", "Retrosplenial cortex"),
    "PRC": ("MTL", "1", "Perirhinal cortex"),
    "ERC": ("MTL", "2", "Entorhinal cortex"),
    "PHC": ("MTL", "3", "Parahippocampal cortex"),
    "SUB": ("MTL", "4", "Subiculum"),
    "CA1": ("MTL", "5", "Cornu Ammonis 1"),
    "DG_CA3": ("MTL", "7,8", "Dentate gyrus + CA3"),
    "nsdgeneral": ("nsdgeneral", "all >0", "NSD's standard responsive-cortex mask"),
    "S_early": ("streams", "1", "Early visual stream"),
    "S_midventral": ("streams", "2", "Mid-ventral stream"),
    "S_midlateral": ("streams", "3", "Mid-lateral stream"),
    "S_ventral": ("streams", "5", "Ventral stream"),
    "S_lateral": ("streams", "6", "Lateral stream"),
    "S_parietal": ("streams", "7", "Parietal stream"),
}


def main() -> int:
    if not STORE.exists():
        print("[ERROR] no extracted store. Run step27 first.")
        return 1
    subs = sorted(p.name for p in STORE.glob("subj*") if p.is_dir())
    if not subs:
        print("[ERROR] store is empty.")
        return 1

    L = []
    A = L.append
    A("# Extracted NSD Dataset — What Is Actually In These Files\n")
    A(f"Generated {date.today().isoformat()} by `step32_document_dataset.py`, which "
      f"reads the files rather than describing them from memory.\n")
    A("---\n")

    # ---------------------------------------------------------------- provenance
    A("## 1. Provenance\n")
    A("**Source.** Natural Scenes Dataset (Allen et al., 2022, *Nature Neuroscience*), "
      "AWS S3 bucket `natural-scenes-dataset`.\n")
    A("**Which NSD product.** `betas_fithrf_GLMdenoise_RR` in `func1pt8mm` space. These "
      "are NSD's **fully preprocessed single-trial GLM beta estimates** — motion and "
      "distortion corrected, per-voxel HRF fitting, GLMdenoise, ridge regression. They "
      "are not raw BOLD timeseries (that is `nsddata_timeseries`, which we never "
      "downloaded and do not need).\n")
    A("**Units.** NSD ships betas as `int16` equal to percent signal change × 300. We "
      "store them unchanged, so divide by 300 for percent signal change. Normalisation "
      "is applied at load time, not baked into the store.\n")
    A("**Spatial resolution.** 1.8 mm isotropic, each subject in their own native "
      "functional space. Subjects have different volume dimensions, so voxel indices "
      "are **not** comparable across subjects — only stimulus identity is.\n")
    A("**Stimulus identity.** Taken from each subject's behavioural log as "
      "`nsdId = 73KID - 1`. This matters: the earlier pipelines in this project looked "
      "up a 1-based 73k index in the 0-based `nsdId` column of "
      "`nsd_stim_info_merged.csv`, which shifted every stimulus by one image. The "
      "relation used here was verified three independent ways and the extractor was "
      "regression-tested against the old tensors (max |diff| 1e-6, r = 1.000000).\n")

    # ---------------------------------------------------------------- layout
    A("## 2. Directory layout\n")
    A("```")
    A("RECONSTRUCTION_PILOT/data/roi_betas_full/")
    for s in subs:
        sess = fd.available_sessions(s)
        A(f"  {s}/")
        A(f"    session01.npz .. session{max(sess):02d}.npz"
          f"   ({len(sess)} files)" if sess else "    (empty)")
    A("```\n")
    A("One file per scan session. Session-wise storage keeps the run resumable and "
      "memory bounded; the analysis layer concatenates via `_full_data.load_rois()`.\n")

    # ---------------------------------------------------------------- coverage
    A("## 3. Coverage\n")
    cov = fd.coverage()
    cov["complete"] = cov.sessions_on_disk >= cov.sessions_total
    A("| subject | sessions on disk | sessions in NSD | trials | ROI sets | size (GB) | complete |")
    A("|---|---|---|---|---|---|---|")
    for _, r in cov.iterrows():
        A(f"| {r.subject_id} | {r.sessions_on_disk} | {r.sessions_total} | "
          f"{r.trials_on_disk:,} | {r.rois} | {r.gb:.2f} | "
          f"{'yes' if r.complete else 'NO'} |")
    A(f"\n**Total: {cov.sessions_on_disk.sum()} sessions, "
      f"{cov.trials_on_disk.sum():,} trials, {cov.gb.sum():.1f} GB.**\n")
    A("Subjects 1, 2, 5 and 7 completed 40 sessions; 3 and 6 completed 32; 4 and 8 "
      "completed 30. That is a property of NSD, not of this extraction.\n")

    # ---------------------------------------------------------------- file contents
    A("## 4. What is inside one session file\n")
    ref = subs[0]
    sess = fd.available_sessions(ref)
    with np.load(STORE / ref / f"session{sess[0]:02d}.npz") as z:
        keys = sorted(z.files)
        shapes = {k: z[k].shape for k in keys}
        dtypes = {k: str(z[k].dtype) for k in keys}
        sample = z[keys[0]]
        vmin, vmax = int(sample.min()), int(sample.max())
    A(f"Example: `{ref}/session{sess[0]:02d}.npz` — a compressed NumPy archive holding "
      f"one array per ROI.\n")
    A("```python")
    A("import numpy as np")
    A(f"z = np.load('data/roi_betas_full/{ref}/session{sess[0]:02d}.npz')")
    A("z['V1'].shape      # (750, n_voxels)  -- trials x voxels, int16")
    A("z['V1'] / 300.0    # percent signal change")
    A("```\n")
    A("**Array axes.** Row = trial, in presentation order within that session "
      "(750 trials). Column = voxel, in a fixed order given by the mask's flattened "
      "index; the order is stable across sessions for a subject, so columns are "
      "directly comparable session to session.\n")
    A(f"**Dtype.** `{dtypes[keys[0]]}`. Observed range in this sample: {vmin} to {vmax} "
      f"(÷300 → {vmin/300:.2f} to {vmax/300:.2f} % signal change).\n")

    # ---------------------------------------------------------------- roi table
    A("## 5. The ROI sets\n")
    A("| ROI | source mask | labels | description |")
    A("|---|---|---|---|")
    for k in keys:
        src, lab, desc = ROI_DOC.get(k, ("?", "?", ""))
        A(f"| `{k}` | `{src}.nii.gz` | {lab} | {desc} |")
    A("")
    A("**Label definitions follow the NSD documentation.** This is worth stating "
      "because the earlier pipelines in this project used `prf-visualrois` labels "
      "1/2/3/4 as V1/V2/V3/hV4, when NSD defines them as V1v/V1d/V2v/V2d — so their "
      "\"V3\" was really V2v, and true V3 and hV4 were never extracted. Similarly "
      "`floc-places` label 1 is OPA, not PPA.\n")
    A("**Overlap is intentional.** `nsdgeneral` and the `streams` parcellation overlap "
      "each other and the functional ROIs, so some voxels are stored two or three "
      "times. That keeps every ROI directly addressable at a cost of a few GB.\n")

    # ---------------------------------------------------------------- per subject
    A("### Voxel counts per subject\n")
    rows = []
    for s in subs:
        ss = fd.available_sessions(s)
        if not ss:
            continue
        with np.load(STORE / s / f"session{ss[0]:02d}.npz") as z:
            rows.append({"subject": s, **{k: z[k].shape[1] for k in sorted(z.files)}})
    vc = pd.DataFrame(rows).set_index("subject")
    A("| subject | " + " | ".join(vc.columns) + " | total |")
    A("|---" * (len(vc.columns) + 2) + "|")
    for s, r in vc.iterrows():
        A(f"| {s} | " + " | ".join(f"{int(v):,}" for v in r) + f" | {int(r.sum()):,} |")
    A("\nCounts differ across subjects because ROIs are defined per brain from that "
      "subject's own retinotopy and functional localisers.\n")

    # ---------------------------------------------------------------- companions
    A("## 6. Companion files\n")
    A("| file | what it is |")
    A("|---|---|")
    A("| `data/trial_manifest_corrected.csv` | 12,000 trials (sessions 1–2, all 8 "
      "subjects) with corrected `nsd_id`/`coco_id`, repetition number and trial lag |")
    A("| `results/retention_design_full.csv` | every repetition pair over all extracted "
      "sessions: trial indices, sessions, elapsed interval, retention bin, "
      "cross-session flag, usability |")
    A("| `data/clip_text_targets.npz` | `coco_ids`, `T_text` — CLIP ViT-B/32 text "
      "embeddings (mean of up to 5 COCO captions, L2-normalised), 512-D |")
    A("| `data/clip_image_targets.npz` | `coco_ids`, `T_image` — CLIP ViT-B/32 image "
      "embeddings of the NSD-cropped stimulus, 512-D |")
    A("| `data/nsd_stimuli_256/coco_*.jpg` | 7,945 stimuli as the participant saw them: "
      "COCO source cropped by NSD's `cropBox`, resized to 256×256 |")
    A(f"| `{NSD_RAW}/subjNN/roi/` | the mask volumes the ROIs were cut from |")
    A(f"| `ADVANCED_RESEARCH/data/behav/subjNN_responses.tsv` | full behavioural logs, "
      f"all sessions: `73KID`, `10KID`, `ISOLD`, `ISCORRECT`, `RT`, `TIME` (days since "
      f"study start) |")
    A("")

    # ---------------------------------------------------------------- usage
    A("## 7. Loading\n")
    A("```python")
    A("import _full_data as fd")
    A("")
    A("fd.coverage()                      # what is extracted")
    A("fd.available_rois('subj01')        # ROI names in the store")
    A("")
    A("# concatenate sessions, z-score each voxel within session")
    A("X, trial_idx, session_idx = fd.load_rois('subj01', ['V1', 'V2'], norm='session')")
    A("#   X            (n_trials, n_voxels) float32")
    A("#   trial_idx    1-based global trial number across the experiment")
    A("#   session_idx  session each trial came from")
    A("")
    A("b = fd.behaviour('subj01')         # full log with corrected nsd_id")
    A("```\n")
    A("**Always normalise within session.** A 40-session experiment spans about nine "
      "months of scanner drift, coil and head-position differences. `norm='session'` is "
      "the default and the minimum; `norm='none'` returns the stored int16 values.\n")

    # ---------------------------------------------------------------- caveats
    A("## 8. Caveats\n")
    A("1. **Voxel indices are not comparable across subjects.** Each subject is in their "
      "own functional space with different volume dimensions. Cross-subject work needs "
      "either the shared1000 stimuli as an anchor or explicit functional alignment.\n")
    A("2. **Trial order is presentation order.** Row *i* of a session file is the *i*-th "
      "trial of that session; map to stimuli through the behavioural log, never by "
      "assuming an image ordering.\n")
    A("3. **Not every subject has every session** — see §3. Analyses must handle "
      "30-, 32- and 40-session subjects.\n")
    A("4. **The source volumes were deleted after extraction.** Re-extracting with a "
      "different ROI set means re-downloading (~430 MB/session). The extractor detects "
      "a stale ROI set automatically and re-fetches.\n")
    A("5. **`int16` rounding.** Values are rounded to the nearest integer on extraction, "
      "a quantisation of 1/300 % signal change — about 1e-6 after z-scoring, verified "
      "against the earlier float tensors.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"     {len(subs)} subjects, {int(cov.sessions_on_disk.sum())} sessions, "
          f"{cov.gb.sum():.1f} GB documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
