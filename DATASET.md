# Extracted NSD Dataset — What Is Actually In These Files

Generated 2026-09-29 by `step32_document_dataset.py`, which reads the files rather than describing them from memory.

---

## 1. Provenance

**Source.** Natural Scenes Dataset (Allen et al., 2022, *Nature Neuroscience*), AWS S3 bucket `natural-scenes-dataset`.

**Which NSD product.** `betas_fithrf_GLMdenoise_RR` in `func1pt8mm` space. These are NSD's **fully preprocessed single-trial GLM beta estimates** — motion and distortion corrected, per-voxel HRF fitting, GLMdenoise, ridge regression. They are not raw BOLD timeseries (that is `nsddata_timeseries`, which we never downloaded and do not need).

**Units.** NSD ships betas as `int16` equal to percent signal change × 300. We store them unchanged, so divide by 300 for percent signal change. Normalisation is applied at load time, not baked into the store.

**Spatial resolution.** 1.8 mm isotropic, each subject in their own native functional space. Subjects have different volume dimensions, so voxel indices are **not** comparable across subjects — only stimulus identity is.

**Stimulus identity.** Taken from each subject's behavioural log as `nsdId = 73KID - 1`. This matters: the earlier pipelines in this project looked up a 1-based 73k index in the 0-based `nsdId` column of `nsd_stim_info_merged.csv`, which shifted every stimulus by one image. The relation used here was verified three independent ways and the extractor was regression-tested against the old tensors (max |diff| 1e-6, r = 1.000000).

## 2. Directory layout

```
RECONSTRUCTION_PILOT/data/roi_betas_full/
  subj01/
    session01.npz .. session40.npz   (40 files)
  subj02/
    session01.npz .. session40.npz   (40 files)
  subj03/
    session01.npz .. session32.npz   (32 files)
  subj04/
    session01.npz .. session30.npz   (30 files)
  subj05/
    session01.npz .. session40.npz   (40 files)
  subj06/
    session01.npz .. session32.npz   (32 files)
  subj07/
    session01.npz .. session40.npz   (40 files)
  subj08/
    session01.npz .. session30.npz   (30 files)
```

One file per scan session. Session-wise storage keeps the run resumable and memory bounded; the analysis layer concatenates via `_full_data.load_rois()`.

## 3. Coverage

| subject | sessions on disk | sessions in NSD | trials | ROI sets | size (GB) | complete |
|---|---|---|---|---|---|---|
| subj01 | 40 | 40 | 30,000 | 21 | 2.60 | yes |
| subj02 | 40 | 40 | 30,000 | 21 | 2.41 | yes |
| subj03 | 32 | 32 | 24,000 | 21 | 2.08 | yes |
| subj04 | 30 | 30 | 22,500 | 21 | 1.70 | yes |
| subj05 | 40 | 40 | 30,000 | 21 | 2.31 | yes |
| subj06 | 32 | 32 | 24,000 | 21 | 2.29 | yes |
| subj07 | 40 | 40 | 30,000 | 21 | 2.08 | yes |
| subj08 | 30 | 30 | 22,500 | 21 | 1.92 | yes |

**Total: 284 sessions, 213,000 trials, 17.4 GB.**

Subjects 1, 2, 5 and 7 completed 40 sessions; 3 and 6 completed 32; 4 and 8 completed 30. That is a property of NSD, not of this extraction.

## 4. What is inside one session file

Example: `subj01/session01.npz` — a compressed NumPy archive holding one array per ROI.

```python
import numpy as np
z = np.load('data/roi_betas_full/subj01/session01.npz')
z['V1'].shape      # (750, n_voxels)  -- trials x voxels, int16
z['V1'] / 300.0    # percent signal change
```

**Array axes.** Row = trial, in presentation order within that session (750 trials). Column = voxel, in a fixed order given by the mask's flattened index; the order is stable across sessions for a subject, so columns are directly comparable session to session.

**Dtype.** `int16`. Observed range in this sample: -15476 to 13974 (÷300 → -51.59 to 46.58 % signal change).

## 5. The ROI sets

| ROI | source mask | labels | description |
|---|---|---|---|
| `CA1` | `MTL.nii.gz` | 5 | Cornu Ammonis 1 |
| `DG_CA3` | `MTL.nii.gz` | 7,8 | Dentate gyrus + CA3 |
| `ERC` | `MTL.nii.gz` | 2 | Entorhinal cortex |
| `FFA` | `floc-faces.nii.gz` | 2,3 (FFA-1,FFA-2) | Fusiform face area |
| `OPA` | `floc-places.nii.gz` | 1 | Occipital place area |
| `PHC` | `MTL.nii.gz` | 3 | Parahippocampal cortex |
| `PPA` | `floc-places.nii.gz` | 2 | Parahippocampal place area |
| `PRC` | `MTL.nii.gz` | 1 | Perirhinal cortex |
| `RSC` | `floc-places.nii.gz` | 3 | Retrosplenial cortex |
| `SUB` | `MTL.nii.gz` | 4 | Subiculum |
| `S_early` | `streams.nii.gz` | 1 | Early visual stream |
| `S_lateral` | `streams.nii.gz` | 6 | Lateral stream |
| `S_midlateral` | `streams.nii.gz` | 3 | Mid-lateral stream |
| `S_midventral` | `streams.nii.gz` | 2 | Mid-ventral stream |
| `S_parietal` | `streams.nii.gz` | 7 | Parietal stream |
| `S_ventral` | `streams.nii.gz` | 5 | Ventral stream |
| `V1` | `prf-visualrois.nii.gz` | 1,2 (V1v+V1d) | Primary visual cortex |
| `V2` | `prf-visualrois.nii.gz` | 3,4 (V2v+V2d) | Secondary visual cortex |
| `V3` | `prf-visualrois.nii.gz` | 5,6 (V3v+V3d) | Third visual area |
| `hV4` | `prf-visualrois.nii.gz` | 7 | Human V4, colour/form |
| `nsdgeneral` | `nsdgeneral.nii.gz` | all >0 | NSD's standard responsive-cortex mask |

**Label definitions follow the NSD documentation.** This is worth stating because the earlier pipelines in this project used `prf-visualrois` labels 1/2/3/4 as V1/V2/V3/hV4, when NSD defines them as V1v/V1d/V2v/V2d — so their "V3" was really V2v, and true V3 and hV4 were never extracted. Similarly `floc-places` label 1 is OPA, not PPA.

**Overlap is intentional.** `nsdgeneral` and the `streams` parcellation overlap each other and the functional ROIs, so some voxels are stored two or three times. That keeps every ROI directly addressable at a cost of a few GB.

### Voxel counts per subject

| subject | CA1 | DG_CA3 | ERC | FFA | OPA | PHC | PPA | PRC | RSC | SUB | S_early | S_lateral | S_midlateral | S_midventral | S_parietal | S_ventral | V1 | V2 | V3 | hV4 | nsdgeneral | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subj01 | 242 | 286 | 241 | 794 | 1,611 | 652 | 1,033 | 345 | 566 | 350 | 5,917 | 7,799 | 834 | 986 | 3,548 | 7,604 | 1,350 | 1,433 | 1,187 | 687 | 15,724 | 53,189 |
| subj02 | 182 | 270 | 204 | 869 | 1,381 | 780 | 994 | 342 | 813 | 277 | 4,611 | 6,752 | 781 | 711 | 3,355 | 8,185 | 1,102 | 1,075 | 1,097 | 483 | 14,278 | 48,542 |
| subj03 | 198 | 234 | 204 | 1,093 | 1,330 | 628 | 1,269 | 329 | 838 | 364 | 5,148 | 8,572 | 815 | 884 | 3,885 | 8,167 | 1,254 | 1,141 | 928 | 426 | 15,226 | 52,933 |
| subj04 | 255 | 267 | 249 | 942 | 1,235 | 526 | 960 | 371 | 813 | 400 | 4,054 | 7,131 | 633 | 881 | 3,346 | 7,645 | 877 | 863 | 808 | 475 | 13,153 | 45,884 |
| subj05 | 242 | 286 | 239 | 907 | 1,332 | 709 | 1,221 | 485 | 771 | 466 | 4,494 | 7,037 | 609 | 1,052 | 2,838 | 6,619 | 1,113 | 1,081 | 925 | 542 | 13,039 | 46,007 |
| subj06 | 238 | 266 | 238 | 826 | 1,362 | 576 | 1,229 | 378 | 845 | 458 | 5,860 | 9,030 | 1,107 | 1,169 | 3,374 | 9,666 | 1,127 | 1,180 | 1,201 | 477 | 17,907 | 58,514 |
| subj07 | 159 | 229 | 173 | 484 | 1,083 | 510 | 912 | 316 | 694 | 300 | 4,559 | 6,118 | 628 | 785 | 2,459 | 6,515 | 1,142 | 986 | 726 | 397 | 12,682 | 41,857 |
| subj08 | 239 | 251 | 227 | 1,204 | 1,360 | 832 | 961 | 332 | 799 | 383 | 4,596 | 8,674 | 655 | 905 | 3,390 | 8,350 | 1,074 | 1,033 | 889 | 495 | 14,386 | 51,035 |

Counts differ across subjects because ROIs are defined per brain from that subject's own retinotopy and functional localisers.

## 6. Companion files

| file | what it is |
|---|---|
| `data/trial_manifest_corrected.csv` | 12,000 trials (sessions 1–2, all 8 subjects) with corrected `nsd_id`/`coco_id`, repetition number and trial lag |
| `results/retention_design_full.csv` | every repetition pair over all extracted sessions: trial indices, sessions, elapsed interval, retention bin, cross-session flag, usability |
| `data/clip_text_targets.npz` | `coco_ids`, `T_text` — CLIP ViT-B/32 text embeddings (mean of up to 5 COCO captions, L2-normalised), 512-D |
| `data/clip_image_targets.npz` | `coco_ids`, `T_image` — CLIP ViT-B/32 image embeddings of the NSD-cropped stimulus, 512-D |
| `data/nsd_stimuli_256/coco_*.jpg` | 7,945 stimuli as the participant saw them: COCO source cropped by NSD's `cropBox`, resized to 256×256 |
| `D:\Desktop\FMRI_NSD\data\raw/subjNN/roi/` | the mask volumes the ROIs were cut from |
| `ADVANCED_RESEARCH/data/behav/subjNN_responses.tsv` | full behavioural logs, all sessions: `73KID`, `10KID`, `ISOLD`, `ISCORRECT`, `RT`, `TIME` (days since study start) |

## 7. Loading

```python
import _full_data as fd

fd.coverage()                      # what is extracted
fd.available_rois('subj01')        # ROI names in the store

# concatenate sessions, z-score each voxel within session
X, trial_idx, session_idx = fd.load_rois('subj01', ['V1', 'V2'], norm='session')
#   X            (n_trials, n_voxels) float32
#   trial_idx    1-based global trial number across the experiment
#   session_idx  session each trial came from

b = fd.behaviour('subj01')         # full log with corrected nsd_id
```

**Always normalise within session.** A 40-session experiment spans about nine months of scanner drift, coil and head-position differences. `norm='session'` is the default and the minimum; `norm='none'` returns the stored int16 values.

## 8. Caveats

1. **Voxel indices are not comparable across subjects.** Each subject is in their own functional space with different volume dimensions. Cross-subject work needs either the shared1000 stimuli as an anchor or explicit functional alignment.

2. **Trial order is presentation order.** Row *i* of a session file is the *i*-th trial of that session; map to stimuli through the behavioural log, never by assuming an image ordering.

3. **Not every subject has every session** — see §3. Analyses must handle 30-, 32- and 40-session subjects.

4. **The source volumes were deleted after extraction.** Re-extracting with a different ROI set means re-downloading (~430 MB/session). The extractor detects a stale ROI set automatically and re-fetches.

5. **`int16` rounding.** Values are rounded to the nearest integer on extraction, a quantisation of 1/300 % signal change — about 1e-6 after z-scoring, verified against the earlier float tensors.
