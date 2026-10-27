# Shared visual representation along the ventral hierarchy

Analysis code, derived result tables, and manuscript source for:

> **Kaila, D., & Aggarwal, M.** (2026). *Shared visual representation increases
> along the ventral hierarchy for semantic but not low-level features.*
> Manuscript in preparation.

---

## ⚠️ How to cite

**This work is licensed under [CC BY 4.0](LICENSE). Attribution is a condition of
the licence, not a courtesy.** If you use this code, these result tables, or the
figures — in a paper, a preprint, a thesis, a talk, a blog post, or a derived
dataset — you must credit the authors and indicate what you changed.

Please cite as:

```
Kaila, D., & Aggarwal, M. (2026). Shared visual representation increases along
the ventral hierarchy for semantic but not low-level features. Manuscript in
preparation. https://github.com/hubdk17/Memory_nsd
```

BibTeX:

```bibtex
@unpublished{kaila2026shared,
  author = {Kaila, Daksh and Aggarwal, Monika},
  title  = {Shared visual representation increases along the ventral hierarchy
            for semantic but not low-level features},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/hubdk17/Memory_nsd}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff) — GitHub's
"Cite this repository" button reads it.

If the manuscript is published, please cite the published version instead; this
README will be updated with the reference.

**Commercial or large-scale reuse:** permitted under CC BY 4.0 with attribution,
but please contact the corresponding author first so we can point you at the
current version — `dkaila_be25@thapar.edu`.

---

## What this repository does *not* contain

**No Natural Scenes Dataset data is redistributed here.** NSD is available from
[naturalscenesdataset.org](https://naturalscenesdataset.org) under its own terms
of use, which you must accept separately. This repository excludes:

- NSD beta values (`roi_betas_full/`)
- NSD stimulus images (`nsd_stimuli_256/`)
- Image feature embeddings derived from those stimuli (CLIP, GIST — ~360 MB)

The scripts read these from a local copy. Paths are set at the top of
[`code/_full_data.py`](code/_full_data.py).

What *is* included are **aggregate derived statistics** — per-region,
per-participant and per-image similarity values — which is what the manuscript's
tables and figures are built from.

---

## What the paper reports

Per-participant decoders map voxels into a common feature space; we ask how much
of one participant's representation is recoverable from the other seven, against
a noise ceiling built from **two decoders trained on disjoint halves** of that
participant's images, so decoder-specific error cancels on both sides of the
ratio.

| shared fraction (V1 → ventral stream) | V1 | ventral stream | gap |
|---|---|---|---|
| semantic target (CLIP) | 0.835 | 0.948 | **+0.113** (p = 1.6×10⁻⁴, 8/8 participants) |
| low-level target (GIST) | 0.889 | 0.883 | **−0.006** (p = 0.89, equivalence supported) |
| CLIP, voxel-count matched | 0.839 | 0.932 | **+0.093** (p = 0.022) |

The gradient holds for semantic content and is **absent** for low-level content
in the same voxels (region × target interaction p = 0.020). Where visual
representation is individual depends on the feature space it is measured in.

Two methodological points worth reusing independently:

1. A noise ceiling that compares **one** decoder's output across two
   presentations is inflated, because image-specific decoder error is common to
   both sides. Here that inflation accounted for about **half** the apparent
   effect, and was largest in early visual cortex.
2. Gradients of this kind should state the feature space they were measured in.

---

## Layout

```
code/                 analysis scripts and helpers
manuscript/           LaTeX source, figures, table/figure generators, validators
results/              derived result tables (CSV) the manuscript is built from
DATASET.md            notes on the dataset and trial indexing
```

### Scripts behind the paper's results

| script | produces |
|---|---|
| `code/_full_data.py` | beta/ROI loading, local data paths |
| `code/_fast_ridge.py` | ridge solver used by the decoders |
| `code/step41_gist_features.py` | 512-d GIST descriptors (low-level target) |
| `code/step42_decoder_confound_controls.py` | split-half ceiling, GIST target, voxel-matched controls |

Other `step*.py` files are earlier analyses from the same project, kept for
transparency; they are not all inputs to this manuscript.

### Regenerating the manuscript

```bash
cd manuscript
python make_tables.py      # tables from results/*.csv
python make_figures.py     # figures from results/*.csv
python check_tex.py        # validates every number against its source table
python wordcount.py        # abstract / main-text limits
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`check_tex.py` re-runs the table generators and compares their output against the
typeset text, so no number reaches the manuscript by transcription. It validates
citation *keys*, not bibliographic content — bibliography entries here were
resolved against Crossref separately.

---

## Status

Manuscript in preparation. `results/decoder_controls_by_subject.csv` is being
extended from the pre-specified ventral sequence to all 16 regions; the five
regions of that sequence (V1, V2, hV4, mid-ventral, ventral stream) are complete
and are what the numbers above rest on. See `results/step42_control_stats.txt`
for the full statistics.

## Licence

[CC BY 4.0](LICENSE) — Copyright © 2026 Daksh Kaila and Monika Aggarwal.

Note on code: CC BY 4.0 is applied to this repository as a whole, including the
scripts. It is not an OSI-approved software licence and grants no patent rights.
It is used deliberately here because attribution is the authors' condition for
reuse. If you need different terms for the code, contact the corresponding
author.
