"""Step 39: build the design for "does a typical representation remember better?"

DESIGN LOGIC
------------
The transfer score from step38 is computed from presentations 1 and 2 of each
shared image. Predicting recognition AT presentation 2 would therefore use the
same trial on both sides: a moment of good attention produces both a cleaner
brain pattern and a more likely correct response, which would manufacture an
association out of nothing.

We avoid this entirely by predicting recognition at PRESENTATION 3, which step38
never touches. The design is then properly ordered in time:

    encoding (pres 1 + 2)  ->  transfer score
                           ->  recognition at pres 3   [outcome]

UNIT OF ANALYSIS
----------------
One row per subject x image. The outcome is a single behavioural response, NOT
duplicated across ROIs; ROI transfer scores enter as separate columns. This is a
deliberate guard against inflating n by the number of regions.

COVARIATES
----------
  lag          time since the previous presentation; drives accuracy strongly
  memorability intrinsic memorability of the image, estimated leave-one-subject
               -out so it cannot absorb the effect we are testing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _full_data as fd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "memory_design.csv"
TARGET_OCC = 2            # 0-indexed: the third presentation

problems: list[str] = []


def main() -> int:
    tr = pd.read_csv(HERE / "results" / "brain_embedding_loso.csv")
    rois = sorted(tr.roi.unique())
    shared = set(tr.nsd_id.unique())
    print(f"transfer scores: {len(tr)} rows, {len(rois)} ROIs, "
          f"{len(shared)} images")

    # ---------------------------------------------------------- behaviour
    beh = []
    for s in sorted(tr.subject_id.unique()):
        b = fd.behaviour(s)
        b = b[b.nsd_id.isin(shared)].sort_values("TIME").copy()
        b["occ"] = b.groupby("nsd_id").cumcount()
        t = b[b.occ == TARGET_OCC].copy()
        beh.append(t[["SUBJECT", "nsd_id", "ISCORRECT", "ISOLD", "RT",
                      "MEMORYRECENT", "MEMORYFIRST", "SESSION"]]
                   .assign(subject_id=s))
    beh = pd.concat(beh, ignore_index=True)
    print(f"presentation-{TARGET_OCC + 1} trials: {len(beh)}")

    if not (beh.ISOLD == 1).all():
        problems.append(f"{int((beh.ISOLD != 1).sum())} target trials are not "
                        f"marked ISOLD=1")
    miss = beh.ISCORRECT.isna().sum()
    if miss:
        print(f"  dropping {miss} trials with missing ISCORRECT")
        beh = beh.dropna(subset=["ISCORRECT"])

    # ------------------------------------------- leave-one-subject-out memorability
    # Intrinsic memorability of each image, from everyone EXCEPT this subject,
    # so the covariate cannot launder this subject's own outcome.
    tot = beh.groupby("nsd_id")["ISCORRECT"].agg(["sum", "count"])
    beh = beh.merge(tot, left_on="nsd_id", right_index=True, how="left")
    beh["memorability"] = (beh["sum"] - beh.ISCORRECT) / (beh["count"] - 1)
    n_single = int((beh["count"] == 1).sum())
    if n_single:
        print(f"  {n_single} images seen at pres-3 by only one subject -> "
              f"memorability undefined, dropped")
        beh = beh[beh["count"] > 1]
    beh = beh.drop(columns=["sum", "count"])

    # ---------------------------------------------------------- transfer, wide
    wide = tr.pivot_table(index=["subject_id", "nsd_id"], columns="roi",
                          values="transfer").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={r: f"tr_{r}" for r in rois})

    d = beh.merge(wide, on=["subject_id", "nsd_id"], how="inner")
    print(f"after join to transfer scores: {len(d)} rows")
    if len(d) != len(beh):
        problems.append(f"join lost {len(beh) - len(d)} behavioural rows")

    tr_cols = [f"tr_{r}" for r in rois]
    if d[tr_cols].isna().any().any():
        problems.append("missing transfer scores after join")

    # ---------------------------------------------------------- covariates
    d["log_lag"] = np.log10(d.MEMORYRECENT.clip(lower=1))
    d["category"] = d.nsd_id.map(
        tr.drop_duplicates("nsd_id").set_index("nsd_id")["category"])

    # z-score predictors WITHIN subject so per-subject betas are comparable
    # and so between-subject differences in level cannot drive the effect.
    for c in tr_cols + ["log_lag", "memorability"]:
        d[f"z_{c}"] = d.groupby("subject_id")[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))

    d["ISCORRECT"] = d.ISCORRECT.astype(int)
    d = d.sort_values(["subject_id", "nsd_id"]).reset_index(drop=True)
    d.to_csv(OUT, index=False)

    # ---------------------------------------------------------- design table
    print("\n" + "=" * 72)
    print("DESIGN TABLE")
    print("=" * 72)
    g = d.groupby("subject_id").agg(
        trials=("ISCORRECT", "size"),
        hits=("ISCORRECT", "sum"),
        hit_rate=("ISCORRECT", "mean"),
        images=("nsd_id", "nunique"),
        median_lag_min=("MEMORYRECENT", "median"))
    g["misses"] = g.trials - g.hits
    print(g[["trials", "hits", "misses", "hit_rate", "images",
             "median_lag_min"]].round(3).to_string())
    print(f"\n  TOTAL {len(d)} trials | {d.ISCORRECT.sum()} hits | "
          f"{len(d) - d.ISCORRECT.sum()} misses | "
          f"{d.nsd_id.nunique()} images | {d.subject_id.nunique()} subjects")

    if (g.misses < 30).any():
        problems.append(f"subject(s) with <30 misses: "
                        f"{g.index[g.misses < 30].tolist()}")
    if (g.hit_rate > 0.95).any() or (g.hit_rate < 0.05).any():
        problems.append("subject at ceiling or floor")

    print("\n" + "=" * 72)
    print("COLLINEARITY AMONG ROI TRANSFER SCORES")
    print("=" * 72)
    cm = d[tr_cols].corr()
    iu = np.triu_indices_from(cm, k=1)
    v = cm.values[iu]
    print(f"  pairwise r: median {np.median(v):.3f}  "
          f"range {v.min():.3f} to {v.max():.3f}")
    print("  -> ROIs are fitted ONE AT A TIME with FDR across ROIs;")
    print("     a joint 16-predictor model would be uninterpretable.")

    print("\n" + "=" * 72)
    print("COVARIATE CHECK")
    print("=" * 72)
    for c in ("memorability", "log_lag"):
        r = d[[c, "ISCORRECT"]].corr().iloc[0, 1]
        print(f"  {c:14s} vs outcome: r = {r:+.3f}")
    r_mem_tr = d[["memorability"] + tr_cols].corr()["memorability"][tr_cols]
    print(f"  memorability vs transfer: |r| max {r_mem_tr.abs().max():.3f} "
          f"({r_mem_tr.abs().idxmax()})")

    print("\n" + ("[OK] design built, no structural problems"
                  if not problems else f"[FAIL] {len(problems)} problem(s):"))
    for p in problems:
        print(f"   - {p}")
    print(f"\nwritten: {OUT.relative_to(HERE)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
