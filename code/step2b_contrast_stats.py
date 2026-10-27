"""Step 2b: paired repetition contrasts across the 8 subjects, from step2's output."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
r = pd.read_csv(HERE / "results" / "ridge_clip_text_retrieval_by_repetition.csv")

rows = []
for analysis, (a_rep, b_rep) in [("lifecycle93", (1, 2)), ("lifecycle93", (1, 3)),
                                 ("repeated323", (1, 2))]:
    for norm in ["session", "repgroup"]:
        sub = r[(r.analysis == analysis) & (r.norm == norm)]
        for roi, g in sub.groupby("roi"):
            a = g[g.repetition == a_rep].set_index("subject_id").two_way
            b = g[g.repetition == b_rep].set_index("subject_id").two_way.reindex(a.index)
            d = (b - a).values
            t, p = stats.ttest_rel(b.values, a.values)
            rows.append(dict(analysis=analysis, norm=norm,
                             contrast=f"Rep{b_rep}-Rep{a_rep}", roi=roi,
                             mean_early=a.mean(), mean_late=b.mean(), delta=d.mean(),
                             sem=d.std(ddof=1) / np.sqrt(len(d)), t=t, p=p,
                             n_subj_negative=int((d < 0).sum()), n_subj=len(d)))

out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "repetition_contrast_stats.csv", index=False)
pd.set_option("display.width", 220)
for (an, norm, c), g in out.groupby(["analysis", "norm", "contrast"]):
    if norm != "repgroup":
        continue
    print(f"### {an}  {c}  (norm=repgroup, paired across N=8)")
    print(g[["roi", "mean_early", "mean_late", "delta", "sem", "t", "p",
             "n_subj_negative"]].sort_values("p").round(5).to_string(index=False))
    print()
