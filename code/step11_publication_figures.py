"""
Step 11: publication figures. Every figure is written to results/figures/ as both PDF
(vector, for the manuscript) and PNG at 300 DPI (for review).

Colour policy, so the figures read as one system:
  - Anatomical tier is the only categorical encoding, and it has THREE levels, which is
    the maximum that clears the all-pairs colour-blind separation floor. Validated with
    the dataviz palette validator (worst all-pairs CVD dE 9.2, normal-vision dE 24.0).
    Aqua sits below 3:1 against the surface, so every marked category also carries a
    visible text label -- the required relief.
  - Repetition number is ORDINAL, so it gets a single-hue light-to-dark ramp rather than
    categorical hues (validated: monotone L, adjacent dL >= 0.06, light end 2.06:1).
  - The accuracy heatmap is a continuous magnitude, so one blue hue, light to dark.
  - No figure uses two y-scales.

Each figure is independent: a missing input CSV is reported and skipped rather than
killing the run.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "legend.frameon": False,
})

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
TIER_COLOR = {"Early visual": "#2a78d6", "Higher visual": "#eb6834", "MTL": "#1baf7a"}
REP_RAMP = {1: "#86b6ef", 2: "#3987e5", 3: "#184f95"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

TIER = {"V1": "Early visual", "V2": "Early visual",
        "FFA": "Higher visual", "PPA": "Higher visual", "RSC": "Higher visual",
        "PHC": "Higher visual",
        "PRC": "MTL", "ERC": "MTL", "SUB": "MTL", "CA1": "MTL", "DG_CA3": "MTL",
        "EarlyVis": "Early visual", "SceneCtx": "Higher visual",
        "AllCortex": "Higher visual", "MTL": "MTL"}
ROI_ORDER = ["V1", "V2", "EarlyVis", "FFA", "PPA", "RSC", "PHC", "SceneCtx",
             "AllCortex", "PRC", "ERC", "SUB", "CA1", "DG_CA3", "MTL"]
ANAT_ORDER = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"  [OK] {name}.pdf / .png")


def read(name):
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"{name} not found -- run the step that produces it")
    return pd.read_csv(p)


def tidy(ax, ylabel=None, xlabel=None, title=None, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e6e5e1", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", color=INK)


def stars(p_unc, fdr_sig):
    s = "**" if p_unc < 0.01 else "*" if p_unc < 0.05 else ""
    return s + ("†" if fdr_sig else "")


def tier_legend(ax, loc="upper right"):
    h = [Line2D([], [], marker="s", linestyle="none", markersize=7, color=c, label=k)
         for k, c in TIER_COLOR.items()]
    ax.legend(handles=h, loc=loc, fontsize=8, handletextpad=0.4, labelspacing=0.3)


# ----------------------------------------------------------------- Figure 2
def figure2():
    r = read("ridge_clip_text_retrieval_by_repetition.csv")
    st = read("statistical_infrastructure.csv")
    fig = plt.figure(figsize=(10.5, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 1.15], hspace=0.48)

    # --- A: Rep1 vs Rep2, repeated323 / repgroup, with per-subject lines
    ax = fig.add_subplot(gs[0])
    sub = r[(r.analysis == "repeated323") & (r.norm == "repgroup")]
    rois = [x for x in ROI_ORDER if x in set(sub.roi)]
    w = 0.34
    for i, roi in enumerate(rois):
        g = sub[sub.roi == roi]
        for rep in (1, 2):
            v = g[g.repetition == rep].two_way
            ax.bar(i + (rep - 1.5) * w, v.mean(), w * 0.92, color=REP_RAMP[rep],
                   edgecolor=SURFACE, linewidth=1.2, zorder=2,
                   label=f"Rep{rep}" if i == 0 else None)
        p1 = g[g.repetition == 1].set_index("subject_id").two_way.sort_index()
        p2 = g[g.repetition == 2].set_index("subject_id").two_way.reindex(p1.index)
        for a, b in zip(p1.values, p2.values):
            ax.plot([i - 0.5 * w, i + 0.5 * w], [a, b], color=MUTED, linewidth=0.55,
                    alpha=0.75, zorder=3)
            ax.plot([i - 0.5 * w, i + 0.5 * w], [a, b], "o", color=INK2, markersize=2.0,
                    zorder=4)
        row = st[(st.roi == roi) & (st.norm == "repgroup")
                 & (st.analysis == "repeated323") & (st.contrast == "Rep2-Rep1")]
        if len(row):
            lab = stars(float(row.p_parametric.iloc[0]), bool(row.fdr_significant.iloc[0]))
            if lab:
                top = max(p1.max(), p2.max())
                ax.text(i, top + 0.004, lab, ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.45, 0.5015, "chance", fontsize=7.5, color=INK2, va="bottom", ha="left")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=38, ha="right", fontsize=8.5)
    for t, roi in zip(ax.get_xticklabels(), rois):
        t.set_color(TIER_COLOR[TIER[roi]])
    ax.set_ylim(0.49, None)
    tidy(ax, "Two-way identification", title="A   Rep1 vs Rep2 (323 stimuli, per-repetition z-scored)")
    n_fdr = int(st[(st.norm == "repgroup") & (st.analysis == "repeated323")
                   & (st.contrast == "Rep2-Rep1")].fdr_significant.sum())
    ax.text(0.0, -0.42, "* p < 0.05, ** p < 0.01, uncorrected; † survives FDR.  "
            + (f"{n_fdr} of 15 ROI sets survive FDR." if n_fdr else
               "No ROI set survives FDR correction across the 15 sets."),
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # --- B: lifecycle trajectory
    ax = fig.add_subplot(gs[1])
    life = r[(r.analysis == "lifecycle93") & (r.norm == "repgroup")]
    show = ["AllCortex", "SceneCtx", "RSC", "PHC", "FFA", "EarlyVis", "MTL"]
    show = [s for s in show if s in set(life.roi)]
    ends = []
    for roi in show:
        g = life[life.roi == roi].groupby("repetition").two_way
        mu, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
        c = TIER_COLOR[TIER[roi]]
        ax.plot(mu.index, mu.values, "-o", color=c, linewidth=2.0, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.14, linewidth=0,
                        zorder=2)
        ends.append([mu.values[-1], roi, c])
    # push apart end labels that would otherwise overprint each other
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = span * 0.055 or 0.002
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for ypos, roi, c in ends:
        ax.annotate(roi, (3.06, ypos), fontsize=8, color=c, va="center")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Rep1", "Rep2", "Rep3"])
    ax.set_xlim(0.72, 3.62)
    tidy(ax, "Two-way identification",
         title="B   Full lifecycle trajectory (93 stimuli, mean ± SEM across 8 subjects)")
    h = [Line2D([], [], marker="s", linestyle="none", markersize=7, color=c, label=k)
         for k, c in TIER_COLOR.items()]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=8, handletextpad=0.4, columnspacing=1.6)

    # --- C: forest plot of Cohen's d
    ax = fig.add_subplot(gs[2])
    d = st[(st.norm == "repgroup") & (st.analysis == "repeated323")
           & (st.contrast == "Rep2-Rep1")].set_index("roi").reindex(rois).dropna(
        subset=["cohens_d"])
    y = np.arange(len(d))[::-1]
    for yy, (roi, row) in zip(y, d.iterrows()):
        c = TIER_COLOR[TIER[roi]]
        ax.plot([row.d_ci_lower, row.d_ci_upper], [yy, yy], color=c, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot(row.cohens_d, yy, "o", color=c, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=8.5)
    for t, roi in zip(ax.get_yticklabels(), d.index):
        t.set_color(TIER_COLOR[TIER[roi]])
    tidy(ax, xlabel="Paired Cohen's d (Rep2 − Rep1), 95% CI", grid_axis="x",
         title="C   Effect size. Negative = accuracy falls with repetition")
    save(fig, "fig2_repetition_effect")


# ----------------------------------------------------------------- Figure 3
def figure3():
    dis = read("text_vs_image_dissociation_stats.csv")
    d = dis[(dis.norm == "repgroup") & (dis.analysis == "repeated323")]
    # A proportional loss divides by (Rep1 - 0.5). Regions decoding at chance have a
    # near-zero denominator and produce meaningless ratios, so they are dropped here
    # rather than plotted as if comparable.
    if "baseline_ok" in d.columns:
        dropped = sorted(d.loc[~d.baseline_ok.astype(bool), "roi"])
        d = d[d.baseline_ok.astype(bool)]
    else:
        dropped = []
    d = d.set_index("roi").reindex([x for x in ROI_ORDER if x in set(d.roi)]).dropna(
        subset=["mean_prop_text"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),
                             gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28})

    ax = axes[0]
    x = np.arange(len(d))
    w = 0.34
    ax.bar(x - w / 2, d.mean_prop_text, w * 0.92, color="#2a78d6", edgecolor=SURFACE,
           linewidth=1.2, label="CLIP text target", zorder=2)
    ax.bar(x + w / 2, d.mean_prop_image, w * 0.92, color="#eb6834", edgecolor=SURFACE,
           linewidth=1.2, label="CLIP image target", zorder=2)
    for i, (roi, row) in enumerate(d.iterrows()):
        if pd.notna(row.p_val) and row.p_val < 0.05:
            lo = min(row.mean_prop_text, row.mean_prop_image)
            ax.text(i, lo - 0.012, stars(row.p_val, bool(row.p_fdr < 0.05)),
                    ha="center", va="top", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=38, ha="right", fontsize=8.5)
    for t, roi in zip(ax.get_xticklabels(), d.index):
        t.set_color(TIER_COLOR[TIER[roi]])
    tidy(ax, "Proportional loss (Rep2−Rep1) / (Rep1−0.5)",
         title="A   Proportional loss by target space")
    ax.legend(fontsize=8, loc="lower left")
    if dropped:
        ax.text(0.0, -0.32, "excluded (Rep1 at chance, ratio undefined): "
                + ", ".join(dropped), transform=ax.transAxes, fontsize=7,
                color=MUTED, va="top")

    ax = axes[1]
    pts = [(row.mean_prop_image, row.mean_prop_text, roi) for roi, row in d.iterrows()]
    for x0, y0, roi in pts:
        ax.plot(x0, y0, "o", color=TIER_COLOR[TIER[roi]], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    lim = [min(d.mean_prop_image.min(), d.mean_prop_text.min()) - 0.025,
           max(d.mean_prop_image.max(), d.mean_prop_text.max()) + 0.035]
    span = lim[1] - lim[0]
    # nudge labels apart: order by y, then enforce a minimum vertical separation
    placed = []
    for x0, y0, roi in sorted(pts, key=lambda t: t[1]):
        ly = y0
        while any(abs(ly - py) < span * 0.045 and abs(x0 - px) < span * 0.22
                  for px, py in placed):
            ly += span * 0.045
        placed.append((x0, ly))
        ax.annotate(roi, (x0, y0), xytext=(6, (ly - y0) / span * 100 * 0.9 + 3),
                    textcoords="offset points", fontsize=7.5,
                    color=TIER_COLOR[TIER[roi]], va="center")
    ax.plot(lim, lim, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(lim[0] + span * 0.02, lim[1] - span * 0.02,
            "dashed: equal proportional loss (uniform SNR)\nabove it: text degrades less",
            fontsize=7.5, color=INK2, ha="left", va="top", linespacing=1.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    tidy(ax, "Proportional loss, CLIP text", "Proportional loss, CLIP image",
         title="B   Every interpretable ROI sits on or below the line", grid_axis="both")
    tier_legend(ax, "lower right")
    save(fig, "fig3_text_vs_image_dissociation")


# ----------------------------------------------------------------- Figure 4
def figure4():
    fig = plt.figure(figsize=(11.5, 11.6))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.3,
                          height_ratios=[1.0, 1.0, 0.85])

    # A: Control B, early vs late on novel trials
    ax = fig.add_subplot(gs[0, :])
    try:
        b = read("controlB_extended.csv")
    except FileNotFoundError:
        b = read("controlB_time_on_task.csv")
    rois = [x for x in ROI_ORDER if x in set(b.roi)]
    w = 0.34
    for i, roi in enumerate(rois):
        g = b[b.roi == roi]
        ax.bar(i - w / 2, g.acc_early.mean(), w * 0.92, color="#86b6ef",
               edgecolor=SURFACE, linewidth=1.2, zorder=2,
               label="early half" if i == 0 else None)
        ax.bar(i + w / 2, g.acc_late.mean(), w * 0.92, color="#184f95",
               edgecolor=SURFACE, linewidth=1.2, zorder=2,
               label="late half" if i == 0 else None)
        for a, c in zip(g.acc_early, g.acc_late):
            ax.plot([i - w / 2, i + w / 2], [a, c], color=MUTED, linewidth=0.5,
                    alpha=0.6, zorder=3)
            ax.plot([i - w / 2, i + w / 2], [a, c], "o", color=INK2, markersize=1.8,
                    zorder=4)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.45, 0.5015, "chance", fontsize=7.5, color=INK2, va="bottom", ha="left")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=38, ha="right", fontsize=8.5)
    for t, roi in zip(ax.get_xticklabels(), rois):
        t.set_color(TIER_COLOR[TIER[roi]])
    ax.set_ylim(0.49, None)
    tidy(ax, "Two-way identification",
         title="A   Control B: novel trials only, early vs late half of the sequence")
    note = ("The gain is ~+0.015 in every region, including those decoding at chance "
            "(PRC, SUB, CA1, DG_CA3).")
    try:
        cn = read("controlB_null_diagnostic_stats.csv")
        pv = cn.pivot_table(index="roi", columns="condition", values="delta")
        frac = 100 * pv["shuffle_rows"].mean() / pv["observed"].mean()
        note += (f"  A null with the voxel rows shuffled reproduces {frac:.0f}% of it "
                 f"(panel D), so the comparison is biased.")
    except FileNotFoundError:
        pass
    ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="top")
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # B: proportional-SNR residuals
    ax = fig.add_subplot(gs[1, 0])
    try:
        st = read("proportional_snr_residual_stats.csv")
        s = st[(st.norm == "repgroup") & (st.analysis == "repeated323")]
        s = s.set_index("roi").reindex([x for x in ROI_ORDER if x in set(s.roi)]).dropna(
            subset=["mean_residual"])
        y = np.arange(len(s))[::-1]
        for yy, (roi, row) in zip(y, s.iterrows()):
            c = TIER_COLOR[TIER[roi]]
            ci = 1.96 * row.sem_residual
            ax.plot([row.mean_residual - ci, row.mean_residual + ci], [yy, yy], color=c,
                    linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.plot(row.mean_residual, yy, "o", color=c, markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s.index, fontsize=8)
        for t, roi in zip(ax.get_yticklabels(), s.index):
            t.set_color(TIER_COLOR[TIER[roi]])
        tidy(ax, xlabel="Residual after removing the proportional component",
             grid_axis="x", title="B   Departure from the uniform-SNR model")
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8, wrap=True)
        ax.axis("off")

    # C: training-size scaling
    ax = fig.add_subplot(gs[1, 1])
    sc = read("training_size_scaling.csv")
    for roi, g in sc.groupby("roi"):
        m = g.groupby("n_train").two_way
        mu, se = m.mean(), m.std(ddof=1) / np.sqrt(m.count())
        c = TIER_COLOR[TIER.get(roi, "Higher visual")]
        ax.plot(mu.index, mu.values, "-o", color=c, linewidth=2.0, markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.14, linewidth=0,
                        zorder=2)
        ax.annotate(roi, (mu.index[-1], mu.values[-1]), fontsize=8, color=c,
                    xytext=(5, -1), textcoords="offset points", va="center")
    ax.set_xscale("log")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(85, 1350)
    tidy(ax, "Two-way identification", "Training trials (log scale)",
         title="C   Training-size scaling", grid_axis="both")

    # D: Control B against its nulls
    ax = fig.add_subplot(gs[2, :])
    try:
        cn = read("controlB_null_diagnostic_stats.csv")
        conds = [("observed", "#184f95", "observed"),
                 ("shuffle_rows", "#eb6834", "null: voxel rows shuffled"),
                 ("perm_targets", "#1baf7a", "null: targets permuted")]
        rois = [r for r in ROI_ORDER if r in set(cn.roi)]
        w = 0.26
        for k, (cond, col, lab) in enumerate(conds):
            for i, roi in enumerate(rois):
                row = cn[(cn.roi == roi) & (cn.condition == cond)]
                if not len(row):
                    continue
                # note: row.sem would resolve to DataFrame.sem(), hence the bracket form
                v, e = float(row["delta"].iloc[0]), 1.96 * float(row["sem"].iloc[0])
                ax.bar(i + (k - 1) * w, v, w * 0.9, color=col, edgecolor=SURFACE,
                       linewidth=1.2, zorder=2, label=lab if i == 0 else None)
                ax.plot([i + (k - 1) * w] * 2, [v - e, v + e], color=INK2,
                        linewidth=1.0, zorder=3)
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, fontsize=9)
        for t, roi in zip(ax.get_xticklabels(), rois):
            t.set_color(TIER_COLOR[TIER[roi]])
        tidy(ax, "Early → late change",
             title="D   Control B fails its own null: shuffling the voxel rows "
                   "reproduces the effect")
        ax.legend(fontsize=8, loc="upper right", ncol=3)
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
        ax.axis("off")
    save(fig, "fig4_confound_controls")


# ----------------------------------------------------------------- Figure 5
def figure5():
    per = read("rsa_decoding_crosslink_stats.csv")
    wi = read("rsa_decoding_within_subject.csv")
    xl = read("rsa_decoding_crosslink.csv")
    space = "S_text"
    cl = xl[(xl.semantic_space == space)] if "semantic_space" in xl else xl
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),
                             gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.3})

    ax = axes[0]
    agg = cl.groupby("roi")[["convergence", "delta_decode"]].mean()
    for roi, row in agg.iterrows():
        c = TIER_COLOR[TIER.get(roi, "MTL")]
        ax.plot(row.convergence, row.delta_decode, "o", color=c, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.annotate(roi, (row.convergence, row.delta_decode), fontsize=7.5, color=c,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    sub = ""
    if len(agg) > 2:
        from scipy import stats as _s
        rr = _s.spearmanr(agg.convergence, agg.delta_decode)
        sub = (f"  ({len(agg)} ROI means: ρ = {rr.statistic:.2f}, "
               f"p = {rr.pvalue:.3f})")
    tidy(ax, "Decoding change (Rep2 − Rep1)",
         f"RSA semantic convergence, {space} (Rep2 − Rep1)", grid_axis="both",
         title="A   RSA convergence vs decoding change" + sub)
    tier_legend(ax, "lower right")

    ax = axes[1]
    w = wi[(wi.semantic_space == space) & (wi.source == "corrected")] \
        if "source" in wi else wi
    if len(w):
        ww = w.sort_values("spearman_rho").reset_index(drop=True)
        xj = np.linspace(-0.55, 0.55, len(ww))
        ax.plot(xj, ww.spearman_rho, "o", color="#2a78d6", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.plot([-0.75, 0.75], [ww.spearman_rho.mean()] * 2, color=INK, linewidth=2.0,
                zorder=4)
        ax.text(0.78, ww.spearman_rho.mean(), "mean", fontsize=7.5, color=INK,
                va="center")
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.95, 1.15)
        ax.set_xticks([])
        from scipy import stats as _s
        t, p = _s.ttest_1samp(ww.spearman_rho, 0.0)
        tidy(ax, "Within-subject Spearman ρ across ROIs",
             title="B   Per-subject bridge")
        ax.text(0.03, 0.04, f"N = {len(ww)}, mean = {ww.spearman_rho.mean():+.2f}, "
                            f"p = {p:.3f}", transform=ax.transAxes, fontsize=8,
                color=INK2, va="bottom")
        for xx, yy, s in zip(xj, ww.spearman_rho, ww.subject_id):
            ax.annotate(s.replace("subj", "S"), (xx, yy), fontsize=7, color=MUTED,
                        ha="center", va="bottom", xytext=(0, 7),
                        textcoords="offset points")
    save(fig, "fig5_rsa_decoding_bridge")


# ----------------------------------------------------------------- Supplements
def figureS1():
    d = read("offbyone_target_demonstration.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    rois = sorted(d.roi.unique())
    w = 0.34
    for i, roi in enumerate(rois):
        g = d[d.roi == roi]
        for k, (lab, col) in enumerate([("correct", "#2a78d6"), ("offby1", "#eb6834")]):
            v = g[g.targets == lab].two_way
            ax.bar(i + (k - 0.5) * w, v.mean(), w * 0.92, color=col, edgecolor=SURFACE,
                   linewidth=1.2, zorder=2,
                   label={"correct": "corrected mapping",
                          "offby1": "off-by-one mapping"}[lab] if i == 0 else None)
            ax.plot([i + (k - 0.5) * w] * len(v), v, "o", color=INK2, markersize=2.2,
                    zorder=3)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(rois) - 0.45, 0.502, "chance", fontsize=7.5, color=INK2, va="bottom",
            ha="right")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, fontsize=9)
    ax.set_ylim(0.49, None)
    tidy(ax, "Two-way identification",
         title="S1   Stimulus-index validation")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "figS1_offbyone_validation")


def figureS2():
    r = read("ridge_clip_text_retrieval_by_repetition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    rois = [x for x in ROI_ORDER if x in set(r.roi)]
    for ax, norm in zip(axes, ["session", "repgroup"]):
        cells, labels = [], []
        for an, tag, reps in [("repeated323", "323 stim", (1, 2)),
                              ("lifecycle93", "93 stim", (1, 2, 3))]:
            for rep in reps:
                s = r[(r.norm == norm) & (r.analysis == an) & (r.repetition == rep)]
                cells.append([s[s.roi == roi].two_way.mean() for roi in rois])
                labels.append(f"{tag}\nRep{rep}")
        M = np.array(cells)
        im = ax.imshow(M, cmap=SEQ_BLUE, aspect="auto", vmin=0.50, vmax=0.69)
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=38, ha="right", fontsize=8)
        for t, roi in zip(ax.get_xticklabels(), rois):
            t.set_color(TIER_COLOR[TIER[roi]])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6.2,
                        color="#ffffff" if M[i, j] > 0.615 else INK)
        ax.set_title(f"norm = {norm}", fontsize=10, loc="left", color=INK)
        ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("Two-way identification", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle("S2   Accuracy across ROI × repetition × normalisation",
                 fontsize=10.5, fontweight="bold", x=0.09, ha="left")
    save(fig, "figS2_accuracy_heatmap")


def figureS3():
    st = read("statistical_infrastructure.csv")
    s = st[(st.norm == "repgroup") & (st.analysis == "repeated323")
           & (st.contrast == "Rep2-Rep1")]
    s = s.set_index("roi").reindex([x for x in ROI_ORDER if x in set(s.roi)]).dropna(
        subset=["bf10"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lb = np.log10(s.bf10.values)
    y = np.arange(len(s))[::-1]
    ax.barh(y, lb, 0.66, color=[TIER_COLOR[TIER[r]] for r in s.index],
            edgecolor=SURFACE, linewidth=1.2, zorder=2)
    for bound, lab in [(np.log10(3), "BF=3"), (np.log10(1 / 3), "BF=1/3"),
                       (1.0, "BF=10"), (-1.0, "BF=1/10")]:
        ax.axvline(bound, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.text(bound, len(s) - 0.35, lab, fontsize=7, color=INK2, ha="center")
    ax.axvline(0, color=INK2, linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=8.5)
    for t, roi in zip(ax.get_yticklabels(), s.index):
        t.set_color(TIER_COLOR[TIER[roi]])
    ax.set_ylim(-0.8, len(s) + 0.1)
    tidy(ax, xlabel="log₁₀ Bayes factor (BF₁₀)", grid_axis="x",
         title="S3   Evidence for the repetition effect. The inconclusive band is wide")
    save(fig, "figS3_bayes_factors")


def main():
    figs = [("Figure 2", figure2), ("Figure 3", figure3), ("Figure 4", figure4),
            ("Figure 5", figure5), ("Figure S1", figureS1), ("Figure S2", figureS2),
            ("Figure S3", figureS3)]
    failed = []
    for name, fn in figs:
        print(f"{name}:")
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            failed.append((name, str(e)))
        except Exception:                                       # noqa: BLE001
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failed.append((name, "exception"))
    print(f"\nfigures written to {FIGDIR}")
    if failed:
        print("skipped or failed: " + ", ".join(n for n, _ in failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
