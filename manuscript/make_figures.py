"""Publication figures for the individuality-gradient manuscript.

Every panel is rebuilt from the trial-level result files, not from numbers typed into
the manuscript. Before anything is drawn the script recomputes the per-region summary
from the trial-level table and checks it against the summary file written by step38;
a mismatch aborts. The same applies to the paired contrasts quoted in the text.

Outputs PDF (for LaTeX) and PNG (for quick inspection) into figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

TOL = 1e-6  # agreement required between recomputed and stored summaries

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # editable/embeddable text, required by most publishers
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Hierarchy tiers drive both colour and the ordering used in every figure, so that a
# reader can follow one region across panels.
TIERS = {
    "early":    ("Early visual",        "#2C5F8A"),
    "mid":      ("Mid-level streams",   "#4A9BA8"),
    "category": ("Category-selective",  "#D4893C"),
    "high":     ("High-level streams",  "#A83E3E"),
    "global":   ("Whole visual cortex", "#6B6B6B"),
}

ROI_META = {
    # key            display          tier
    "V1":           ("V1",            "early"),
    "V2":           ("V2",            "early"),
    "V3":           ("V3",            "early"),
    "hV4":          ("hV4",           "early"),
    "S_early":      ("Early stream",  "early"),
    "S_midventral": ("Mid-ventral",   "mid"),
    "S_midlateral": ("Mid-lateral",   "mid"),
    "S_parietal":   ("Parietal",      "mid"),
    "FFA":          ("FFA",           "category"),
    "PPA":          ("PPA",           "category"),
    "OPA":          ("OPA",           "category"),
    "RSC":          ("RSC",           "category"),
    "PHC":          ("PHC",           "category"),
    "S_ventral":    ("Ventral",       "high"),
    "S_lateral":    ("Lateral",       "high"),
    "nsdgeneral":   ("nsdgeneral",    "global"),
}
DISP = {k: v[0] for k, v in ROI_META.items()}
COLOR = {k: TIERS[v[1]][1] for k, v in ROI_META.items()}

CAT_ORDER = ["person", "animal", "vehicle", "food", "indoor", "kitchen",
             "furniture", "accessory", "outdoor", "electronic", "appliance",
             "sports"]

problems: list[str] = []


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=400)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+png)")


# ---------------------------------------------------------------- load + validate
def per_subject_fraction(trials: pd.DataFrame) -> pd.DataFrame:
    """Fraction of ceiling per (roi, subject): ratio of means, not mean of ratios.

    Per-image ratios are unusable -- the denominator crosses zero -- so the subject is
    the unit and the ratio is formed once per subject from their image-averaged cosines.
    """
    g = (trials.groupby(["roi", "subject_id"], observed=True)[["transfer", "ceiling"]]
         .mean().reset_index())
    g["frac"] = g["transfer"] / g["ceiling"]
    return g


def validate(trials: pd.DataFrame, by_roi: pd.DataFrame) -> pd.DataFrame:
    """Recompute the stored per-region summary from trial level and compare."""
    print("=== validating summary against trial-level data ===")
    subj = per_subject_fraction(trials)
    rec = (subj.groupby("roi", observed=True)
           .agg(transfer=("transfer", "mean"), ceiling=("ceiling", "mean"),
                frac=("frac", "mean"), sem=("frac", lambda x: stats.sem(x, ddof=1)),
                n=("frac", "size")).reset_index())

    m = rec.merge(by_roi, on="roi", suffixes=("_rec", "_file"))
    if len(m) != len(by_roi):
        problems.append(f"roi merge lost rows: {len(m)} vs {len(by_roi)}")

    worst = {}
    for a, b in (("transfer_rec", "transfer_file"), ("ceiling_rec", "ceiling_file"),
                 ("frac", "frac_of_ceiling"), ("sem_rec", "sem_file")):
        d = float(np.abs(m[a] - m[b]).max())
        worst[a.replace("_rec", "")] = d
        if d > TOL:
            problems.append(f"{a} vs {b}: max abs diff {d:.3e} exceeds {TOL:g}")
    for k, v in worst.items():
        print(f"  {k:9s} max|recomputed - stored| = {v:.3e}")

    if not (m["n"] == 8).all():
        problems.append(f"expected 8 subjects per roi, got {sorted(m['n'].unique())}")
    print(f"  subjects per roi: {sorted(m['n'].unique())}")
    print(f"  regions: {len(m)}")
    return subj


def check_contrasts(subj: pd.DataFrame) -> dict:
    """Recompute the paired V1 contrasts quoted in the Results."""
    print("\n=== validating paired contrasts quoted in the text ===")
    wide = subj.pivot(index="subject_id", columns="roi", values="frac")
    quoted = {"S_ventral": (0.218, 17.19, 5.5e-7),
              "PPA": (0.196, 10.72, 1.4e-5),
              "FFA": (0.196, 11.10, 1.1e-5)}
    out = {}
    for roi, (d_q, t_q, p_q) in quoted.items():
        d = wide[roi] - wide["V1"]
        t, p = stats.ttest_rel(wide[roi], wide["V1"])
        out[roi] = (float(d.mean()), float(t), float(p))
        ok = (abs(d.mean() - d_q) < 5e-4 and abs(t - t_q) < 0.05
              and abs(np.log10(p) - np.log10(p_q)) < 0.05)
        if not ok:
            problems.append(f"contrast {roi} vs V1 recomputed "
                            f"d={d.mean():.4f} t={t:.2f} p={p:.2e}, "
                            f"manuscript says d={d_q} t={t_q} p={p_q:.1e}")
        print(f"  {roi:10s} vs V1  d={d.mean():.4f} t({len(d)-1})={t:.2f} "
              f"p={p:.2e}   {'OK' if ok else 'MISMATCH'}")
    return out


# ---------------------------------------------------------------- figures
def fig1(by_roi: pd.DataFrame, subj: pd.DataFrame) -> None:
    """The gradient: fraction of ceiling by region, with the raw quantities beside it."""
    order = by_roi.sort_values("frac_of_ceiling")["roi"].tolist()
    y = np.arange(len(order))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 4.7), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- A: fraction of ceiling, per-subject points over the group mean
    idx = by_roi.set_index("roi")
    for i, roi in enumerate(order):
        c = COLOR[roi]
        pts = subj.loc[subj.roi == roi, "frac"].to_numpy()
        jit = (np.arange(len(pts)) - (len(pts) - 1) / 2) * 0.055
        axA.scatter(pts, i + jit, s=7, color=c, alpha=0.40, linewidths=0, zorder=2)
        axA.errorbar(idx.loc[roi, "frac_of_ceiling"], i,
                     xerr=idx.loc[roi, "sem"], fmt="o", ms=5.0, color=c,
                     ecolor=c, elinewidth=1.4, capsize=2.5, zorder=3)
    axA.set_yticks(y, [DISP[r] for r in order])
    axA.set_xlabel("Transfer as fraction of within-participant ceiling")
    axA.set_xlim(0.55, 1.03)
    axA.axvline(1.0, color="#333333", lw=0.7, ls=":", zorder=1)
    axA.text(1.007, len(order) - 1.5, "fully\nshared", fontsize=6, ha="left",
             va="center", color="#333333", linespacing=0.95)
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.grid(axis="x", lw=0.35, color="#DDDDDD", zorder=0)
    axA.set_axisbelow(True)
    axA.set_title("A   Shared fraction of decodable representation", loc="left")

    handles = [Line2D([], [], marker="o", ls="", ms=4.5, color=c, label=lab)
               for lab, c in TIERS.values()]
    axA.legend(handles=handles, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, -0.10), ncol=3, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0.0)

    # --- B: the two raw cosines that form the ratio
    for i, roi in enumerate(order):
        c = COLOR[roi]
        t, ce = idx.loc[roi, "transfer"], idx.loc[roi, "ceiling"]
        axB.plot([t, ce], [i, i], lw=1.0, color=c, alpha=0.55, zorder=2)
        axB.scatter([ce], [i], s=26, facecolor="white", edgecolor=c,
                    linewidths=1.0, zorder=3)
        axB.scatter([t], [i], s=18, color=c, linewidths=0, zorder=4)
    axB.set_yticks(y, [])
    axB.set_xlabel("Mean centred cosine similarity")
    axB.set_ylim(-0.6, len(order) - 0.4)
    axB.grid(axis="x", lw=0.35, color="#DDDDDD", zorder=0)
    axB.set_axisbelow(True)
    axB.set_title("B   Transfer and ceiling separately", loc="left")
    axB.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.5, color="#444444",
               label="cross-participant transfer"),
        Line2D([], [], marker="o", ls="", ms=5.0, markerfacecolor="white",
               markeredgecolor="#444444", color="#444444",
               label="within-participant ceiling")],
        loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.10),
        ncol=1, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.35)

    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig1_gradient")


def fig2(subj: pd.DataFrame, contrasts: dict) -> None:
    """Every participant individually: V1 against ventral stream, and the full profile."""
    wide = subj.pivot(index="subject_id", columns="roi", values="frac")
    subs = sorted(wide.index)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [0.72, 1.28]})

    # --- A: paired V1 -> ventral, one line per participant
    d, t, p = contrasts["S_ventral"]
    for s in subs:
        axA.plot([0, 1], [wide.loc[s, "V1"], wide.loc[s, "S_ventral"]],
                 "-o", ms=3.4, lw=0.9, color="#555555", alpha=0.75,
                 markerfacecolor="white", markeredgewidth=0.9)
    axA.set_xticks([0, 1], ["V1", "Ventral\nstream"])
    axA.set_xlim(-0.35, 1.35)
    # Open a clear band below the lowest participant line so the statistics
    # block has somewhere to sit without covering the data.
    lo = min(wide["V1"].min(), wide["S_ventral"].min())
    hi = max(wide["V1"].max(), wide["S_ventral"].max())
    axA.set_ylim(lo - 0.16 * (hi - lo), hi + 0.04 * (hi - lo))
    axA.set_ylabel("Fraction of ceiling")
    axA.set_title("A   Paired within participant", loc="left")
    mant, expo = f"{p:.1e}".split("e")
    axA.text(0.5, 0.015,
             f"$\\Delta$ = {d:.3f}\n$t$(7) = {t:.2f}\n"
             f"$p$ = {mant}$\\times10^{{{int(expo)}}}$",
             transform=axA.transAxes, fontsize=7, ha="center", va="bottom",
             linespacing=1.35)
    axA.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    axA.set_axisbelow(True)

    # --- B: full per-participant profile across regions
    order = (subj.groupby("roi", observed=True)["frac"].mean()
             .sort_values().index.tolist())
    x = np.arange(len(order))
    for s in subs:
        axB.plot(x, [wide.loc[s, r] for r in order], "-", lw=0.7,
                 color="#888888", alpha=0.55, zorder=2)
    mean = [wide[r].mean() for r in order]
    axB.plot(x, mean, "-", lw=1.6, color="#1A1A1A", zorder=3, label="group mean")
    for i, roi in enumerate(order):
        axB.scatter([i], [mean[i]], s=22, color=COLOR[roi], zorder=4,
                    edgecolor="#1A1A1A", linewidths=0.5)
    axB.set_xticks(x, [DISP[r] for r in order], rotation=55, ha="right")
    axB.set_ylabel("Fraction of ceiling")
    axB.set_xlim(-0.6, len(order) - 0.4)
    axB.set_title("B   Each participant, all regions", loc="left")
    axB.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    axB.set_axisbelow(True)
    axB.legend(loc="lower right", frameon=False, borderaxespad=0.3)

    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig2_per_participant")


def fig3(by_roi: pd.DataFrame, geom: pd.DataFrame) -> dict:
    """Do the two measures agree? Ordering yes, magnitude no -- shown as such."""
    m = by_roi.merge(geom[["roi", "shared", "sem_shared"]], on="roi", how="inner")
    m = m.dropna(subset=["shared"])
    rho, p_rho = stats.spearmanr(m["frac_of_ceiling"], m["shared"])
    r, p_r = stats.pearsonr(m["frac_of_ceiling"], m["shared"])
    print("\n=== cross-method agreement (recomputed) ===")
    print(f"  n regions {len(m)}")
    print(f"  Spearman rho = {rho:.3f}, p = {p_rho:.3f}")
    print(f"  Pearson   r  = {r:.3f}, p = {p_r:.3f}")
    for nm, val, q in (("rho", rho, 0.556), ("p_rho", p_rho, 0.025),
                       ("r", r, 0.350), ("p_r", p_r, 0.18)):
        if abs(val - q) > 0.006:
            problems.append(f"cross-method {nm} recomputed {val:.4f}, "
                            f"manuscript says {q}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 4.5),
                                   gridspec_kw={"width_ratios": [1.3, 0.7]})

    # --- A: magnitude scatter. Label offsets are hand-set: the upper-right corner
    # is crowded and automatic placement puts labels on top of each other.
    # PPA/FFA/OPA/RSC sit in one tight cluster. All four are pushed LEFT (the
    # right side is occupied by Ventral and Lateral and their error bars) and
    # staggered vertically so no two labels share a band.
    OFF = {"S_ventral": (7, 2), "nsdgeneral": (-7, 5), "S_lateral": (7, -1),
           "PPA": (-7, 7), "FFA": (-7, 1), "OPA": (7, -1), "RSC": (-7, -4),
           "S_early": (-6, 4), "S_parietal": (7, -1), "S_midlateral": (7, 2),
           "S_midventral": (-6, -7), "hV4": (7, -1), "V3": (7, 2),
           "V2": (2, -9), "V1": (2, -9), "PHC": (-7, 2)}
    for _, row in m.iterrows():
        roi, c = row["roi"], COLOR[row["roi"]]
        axA.errorbar(row["frac_of_ceiling"], row["shared"],
                     xerr=row["sem"], yerr=row["sem_shared"],
                     fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=0.8,
                     capsize=0, alpha=0.9, zorder=3)
        dx, dy = OFF.get(roi, (5, 3))
        axA.annotate(DISP[roi], (row["frac_of_ceiling"], row["shared"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=6,
                     ha="right" if dx < 0 else "left", color="#333333")
    axA.plot([0.45, 1.08], [0.45, 1.08], ls=":", lw=0.7, color="#999999", zorder=1)
    axA.set_xlim(0.66, 1.005)
    axA.set_ylim(0.22, 1.12)
    axA.set_xlabel("Decoding measure (fraction of ceiling)")
    axA.set_ylabel("Geometry measure (shared fraction)")
    axA.set_title("A   Magnitude", loc="left")
    axA.text(0.03, 0.97,
             f"Pearson $r$ = {r:.2f}, $p$ = {p_r:.2f}\n"
             f"(bars: SEM over 8 participants)",
             transform=axA.transAxes, fontsize=7, va="top", linespacing=1.35)

    # --- B: rank agreement, the claim we actually make
    m2 = m.copy()
    m2["rk_dec"] = m2["frac_of_ceiling"].rank()
    m2["rk_geo"] = m2["shared"].rank()
    for _, row in m2.iterrows():
        axB.plot([0, 1], [row["rk_dec"], row["rk_geo"]], "-", lw=0.9,
                 color=COLOR[row["roi"]], alpha=0.85, zorder=2)
        axB.scatter([0, 1], [row["rk_dec"], row["rk_geo"]], s=16,
                    color=COLOR[row["roi"]], zorder=3, linewidths=0)
    for _, row in m2.iterrows():
        axB.annotate(DISP[row["roi"]], (0, row["rk_dec"]),
                     textcoords="offset points", xytext=(-4, 0), fontsize=6,
                     ha="right", va="center", color="#333333")
        axB.annotate(DISP[row["roi"]], (1, row["rk_geo"]),
                     textcoords="offset points", xytext=(4, 0), fontsize=6,
                     ha="left", va="center", color="#333333")
    axB.set_xticks([0, 1], ["Decoding", "Geometry"])
    axB.set_xlim(-0.80, 1.62)
    axB.set_ylim(-0.6, len(m2) + 1.9)
    axB.set_ylabel("Rank (1 = most individual)")
    axB.set_title(f"B   Rank order  "
                  f"($\\rho$ = {rho:.2f}, $p$ = {p_rho:.3f})", loc="left")
    axB.spines["bottom"].set_visible(False)
    axB.tick_params(axis="x", length=0)

    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig3_cross_method")
    return {"rho": rho, "p_rho": p_rho, "r": r, "p_r": p_r, "n": len(m)}


def fig4(by_cat: pd.DataFrame, by_roi: pd.DataFrame) -> pd.DataFrame:
    """Category dependence: strong in V1, flat in ventral stream."""
    g = (by_cat.groupby(["roi", "category"], observed=True)["frac"]
         .mean().reset_index())
    piv = g.pivot(index="roi", columns="category", values="frac")
    roi_order = by_roi.sort_values("frac_of_ceiling", ascending=False)["roi"].tolist()
    cats = [c for c in CAT_ORDER if c in piv.columns]
    missing = sorted(set(piv.columns) - set(cats))
    if missing:
        problems.append(f"categories present in data but not in CAT_ORDER: {missing}")
    piv = piv.loc[roi_order, cats]

    spans = (piv.max(axis=1) - piv.min(axis=1)).rename("span")
    print("\n=== category span by region (max - min across categories) ===")
    print(f"  V1       {spans['V1']:.3f}  "
          f"range {piv.loc['V1'].min():.3f}-{piv.loc['V1'].max():.3f}")
    print(f"  Ventral  {spans['S_ventral']:.3f}  "
          f"range {piv.loc['S_ventral'].min():.3f}-{piv.loc['S_ventral'].max():.3f}")
    for nm, got, want in (("V1 span", spans["V1"], 0.178),
                          ("ventral span", spans["S_ventral"], 0.079),
                          ("V1 min", piv.loc["V1"].min(), 0.625),
                          ("V1 max", piv.loc["V1"].max(), 0.803),
                          ("ventral min", piv.loc["S_ventral"].min(), 0.886),
                          ("ventral max", piv.loc["S_ventral"].max(), 0.966)):
        if abs(got - want) > 6e-4:
            problems.append(f"category {nm} recomputed {got:.4f}, "
                            f"manuscript says {want}")

    fig = plt.figure(figsize=(7.2, 6.4))
    # Panel C's region names are long; without a wide gutter they are drawn on top
    # of panel B's data.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.0, 0.54],
                          hspace=0.52, wspace=0.46)
    axH = fig.add_subplot(gs[0, :])
    axL = fig.add_subplot(gs[1, 0])
    axS = fig.add_subplot(gs[1, 1])

    # --- heatmap
    im = axH.imshow(piv.to_numpy(), aspect="auto", cmap="magma",
                    vmin=0.60, vmax=1.00)
    axH.set_xticks(np.arange(len(cats)), cats, rotation=48, ha="right")
    axH.set_yticks(np.arange(len(roi_order)), [DISP[r] for r in roi_order])
    for tick, roi in zip(axH.get_yticklabels(), roi_order):
        tick.set_color(COLOR[roi])
    cb = fig.colorbar(im, ax=axH, fraction=0.022, pad=0.012)
    cb.set_label("Fraction of ceiling", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    axH.set_title("A   Transfer by region and image category", loc="left")
    for sp in axH.spines.values():
        sp.set_visible(False)

    # --- V1 vs ventral profiles across categories
    v1 = piv.loc["V1"].sort_values()
    order_c = v1.index.tolist()
    xc = np.arange(len(order_c))
    axL.plot(xc, piv.loc["V1", order_c].to_numpy(), "-o", ms=3.4, lw=1.1,
             color=COLOR["V1"], label="V1")
    axL.plot(xc, piv.loc["S_ventral", order_c].to_numpy(), "-o", ms=3.4, lw=1.1,
             color=COLOR["S_ventral"], label="Ventral stream")
    axL.set_xticks(xc, order_c, rotation=48, ha="right")
    axL.set_ylabel("Fraction of ceiling")
    axL.set_ylim(0.58, 1.02)
    axL.legend(frameon=False, loc="center left", borderaxespad=0.4,
           handlelength=1.4)
    axL.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    axL.set_axisbelow(True)
    axL.set_title("B   V1 varies with content; ventral stream does not", loc="left")

    # --- span per region
    sp_order = spans.sort_values(ascending=False)
    axS.barh(np.arange(len(sp_order)), sp_order.to_numpy(),
             color=[COLOR[r] for r in sp_order.index], height=0.72)
    # The trailing space keeps a final "l" off the spine, where it reads as clipped.
    axS.set_yticks(np.arange(len(sp_order)),
                   [DISP[r] + " " for r in sp_order.index], fontsize=7.5)
    axS.tick_params(axis="y", length=2)
    axS.invert_yaxis()
    axS.set_xlabel("Span across categories")
    axS.grid(axis="x", lw=0.35, color="#DDDDDD", zorder=0)
    axS.set_axisbelow(True)
    axS.set_title("C   Category sensitivity", loc="left")

    save(fig, "fig4_category")
    return piv


def fig5(trials: pd.DataFrame, by_roi: pd.DataFrame) -> pd.DataFrame:
    """Identification accuracy: the gate, and the within/cross gap that drives Fig 1."""
    # ident_* are per (roi, subject) constants in the trial table; take the first.
    ident = (trials.groupby(["roi", "subject_id"], observed=True)
             [["ident_transfer", "ident_ceiling"]].agg(
                 lambda x: x.iloc[0]).reset_index())
    nun = (trials.groupby(["roi", "subject_id"], observed=True)["ident_ceiling"]
           .nunique())
    if (nun > 1).any():
        problems.append("ident_ceiling is not constant within (roi, subject)")

    agg = (ident.groupby("roi", observed=True)
           .agg(ic=("ident_ceiling", "mean"),
                ic_sem=("ident_ceiling", lambda x: stats.sem(x, ddof=1)),
                it=("ident_transfer", "mean"),
                it_sem=("ident_transfer", lambda x: stats.sem(x, ddof=1)))
           .reset_index())
    agg["gap"] = agg["ic"] - agg["it"]

    print("\n=== identification gate (within-participant, chance 0.5) ===")
    print(f"  min {agg['ic'].min():.3f} ({DISP[agg.loc[agg.ic.idxmin(), 'roi']]}), "
          f"max {agg['ic'].max():.3f} "
          f"({DISP[agg.loc[agg.ic.idxmax(), 'roi']]})")
    below = agg.loc[agg["ic"] < 0.55, "roi"].tolist()
    print(f"  regions below the 0.55 criterion: {below or 'none'}")
    if below:
        problems.append(f"regions below identification criterion: {below}")
    for nm, got, want in (("min ident", agg["ic"].min(), 0.675),
                          ("max ident", agg["ic"].max(), 0.962)):
        if abs(got - want) > 6e-4:
            problems.append(f"{nm} recomputed {got:.4f}, manuscript says {want}")
    v1 = agg.set_index("roi").loc["V1"]
    print(f"  V1 within {v1['ic']:.3f} vs cross {v1['it']:.3f} "
          f"(gap {v1['gap']:.3f}, largest = {agg['gap'].max():.3f} in "
          f"{DISP[agg.loc[agg.gap.idxmax(), 'roi']]})")
    for nm, got, want in (("V1 within ident", v1["ic"], 0.837),
                          ("V1 cross ident", v1["it"], 0.756)):
        if abs(got - want) > 6e-4:
            problems.append(f"{nm} recomputed {got:.4f}, manuscript says {want}")
    if agg.loc[agg.gap.idxmax(), "roi"] != "V1":
        problems.append("manuscript claims V1 has the widest within/cross "
                        f"identification gap, but it is "
                        f"{agg.loc[agg.gap.idxmax(), 'roi']}")

    order = by_roi.sort_values("frac_of_ceiling")["roi"].tolist()
    agg = agg.set_index("roi").loc[order].reset_index()
    x = np.arange(len(order))

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1.0]})

    axA.axhline(0.5, color="#666666", lw=0.7, ls="-", zorder=1)
    axA.axhline(0.55, color="#B03030", lw=0.7, ls="--", zorder=1)
    axA.text(len(order) - 0.45, 0.505, "chance", fontsize=6, color="#666666",
             va="bottom", ha="right")
    axA.text(len(order) - 0.45, 0.556, "criterion", fontsize=6, color="#B03030",
             va="bottom", ha="right")
    axA.errorbar(x, agg["ic"], yerr=agg["ic_sem"], fmt="o", ms=4.5,
                 color="#1A1A1A", ecolor="#1A1A1A", elinewidth=1.0, capsize=2,
                 label="within participant (ceiling)", zorder=3)
    axA.errorbar(x, agg["it"], yerr=agg["it_sem"], fmt="s", ms=4.0,
                 markerfacecolor="white", color="#C05A2A", ecolor="#C05A2A",
                 elinewidth=1.0, capsize=2, label="from seven other brains",
                 zorder=3)
    axA.set_ylabel("Two-way identification accuracy")
    axA.set_ylim(0.46, 1.0)
    axA.legend(frameon=False, loc="upper left", borderaxespad=0.3)
    axA.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    axA.set_axisbelow(True)
    axA.set_title("A   Every region clears the identification criterion", loc="left")

    axB.bar(x, agg["gap"], color=[COLOR[r] for r in order], width=0.68, zorder=2)
    axB.set_xticks(x, [DISP[r] for r in order], rotation=55, ha="right")
    axB.set_ylabel("Within $-$ cross\naccuracy", linespacing=1.3)
    axB.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    axB.set_axisbelow(True)
    axB.set_title("B   Information present but not transferable", loc="left")

    fig.subplots_adjust(hspace=0.18)
    save(fig, "fig5_identification")
    return agg


# ---------------------------------------------------------------- main
def main() -> int:
    print("loading result tables ...")
    trials = pd.read_csv(RESULTS / "brain_embedding_loso.csv")
    by_roi = pd.read_csv(RESULTS / "brain_embedding_loso_by_roi.csv")
    by_cat = pd.read_csv(RESULTS / "brain_embedding_loso_by_category.csv")
    geom = pd.read_csv(RESULTS / "intersubject_mapping_breakdown_summary.csv")
    print(f"  trials {trials.shape}, by_roi {by_roi.shape}, "
          f"by_cat {by_cat.shape}, geometry {geom.shape}")

    unknown = sorted(set(by_roi["roi"]) - set(ROI_META))
    if unknown:
        problems.append(f"regions with no display metadata: {unknown}")

    subj = validate(trials, by_roi)
    contrasts = check_contrasts(subj)

    print("\n=== drawing figures ===")
    fig1(by_roi, subj)
    fig2(subj, contrasts)
    cm = fig3(by_roi, geom)
    piv = fig4(by_cat, by_roi)
    ident = fig5(trials, by_roi)

    # Machine-readable companions to the figures, for the tables in the manuscript.
    piv.round(4).to_csv(FIGDIR / "table_category_by_roi.csv")
    ident.round(4).to_csv(FIGDIR / "table_identification.csv", index=False)
    pd.DataFrame([cm]).round(4).to_csv(FIGDIR / "table_cross_method.csv",
                                       index=False)

    print("\n" + ("[OK] all figure data validated against stored summaries"
                  if not problems else f"[FAIL] {len(problems)} problem(s):"))
    for p in problems:
        print(f"   - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
