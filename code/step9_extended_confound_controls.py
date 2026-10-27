"""
Step 9: fill the gaps in the confound controls.

9a  Control B (time-on-task) extended to the MTL and hippocampal ROIs, which step3
    skipped entirely.

9b  Proportional-SNR residuals computed WITHIN subject. Control A regressed the
    subject-averaged delta on baseline-above-chance and found r = -0.859, which is
    exactly what a uniform SNR account predicts. Averaging over subjects first discards
    the per-subject structure, so here each subject gets their own delta ~ baseline
    regression across ROIs, and we test whether any ROI's residual departs from the
    proportional model consistently across the 8 subjects. A genuine representational
    effect has to show up here; pure SNR loss cannot produce it.

9c  Same-session repetition analysis. The brief assumed NSD spaces repetitions across
    sessions. It does not always: 260 of the 323 repeated stimuli have Rep1 and Rep2
    inside the SAME session (136 both in session 1, 124 both in session 2), so the
    contrast can be run with session, scanner drift and scan day held constant. A
    leave-one-session-out decoder is also fitted, training on one session's novel trials
    and testing the other session's repeats, so train and test never share a session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from _fast_ridge import SVDRidge

HERE = Path(__file__).resolve().parent
BETAS = Path("D:/Desktop/Memory_retention/ADVANCED_RESEARCH/data/roi_betas")
RESULTS = HERE / "results"
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
ROIS = ["V1", "V2", "FFA", "PPA", "RSC", "PHC", "PRC", "ERC", "SUB", "CA1", "DG_CA3"]
ROI_SETS = {r: [r] for r in ROIS}
ROI_SETS.update({"EarlyVis": ["V1", "V2"], "SceneCtx": ["PPA", "RSC", "PHC"],
                 "MTL": ["PRC", "ERC", "SUB", "CA1", "DG_CA3"],
                 "AllCortex": ["V1", "V2", "FFA", "PPA", "RSC", "PHC"]})
MTL_SETS = {"PRC", "ERC", "SUB", "CA1", "DG_CA3", "MTL"}
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
SEED = 42


def zs(X, g):
    Z = np.empty_like(X)
    for v in np.unique(g):
        m = g == v
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def two_way(P, pool, ti):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ pool.T
    ts = S[np.arange(len(ti)), ti]
    S[np.arange(len(ti)), ti] = -np.inf
    return float((ts[:, None] > S).sum() / (len(ti) * (pool.shape[0] - 1)))


def cv_alpha(X, Y, folds=5):
    folds = min(folds, max(2, len(X) // 30))
    acc = np.zeros(len(ALPHAS))
    for tr, va in KFold(folds, shuffle=True, random_state=SEED).split(X):
        m = SVDRidge().fit(X[tr], Y[tr])
        for j, a in enumerate(ALPHAS):
            p = m.predict(X[va], a)
            acc[j] += float(((p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8))
                             * Y[va]).sum(1).mean())
    return ALPHAS[int(np.argmax(acc))]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    adj, prev = np.empty(n), 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def subject_trial_sets(m):
    """Precompute the trial index sets each analysis needs, once per subject."""
    rep = m.repeat_number.values
    sess = m.session.values
    nsd = m.nsd_id.values

    pair = m[m.n_presentations >= 2].pivot_table(index="nsd_id", columns="repeat_number",
                                                values="session")
    same_ids = set(pair.index[pair[1] == pair[2]])
    same_mask = np.isin(nsd, list(same_ids))

    # per held-out session: stimuli whose Rep1 AND Rep2 both fall in that session
    loso = {}
    for held in (1, 2):
        ids = sorted(pair.index[(pair[1] == held) & (pair[2] == held)])
        loso[held] = ids
    return rep, sess, nsd, same_ids, same_mask, loso


def per_subject_analyses():
    man = pd.read_csv(HERE / "data" / "trial_manifest_corrected.csv")
    tg = np.load(HERE / "data" / "clip_text_targets.npz")
    c2r = {int(c): i for i, c in enumerate(tg["coco_ids"])}
    T = tg["T_text"]
    rows_b, rows_same, rows_loso = [], [], []

    for subj in SUBJECTS:
        z = np.load(BETAS / f"{subj}_roi_betas_1500trials.npz")
        m = man[man.subject_id == subj].sort_values("global_trial").reset_index(drop=True)
        Y = T[[c2r[c] for c in m.coco_id]]
        novel = (m.n_presentations == 1).values
        rep, sess, nsd, same_ids, same_mask, loso = subject_trial_sets(m)
        gt = m.global_trial.values
        # nsd_id -> target row, for pool construction
        id2target = {int(i): c2r[int(c)] for i, c in zip(m.nsd_id, m.coco_id)}

        for roi, parts in ROI_SETS.items():
            Xraw = np.concatenate([z[p] for p in parts], 1).astype(np.float64)

            # --- 9a: Control B for every ROI set, including MTL/hippocampus
            X = zs(Xraw, sess)
            Xn, Yn, gtn = X[novel], Y[novel], gt[novel]
            alpha = cv_alpha(Xn, Yn)
            pred = np.zeros_like(Yn)
            for tr, te in KFold(5, shuffle=True, random_state=7).split(Xn):
                pred[te] = SVDRidge().fit(Xn[tr], Yn[tr]).predict(Xn[te], alpha)
            med = np.median(gtn)
            early, late = gtn <= med, gtn > med
            pool_idx = np.arange(len(Yn))
            rows_b.append(dict(subject_id=subj, roi=roi, is_mtl=roi in MTL_SETS,
                               acc_early=two_way(pred[early], Yn, pool_idx[early]),
                               acc_late=two_way(pred[late], Yn, pool_idx[late]),
                               n_early=int(early.sum()), n_late=int(late.sum())))

            # --- 9c-i: same-session Rep1 vs Rep2, decoder trained on the novel trials
            Xg = zs(Xraw, rep)                    # per-repetition-group normalisation
            a_used = cv_alpha(Xg[novel], Y[novel])
            mdl = SVDRidge().fit(Xg[novel], Y[novel])
            sel1 = same_mask & (rep == 1)
            imgs = nsd[sel1]
            pool = T[[id2target[int(i)] for i in imgs]]
            idx_of = {int(im): k for k, im in enumerate(imgs)}
            accs = {}
            for r in (1, 2):
                s = same_mask & (rep == r)
                ti = np.array([idx_of[int(i)] for i in nsd[s]])
                accs[r] = two_way(mdl.predict(Xg[s], a_used), pool, ti)
            rows_same.append(dict(subject_id=subj, roi=roi, n_stimuli=len(imgs),
                                  acc_rep1=accs[1], acc_rep2=accs[2],
                                  delta=accs[2] - accs[1]))

            # --- 9c-ii: leave-one-session-out; train and test share no session
            for held in (1, 2):
                te_ids = loso[held]
                tr_mask = novel & (sess != held)
                if tr_mask.sum() < 100 or len(te_ids) < 20:
                    continue
                al = cv_alpha(Xg[tr_mask], Y[tr_mask])
                mdl2 = SVDRidge().fit(Xg[tr_mask], Y[tr_mask])
                pool2 = T[[id2target[int(i)] for i in te_ids]]
                idx2 = {int(im): k for k, im in enumerate(te_ids)}
                te_mask = np.isin(nsd, te_ids) & (sess == held)
                acc2 = {}
                for r in (1, 2):
                    s = te_mask & (rep == r)
                    ti = np.array([idx2[int(i)] for i in nsd[s]])
                    acc2[r] = two_way(mdl2.predict(Xg[s], al), pool2, ti)
                rows_loso.append(dict(subject_id=subj, roi=roi, held_session=held,
                                      n_stimuli=len(te_ids), acc_rep1=acc2[1],
                                      acc_rep2=acc2[2], delta=acc2[2] - acc2[1],
                                      n_train=int(tr_mask.sum())))
        print(f"  [OK] {subj}", flush=True)

    return pd.DataFrame(rows_b), pd.DataFrame(rows_same), pd.DataFrame(rows_loso)


def proportional_residuals():
    """Within-subject regression of delta on baseline-above-chance across ROIs."""
    r = pd.read_csv(RESULTS / "ridge_clip_text_retrieval_by_repetition.csv")
    out = []
    for (norm, analysis), sub in r.groupby(["norm", "analysis"]):
        if analysis != "repeated323":
            continue
        piv = sub.pivot_table(index=["subject_id", "roi"], columns="repetition",
                              values="two_way").reset_index().rename(
            columns={1: "rep1", 2: "rep2"})
        piv["baseline"] = piv.rep1 - 0.5
        piv["delta"] = piv.rep2 - piv.rep1
        for subj, g in piv.groupby("subject_id"):
            sl, ic, rr, pp, se = stats.linregress(g.baseline, g.delta)
            resid = g.delta - (ic + sl * g.baseline)
            for roi, rs, base, dl in zip(g.roi, resid, g.baseline, g.delta):
                out.append(dict(subject_id=subj, norm=norm, analysis=analysis, roi=roi,
                                baseline=base, delta=dl, fitted=dl - rs, residual=rs,
                                subj_slope=sl, subj_intercept=ic, subj_r2=rr ** 2))
    res = pd.DataFrame(out)

    stat = []
    for (norm, analysis, roi), g in res.groupby(["norm", "analysis", "roi"]):
        t, p = stats.ttest_1samp(g.residual, 0.0)
        sd = g.residual.std(ddof=1)
        stat.append(dict(norm=norm, analysis=analysis, roi=roi, n_subj=len(g),
                         mean_residual=g.residual.mean(),
                         sem_residual=sd / np.sqrt(len(g)), t_stat=t, p_val=p,
                         cohens_d=g.residual.mean() / sd if sd > 0 else 0.0,
                         n_subj_negative=int((g.residual < 0).sum())))
    st = pd.DataFrame(stat)
    st["p_fdr"] = np.nan
    for _, idx in st.groupby(["norm", "analysis"]).groups.items():
        st.loc[idx, "p_fdr"] = bh_fdr(st.loc[idx, "p_val"].values)
    return res, st


def main():
    try:
        b, same, loso = per_subject_analyses()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    pd.set_option("display.width", 250)

    b["delta_late_minus_early"] = b.acc_late - b.acc_early
    b.to_csv(RESULTS / "controlB_extended.csv", index=False)
    print("\n=== 9a  Control B extended: time-on-task on novel trials (N=8 paired) ===")
    rows = [(roi, roi in MTL_SETS, g.acc_early.mean(), g.acc_late.mean(),
             g.delta_late_minus_early.mean(), *stats.ttest_rel(g.acc_late, g.acc_early))
            for roi, g in b.groupby("roi")]
    tb = pd.DataFrame(rows, columns=["roi", "is_mtl", "early", "late", "delta", "t", "p"])
    tb["p_fdr"] = bh_fdr(tb.p.values)
    print(tb.sort_values("p").round(5).to_string(index=False))
    tb.to_csv(RESULTS / "controlB_extended_stats.csv", index=False)

    same.to_csv(RESULTS / "session_balanced_analysis.csv", index=False)
    print("\n=== 9c-i  Same-session Rep1 vs Rep2 (session, drift and day held constant) ===")
    rows = []
    for roi, g in same.groupby("roi"):
        t, p = stats.ttest_rel(g.acc_rep2, g.acc_rep1)
        rows.append((roi, int(g.n_stimuli.mean()), g.acc_rep1.mean(), g.acc_rep2.mean(),
                     g.delta.mean(), g.delta.std(ddof=1) / np.sqrt(len(g)), t, p,
                     int((g.delta < 0).sum())))
    ts = pd.DataFrame(rows, columns=["roi", "n_stim", "rep1", "rep2", "delta", "sem",
                                     "t", "p", "n_neg"])
    ts["p_fdr"] = bh_fdr(ts.p.values)
    print(ts.sort_values("p").round(5).to_string(index=False))
    ts.to_csv(RESULTS / "session_balanced_stats.csv", index=False)

    if not loso.empty:
        loso.to_csv(RESULTS / "loso_session_analysis.csv", index=False)
        print("\n=== 9c-ii  Leave-one-session-out (train and test share no session) ===")
        agg = loso.groupby(["subject_id", "roi"]).delta.mean().reset_index()
        rows = []
        for roi, g in agg.groupby("roi"):
            t, p = stats.ttest_1samp(g.delta, 0.0)
            rows.append((roi, g.delta.mean(), g.delta.std(ddof=1) / np.sqrt(len(g)), t, p,
                         int((g.delta < 0).sum())))
        tl = pd.DataFrame(rows, columns=["roi", "delta", "sem", "t", "p", "n_neg"])
        tl["p_fdr"] = bh_fdr(tl.p.values)
        print(tl.sort_values("p").round(5).to_string(index=False))
        tl.to_csv(RESULTS / "loso_session_stats.csv", index=False)
    else:
        print("\n[WARN] leave-one-session-out produced no rows.")

    res, st = proportional_residuals()
    res.to_csv(RESULTS / "proportional_snr_residuals.csv", index=False)
    st.to_csv(RESULTS / "proportional_snr_residual_stats.csv", index=False)
    print("\n=== 9b  Proportional-SNR residuals, within subject (repeated323) ===")
    for norm, g in st[st.analysis == "repeated323"].groupby("norm"):
        mr2 = res[res.norm == norm].subj_r2.mean()
        print(f"\n  norm={norm}   mean within-subject proportional-model R2 = {mr2:.3f}")
        print(g.sort_values("p_val")[["roi", "mean_residual", "sem_residual", "t_stat",
                                      "p_val", "p_fdr", "cohens_d",
                                      "n_subj_negative"]].round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
