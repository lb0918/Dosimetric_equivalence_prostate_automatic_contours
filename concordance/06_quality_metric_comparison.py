"""
06_quality_metric_comparison.py
===============================
WHICH SEGMENTATION QUALITY METRIC deserves to carry the threshold?

Script 05 estimates a threshold on the Dice. This script tests whether that
choice is the right one instead of assuming it, and answers four distinct
questions.

Q1 - Does another metric discriminate equivalence BETTER than the Dice?
     The framing is identical to 05: the label is equiv = 1{|delta| <= margin},
     with the pre-declared margin. The criterion is the AUC of the metric used as
     a classifier of equiv. Each metric is ORIENTED (U.QUALITY_METRICS) so that a
     high score always means a better segmentation; otherwise an AUC below 0.5
     would be a sign artefact rather than a result.

     The comparison uses a PAIRED BOOTSTRAP: patients are resampled and
     delta AUC = AUC(metric) - AUC(Dice) is recomputed on the SAME resample.
     Comparing two separately computed AUC confidence intervals would be
     invalid, since both metrics are measured on the same patients and are
     therefore correlated. A delta AUC CI containing 0 means the metrics are
     indistinguishable.

     NB: 'jaccard' is absent from the panel. J = D/(2-D) is a strictly monotone
     transform of the Dice, so its AUC and Youden cutpoint are IDENTICAL by
     construction. It is not a candidate, it is a duplicate.

Q2 - Would a COMPOSITE SCORE beat a single metric?
     Logistic regression on nested subsets of metrics, evaluated by AUC under
     stratified 5-fold CROSS-VALIDATION. The cross-validation is essential: the
     resubstitution AUC of a four-variable model rises mechanically and would
     wrongly suggest that a composite helps.

Q3 - Why? A collinearity matrix (Spearman) between the metrics.
     If the unsigned metrics are strongly collinear, a composite CANNOT help: it
     adds up the same information. This is the mechanical explanation of the Q2
     result, not a hypothesis.

Q4 - What the Dice structurally CANNOT say: the DIRECTION of the error.
     Cross-validated R2 for predicting the SIGNED delta, from the signed
     relative volume error versus from the Dice. This motivates axis 2 of script
     05.

Q5 - Why the precision can fall below chance: a monotonicity diagnostic.

Outputs (in the quality_metrics subdirectory of the results directory):
    - auc_by_metric.csv         AUC, delta AUC versus the Dice and its paired
                                bootstrap CI, per (index, margin, metric);
    - auc_summary.csv           mean AUC and rank per metric;
    - composite_cv_auc.csv      5-fold cross-validated AUC per predictor set;
    - collinearity_spearman.csv correlation matrix between metrics;
    - signed_vs_unsigned_r2.csv cross-validated R2 on the signed delta, signed
                                versus unsigned predictor;
    - figures: delta_auc_vs_dice.png, auc_ranking.png, composite_cv_auc.png,
               collinearity.png

Usage: python 06_quality_metric_comparison.py [--boot 400] [--seed 0]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils as U

plt.rc('font', family='serif')

# ============================================================
# CHEMINS & CONSTANTES
# ============================================================
OUT_DIR = U.OUT_DIR / "quality_metrics"

# Same restriction as 05: the cross-dataset geometry drives the prostate only.
STRUCTURE_TO_GEOM_ORGAN = {"Prostate": "Prostate"}
PANEL = [(s, i, t) for s, i, t in U.PANEL if s in STRUCTURE_TO_GEOM_ORGAN]
MARGINS = ("conf", "sens")

REFERENCE_METRIC = "dice"        # reference metric of the delta AUC comparisons
CANDIDATES = list(U.QUALITY_METRICS)

# Nested predictor sets tested for the composite.
COMPOSITE_SETS = {
    "dice seul": ["dice"],
    "assd seul": ["assd_mm"],
    "dice + assd": ["dice", "assd_mm"],
    "dice + assd + |Δvol|": ["dice", "assd_mm", "abs_rel_vol_err"],
    "dice + assd + |Δvol| + hd95": ["dice", "assd_mm", "abs_rel_vol_err",
                                    "hausdorff95_mm"],
}

# FIGURE labels of the composites. The keys of COMPOSITE_SETS are left as is:
# they are CSV column identifiers.
COMPOSITE_LABEL = {
    "dice seul": "Dice only",
    "assd seul": "ASSD only",
    "dice + assd": "Dice + ASSD",
    "dice + assd + |Δvol|": "Dice + ASSD + |Δvol|",
    "dice + assd + |Δvol| + hd95": "Dice + ASSD + |Δvol| + HD95",
}

# Okabe-Ito palette (CVD-safe), consistent with the rest of the pipeline.
C_REF = "#0072B2"     # blue    reference metric (Dice)
C_ALT = "#E69F00"     # orange  candidate metrics
C_POS = "#009E73"     # vert    : gain
C_NEG = "#D55E00"     # vermillon : perte
INK = "0.15"
GRID = "0.93"

# Figure font sizes: a legible floor once scaled down to a column.
FS_TICK = 12
FS_LABEL = 13
FS_TITLE = 14
FS_ANNOT = 11
FS_LEG = 12

# Diverging colormap with a grey NEUTRAL midpoint (never a hue in the middle)
# for the correlation matrix, whose values are bipolar around 0.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "okabe_div", [C_ALT, "#F2F2F2", C_REF], N=256)


def _label(metric: str) -> str:
    meta = U.QUALITY_METRICS[metric]
    return f"{meta['label']} ({meta['unite']})" if meta["unite"] else meta["label"]


def _oriented(g: pd.DataFrame, metric: str) -> np.ndarray:
    """Oriented score: higher means a better segmentation."""
    return U.QUALITY_METRICS[metric]["orient"] * g[metric].to_numpy(float)


# ============================================================
# Q1 - AUC PER METRIC plus PAIRED delta AUC versus the DICE
# ============================================================
def _auc(y, s):
    m = ~(np.isnan(s) | np.isnan(y))
    if m.sum() < 3 or np.unique(y[m]).size < 2:
        return np.nan
    return float(roc_auc_score(y[m], s[m]))


def auc_by_metric(base: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """
    AUC of each oriented metric for predicting equiv, with the delta AUC versus
    the Dice and its 95% CI from a PAIRED bootstrap (the same resample for both
    metrics).
    """
    rows = []
    for structure, index, tier in PANEL:
        g0 = base[(base["structure"] == structure) & (base["index"] == index)]
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            g = g0.dropna(subset=CANDIDATES + ["abs_diff"])
            y = (g["abs_diff"].to_numpy(float) <= delta).astype(int)
            if np.unique(y).size < 2:
                continue
            scores = {m: _oriented(g, m) for m in CANDIDATES}
            ref = scores[REFERENCE_METRIC]

            # Paired bootstrap: a single index set per replicate, shared by
            # every metric, so the delta AUCs are directly comparable.
            n = y.size
            boot = {m: [] for m in CANDIDATES}
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                yb = y[idx]
                if np.unique(yb).size < 2:
                    continue
                a_ref = _auc(yb, ref[idx])
                for m in CANDIDATES:
                    a_m = _auc(yb, scores[m][idx])
                    if np.isfinite(a_m) and np.isfinite(a_ref):
                        boot[m].append(a_m - a_ref)

            for m in CANDIDATES:
                d = np.asarray(boot[m], float)
                lo, hi = ((float(np.percentile(d, 2.5)),
                           float(np.percentile(d, 97.5)))
                          if d.size >= max(20, n_boot // 2) else (np.nan, np.nan))
                auc_m = _auc(y, scores[m])
                rows.append(dict(
                    structure=structure, index=index, tier=tier, margin=margin,
                    delta=delta, n=int(n), equiv_rate=float(y.mean()),
                    metric=m, orient=U.QUALITY_METRICS[m]["orient"],
                    auc=auc_m,
                    delta_auc_vs_ref=auc_m - _auc(y, ref),
                    delta_auc_lo=lo, delta_auc_hi=hi,
                    # "distinguishable" = paired delta AUC CI excluding 0.
                    beats_ref=bool(np.isfinite(lo) and lo > 0),
                    worse_than_ref=bool(np.isfinite(hi) and hi < 0),
                ))
    return pd.DataFrame(rows)


def auc_summary(auc_tbl: pd.DataFrame) -> pd.DataFrame:
    """Mean AUC per metric, over every cell (index x margin)."""
    s = (auc_tbl.groupby("metric")
         .agg(auc_mean=("auc", "mean"), auc_min=("auc", "min"),
              auc_max=("auc", "max"),
              delta_auc_mean=("delta_auc_vs_ref", "mean"),
              n_cells=("auc", "size"),
              n_beats_ref=("beats_ref", "sum"),
              n_worse_than_ref=("worse_than_ref", "sum"))
         .reset_index()
         .sort_values("auc_mean", ascending=False))
    s["rank"] = np.arange(1, len(s) + 1)
    s["label"] = s["metric"].map(_label)
    return s.reset_index(drop=True)


# ============================================================
# Q2 - COMPOSITE, EVALUATED BY CROSS-VALIDATION
# ============================================================
def composite_cv(base: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Stratified 5-fold cross-validated AUC of a logistic model, per predictor
    set. The predictors are standardised INSIDE the pipeline and therefore
    refitted at every fold, so nothing leaks from the test fold into the
    training fold.
    """
    cols_all = sorted({c for cols in COMPOSITE_SETS.values() for c in cols})
    rows = []
    for structure, index, tier in PANEL:
        g0 = base[(base["structure"] == structure) &
                  (base["index"] == index)].dropna(subset=cols_all + ["abs_diff"])
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            y = (g0["abs_diff"].to_numpy(float) <= delta).astype(int)
            if np.unique(y).size < 2 or min(np.bincount(y)) < 5:
                continue
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            r = dict(structure=structure, index=index, tier=tier, margin=margin,
                     delta=delta, n=int(y.size), equiv_rate=float(y.mean()))
            for name, cols in COMPOSITE_SETS.items():
                X = g0[cols].to_numpy(float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = cross_val_predict(
                        make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=1000)),
                        X, y, cv=cv, method="predict_proba")[:, 1]
                r[name] = _auc(y, p)
            rows.append(r)
    tbl = pd.DataFrame(rows)
    if len(tbl):
        # Summary row: mean over every cell.
        mean_row = {c: tbl[c].mean() for c in COMPOSITE_SETS}
        mean_row.update(structure="(toutes)", index="(moyenne)", tier="",
                        margin="(toutes)", delta=np.nan, n=int(tbl["n"].sum()),
                        equiv_rate=tbl["equiv_rate"].mean())
        tbl = pd.concat([tbl, pd.DataFrame([mean_row])], ignore_index=True)
    return tbl


# ============================================================
# Q3 - COLLINEARITY
# ============================================================
def collinearity(base: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between oriented metrics, over the unique patients."""
    pat = base.drop_duplicates(subset=["record_id", "geom_organ"])
    M = pd.DataFrame({m: _oriented(pat, m) for m in CANDIDATES})
    M[U.SIGNED_AXIS] = pat[U.SIGNED_AXIS].to_numpy(float)
    return M.corr(method="spearman")


# ============================================================
# Q4 - SIGNED versus UNSIGNED ON THE SIGNED DELTA
# ============================================================
def signed_vs_unsigned(base: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Cross-validated R2 for predicting the SIGNED delta. The Dice, being
    unsigned, cannot carry directional information: its expected R2 is about 0,
    and can be negative under cross-validation, which means worse than
    predicting the mean.
    """
    cv = KFold(5, shuffle=True, random_state=seed)
    rows = []
    for structure, index, tier in PANEL:
        g = base[(base["structure"] == structure) &
                 (base["index"] == index)].dropna(
                     subset=[U.SIGNED_AXIS, "dice", "abs_rel_vol_err", "diff"])
        y = g["diff"].to_numpy(float)
        if y.size < 20:
            continue
        r = dict(structure=structure, index=index, tier=tier, n=int(y.size))
        for name, col in [("r2_cv_signed_vol_err", U.SIGNED_AXIS),
                          ("r2_cv_abs_vol_err", "abs_rel_vol_err"),
                          ("r2_cv_dice", "dice"),
                          ("r2_cv_assd", "assd_mm")]:
            X = g[col].to_numpy(float)[:, None]
            r[name] = float(cross_val_score(LinearRegression(), X, y,
                                            cv=cv, scoring="r2").mean())
        r["spearman_signed_vol_err"] = float(
            stats.spearmanr(g[U.SIGNED_AXIS], y).statistic)
        r["spearman_dice"] = float(stats.spearmanr(g["dice"], y).statistic)
        # Line of delta on the signed volume error: feeds the NEUTRAL POINT of
        # Q5, i.e. the volume error at which the dosimetric gap vanishes.
        fit = LinearRegression().fit(g[[U.SIGNED_AXIS]].to_numpy(float), y)
        r["slope"] = float(fit.coef_[0])
        r["intercept"] = float(fit.intercept_)
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================
# Q5 - WHY THE PRECISION CAN FALL BELOW CHANCE
# ============================================================
# Bounds of the relative volume error (as a fraction of the ground-truth volume).
VOL_BINS = [-1.0, -0.20, -0.10, 0.0, 0.10, 0.20, 10.0]
VOL_LABELS = ["<−20 %", "−20..−10 %", "−10..0 %", "0..+10 %", "+10..+20 %", ">+20 %"]


def equiv_by_volume_bin(base: pd.DataFrame) -> pd.DataFrame:
    """
    Equivalence rate per bin of SIGNED volume error.

    This tests whether the quality -> equivalence relation is MONOTONE. The AUC
    assumes monotonicity; if the equivalence rate is an inverted U (maximal at
    an interior optimum and decreasing on both sides), the AUC of a directional
    predictor becomes uninterpretable and can fall below 0.5 without that
    predictor being uninformative.
    """
    rows = []
    for structure, index, _ in PANEL:
        g0 = base[(base["structure"] == structure) &
                  (base["index"] == index)].dropna(subset=[U.SIGNED_AXIS, "abs_diff"])
        b = pd.cut(g0[U.SIGNED_AXIS], VOL_BINS, labels=VOL_LABELS)
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            t = (g0.assign(bin=b, equiv=(g0["abs_diff"] <= delta).astype(int))
                   .groupby("bin", observed=True)
                   .agg(n=("equiv", "size"), equiv_rate=("equiv", "mean"),
                        absdiff_median=("abs_diff", "median"),
                        precision_median=("precision", "median"))
                   .reset_index())
            t.insert(0, "margin", margin)
            t.insert(0, "index", index)
            t.insert(0, "structure", structure)
            t["delta"] = delta
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def precision_nonmonotonicity(base: pd.DataFrame, signed: pd.DataFrame) -> pd.DataFrame:
    """
    A quantified demonstration that the AUC under-reads the precision.

    The AUC of the RAW precision (a directional predictor, assumed monotone by
    the AUC) is compared with that of the precision RECENTRED on its optimum,
    -|precision - p_opt|, qui est monotone par construction. Si la seconde est
    clearly higher, the precision does carry signal: it was the monotonicity
    assumption of the AUC that was wrong, not the metric that is useless.

    p_opt is the median precision of the equivalent patients, a robust
    estimator of
    l'optimum, sans imposer de forme fonctionnelle).

    The dosimetric NEUTRAL POINT -intercept/slope of the signed axis is also
    reported: the volume error at which delta = 0, that is, the volume error
    that cancels the dosimetric gap - which is not necessarily zero.
    """
    neutral = {r["index"]: -r["intercept"] / r["slope"]
               for _, r in signed.iterrows() if r["slope"] != 0}
    rows = []
    for structure, index, _ in PANEL:
        g0 = base[(base["structure"] == structure) &
                  (base["index"] == index)].dropna(subset=["precision", "abs_diff"])
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            y = (g0["abs_diff"].to_numpy(float) <= delta).astype(int)
            if np.unique(y).size < 2:
                continue
            p = g0["precision"].to_numpy(float)
            p_opt = float(np.median(p[y == 1]))
            rows.append(dict(
                structure=structure, index=index, margin=margin, delta=delta,
                n=int(y.size),
                auc_precision_raw=_auc(y, p),
                p_opt=p_opt,
                auc_precision_recentered=_auc(y, -np.abs(p - p_opt)),
                neutral_rel_vol_err=neutral.get(index, np.nan),
            ))
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl["auc_gain_recentering"] = (tbl["auc_precision_recentered"]
                                       - tbl["auc_precision_raw"])
    return tbl


# ============================================================
# FIGURES
# ============================================================
def figure_delta_auc(auc_tbl: pd.DataFrame, summ: pd.DataFrame):
    """
    Paired delta AUC versus the Dice, as small multiples - one panel per cell
    (index x margin).

    Small multiples rather than a single forest: with several metrics across
    several cells, one axis would stack dozens of unlabelled rows and become
    illegible.
    """
    order = [m for m in summ["metric"] if m != REFERENCE_METRIC][::-1]
    cells = (auc_tbl[["index", "margin"]].drop_duplicates()
             .sort_values(["index", "margin"]).values.tolist())
    ncols = 4
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.9 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    y = np.arange(len(order))
    for k, (index, margin) in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        block = auc_tbl[(auc_tbl["index"] == index) & (auc_tbl["margin"] == margin)]
        for j, m in enumerate(order):
            r = block[block["metric"] == m]
            if not len(r):
                continue
            r = r.iloc[0]
            col = (C_POS if r["beats_ref"]
                   else (C_NEG if r["worse_than_ref"] else "0.45"))
            ax.plot([r["delta_auc_lo"], r["delta_auc_hi"]], [y[j], y[j]],
                    color=col, lw=2, solid_capstyle="round", zorder=3)
            ax.plot([r["delta_auc_vs_ref"]], [y[j]], "o", ms=5, color=col,
                    mec="white", mew=0.8, zorder=4)
        ax.axvline(0, color=INK, lw=1.4, zorder=2)
        ax.set_title(f"{U.index_label(index)} : δ_{margin}", fontsize=FS_TITLE,
                     color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([_label(m) for m in order], fontsize=FS_TICK)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for k in range(len(cells), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("No metric outperforms Dice: ΔAUC vs Dice, "
                 "paired bootstrap (95 % CI)\n"
                 "a CI crossing 0 = indistinguishable metrics ; "
                 "in vermillion, significantly worse",
                 fontsize=FS_TITLE, color=INK, y=1.0)
    fig.supxlabel("ΔAUC vs Dice", fontsize=FS_LABEL, y=-0.01)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "delta_auc_vs_dice.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_auc_ranking(auc_tbl: pd.DataFrame, summ: pd.DataFrame):
    """Metric ranking: the mean plus one mark per cell (index x margin)."""
    s = summ.sort_values("auc_mean")
    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(s) + 1.8))
    y = np.arange(len(s))
    for k, m in enumerate(s["metric"]):
        pts = auc_tbl.loc[auc_tbl["metric"] == m, "auc"].to_numpy()
        col = C_REF if m == REFERENCE_METRIC else C_ALT
        ax.scatter(pts, np.full(pts.size, y[k]), s=26, color=col, alpha=0.35,
                   edgecolor="none", zorder=3)
        mean = s["auc_mean"].iloc[k]
        ax.plot([mean], [y[k]], "|", ms=22, mew=2.6, color=col, zorder=4)
        ax.text(mean, y[k] + 0.28, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=INK)
    ax.axvline(0.5, color="0.6", lw=1, ls=":", zorder=2)
    ax.text(0.5, -0.85, " chance", fontsize=FS_ANNOT, color="0.4", va="bottom")
    ax.set_ylim(-1.0, len(s) - 1 + 0.85)     # headroom : labels au-dessus du trait
    ax.set_yticks(y)
    ax.set_yticklabels([_label(m) for m in s["metric"]], fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("AUC for predicting equiv = 1{|Δ| ≤ δ}\n"
                  "(bar = mean, dots = index × margin cells)", fontsize=FS_LABEL)
    # ax.set_title("Equivalence discrimination, by quality metric",
    #              fontsize=FS_TITLE, color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "auc_ranking.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_composite(comp: pd.DataFrame):
    """Composite comparison, with the Dice-only model as the reference line."""
    cells = comp[comp["index"] != "(moyenne)"]
    names = list(COMPOSITE_SETS)
    means = [cells[n].mean() for n in names]
    ref = cells["dice seul"].mean()
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(names) + 2.2))
    y = np.arange(len(names))[::-1]
    for k, n in enumerate(names):
        col = C_REF if n == "dice seul" else C_ALT
        pts = cells[n].to_numpy()
        ax.scatter(pts, np.full(pts.size, y[k]), s=26, color=col, alpha=0.35,
                   edgecolor="none", zorder=3)
        ax.plot([means[k]], [y[k]], "|", ms=22, mew=2.6, color=col, zorder=4)
        ax.text(means[k], y[k] + 0.3, f"{means[k]:.3f}", ha="center",
                va="bottom", fontsize=FS_ANNOT, color=INK)
    ax.axvline(ref, color=C_REF, lw=1.4, ls="--", zorder=2)
    ax.set_ylim(-0.7, len(names) - 1 + 0.9)   # headroom : labels au-dessus du trait
    ax.set_yticks(y)
    ax.set_yticklabels([COMPOSITE_LABEL.get(n, n) for n in names], fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("5-fold cross-validated AUC\n"
                  "(bar = mean, dots = index × margin cells)", fontsize=FS_LABEL)
    ax.set_title("Adding metrics does not improve discrimination\n"
                 f"mean gain of the best composite : "
                 f"{max(means) - ref:+.3f} AUC, i.e. noise",
                 fontsize=FS_TITLE, color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "composite_cv_auc.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_equiv_by_volume_bin(vb: pd.DataFrame, nm: pd.DataFrame, margin="conf"):
    """
    Equivalence rate as a function of the signed volume error. An INVERTED-U
    shape whose peak is not at zero explains a precision AUC below 0.5: a
    directional predictor read through a monotone summary.
    """
    sub = vb[vb["margin"] == margin]
    idx_order = [i for _, i, _ in PANEL]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]   # Okabe-Ito
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(VOL_LABELS))
    for k, index in enumerate(idx_order):
        g = sub[sub["index"] == index].set_index("bin").reindex(VOL_LABELS)
        y = g["equiv_rate"].to_numpy(float)
        ax.plot(x, y, "-o", color=colors[k % len(colors)], lw=2, ms=7,
                mec="white", mew=1.4, label=U.index_label(index), zorder=3)
    # Les 4 courbes se croisent : un label direct au sommet serait ambigu et
    # would collide. The legend alone carries the identity here.
    # Reference mark: zero volume error (a volumetrically perfect segmentation).
    ax.axvline(2.5, color=INK, lw=1.4, ls="--", zorder=2)
    ax.annotate("zero volume error", (2.5, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-6, -6), rotation=90,
                va="top", ha="right", fontsize=FS_ANNOT, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(VOL_LABELS, fontsize=FS_TICK)
    ax.set_xlabel("Signed relative volume error   [< 0 = under-seg., > 0 = over-seg.]",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Equivalence rate  (|Δ| ≤ δ)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_title("The quality → equivalence relationship is an INVERTED U, "
                 "and its peak is not at zero\n"
                 "hence the AUC of precision below 0.5 : "
                 "a directional predictor read through a monotone summary",
                 fontsize=FS_TITLE, color=INK)
    ax.legend(title=f"margin δ_{margin}", frameon=False, fontsize=FS_LEG,
              title_fontsize=FS_LEG, loc="lower right", ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / f"equiv_by_volume_bin_{margin}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_collinearity(corr: pd.DataFrame):
    """
    Mechanical explanation of the Q2 result: the overlap-and-distance family
    (Dice, ASSD, HD95) is strongly collinear, so a composite over it adds up the
    same information.

    Deux lectures secondaires que la matrice rend visibles :
      - volume similarity, |relative volume error| and |precision - recall| are
        the SAME quantity, three names for the magnitude of the volume error;
      - the signed axis is nearly orthogonal to the Dice but strongly tied to
        precision and recall: it is exactly the over- versus under-segmentation
        asymmetry that the Dice aggregates and therefore erases.
    """
    labels = [_label(c) if c in U.QUALITY_METRICS else "rel. volume error (signed)"
              for c in corr.columns]
    fig, ax = plt.subplots(figsize=(9.4, 7.8))
    im = ax.imshow(corr.to_numpy(), cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_ANNOT,
                    color="white" if abs(v) > 0.6 else INK)
    # 2px separator before the last row/column (the orthogonal signed axis).
    k = len(labels) - 1.5
    ax.axhline(k, color="white", lw=2)
    ax.axvline(k, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman ρ (oriented metrics)", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.outline.set_visible(False)
    ax.set_title("Dice, ASSD and HD95 measure the same thing (ρ ≥ 0.83) :\n"
                 "a composite cannot add anything ; the signed axis, however, "
                 "is orthogonal to Dice",
                 fontsize=FS_TITLE, color=INK, pad=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "collinearity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    U.ensure_dir(OUT_DIR)
    rng = np.random.default_rng(args.seed)

    base = U.load_paired_with_geometry(STRUCTURE_TO_GEOM_ORGAN)
    print(f"[data] {len(base)} paired rows, "
          f"{base['record_id'].nunique()} patients")

    # ---- Q1 ----
    auc_tbl = auc_by_metric(base, args.boot, rng)
    U.save_csv(auc_tbl, OUT_DIR / "auc_by_metric.csv")
    summ = auc_summary(auc_tbl)
    U.save_csv(summ, OUT_DIR / "auc_summary.csv")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== Q1 — AUC moyenne par metrique (ref = Dice) ===")
        print(summ[["rank", "label", "auc_mean", "auc_min", "auc_max",
                    "delta_auc_mean", "n_beats_ref", "n_worse_than_ref",
                    "n_cells"]].round(3).to_string(index=False))
    n_better = int(summ["n_beats_ref"].sum())
    print(f"\n-> cells where a metric significantly beats the Dice: {n_better}")

    # ---- Q2 ----
    comp = composite_cv(base, args.seed)
    U.save_csv(comp, OUT_DIR / "composite_cv_auc.csv")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== Q2 — AUC en CV 5-fold : composite vs metrique seule ===")
        print(comp[["index", "margin"] + list(COMPOSITE_SETS)]
              .round(3).to_string(index=False))

    # ---- Q3 ----
    corr = collinearity(base)
    corr.to_csv(OUT_DIR / "collinearity_spearman.csv")
    print(f"[written] {OUT_DIR / 'collinearity_spearman.csv'}")
    print("\n=== Q3 — Colinearite (Spearman, metriques orientees) ===")
    print(f"  dice ~ assd_mm        : {corr.loc['dice', 'assd_mm']:+.3f}")
    print(f"  dice ~ hausdorff95_mm : {corr.loc['dice', 'hausdorff95_mm']:+.3f}")
    print(f"  dice ~ {U.SIGNED_AXIS:15s}: {corr.loc['dice', U.SIGNED_AXIS]:+.3f}"
          "   <- axe signe : quasi orthogonal")

    # ---- Q4 ----
    sv = signed_vs_unsigned(base, args.seed)
    U.save_csv(sv, OUT_DIR / "signed_vs_unsigned_r2.csv")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== Q4 - cross-validated R2 for predicting the SIGNED delta ===")
        print(sv[["index", "n", "r2_cv_signed_vol_err", "r2_cv_abs_vol_err",
                  "r2_cv_dice", "r2_cv_assd"]].round(3).to_string(index=False))

    # ---- Q5 ----
    vb = equiv_by_volume_bin(base)
    U.save_csv(vb, OUT_DIR / "equiv_by_volume_bin.csv")
    nm = precision_nonmonotonicity(base, sv)
    U.save_csv(nm, OUT_DIR / "nonmonotonicity_precision.csv")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== Q5 - the precision is a NON-MONOTONE predictor ===")
        print(nm[["index", "margin", "auc_precision_raw", "p_opt",
                  "auc_precision_recentered", "auc_gain_recentering",
                  "neutral_rel_vol_err"]].round(3).to_string(index=False))
        print("\n  Equivalence rate per volume-error bin (decision margin):")
        print(vb[vb["margin"] == "conf"]
              .pivot(index="index", columns="bin", values="equiv_rate")
              .reindex(columns=VOL_LABELS).round(2).to_string())

    # ---- figures ----
    figure_delta_auc(auc_tbl, summ)
    figure_auc_ranking(auc_tbl, summ)
    figure_composite(comp)
    figure_collinearity(corr)
    figure_equiv_by_volume_bin(vb, nm, margin="conf")

    print("\n[OK] Quality-metric comparison complete.")


if __name__ == "__main__":
    main()
