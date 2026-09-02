"""
05_dice_threshold.py
====================
Extension of the concordance arm: SEGMENTATION QUALITY THRESHOLD for the
equivalence of the DVH indices AT THE PATIENT LEVEL (rather than the cohort
level).

Conceptual pivot
----------------
The TOST (scripts 01-02) is a COHORT statement: does the MEAN of the paired
deltas lie inside +/- delta? For ONE patient there is a single delta per index,
so there is no mean and no TOST. Patient-level equivalence reduces to a margin:

        equiv_i = 1  if |delta_i| <= margin,  0 otherwise

"Where is the quality threshold for equivalence?" then becomes a
CLASSIFICATION / TOLERANCE problem: above what contour quality is a patient
equivalent?

The PANEL and the margin table (delta_conf / delta_sens) of utils.py are reused
unchanged, as is the convention delta = val_manual - val_auto (manual =
reference).

TWO ORTHOGONAL AXES
-------------------
This script reports two complementary readings, NOT a composite score; script 06
shows empirically that a composite adds nothing, the unsigned metrics being
collinear.

  AXIS 1 - THRESHOLD (unsigned): "can I trust this index?"
        Estimated on each metric of U.THRESHOLD_METRICS, whose discrimination is
        indistinguishable (see 06). The Dice remains the conventional axis; the
        ASSD gives the PHYSICAL reading, in mm, hence a review criterion that is
        directly actionable in the clinic.

  AXIS 2 - DIRECTION (signed): "in which direction, and by how much, am I
        wrong?" A regression of the SIGNED delta on the signed relative volume
        error. No unsigned metric can answer this: the Dice has no explanatory
        power on the signed delta by construction, since it ignores the
        direction of the contouring error. This axis is nearly orthogonal to the
        Dice, so it is genuinely new information.

Three DUAL threshold estimators, per (structure, index, metric, margin)
-----------------------------------------------------------------------
Everything is computed on the ORIENTED SCORE s = orient * x (orient from
U.QUALITY_METRICS), so that a high score always means a better segmentation.
Thresholds are converted back to the native unit before being reported.

  (1) LOGISTIC - P(equiv=1 | s) = sigmoid(b0 + b1*s).
        Threshold s* = (logit(p*) - b0)/b1 for p* in {0.80, 0.90, 0.95}.
        Reading: "above x*, this patient has at least p* chance of being
        equivalent". A direct generalisation of the TOST flag to the patient
        level.

  (2) QUANTILE / TOLERANCE - the tau=0.95 quantile of |delta| regressed on s.
        Q95(|delta| | s) = a + b*s; the threshold s* satisfies Q95 = margin,
        i.e. s* = (margin - a)/b.
        Reading: "above x*, 95% of patients have |delta| <= margin".

  (3) Non-parametric CONTROL - Youden: s used as a classifier of equiv, with the
        cutpoint maximising sensitivity + specificity (plus the AUC). It
        validates the monotone shape assumed by (1) and (2).

Each threshold comes with a 95% bootstrap CI (resampling patients) and an
EXTRAPOLATION flag: a threshold outside the quality range actually observed is
an artefact of the functional form, not a quotable value.

Structural caveat: a threshold exists only where the metric DRIVES the
discordance. If the logistic slope is about 0, or the quantile slope is
non-negative, the script reports "no identifiable threshold" rather than a
misleading number.

Outputs (in the quality_threshold subdirectory of the results directory):
    - quality_threshold_table.csv  thresholds, CIs and flags, per
                                   (structure, index, metric, margin);
    - signed_volume_axis.csv       axis 2 (direction of the error), no margin;
    - equiv_logistic_<metric>_<margin>.png
    - quantile_<metric>_<margin>.png
    - signed_volume_axis.png

Usage: python 05_dice_threshold.py [--boot 400] [--seed 0]
       [--margins conf sens] [--p-target 0.90] [--only-logistic] [--tag ...]

The restriction options (--margins / --p-target / --only-logistic) automatically
suffix the output files, so a partial run never overwrites the reference table.
--p-target selects the HIGHLIGHTED logistic target (bootstrap CI, extrapolation
flag, vertical line in the figures, printed summary); the thr_p80/thr_p90/thr_p95
columns are all reported regardless.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils as U

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

# ============================================================
# CHEMINS & CONSTANTES
# ============================================================
OUT_DIR = U.OUT_DIR / "quality_threshold"

# Local panel: Prostate only. The cross-dataset geometry drives the prostate
# alone; BladderNeck (a derived mask whose geometry is only a prostate proxy) is
# excluded -
# its chance-level AUC shows that the wrong ORGAN is being used, not the wrong
# metric, and switching metric would not save it.
STRUCTURE_TO_GEOM_ORGAN = {
    "Prostate": "Prostate",
    # "BladderNeck": "Prostate",   # mask derived from the prostate model (proxy)
}
PANEL = [(s, i, t) for s, i, t in U.PANEL if s in STRUCTURE_TO_GEOM_ORGAN]

P_TARGETS = (0.80, 0.90, 0.95)   # probability targets for the logistic model
TAU = 0.95                        # quantile of |delta| for the tolerance bound
MARGINS = ("conf", "sens")        # delta_conf / delta_sens

ORGAN_COLORS = {"Prostate": "#0072B2", "BladderNeck": "#CC79A7",
                "Rectum": "#D55E00", "Bladder": "#009E73"}

# Figure font sizes: a legible floor once scaled down to a column.
FS_TICK = 12
FS_LABEL = 14
FS_TITLE = 15
FS_ANNOT = 12
FS_SUPTITLE = 16


def _p_key(p: float) -> str:
    """Column name of a probability target, e.g. 0.95 -> 'thr_p95'."""
    return f"thr_p{int(round(p * 100))}"


def _metric_label(metric: str) -> str:
    """Axis label with its unit, e.g. 'ASSD (mm)'."""
    meta = U.QUALITY_METRICS[metric]
    return f"{meta['label']} ({meta['unite']})" if meta["unite"] else meta["label"]


# ============================================================
# ORIENTATION : score s = orient * x, « plus haut = meilleur »
# ============================================================
def _to_native(score, orient):
    """Convert a threshold from oriented space back to the native unit
    (x = orient*s)."""
    return orient * np.asarray(score, float)


def _ci_native(lo, hi, orient):
    """CI converted back to native units; orient=-1 swaps the bound order."""
    lo_n, hi_n = _to_native(lo, orient), _to_native(hi, orient)
    return (lo_n, hi_n) if orient > 0 else (hi_n, lo_n)


# ============================================================
# THRESHOLD ESTIMATORS (on an ORIENTED score plus label/value)
# ============================================================
def logistic_thresholds(score, equiv, p_targets=P_TARGETS):
    """
    Ajuste P(equiv|s)=σ(b0+b1·s). Renvoie b0,b1, converged, et s* pour chaque
    p_target (NaN si pente non positive / non identifiable).
    """
    score = np.asarray(score, float)
    equiv = np.asarray(equiv, float)
    res = dict(b0=np.nan, b1=np.nan, converged=False,
               **{f"thr_p{int(p*100)}": np.nan for p in p_targets})
    if np.unique(equiv).size < 2:      # a single class: no boundary
        res["flag"] = "single_class"
        return res
    X = sm.add_constant(score)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(equiv, X).fit(disp=0, maxiter=200)
        b0, b1 = float(fit.params[0]), float(fit.params[1])
        conv = bool(fit.mle_retvals.get("converged", False))
    except (PerfectSeparationError, np.linalg.LinAlgError, Exception):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.Logit(equiv, X).fit_regularized(disp=0, alpha=1.0)
            b0, b1 = float(fit.params[0]), float(fit.params[1])
            conv = True
        except Exception:
            res["flag"] = "fit_failed"
            return res
    res.update(b0=b0, b1=b1, converged=conv)
    if b1 <= 0:                        # inverted relation -> no useful threshold
        res["flag"] = "slope_nonpositive"
        return res
    for p in p_targets:
        res[f"thr_p{int(p*100)}"] = (np.log(p / (1 - p)) - b0) / b1
    res["flag"] = "ok"
    return res


def quantile_threshold(score, abs_diff, delta, tau=TAU):
    """
    Quantile regression at level tau of |delta| on the oriented score:
    Q_tau = a + b*s. The threshold s* satisfies Q_tau = delta, i.e.
    s* = (delta - a)/b. Returns NaN when b >= 0, since the tolerance then fails
    to improve as the segmentation improves and the relation is unusable.
    """
    score = np.asarray(score, float)
    y = np.asarray(abs_diff, float)
    res = dict(q_a=np.nan, q_b=np.nan, thr_tol=np.nan, flag="")
    if np.ptp(score) == 0 or score.size < 10:
        res["flag"] = "insufficient"
        return res
    X = sm.add_constant(score)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QuantReg(y, X).fit(q=tau, max_iter=2000)
        a, b = float(fit.params[0]), float(fit.params[1])
    except Exception:
        res["flag"] = "fit_failed"
        return res
    res.update(q_a=a, q_b=b)
    if b >= 0:
        res["flag"] = "slope_nonnegative"
        return res
    res["thr_tol"] = (delta - a) / b
    res["flag"] = "ok"
    return res


def youden(score, equiv):
    """Cutpoint of the oriented score maximising Youden J = sens+spec-1, plus AUC."""
    score = np.asarray(score, float)
    equiv = np.asarray(equiv, int)
    res = dict(thr_youden=np.nan, youden_j=np.nan, auc=np.nan,
               sens=np.nan, spec=np.nan, flag="")
    if np.unique(equiv).size < 2:
        res["flag"] = "single_class"
        return res
    fpr, tpr, thr = roc_curve(equiv, score)   # oriented score, positive = equiv
    j = tpr - fpr
    k = int(np.argmax(j))
    res.update(thr_youden=float(thr[k]), youden_j=float(j[k]),
               auc=float(roc_auc_score(equiv, score)),
               sens=float(tpr[k]), spec=float(1 - fpr[k]), flag="ok")
    return res


def _bootstrap_ci(score, y, estim_fn, keys, n_boot, rng):
    """95% percentile bootstrap CI (resampling patients), in oriented space."""
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    n = score.size
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = estim_fn(score[idx], y[idx])
        for k in keys:
            v = r.get(k, np.nan)
            if v is not None and np.isfinite(v):
                acc[k].append(v)
    out = {}
    for k in keys:
        vals = np.asarray(acc[k], float)
        if vals.size >= max(20, int(0.5 * n_boot)):   # exige assez de fits valides
            out[f"{k}_lo"] = float(np.percentile(vals, 2.5))
            out[f"{k}_hi"] = float(np.percentile(vals, 97.5))
        else:
            out[f"{k}_lo"] = np.nan
            out[f"{k}_hi"] = np.nan
    return out


def _extrapolation_flag(thr_native, x_native) -> str:
    """
    'ok' when the threshold falls inside the OBSERVED quality range, otherwise
    'extrapolated_low' or 'extrapolated_high'. An extrapolated threshold only
    reflects the imposed functional form (sigmoid, linear quantile) and must not
    be quoted as is.
    """
    if not np.isfinite(thr_native):
        return ""
    lo, hi = float(np.nanmin(x_native)), float(np.nanmax(x_native))
    if thr_native < lo:
        return "extrapolated_low"
    if thr_native > hi:
        return "extrapolated_high"
    return "ok"


# ============================================================
# AXIS 1 - PIPELINE PER (structure, index, metric, margin)
# ============================================================
def analyse_group(g: pd.DataFrame, structure, index, metric, margin, delta,
                  n_boot, rng, p_target=0.90, only_logistic=False):
    """Thresholds for one group. `p_target` designates the HIGHLIGHTED logistic
    AVANT (IC bootstrap, flag d'extrapolation, trait vertical des figures) ; les
    trois cibles de P_TARGETS restent rapportees en colonnes. `only_logistic`
    skips estimators (2) and (3)."""
    orient = U.QUALITY_METRICS[metric]["orient"]
    x = g[metric].to_numpy(float)            # native unit
    absd = g["abs_diff"].to_numpy(float)
    keep = ~(np.isnan(x) | np.isnan(absd))
    x, absd = x[keep], absd[keep]
    s = orient * x                            # oriented score
    equiv = (absd <= delta).astype(int)

    row = dict(structure=structure, index=index,
               tier=U.DELTA[(structure, index)]["tier"],
               unite=U.DELTA[(structure, index)]["unite"],
               metric=metric, metric_unite=U.QUALITY_METRICS[metric]["unite"],
               metric_orient=orient,
               margin=margin, delta=delta, n=int(x.size),
               geom_organ=g["geom_organ"].iloc[0],
               equiv_rate=float(equiv.mean()) if equiv.size else np.nan,
               metric_min=float(np.min(x)), metric_max=float(np.max(x)),
               metric_median=float(np.median(x)))

    # --- (1) logistique ---
    pkey = _p_key(p_target)
    lg = logistic_thresholds(s, equiv)
    row.update(logit_b1=lg["b1"], logit_flag=lg["flag"], p_target=p_target)
    for p in P_TARGETS:
        k = _p_key(p)
        row[k] = _to_native(lg[k], orient)
    row[f"{pkey}_extrap"] = _extrapolation_flag(row[pkey], x)
    ci = _bootstrap_ci(s, equiv, lambda d, y: logistic_thresholds(d, y),
                       [pkey], n_boot, rng)
    lo, hi = _ci_native(ci[f"{pkey}_lo"], ci[f"{pkey}_hi"], orient)
    row.update({f"{pkey}_lo": lo, f"{pkey}_hi": hi})

    if only_logistic:
        return row, lg, None

    # --- (2) quantile / tolerance ---
    qt = quantile_threshold(s, absd, delta)
    tol = _to_native(qt["thr_tol"], orient)
    row.update(quant_b=qt["q_b"], quant_flag=qt["flag"], thr_tol=tol,
               thr_tol_extrap=_extrapolation_flag(tol, x))
    ciq = _bootstrap_ci(s, absd, lambda d, y: quantile_threshold(d, y, delta),
                        ["thr_tol"], n_boot, rng)
    lo, hi = _ci_native(ciq["thr_tol_lo"], ciq["thr_tol_hi"], orient)
    row.update(thr_tol_lo=lo, thr_tol_hi=hi)

    # --- (3) Youden ---
    yd = youden(s, equiv)
    row.update(thr_youden=_to_native(yd["thr_youden"], orient),
               youden_j=yd["youden_j"], auc=yd["auc"],
               youden_sens=yd["sens"], youden_spec=yd["spec"],
               youden_flag=yd["flag"])
    return row, lg, qt


# ============================================================
# AXIS 2 - DIRECTION OF THE ERROR (SIGNED relative volume error)
# ============================================================
def signed_axis(base: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Regress the SIGNED delta on the SIGNED relative volume error.

    This axis does not answer "can I trust this?" (axis 1) but "in which
    direction is the automatic index biased, and by how much?". The R2 is
    reported under 5-fold cross-validation, hence honest and free of
    overfitting, alongside that of the Dice on the same target: their gap
    quantifies exactly what
    que le Dice ne peut PAS dire.

    Convention : rel_vol_err > 0 <=> SUR-segmentation (pred > gt) ;
    Δ = manuel - auto. Une pente positive signifie donc « plus le contour auto
    est gros, plus l'indice auto sous-estime l'indice manuel ».
    """
    axis = U.SIGNED_AXIS
    cv = KFold(5, shuffle=True, random_state=seed)
    rows, fits = [], {}
    for structure, index, tier in PANEL:
        g = base[(base["structure"] == structure) & (base["index"] == index)]
        g = g.dropna(subset=[axis, "dice", "diff"])
        x = g[axis].to_numpy(float)
        y = g["diff"].to_numpy(float)
        if x.size < 20 or np.ptp(x) == 0:
            continue
        lr = sm.OLS(y, sm.add_constant(x)).fit()
        a, b = float(lr.params[0]), float(lr.params[1])
        r2_cv = float(cross_val_score(LinearRegression(), x[:, None], y,
                                      cv=cv, scoring="r2").mean())
        r2_cv_dice = float(cross_val_score(
            LinearRegression(), g["dice"].to_numpy(float)[:, None], y,
            cv=cv, scoring="r2").mean())
        rows.append(dict(
            structure=structure, index=index, tier=tier, n=int(x.size),
            axis=axis, slope=b, intercept=a,
            slope_p=float(lr.pvalues[1]),
            r2_insample=float(lr.rsquared), r2_cv=r2_cv,
            r2_cv_dice=r2_cv_dice, r2_gain_vs_dice=r2_cv - r2_cv_dice,
            # Actionable reading: expected bias of the automatic index for a
            # over- or under-segmentation of 10% of the volume.
            pred_diff_at_plus10pct=a + b * 0.10,
            pred_diff_at_minus10pct=a - b * 0.10,
        ))
        fits[(structure, index)] = dict(a=a, b=b, r2_cv=r2_cv)
    return pd.DataFrame(rows), fits


# ============================================================
# FIGURES
# ============================================================
def _panels(pairs):
    n = len(pairs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.7 * nrows),
                             squeeze=False)
    return fig, axes, ncols, nrows


def _anchor_ha(xv, x):
    """Horizontal anchoring of a label placed at xv over the range of x.

    Une boite centree sur un seuil proche d'un bord chevauche l'axe (ou sort du
    cadre) — d'autant plus depuis que les polices ont ete agrandies. On ancre
    donc du cote oppose au bord le plus proche.
    """
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi <= lo:
        return "center"
    frac = (xv - lo) / (hi - lo)
    if frac < 0.20:
        return "left"
    if frac > 0.80:
        return "right"
    return "center"


def _draw_logistic(axes, tbl, fits, metric, ncols, nrows, p_target=0.90):
    """Draw the logistic panels into an already built axis grid.

    Extrait de figure_logistic pour que la figure composite reutilise le MEME
    code de trace, et non une copie qui deriverait a la premiere retouche.
    """
    orient = U.QUALITY_METRICS[metric]["orient"]
    pairs = [(s, i) for s, i, _ in PANEL]
    for k, (structure, index) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        g = tbl[(tbl["structure"] == structure) & (tbl["index"] == index)]
        x = g[metric].to_numpy(float)
        equiv = g["equiv"].to_numpy()
        color = ORGAN_COLORS.get(structure, "#555")
        jit = (np.random.default_rng(0).random(equiv.size) - 0.5) * 0.06
        ax.scatter(x, equiv + jit, s=12, color=color, alpha=0.5,
                   edgecolor="none", zorder=2)
        lg = fits[(structure, index)]["logit"]
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        if np.isfinite(lg["b1"]):
            ys = 1 / (1 + np.exp(-(lg["b0"] + lg["b1"] * orient * xs)))
            ax.plot(xs, ys, color="0.15", lw=2, zorder=4)
        dstar = _to_native(lg.get(_p_key(p_target), np.nan), orient)
        if np.isfinite(dstar) and np.nanmin(x) <= dstar <= np.nanmax(x):
            ax.axvline(dstar, color="0.15", ls="--", lw=1.4, zorder=3)
            # Label above the cloud: at mid-height it collided with the inset.
            ax.text(dstar, 1.20, f"threshold = {dstar:.2f}", va="center",
                    ha=_anchor_ha(dstar, x), fontsize=FS_ANNOT, color="0.1",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8"))
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axhline(1, color="0.85", lw=0.8)
        ax.set_ylim(-0.15, 1.35)
        ax.set_title(f"{structure} : {U.index_label(index)}", fontsize=FS_TITLE)
        ax.set_xlabel(_metric_label(metric), fontsize=FS_LABEL)
        ax.set_ylabel("equivalent (|Δ| ≤ δ)", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.text(0.97, 0.55, f"equivalence rate = {equiv.mean():.2f}\nn = {equiv.size}",
                transform=ax.transAxes, va="center", ha="right", fontsize=FS_ANNOT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
        ax.grid(True, color="0.93", lw=0.6)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")


def figure_logistic(tbl, fits, metric, margin, p_target=0.90, tag=""):
    pairs = [(s, i) for s, i, _ in PANEL]
    fig, axes, ncols, nrows = _panels(pairs)
    _draw_logistic(axes, tbl, fits, metric, ncols, nrows, p_target=p_target)
    # fig.suptitle(f"P(equivalent patient | {U.QUALITY_METRICS[metric]['label']})"
                #  f" : margin δ_{margin}", fontsize=FS_SUPTITLE, y=1.0)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / f"equiv_logistic_{metric}_{margin}{tag}.png",
                      bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def _draw_quantile(axes, tbl, fits, metric, ncols, nrows):
    """Draw the Q95 tolerance panels into an already built axis grid."""
    orient = U.QUALITY_METRICS[metric]["orient"]
    pairs = [(s, i) for s, i, _ in PANEL]
    for k, (structure, index) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        g = tbl[(tbl["structure"] == structure) & (tbl["index"] == index)]
        x = g[metric].to_numpy(float)
        absd = g["abs_diff"].to_numpy(float)
        delta = g["delta"].iloc[0]
        color = ORGAN_COLORS.get(structure, "#555")
        ax.scatter(x, absd, s=12, color=color, alpha=0.5, edgecolor="none")
        qt = fits[(structure, index)]["quant"]
        dstar = _to_native(qt.get("thr_tol", np.nan), orient)

        ax.axhline(delta, color="0.15", lw=1.4, ls=":")
        # The margin label goes on the side OPPOSITE the threshold verticalthe
        # metric orientation puts that line on the left (ASSD) or on the right
        # (Dice), so a fixed side would collide with it half the time.
        if _anchor_ha(dstar, x) == "right":
            ax.text(np.nanmin(x), delta, f" δ = {delta:g}", va="bottom", ha="left",
                    fontsize=FS_ANNOT, color="0.1")
        else:
            ax.text(np.nanmax(x), delta, f"δ = {delta:g} ", va="bottom", ha="right",
                    fontsize=FS_ANNOT, color="0.1")

        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        if np.isfinite(qt["q_b"]):
            ax.plot(xs, qt["q_a"] + qt["q_b"] * orient * xs, color="0.15", lw=2,
                    label=f"Q{int(TAU*100)}(|Δ|)")
        if np.isfinite(dstar) and np.nanmin(x) <= dstar <= np.nanmax(x):
            ax.axvline(dstar, color="0.15", ls="--", lw=1.4)
            ax.text(dstar, ax.get_ylim()[1] * 0.97, f"threshold = {dstar:.2f}",
                    va="top", ha=_anchor_ha(dstar, x), fontsize=FS_ANNOT, color="0.1",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8"))
        ax.set_title(f"{structure} : {U.index_label(index)}", fontsize=FS_TITLE)
        ax.set_xlabel(_metric_label(metric), fontsize=FS_LABEL)
        ax.set_ylabel("|Δ|", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color="0.93", lw=0.6)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")


def figure_quantile(tbl, fits, metric, margin):
    pairs = [(s, i) for s, i, _ in PANEL]
    fig, axes, ncols, nrows = _panels(pairs)
    _draw_quantile(axes, tbl, fits, metric, ncols, nrows)
    fig.suptitle(f"Tolerance bound Q{int(TAU*100)}(|Δ|) vs "
                 f"{U.QUALITY_METRICS[metric]['label']} : margin δ_{margin}",
                 fontsize=FS_SUPTITLE, y=1.0)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / f"quantile_{metric}_{margin}.png",
                      bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_logistic_quantile(tbl, fits, metric, margin):
    """Figure composite : (a) logistique P(equiv), (b) tolerance Q95 de |Δ|.

    Les deux blocs partagent l'axe des abscisses (la metrique de qualite) et le
    meme seuil vertical : les reunir montre que les deux estimateurs, l'un a
    fixed quality reading a probability, the other at fixed coverage reading a
    qualite, tombent au meme endroit. Sortie PDF seule (figure d'article).
    """
    pairs = [(s, i) for s, i, _ in PANEL]
    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig = plt.figure(figsize=(4.8 * ncols, 2 * 3.7 * nrows + 0.5))
    blocks = fig.subfigures(2, 1, hspace=0.02)

    for sub, draw in ((blocks[0], _draw_logistic), (blocks[1], _draw_quantile)):
        axes = sub.subplots(nrows, ncols, squeeze=False)
        draw(axes, tbl, fits, metric, ncols, nrows)

    # Sub-figure labels: placed on the subfigure, they stay attached to their
    # block whatever the number of panels.
    for sub, tag in zip(blocks, ("(a)", "(b)")):
        sub.text(0.005, 0.995, tag, fontsize=FS_SUPTITLE, fontweight="bold",
                 color="0.1", ha="left", va="top")

    for sub in blocks:
        sub.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.11,
                            hspace=0.45, wspace=0.28)

    f = OUT_DIR / f"equiv_logistic_quantile_{metric}_{margin}.pdf"
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def signed_axis_pairs(fits):
    """Couples (structure, indice) reellement ajustes sur l'axe signe."""
    return [(s, i) for s, i, _ in PANEL if (s, i) in fits]


def _draw_signed_axis(axes, base, fits, ncols, nrows):
    """Draw the signed-axis panels into an already built axis grid."""
    axis = U.SIGNED_AXIS
    pairs = signed_axis_pairs(fits)
    for k, (structure, index) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        g = base[(base["structure"] == structure) &
                 (base["index"] == index)].dropna(subset=[axis, "diff"])
        x = g[axis].to_numpy(float) * 100.0     # en % du volume GT
        y = g["diff"].to_numpy(float)
        color = ORGAN_COLORS.get(structure, "#555")
        ax.axhline(0, color="0.6", lw=1, zorder=1)
        ax.axvline(0, color="0.6", lw=1, zorder=1)
        ax.scatter(x, y, s=12, color=color, alpha=0.5, edgecolor="none", zorder=2)
        f = fits[(structure, index)]
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, f["a"] + f["b"] * xs / 100.0, color="0.15", lw=2, zorder=4)
        ax.text(0.03, 0.96, f"R² (CV) = {f['r2_cv']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=FS_ANNOT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
        ax.set_title(f"{structure} : {U.index_label(index)}", fontsize=FS_TITLE)
        ax.set_xlabel("Relative volume error (%)\n[> 0 = over-seg.]", fontsize=FS_LABEL)
        ax.set_ylabel("Δ = manual − auto", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color="0.93", lw=0.6)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")


def figure_signed_axis(base, fits):
    """Signed delta vs signed relative volume error - the axis the Dice cannot see."""
    pairs = signed_axis_pairs(fits)
    if not pairs:
        return
    fig, axes, ncols, nrows = _panels(pairs)
    _draw_signed_axis(axes, base, fits, ncols, nrows)
    # fig.suptitle("SIGNED axis: direction of the dosimetric error ", fontsize=FS_SUPTITLE, y=1.0)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "signed_volume_axis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


# ============================================================
# MAIN
# ============================================================
def _run_tag(args) -> str:
    """Output suffix: empty in the default configuration, so the reference files
    are never overwritten by a restricted run; otherwise an explicit marker of
    the requested subset."""
    if args.tag:
        return "_" + args.tag.lstrip("_")
    bits = []
    if args.only_logistic:
        bits.append("logit")
    if abs(args.p_target - 0.90) > 1e-9:
        bits.append(f"p{int(round(args.p_target * 100))}")
    if tuple(args.margins) != MARGINS:
        bits.append("-".join(args.margins))
    return "_" + "_".join(bits) if bits else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400, help="bootstrap iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margins", nargs="+", choices=MARGINS, default=list(MARGINS),
                    help="margins to process (default: conf and sens)")
    ap.add_argument("--p-target", type=float, default=0.90, dest="p_target",
                    help="highlighted logistic probability target "
                         "(IC bootstrap, flag d'extrapolation, figures)")
    ap.add_argument("--only-logistic", action="store_true", dest="only_logistic",
                    help="estimate only the logistic model: skip the quantile, "
                         "Youden and axis 2 (direction of the error)")
    ap.add_argument("--tag", default="", help="suffixe de fichiers de sortie")
    args = ap.parse_args()

    if _p_key(args.p_target) not in {_p_key(p) for p in P_TARGETS}:
        ap.error(f"--p-target must be one of {P_TARGETS}")

    U.ensure_dir(OUT_DIR)
    rng = np.random.default_rng(args.seed)
    tag = _run_tag(args)
    pkey = _p_key(args.p_target)

    base = U.load_paired_with_geometry(STRUCTURE_TO_GEOM_ORGAN)
    print(f"[data] {len(base)} paired rows, "
          f"{base['record_id'].nunique()} patients")

    # ---------- AXE 1 : seuils ----------
    rows = []
    for metric in U.THRESHOLD_METRICS:
        for margin in args.margins:
            fits = {}
            parts = []
            for structure, index, _ in PANEL:
                g = base[(base["structure"] == structure) &
                         (base["index"] == index)].copy()
                delta = U.DELTA[(structure, index)][f"delta_{margin}"]
                row, lg, qt = analyse_group(g, structure, index, metric, margin,
                                            delta, args.boot, rng,
                                            p_target=args.p_target,
                                            only_logistic=args.only_logistic)
                rows.append(row)
                fits[(structure, index)] = {"logit": lg, "quant": qt}
                g["equiv"] = (g["abs_diff"] <= delta).astype(int)
                g["delta"] = delta
                parts.append(g)
            mt = pd.concat(parts, ignore_index=True)
            figure_logistic(mt, fits, metric, margin, p_target=args.p_target,
                            tag=tag)
            if not args.only_logistic:
                figure_quantile(mt, fits, metric, margin)
                figure_logistic_quantile(mt, fits, metric, margin)

    summary = pd.DataFrame(rows)
    U.save_csv(summary, OUT_DIR / f"quality_threshold_table{tag}.csv")

    show = ["index", "metric", "margin", "delta", "equiv_rate", pkey,
            f"{pkey}_lo", f"{pkey}_hi", f"{pkey}_extrap"]
    if not args.only_logistic:
        show += ["thr_tol", "thr_youden", "auc"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        est = "logistique seule" if args.only_logistic else "3 estimateurs"
        print(f"\n=== AXIS 1 - Quality thresholds for patient-level equivalence "
              f"({est}, p*={args.p_target:.2f}, marges={'/'.join(args.margins)}) ===")
        print(summary[show].round(3).to_string(index=False))

    # ---------- AXIS 2: direction of the error ----------
    signed, sfits = (pd.DataFrame(), {}) if args.only_logistic \
        else signed_axis(base, seed=args.seed)
    if len(signed):
        U.save_csv(signed, OUT_DIR / "signed_volume_axis.csv")
        figure_signed_axis(base, sfits)
        cols = ["index", "n", "slope", "r2_cv", "r2_cv_dice", "r2_gain_vs_dice",
                "pred_diff_at_plus10pct", "pred_diff_at_minus10pct"]
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print("\n=== AXIS 2 - Direction of the error (signed rel. volume error) ===")
            print(signed[cols].round(3).to_string(index=False))

    print("\n[OK] Quality-threshold analysis complete.")


if __name__ == "__main__":
    main()
