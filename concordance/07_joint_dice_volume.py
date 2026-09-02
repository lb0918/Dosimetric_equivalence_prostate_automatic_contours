"""
07_joint_dice_volume.py
=======================
JOINT ANALYSIS: does the PATIENT-level equivalence of the DVH indices depend on
the segmentation quality (Dice) AND on the PROSTATE VOLUME - and is the Dice
threshold the same for every prostate size?

Position relative to script 05
------------------------------
Script 05 estimates a SINGLE quality threshold per index: "above a given Dice,
this patient has a given chance of having |delta| within the margin". That
threshold is implicitly assumed VALID FOR EVERY PATIENT. Here that assumption is
lifted by adding a PATIENT covariate - the clinical prostate volume
(`ldr_post_vol`) - and two nested questions are asked:

  Q1 - At equal Dice, does the prostate volume change the probability of
       equivalence? (MAIN effect of volume: likelihood-ratio test M1 -> M2)

  Q2 - Does the Dice THRESHOLD itself depend on the volume?
       (Dice x volume INTERACTION: likelihood-ratio test M2 -> M3)

Q2 is the clinically decisive question. A main effect alone shifts the curve; an
interaction shifts the THRESHOLD, and it is the threshold that a
quality-assurance protocol quotes. If the interaction is real, a single
threshold is too lax for part of the cohort and too strict for the rest.

Why the question is not gratuitous
----------------------------------
The Dice is mechanically penalised on small organs: at a constant contouring
error in millimetres, the surface-to-volume ratio drives the Dice down as the
organ shrinks. The same Dice on a small and on a large prostate therefore does
NOT describe the same geometric error, and there is no reason for it to imply
the same dosimetric reliability. The script quantifies that coupling
(descriptive block) before modelling it.

The prostate volume is moreover known BEFORE any segmentation: if the threshold
depends on it, that yields an a-priori triage rule (require a higher Dice, or
systematically review, on small prostates) usable upstream of the automatic
pipeline.

Models (per structure x index x metric x margin)
------------------------------------------------
The label is identical to 05: equiv = 1{|delta| <= margin}, with the
pre-declared margin, and the ORIENTED score s = orient*x (U.QUALITY_METRICS) so
that a high s means a better segmentation. The volume is centred and scaled as
v = (V - V0)/VOL_SCALE (V0 = median of the cell), so the coefficients read "per
VOL_SCALE cc", and the thresholds are still reported in the NATIVE unit.

    M1:  logit P(equiv) = b0 + b1*s                      (the model of 05)
    M2:  logit P(equiv) = b0 + b1*s + b2*v               (volume main effect)
    M3:  logit P(equiv) = b0 + b1*s + b2*v + b3*s*v      (interaction)

The quality threshold then becomes a CURVE rather than a number:

        s*(v) = [logit(p*) - b0 - b2*v] / (b1 + b3*v)

    - under M2 (b3 = 0) the threshold shifts linearly with the volume, at slope
      -b2/b1 (in score units per VOL_SCALE cc);
    - under M3 the slope itself depends on the volume.

DUAL estimator, as in 05 - quantile regression at tau = 0.95 of |delta| on
(s, v, s*v):

        Q95(|delta| | s, v) = a0 + a1*s + a2*v + a3*s*v = margin
        =>  s*(v) = [margin - a0 - a2*v] / (a1 + a3*v)

Reading: "at this volume, above x*, 95% of patients have |delta| within the
margin". The logistic fixes the quality and reads a probability; the quantile
fixes a coverage and reads the quality. Their convergence is the robustness
guarantee.

NON-PARAMETRIC CONTROL - volume tertiles. Within each tertile the UNIVARIATE
estimators of 05 (logistic, Q95 tolerance, Youden, AUC) are re-applied, imported
as is from module 05, so no functional form is imposed on the volume. If the
per-tertile threshold follows the curve s*(v), the interaction is real and not a
specification artefact.

Guards
------
  - 95% bootstrap CI (resampling PATIENTS) over the whole curve s*(v);
  - EXTRAPOLATION flag per volume point: a threshold outside the quality range
    actually observed is not quotable (the same rule as in 05);
  - SUPPORT flag: the figures mask the regions of the (quality x volume) plane
    where no patient was observed - a model produces a surface there, not a
    result;
  - 5-fold CROSS-VALIDATED AUC for M1/M2/M3. The likelihood-ratio test is a
    resubstitution statistic: it says whether the term is statistically
    detectable, not whether it improves prediction. Both readings are reported
    side by side, and it is their possible DISAGREEMENT that must be discussed
    (a real effect too small to change a decision).

Outputs (in the joint_dice_volume subdirectory of the results directory):
    - volume_quality_descriptive.csv  coupling between volume and the quality
                                      metrics;
    - joint_models.csv                M1/M2/M3 coefficients, odds ratios,
                                      likelihood-ratio tests, cross-validated AUC;
    - threshold_vs_volume.csv         threshold (logistic p*, Q95 tolerance)
                                      evaluated at the volume quantiles, with
                                      bootstrap CIs and flags;
    - volume_strata.csv               univariate thresholds of 05 per volume
                                      tertile;
    - joint_pequiv_<metric>_<margin>.png        P(equiv | quality, volume) surface;
    - threshold_vs_volume_<metric>_<margin>.png threshold vs volume, bootstrap CI;
    - volume_vs_quality.png                     the descriptive coupling.

Validation: `07_validate_joint_synthetic.py` builds a synthetic cohort whose
dependence on both the Dice and the volume is known in closed form (three
regimes: a truth with interaction, a truth with no volume effect, and a quantile
truth). Run it BEFORE any real run, as 00_validate_synthetic.py does for scripts
01-04.

Usage: python 07_joint_dice_volume.py [--boot 400] [--seed 0] [--p-target 0.90]
                                      [--metrics dice assd_mm]
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils as U

plt.rc('font', family='serif')

# ============================================================
# CHEMINS & CONSTANTES
# ============================================================
OUT_DIR = U.OUT_DIR / "joint_dice_volume"

# Same restriction as in 05/06: the cross-dataset geometry drives the prostate
# only, and the prostate volume is a relevant covariate there alone.
STRUCTURE_TO_GEOM_ORGAN = {"Prostate": "Prostate"}
PANEL = [(s, i, t) for s, i, t in U.PANEL if s in STRUCTURE_TO_GEOM_ORGAN]
MARGINS = ("conf", "sens")

VOL_SCALE = 10.0          # cc: reading unit of the volume coefficients
TAU = 0.95                # quantile of |delta| for the tolerance bound (as in 05)
N_TERTILES = 3            # volume strata for the non-parametric control
VOL_Q_TABLE = (0.10, 0.25, 0.50, 0.75, 0.90)   # volumes where the threshold is tabulated
VOL_Q_FIG = (0.05, 0.95)                        # plotted volume range
SUPPORT_RADIUS = 0.12     # radius (in normalised coordinates) of the support mask

# Significance level for RETAINING the interaction term in the threshold curve.
# Deliberately stricter than the usual 5%, and not out of timidity: the threshold
# s*(v) = [logit(p*) - b0 - b2*v]/(b1 + b3*v) DIVERGES as b1 + b3*v approaches 0.
# On null data, 07_validate_joint_synthetic.py shows that an interaction retained
# in error - an ordinary false positive just under 0.05 - produces a curve with
# a large apparent amplitude where the truth is flat: a division artefact, not a
# result. The cost of a false positive is therefore catastrophic and asymmetric
# here, whereas a false negative only costs a slightly biased threshold, the
# additive model remaining a good local approximation. Hence alpha = 1%.
INTERACTION_ALPHA = 0.01
VOLUME_ALPHA = 0.05       # main effect: the usual test, no divergence possible

# Okabe-Ito palette (CVD-safe), consistent with 05/06.
C_REF = "#0072B2"     # blue      threshold curve of the joint model
C_ALT = "#E69F00"     # orange    iso-p*, and the single threshold of 05
C_NEG = "#D55E00"     # vermilion NON-equivalent patients
INK = "0.15"
GRID = "0.93"

# Figure font sizes: a legible floor once scaled down to a column.
FS_TICK = 12
FS_LABEL = 13
FS_TITLE = 14
FS_ANNOT = 11
FS_LEG = 11

# MONOHUE sequential colormap, light to dark, for a probability (a magnitude);
# never a rainbow, whose luminance bands invent boundaries.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "prob_blue", ["#F7FAFD", "#CFE2F0", "#8FC0DE", "#4E97C6", C_REF, "#004E7C"], N=256)


# ============================================================
# REUSE OF THE UNIVARIATE ESTIMATORS OF 05
# ============================================================
# Module 05 cannot be imported by name (it starts with a digit), so it is loaded
# explicitly, as 00_validate_synthetic.py does. The point is that the per-tertile
# control uses EXACTLY the code of 05, not a rewrite.
def _load_module(fname: str, modname: str):
    spec = importlib.util.spec_from_file_location(
        modname, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M05 = _load_module("05_dice_threshold.py", "dice_threshold_05")


def _metric_label(metric: str) -> str:
    meta = U.QUALITY_METRICS[metric]
    return f"{meta['label']} ({meta['unite']})" if meta["unite"] else meta["label"]


def _vol_label() -> str:
    return f"Prostate volume ({U.PROSTATE_VOL_UNITE})"


# ============================================================
# DATA
# ============================================================
def load_base(strict_volume: bool = True) -> pd.DataFrame:
    """
    The paired table of 05/06 (manual versus deterministic, plus geometry),
    enriched with the per-patient clinical prostate volume.

    A LEFT join followed by an explicit count of the patients without a volume:
    the loss must be visible and quantified rather than silent (the convention of
    this arm being no imputation).
    """
    base = U.load_paired_with_geometry(STRUCTURE_TO_GEOM_ORGAN)
    vol = U.load_prostate_volume(strict=strict_volume)
    merged = base.merge(vol, on="record_id", how="left")

    pat_all = merged["record_id"].nunique()
    pat_vol = merged.loc[merged["prostate_vol"].notna(), "record_id"].nunique()
    print(f"[data] {len(base)} paired rows, {pat_all} patients")
    print(f"[data] prostate volume available: {pat_vol}/{pat_all} patients "
          f"({pat_all - pat_vol} without a volume -> dropped)")
    return merged.dropna(subset=["prostate_vol"]).reset_index(drop=True)


def cell(base: pd.DataFrame, structure: str, index: str, metric: str) -> pd.DataFrame:
    """Sub-table of one cell (structure, index), complete rows only."""
    return (base[(base["structure"] == structure) & (base["index"] == index)]
            .dropna(subset=[metric, "abs_diff", "prostate_vol"])
            .reset_index(drop=True))


# ============================================================
# DESCRIPTIVE BLOCK - is the volume coupled to the quality?
# ============================================================
def volume_quality_descriptive(base: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlations between the prostate volume and the quality metrics,
    over
    the unique PATIENTS (not the index-by-patient rows, which would duplicate
    each patient once per index and distort the p-values).

    This motivates the joint analysis: a positive correlation between volume and
    Dice means the Dice is mechanically stricter on small prostates, so a single
    threshold cannot be neutral with respect to organ size.
    """
    pat = base.drop_duplicates(subset=["record_id", "geom_organ"])
    v = pat["prostate_vol"].to_numpy(float)
    rows = []
    targets = list(U.QUALITY_METRICS) + ["gt_volume_ml", "pred_volume_ml",
                                         U.SIGNED_AXIS]
    for m in targets:
        if m not in pat.columns:
            continue
        x = pat[m].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(v))
        if ok.sum() < 10:
            continue
        r = stats.spearmanr(v[ok], x[ok])
        rows.append(dict(
            variable=m,
            label=(_metric_label(m) if m in U.QUALITY_METRICS else m),
            is_quality_metric=m in U.QUALITY_METRICS,
            n=int(ok.sum()),
            spearman_vs_volume=float(r.statistic),
            p_value=float(r.pvalue),
        ))
    tbl = pd.DataFrame(rows)
    # Summary of the volume itself: the range actually covered by the cohort.
    tbl.attrs["volume_summary"] = dict(
        n=int(np.isfinite(v).sum()), mean=float(np.nanmean(v)),
        sd=float(np.nanstd(v, ddof=1)), q10=float(np.nanpercentile(v, 10)),
        median=float(np.nanmedian(v)), q90=float(np.nanpercentile(v, 90)))
    return tbl


# ============================================================
# AJUSTEMENTS
# ============================================================
def _fit_logit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Logistic model with a regularised fallback under separation (the same
    strategy as in 05).

    `method` is reported because it conditions the validity of the
    likelihood-ratio test: a penalised log-likelihood is not comparable to a
    maximised one, so a test involving a 'ridge' fit is invalidated rather than
    reported anyway.
    """
    res = dict(params=None, llf=np.nan, converged=False, method="", flag="ok")
    if np.unique(y).size < 2:
        res["flag"] = "single_class"
        return res
    Xc = sm.add_constant(X, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
        res.update(params=np.asarray(fit.params, float), llf=float(fit.llf),
                   converged=bool(fit.mle_retvals.get("converged", False)),
                   method="mle")
        return res
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y, Xc).fit_regularized(disp=0, alpha=1.0)
        res.update(params=np.asarray(fit.params, float), llf=np.nan,
                   converged=True, method="ridge", flag="regularized")
    except Exception:
        res["flag"] = "fit_failed"
    return res


def _fit_quantreg(X: np.ndarray, y: np.ndarray, tau: float = TAU) -> dict:
    """Quantile regression at level tau of |delta| on the supplied design (the
    constant is added here)."""
    res = dict(params=None, flag="ok")
    if y.size < 20:
        res["flag"] = "insufficient"
        return res
    Xc = sm.add_constant(X, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = QuantReg(y, Xc).fit(q=tau, max_iter=5000)
        res["params"] = np.asarray(fit.params, float)
    except Exception:
        res["flag"] = "fit_failed"
    return res


def _lrt(llf_small: float, llf_big: float, df_diff: int) -> tuple[float, float]:
    """Likelihood-ratio test: statistic and p, NaN when a log-likelihood is unavailable."""
    if not (np.isfinite(llf_small) and np.isfinite(llf_big)):
        return np.nan, np.nan
    stat = 2.0 * (llf_big - llf_small)
    if stat < 0:                      # optimisation imparfaite : ne pas maquiller
        return float(stat), np.nan
    return float(stat), float(stats.chi2.sf(stat, df_diff))


# ============================================================
# COURBE DE SEUIL s*(v)
# ============================================================
def _threshold_curve(params, v_grid, target, kind: str):
    """
    Threshold in ORIENTED score space, for each centred volume of the grid.

        logistique : s*(v) = [logit(p*) - b0 - b2·v] / (b1 + b3·v)
        quantile   : s*(v) = [δ     - a0 - a2·v] / (a1 + a3·v)

    `params` = [c0, c1, c2, c3] (c3 = 0 when the model has no interaction).
    `target` is logit(p*) for the logistic model, and the margin for the quantile.

    Returns a vector with NaN wherever the threshold is NOT identifiable:
      - logistic: a non-positive denominator means the probability of
        equivalence does not increase with quality at that volume, so no
        threshold is meaningful;
      - quantile: a non-negative denominator means the tolerance bound does not
        decrease as the segmentation improves.
    Reporting a number in those cases would produce a division artefact.
    """
    c0, c1, c2, c3 = params
    v = np.asarray(v_grid, float)
    denom = c1 + c3 * v
    num = target - c0 - c2 * v
    bad = (denom <= 0) if kind == "logit" else (denom >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = num / denom
    s[bad] = np.nan
    return s


def _design(s, v, interaction: bool) -> np.ndarray:
    cols = [s, v] + ([s * v] if interaction else [])
    return np.column_stack(cols)


def _params4(p, interaction: bool):
    """Pad the coefficients to [c0, c1, c2, c3] (c3 = 0 without interaction)."""
    if p is None:
        return None
    return np.array([p[0], p[1], p[2], p[3] if interaction else 0.0], float)


def joint_fit(g: pd.DataFrame, metric: str, delta: float, p_target: float,
              interaction: bool):
    """
    Fit the joint logistic model and its quantile dual on one cell,
    and return the threshold functions ready to evaluate.
    """
    orient = U.QUALITY_METRICS[metric]["orient"]
    x = g[metric].to_numpy(float)
    s = orient * x
    V = g["prostate_vol"].to_numpy(float)
    v0 = float(np.median(V))
    v = (V - v0) / VOL_SCALE
    absd = g["abs_diff"].to_numpy(float)
    y = (absd <= delta).astype(int)

    lg = _fit_logit(_design(s, v, interaction), y)
    qt = _fit_quantreg(_design(s, v, interaction), absd)
    return dict(orient=orient, x=x, s=s, V=V, v=v, v0=v0, y=y, absd=absd,
                logit=lg, quant=qt, interaction=interaction,
                logit_params=_params4(lg["params"], interaction),
                quant_params=_params4(qt["params"], interaction),
                logit_target=float(np.log(p_target / (1 - p_target))),
                quant_target=float(delta))


def thresholds_at(fit: dict, V_native) -> tuple[np.ndarray, np.ndarray]:
    """Thresholds (logistic, tolerance) in the NATIVE unit, at the given native
    volumes."""
    v = (np.asarray(V_native, float) - fit["v0"]) / VOL_SCALE
    o = fit["orient"]
    if fit["logit_params"] is None:
        thr_p = np.full(v.shape, np.nan)
    else:
        thr_p = o * _threshold_curve(fit["logit_params"], v,
                                     fit["logit_target"], "logit")
    if fit["quant_params"] is None:
        thr_q = np.full(v.shape, np.nan)
    else:
        thr_q = o * _threshold_curve(fit["quant_params"], v,
                                     fit["quant_target"], "quant")
    return thr_p, thr_q


def bootstrap_fits(g: pd.DataFrame, metric: str, delta: float, p_target: float,
                   interaction: bool, n_boot: int, rng) -> list[dict]:
    """
    FITTED bootstrap replicates, by resampling PATIENTS.

    Each row of `g` is one patient (a cell being one index), so resampling rows
    does resample patients. The model is REFITTED at every replicate, so the
    propagated uncertainty is that of the whole curve, not that of a single point
    with frozen coefficients.

    The FITS are returned rather than a band, because the same cell is evaluated on
    two
    volume grids (a few points for the table, many for the figure), and redoing
    the fits for each would double the runtime without changing the
    grids. The coefficients are estimated once and evaluated as often as needed.
    """
    # Only these keys are used by thresholds_at; keeping each replicate's data
    # vectors would multiply memory by n_boot for no purpose.
    keep = ("orient", "v0", "logit_params", "quant_params",
            "logit_target", "quant_target")
    n = len(g)
    fits = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gb = g.iloc[idx]
        if (gb["abs_diff"] <= delta).nunique() < 2:
            continue
        try:
            f = joint_fit(gb, metric, delta, p_target, interaction)
        except Exception:
            continue
        fits.append({k: f[k] for k in keep})
    return fits


def band_from_fits(fits: list[dict], V_grid: np.ndarray):
    """95% percentile CI of both threshold curves, at the given native volumes."""
    acc_p, acc_q = [], []
    for f in fits:
        tp, tq = thresholds_at(f, V_grid)
        acc_p.append(tp)
        acc_q.append(tq)

    def _band(acc):
        if not acc:
            return (np.full(V_grid.shape, np.nan),) * 2
        A = np.vstack(acc)
        n_ok = np.sum(np.isfinite(A), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo = np.nanpercentile(A, 2.5, axis=0)
            hi = np.nanpercentile(A, 97.5, axis=0)
        # A CI estimated from a handful of valid replicates is not a CI.
        weak = n_ok < max(20, int(0.5 * len(acc)))
        lo[weak] = np.nan
        hi[weak] = np.nan
        return lo, hi

    return _band(acc_p), _band(acc_q)


# ============================================================
# CROSS-VALIDATED AUC - M1 / M2 / M3
# ============================================================
def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    if np.unique(y).size < 2 or min(np.bincount(y)) < 5:
        return np.nan
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


# ============================================================
# JOINT MODEL TABLE
# ============================================================
def joint_models(base: pd.DataFrame, metric: str, seed: int) -> pd.DataFrame:
    """
    For each (index, margin): M1 (quality only), M2 (+ volume), M3
    (+ interaction), with the coefficients, readable odds ratios, nested
    likelihood-ratio tests and cross-validated AUC.

    The odds ratios are given in an INTERPRETABLE unit:
      - quality: per a small step of the metric, following its orientation;
      - volume  : par +10 cc de prostate.
    An odds ratio per full unit of Dice (0 to 1) has no clinical meaning.
    """
    rows = []
    for structure, index, tier in PANEL:
        g = cell(base, structure, index, metric)
        if len(g) < 30:
            continue
        orient = U.QUALITY_METRICS[metric]["orient"]
        s = orient * g[metric].to_numpy(float)
        V = g["prostate_vol"].to_numpy(float)
        v = (V - np.median(V)) / VOL_SCALE
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            y = (g["abs_diff"].to_numpy(float) <= delta).astype(int)
            if np.unique(y).size < 2:
                continue
            m1 = _fit_logit(s[:, None], y)
            m2 = _fit_logit(_design(s, v, False), y)
            m3 = _fit_logit(_design(s, v, True), y)
            lr_vol_stat, lr_vol_p = _lrt(m1["llf"], m2["llf"], 1)
            lr_int_stat, lr_int_p = _lrt(m2["llf"], m3["llf"], 1)

            p2 = m2["params"]
            p3 = m3["params"]
            # Pas de score-unit universelle : 0,05 de Dice, 0,5 mm d'ASSD.
            step = 0.05 if metric == "dice" else 0.5
            rows.append(dict(
                structure=structure, index=index, tier=tier, metric=metric,
                margin=margin, delta=delta, n=int(len(g)),
                equiv_rate=float(y.mean()),
                vol_median=float(np.median(V)),
                vol_q10=float(np.percentile(V, 10)),
                vol_q90=float(np.percentile(V, 90)),
                # --- M2: volume main effect, adjusted on quality ---
                m2_b_quality=float(p2[1]) if p2 is not None else np.nan,
                m2_b_volume=float(p2[2]) if p2 is not None else np.nan,
                m2_or_quality_per_step=(float(np.exp(p2[1] * step))
                                        if p2 is not None else np.nan),
                m2_or_volume_per_10cc=(float(np.exp(p2[2]))
                                       if p2 is not None else np.nan),
                # Threshold shift induced by one VOL_SCALE step under M2:
                # -b2/b1 in oriented score, converted back to the native unit.
                m2_thr_shift_per_10cc=(orient * (-p2[2] / p2[1])
                                       if p2 is not None and p2[1] != 0 else np.nan),
                # --- M3: quality x volume interaction ---
                m3_b_interaction=float(p3[3]) if p3 is not None else np.nan,
                # --- nested tests ---
                lrt_volume_stat=lr_vol_stat, lrt_volume_p=lr_vol_p,
                lrt_interaction_stat=lr_int_stat, lrt_interaction_p=lr_int_p,
                volume_matters=bool(np.isfinite(lr_vol_p)
                                    and lr_vol_p < VOLUME_ALPHA),
                threshold_shifts=bool(np.isfinite(lr_int_p)
                                      and lr_int_p < INTERACTION_ALPHA),
                # --- honest discrimination ---
                cv_auc_m1=_cv_auc(s[:, None], y, seed),
                cv_auc_m2=_cv_auc(_design(s, v, False), y, seed),
                cv_auc_m3=_cv_auc(_design(s, v, True), y, seed),
                fit_flag_m1=m1["flag"], fit_flag_m2=m2["flag"],
                fit_flag_m3=m3["flag"],
            ))
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl["cv_auc_gain_m2_vs_m1"] = tbl["cv_auc_m2"] - tbl["cv_auc_m1"]
        tbl["cv_auc_gain_m3_vs_m1"] = tbl["cv_auc_m3"] - tbl["cv_auc_m1"]
    return tbl


# ============================================================
# TABLE DU SEUIL EN FONCTION DU VOLUME
# ============================================================
def threshold_vs_volume(base: pd.DataFrame, metric: str, p_target: float,
                        n_boot: int, rng, models: pd.DataFrame):
    """
    Returns (table, cache). The cache of bootstrap replicates per (index,
    margin) is reused as is by the figure, which evaluates the same uncertainty
    on a
    grille de volume plus fine.

    Quality threshold tabulated at the volume quantiles, for both estimators.

    The retained model (with or without interaction) is chosen PER CELL by the
    likelihood-ratio test
    on the interaction term, at the INTERACTION_ALPHA level (see its comment: an
    s*v term retained in error makes the threshold curve DIVERGE). This is the
    reading
    parsimonious: a threshold varying non-linearly with volume on the strength of
    a poorly identified term would be over-interpreted.
    """
    rows, cache = [], {}
    for structure, index, tier in PANEL:
        g = cell(base, structure, index, metric)
        if len(g) < 30:
            continue
        V = g["prostate_vol"].to_numpy(float)
        V_grid = np.percentile(V, [q * 100 for q in VOL_Q_TABLE])
        x_obs = g[metric].to_numpy(float)
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            mrow = models[(models["index"] == index) & (models["margin"] == margin)]
            interaction = bool(mrow["threshold_shifts"].iloc[0]) if len(mrow) else False

            fit = joint_fit(g, metric, delta, p_target, interaction)
            thr_p, thr_q = thresholds_at(fit, V_grid)
            boot = bootstrap_fits(g, metric, delta, p_target, interaction,
                                  n_boot, rng)
            cache[(index, margin)] = boot
            (plo, phi), (qlo, qhi) = band_from_fits(boot, V_grid)

            for k, q in enumerate(VOL_Q_TABLE):
                rows.append(dict(
                    structure=structure, index=index, tier=tier, metric=metric,
                    metric_unite=U.QUALITY_METRICS[metric]["unite"],
                    margin=margin, delta=delta, n=int(len(g)),
                    model=("M3 (interaction)" if interaction else "M2 (additif)"),
                    vol_quantile=q, volume_cc=float(V_grid[k]),
                    p_target=p_target,
                    thr_logistic=float(thr_p[k]),
                    thr_logistic_lo=float(plo[k]), thr_logistic_hi=float(phi[k]),
                    thr_logistic_extrap=M05._extrapolation_flag(thr_p[k], x_obs),
                    thr_tolerance=float(thr_q[k]),
                    thr_tolerance_lo=float(qlo[k]), thr_tolerance_hi=float(qhi[k]),
                    thr_tolerance_extrap=M05._extrapolation_flag(thr_q[k], x_obs),
                ))
    return pd.DataFrame(rows), cache


# ============================================================
# NON-PARAMETRIC CONTROL - VOLUME TERTILES
# ============================================================
def volume_strata(base: pd.DataFrame, metric: str, p_target: float) -> pd.DataFrame:
    """
    UNIVARIATE thresholds of 05, re-estimated independently within each volume tertile.

    No functional form is imposed on the volume: if the per-tertile threshold
    reproduit la tendance de la courbe s*(v), l'interaction n'est pas un artefact
    a specification artefact. This plays the same role as the Youden cutpoint
    does with respect to the sigmoid in 05.
    """
    orient = U.QUALITY_METRICS[metric]["orient"]
    rows = []
    for structure, index, tier in PANEL:
        g = cell(base, structure, index, metric)
        if len(g) < 30:
            continue
        try:
            strata = pd.qcut(g["prostate_vol"], N_TERTILES,
                             labels=[f"T{i+1}" for i in range(N_TERTILES)])
        except ValueError:            # volumes too homogeneous to cut
            continue
        for margin in MARGINS:
            delta = U.DELTA[(structure, index)][f"delta_{margin}"]
            for lab in strata.cat.categories:
                gs = g[strata == lab]
                if len(gs) < 20:
                    continue
                x = gs[metric].to_numpy(float)
                s = orient * x
                absd = gs["abs_diff"].to_numpy(float)
                equiv = (absd <= delta).astype(int)
                lg = M05.logistic_thresholds(s, equiv, p_targets=(p_target,))
                qt = M05.quantile_threshold(s, absd, delta)
                yd = M05.youden(s, equiv)
                thr_p = orient * lg[f"thr_p{int(p_target * 100)}"]
                thr_t = orient * qt["thr_tol"]
                rows.append(dict(
                    structure=structure, index=index, metric=metric,
                    margin=margin, delta=delta, stratum=str(lab), n=int(len(gs)),
                    vol_lo=float(gs["prostate_vol"].min()),
                    vol_median=float(gs["prostate_vol"].median()),
                    vol_hi=float(gs["prostate_vol"].max()),
                    equiv_rate=float(equiv.mean()),
                    metric_min=float(np.min(x)), metric_max=float(np.max(x)),
                    metric_median=float(np.median(x)),
                    thr_logistic=float(thr_p),
                    thr_logistic_extrap=M05._extrapolation_flag(thr_p, x),
                    thr_tolerance=float(thr_t),
                    thr_youden=float(orient * yd["thr_youden"]),
                    auc=float(yd["auc"]),
                    logit_flag=lg["flag"], quant_flag=qt["flag"],
                ))
    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================
def _support_mask(XX, YY, x_obs, y_obs, radius=SUPPORT_RADIUS):
    """
    Mask the regions of the plane with no empirical support.

    Both axes are normalised over the observed range before the distance is
    computed; otherwise the radius would mix cc and Dice points. A cell with no
    neighbour within `radius` is a region where the model EXTRAPOLATES, and it is
    left blank rather than painted with a probability nothing supports.
    """
    def _norm(a, ref):
        lo, hi = np.nanmin(ref), np.nanmax(ref)
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    pts = np.column_stack([_norm(x_obs, x_obs), _norm(y_obs, y_obs)])
    grid = np.column_stack([_norm(XX.ravel(), x_obs), _norm(YY.ravel(), y_obs)])
    d, _ = cKDTree(pts).query(grid, k=1)
    return (d.reshape(XX.shape) > radius)


def figure_joint_surface(base, metric, margin, p_target, models):
    """
    P(equiv | quality, volume) surface - the central figure of the joint
    analysis.

    Encoding: a probability is a MAGNITUDE, hence a monohue sequential ramp from
    light to dark. Patients are overlaid with two distinct markers (filled =
    equivalent, cross = not), so their status does not rest on colour alone. The
    thick line is the iso-p*, that is, the threshold curve: its SLOPE is the
    result - horizontal means a single threshold valid for every size, tilted
    means the threshold depends on the volume.
    """
    orient = U.QUALITY_METRICS[metric]["orient"]
    pairs = [(s, i) for s, i, _ in PANEL]
    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                             squeeze=False, layout="constrained")
    im = None
    for k, (structure, index) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        g = cell(base, structure, index, metric)
        delta = U.DELTA[(structure, index)][f"delta_{margin}"]
        mrow = models[(models["index"] == index) & (models["margin"] == margin)]
        interaction = bool(mrow["threshold_shifts"].iloc[0]) if len(mrow) else False
        fit = joint_fit(g, metric, delta, p_target, interaction)
        if fit["logit_params"] is None:
            ax.axis("off")
            continue

        x = fit["x"]
        V = fit["V"]
        # Grid over the FULL range of both axes: it is the support mask, not a
        # percentile crop, that decides where the surface is painted. Otherwise
        # real patients would fall outside the coloured background.
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 220)
        Vs = np.linspace(np.nanmin(V), np.nanmax(V), 220)
        XX, VV = np.meshgrid(xs, Vs)
        c0, c1, c2, c3 = fit["logit_params"]
        S = orient * XX
        vv = (VV - fit["v0"]) / VOL_SCALE
        P = 1.0 / (1.0 + np.exp(-(c0 + c1 * S + c2 * vv + c3 * S * vv)))
        P = np.ma.array(P, mask=_support_mask(XX, VV, x, V))

        im = ax.pcolormesh(XX, VV, P, cmap=CMAP_SEQ, vmin=0, vmax=1,
                           shading="auto", zorder=1)
        # The secondary iso-contours carry a label; the iso-p* is identified by
        # the legend instead. A clabel on it would sit on the thick line and
        # become illegible exactly where it matters.
        cs = ax.contour(XX, VV, P, levels=[0.80, 0.95], colors=INK,
                        linewidths=1.0, linestyles=["--", ":"], zorder=3)
        ax.clabel(cs, fmt=lambda v: f"{v:.2f}", fontsize=FS_ANNOT, inline=True)
        ax.contour(XX, VV, P, levels=[p_target], colors=C_ALT, linewidths=2.6,
                   zorder=6)

        eq = fit["y"].astype(bool)
        ax.scatter(x[eq], V[eq], s=16, facecolor="white", edgecolor=INK,
                   linewidth=0.7, alpha=0.85, zorder=4, label="|Δ| ≤ δ")
        ax.scatter(x[~eq], V[~eq], s=26, marker="x", color=C_NEG, linewidth=1.3,
                   alpha=0.9, zorder=5, label="|Δ| > δ")
        ax.plot([], [], color=C_ALT, lw=2.6, label=f"iso-{p_target:.0%}")

        ax.set_title(f"{structure} : {U.index_label(index)}   (δ_{margin} = {delta:g})",
                     fontsize=FS_TITLE, color=INK)
        ax.set_xlabel(_metric_label(metric), fontsize=FS_LABEL)
        ax.set_ylabel(_vol_label(), fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.text(0.98, 0.03,
                f"n={len(g)} · {'M3 interaction' if interaction else 'M2 additive'}",
                transform=ax.transAxes, fontsize=FS_ANNOT, color=INK, va="bottom",
                ha="right", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85"))
        if k == 0:
            leg = ax.legend(fontsize=FS_LEG, loc="upper left", handletextpad=0.4,
                            framealpha=0.92, facecolor="white", edgecolor="0.85")
            leg.set_zorder(10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
        cb.set_label(f"P(equiv = 1 | {U.QUALITY_METRICS[metric]['label']}, volume)",
                     fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
        cb.outline.set_visible(False)
    f = U.save_figure(fig, OUT_DIR / f"joint_pequiv_{metric}_{margin}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_threshold_vs_volume(base, metric, margin, p_target, models,
                               boot_cache):
    """
    The quality threshold as a function of the volume, with its bootstrap CI.

    The figure carries the model only: the joint logistic curve, the bootstrap
    band, and the single threshold of 05 as a reference mark. The non-parametric
    per-tertile control is still computed and reported in volume_strata.csv; it
    reads on the
    table, with its per-stratum n and its extrapolation flag, which a marker
    placed on the axis cannot say. The dual Q95 tolerance curve is likewise not
    drawn; it lives in threshold_vs_volume.csv (thr_tolerance).

    The y axis follows what is drawn, never exceeding the range of
    the widened observed quality range (see _threshold_ylim): outside it a
    threshold is extrapolated (see the table flags), and letting it stretch the
    axis would make the figure illegible for every other cell. The shaded band
    remains the marker of the observed range; when it covers the whole panel, the
    entire frame lies in the interpolated region.
    """
    fig, axes, ncols, nrows = _threshold_panels()
    _draw_threshold_vs_volume(axes, base, metric, margin, p_target, models,
                              boot_cache, ncols, nrows)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / f"threshold_vs_volume_{metric}_{margin}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def _threshold_panels():
    pairs = [(s, i) for s, i, _ in PANEL]
    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    return fig, axes, ncols, nrows


def _finite(series):
    vals = [np.asarray(s, float).ravel() for s in series if s is not None]
    vals = np.concatenate(vals) if vals else np.array([], float)
    return vals[np.isfinite(vals)]


def _threshold_ylim(x_lo, x_hi, pad, anchor, elastic,
                    frac_pad=0.08, min_frac=0.12, stretch=0.5):
    """Vertical frame of a threshold-versus-volume panel: what is DRAWN, not the
    whole observed quality range.

    Framing on the full observed range stretches the axis by the few heavily
    degraded cases that none of the series approaches: the curves would be
    squeezed into the top tenth of the panel and the
    quarts du cadre ne portaient aucune information.

    Not every drawn series can dictate the frame, though. The logistic threshold
    and its bootstrap band run to an asymptote as soon as the quality slope
    approaches zero: a single grid point at minus infinity would blow the axis
    wide open. Hence the separation:

    - `anchor`: quantities bounded by construction (the single threshold). They
      set the scale and are ALWAYS visible;
    - `elastic`: the logistic threshold and its band. They may widen the frame,
      but by at most `stretch` times the amplitude of the anchors on each side;
      beyond that the curve simply leaves the frame, which is precisely the
      message (a threshold unreachable within the observed quality).

    Two guards complete the computation: the view never zooms out beyond
    the previous frame [x_lo - pad, x_hi + pad], and a minimum amplitude is imposed
    (min_frac of the observed range), without which a perfectly flat panel would
    give a nearly degenerate axis where the slightest noise would look like a
    pente.
    """
    lo_bound, hi_bound = x_lo - pad, x_hi + pad
    floor = min_frac * (x_hi - x_lo)

    a = np.clip(_finite(anchor), lo_bound, hi_bound)
    e = np.clip(_finite(elastic), lo_bound, hi_bound)
    if not a.size and not e.size:
        return lo_bound, hi_bound

    if a.size:
        lo, hi = float(a.min()), float(a.max())
        room = stretch * max(hi - lo, floor)
        if e.size:
            e = e[(e >= lo - room) & (e <= hi + room)]
            if e.size:
                lo, hi = min(lo, float(e.min())), max(hi, float(e.max()))
    else:
        lo, hi = float(e.min()), float(e.max())

    span = max(hi - lo, floor)
    mid = 0.5 * (lo + hi)
    lo, hi = mid - 0.5 * span, mid + 0.5 * span
    m = frac_pad * span
    return max(lo - m, lo_bound), min(hi + m, hi_bound)


def _draw_threshold_vs_volume(axes, base, metric, margin, p_target, models,
                              boot_cache, ncols, nrows):
    """Draw the threshold-versus-volume panels into an already built axis grid.

    Extrait de figure_threshold_vs_volume pour que la figure composite partage le
    meme code de trace.
    """
    pairs = [(s, i) for s, i, _ in PANEL]
    for k, (structure, index) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        g = cell(base, structure, index, metric)
        delta = U.DELTA[(structure, index)][f"delta_{margin}"]
        mrow = models[(models["index"] == index) & (models["margin"] == margin)]
        interaction = bool(mrow["threshold_shifts"].iloc[0]) if len(mrow) else False
        V = g["prostate_vol"].to_numpy(float)
        V_grid = np.linspace(np.percentile(V, VOL_Q_FIG[0] * 100),
                             np.percentile(V, VOL_Q_FIG[1] * 100), 60)
        fit = joint_fit(g, metric, delta, p_target, interaction)
        thr_p, _ = thresholds_at(fit, V_grid)
        (plo, phi), _ = band_from_fits(boot_cache.get((index, margin), []), V_grid)

        # A single colour for the curve across the panels: each
        # panel carries a single series, already named by its title; colouring by
        # panel would add a dimension encoding nothing.
        ax.fill_between(V_grid, plo, phi, color=C_REF, alpha=0.16, lw=0,
                        zorder=2, label="95 % bootstrap CI")
        ax.plot(V_grid, thr_p, color=C_REF, lw=2.4, zorder=4,
                label=f"logistic p={p_target:.2f}")

        # Reference: the SINGLE threshold of 05, estimated without the volume.
        # That is the hypothesis this figure tests - if the curve departs from it
        # across the axis, a single threshold does not cover the cohort.
        orient = U.QUALITY_METRICS[metric]["orient"]
        equiv = (g["abs_diff"].to_numpy(float) <= delta).astype(int)
        uni = M05.logistic_thresholds(orient * g[metric].to_numpy(float), equiv,
                                      p_targets=(p_target,))
        thr_uni = orient * uni[f"thr_p{int(p_target * 100)}"]
        if np.isfinite(thr_uni):
            ax.axhline(thr_uni, color=C_ALT, lw=1.5, ls=":", zorder=5,
                       label="single threshold")

        # Observed quality range: outside this band, any threshold is extrapolated.
        x = g[metric].to_numpy(float)
        x_lo, x_hi = float(np.nanmin(x)), float(np.nanmax(x))
        ax.axhspan(x_lo, x_hi, color="0.5", alpha=0.06, lw=0, zorder=1)
        pad = 0.12 * (x_hi - x_lo)
        y_lo, y_hi = _threshold_ylim(
            x_lo, x_hi, pad,
            anchor=[np.array([thr_uni], float)],
            elastic=[thr_p, plo, phi])
        ax.set_ylim(y_lo, y_hi)
        # A panel whose curve leaves the frame entirely is not an empty panel:
        # it is a threshold unreachable within the observed quality. Say so,
        # rather than letting the reader conclude there is a display bug.
        out_high = np.isfinite(thr_p).any() and np.nanmin(thr_p) > x_hi + pad
        out_low = np.isfinite(thr_p).any() and np.nanmax(thr_p) < x_lo - pad
        if out_high or out_low or not np.isfinite(thr_p).any():
            msg = ("not identifiable (non-positive slope)"
                   if not np.isfinite(thr_p).any()
                   else "extrapolated, not quotable, cf. flags")
            ax.text(0.5, 0.5, f"p={p_target:.2f} threshold outside the observed "
                              f"{U.QUALITY_METRICS[metric]['label']} range\n"
                              f"({msg})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=FS_ANNOT, color="0.35", zorder=8)
        # Bare margin value, without the margin suffix: the margin is already
        # carried by the name of
        # fichier, et 'δ_conf' se lit mal en indice.
        ax.set_title(f"{structure} : {U.index_label(index)}   (δ = {delta:g})",
                     fontsize=FS_TITLE, color=INK)
        ax.set_xlabel(_vol_label(), fontsize=FS_LABEL)
        ax.set_ylabel(f"Threshold: {_metric_label(metric)}", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k == 0:
            # Cadre blanc translucide : le panneau n'a plus de coin vide depuis
            # the axis follows the content, so the legend necessarily sits on a
            # series and must stay legible without erasing it.
            leg = ax.legend(fontsize=FS_LEG, loc="best", frameon=True,
                            framealpha=0.88, facecolor="white", edgecolor="0.85")
            leg.set_zorder(10)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")


def figure_signed_axis_and_threshold(base, metric, margin, p_target, models,
                                     boot_cache, seed=0):
    """Composite figure: (a) signed axis (script 05), (b) threshold vs volume.

    Les deux blocs disent ce qu'une metrique non signee et sans covariable de
    patient ne peut pas dire : le SENS de l'erreur dosimetrique (a) et la
    dependance du SEUIL a la taille de l'organe (b).

    Le bloc (a) est recalcule sur la table de 05 (sans filtre de volume), pour
    stay identical to the standalone signed_volume_axis figure.
    """
    base05 = U.load_paired_with_geometry(STRUCTURE_TO_GEOM_ORGAN)
    _, sfits = M05.signed_axis(base05, seed=seed)
    pairs_a = M05.signed_axis_pairs(sfits)
    if not pairs_a:
        return

    ncols = 2
    nrows = int(np.ceil(len(pairs_a) / ncols))
    fig = plt.figure(figsize=(5.2 * ncols, 2 * 4.0 * nrows + 0.5))
    blocks = fig.subfigures(2, 1, hspace=0.02)

    axes_a = blocks[0].subplots(nrows, ncols, squeeze=False)
    M05._draw_signed_axis(axes_a, base05, sfits, ncols, nrows)

    nrows_b = int(np.ceil(len([(s, i) for s, i, _ in PANEL]) / ncols))
    axes_b = blocks[1].subplots(nrows_b, ncols, squeeze=False)
    _draw_threshold_vs_volume(axes_b, base, metric, margin, p_target, models,
                              boot_cache, ncols, nrows_b)

    for sub, tag in zip(blocks, ("(a)", "(b)")):
        sub.text(0.005, 0.995, tag, fontsize=FS_TITLE + 2, fontweight="bold",
                 color="0.1", ha="left", va="top")
        sub.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12,
                            hspace=0.55, wspace=0.30)

    f = OUT_DIR / f"signed_axis_threshold_{metric}_{margin}.pdf"
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


def figure_volume_vs_quality(base, desc):
    """
    The descriptive volume-to-quality coupling: the premise of the whole script.

    Panneau de gauche : le nuage Dice ~ volume avec sa tendance LOWESS-libre (une
    a linear regression suffices here, no functional form is claimed). Right
    panel: the Spearman correlations of each quality metric with the volume,
    sorted - an at-a-glance reading of what organ size explains.
    """
    pat = base.drop_duplicates(subset=["record_id", "geom_organ"])
    V = pat["prostate_vol"].to_numpy(float)
    d = pat["dice"].to_numpy(float)
    ok = ~(np.isnan(V) | np.isnan(d))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8),
                             gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    ax = axes[0]
    ax.scatter(V[ok], d[ok], s=18, color=C_REF, alpha=0.45, edgecolor="none",
               zorder=3)
    if ok.sum() > 3:
        lr = stats.linregress(V[ok], d[ok])
        xs = np.linspace(np.nanmin(V[ok]), np.nanmax(V[ok]), 100)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=INK, lw=2, zorder=4)
        rho = stats.spearmanr(V[ok], d[ok])
        ax.text(0.03, 0.05, f"Spearman ρ = {rho.statistic:+.2f}  (p = {rho.pvalue:.1e})"
                            f"\nslope = {lr.slope * 10:+.4f} Dice / 10 cc",
                transform=ax.transAxes, fontsize=FS_ANNOT, color=INK, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85"))
    ax.set_xlabel(_vol_label(), fontsize=FS_LABEL)
    ax.set_ylabel("Dice", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    s = (desc[desc["is_quality_metric"]]
         .sort_values("spearman_vs_volume"))
    y = np.arange(len(s))
    cols = [C_REF if v >= 0 else C_ALT for v in s["spearman_vs_volume"]]
    ax.barh(y, s["spearman_vs_volume"], color=cols, height=0.62, zorder=3)
    for j, (val, p) in enumerate(zip(s["spearman_vs_volume"], s["p_value"])):
        off = 0.012 if val >= 0 else -0.012
        ax.text(val + off, y[j], f"{val:+.2f}{'*' if p < 0.05 else ''}",
                va="center", ha="left" if val >= 0 else "right", fontsize=FS_ANNOT,
                color=INK)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    # Explicit margin: without it, the label of a negative bar lands on the
    # metric name.
    span = float(np.nanmax(np.abs(s["spearman_vs_volume"]))) or 0.1
    ax.set_xlim(-1.45 * span, 1.45 * span)
    ax.set_yticks(y)
    ax.set_yticklabels(s["label"], fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("Spearman ρ with prostate volume  (* : p < 0.05)",
                  fontsize=FS_LABEL)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    fig.tight_layout()
    f = U.save_figure(fig, OUT_DIR / "volume_vs_quality.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {f}")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-target", type=float, default=0.90,
                    help="target probability of equivalence for the threshold")
    ap.add_argument("--metrics", nargs="*", default=list(U.THRESHOLD_METRICS),
                    help="quality metrics carrying the threshold")
    args = ap.parse_args()

    U.ensure_dir(OUT_DIR)
    rng = np.random.default_rng(args.seed)

    base = load_base()

    # ---------- bloc descriptif ----------
    desc = volume_quality_descriptive(base)
    U.save_csv(desc, OUT_DIR / "volume_quality_descriptive.csv")
    vs = desc.attrs["volume_summary"]
    print(f"\n=== Volume prostatique ({U.PROSTATE_VOL_COL}) ===")
    print(f"  n={vs['n']}  moyenne={vs['mean']:.1f}  ecart-type={vs['sd']:.1f}  "
          f"mediane={vs['median']:.1f}  [q10={vs['q10']:.1f} ; q90={vs['q90']:.1f}] "
          f"{U.PROSTATE_VOL_UNITE}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n=== Volume-to-quality coupling (Spearman, unique patients) ===")
        print(desc[["variable", "n", "spearman_vs_volume", "p_value"]]
              .round(4).to_string(index=False))

    figure_volume_vs_quality(base, desc)

    all_models, all_thr, all_strata = [], [], []
    for metric in args.metrics:
        if metric not in U.QUALITY_METRICS:
            raise SystemExit(f"[STOP] Metrique inconnue : {metric}")
        print(f"\n########## METRIQUE : {metric} ##########")

        models = joint_models(base, metric, args.seed)
        if not len(models):
            print("[skip] aucune cellule exploitable.")
            continue
        all_models.append(models)
        with pd.option_context("display.width", 240, "display.max_columns", None):
            print("\n=== Modeles conjoints : le volume ajoute-t-il quelque chose ? ===")
            print(models[["index", "margin", "n", "equiv_rate",
                          "m2_or_volume_per_10cc", "lrt_volume_p",
                          "lrt_interaction_p", "m2_thr_shift_per_10cc",
                          "cv_auc_m1", "cv_auc_m2", "cv_auc_m3"]]
                  .round(4).to_string(index=False))

        strata = volume_strata(base, metric, args.p_target)
        all_strata.append(strata)
        if len(strata):
            with pd.option_context("display.width", 240, "display.max_columns", None):
                print("\n=== Controle non parametrique : seuils par tertile de volume ===")
                print(strata[strata["margin"] == "conf"][
                    ["index", "stratum", "n", "vol_median", "equiv_rate",
                     "thr_logistic", "thr_tolerance", "thr_youden", "auc"]]
                    .round(3).to_string(index=False))

        thr, boot_cache = threshold_vs_volume(base, metric, args.p_target,
                                              args.boot, rng, models)
        all_thr.append(thr)
        if len(thr):
            with pd.option_context("display.width", 240, "display.max_columns", None):
                print("\n=== Quality threshold as a function of volume (decision margin) ===")
                print(thr[thr["margin"] == "conf"][
                    ["index", "model", "volume_cc", "thr_logistic",
                     "thr_logistic_lo", "thr_logistic_hi", "thr_logistic_extrap",
                     "thr_tolerance"]].round(3).to_string(index=False))

        for margin in MARGINS:
            figure_joint_surface(base, metric, margin, args.p_target, models)
            figure_threshold_vs_volume(base, metric, margin, args.p_target,
                                       models, boot_cache)
            figure_signed_axis_and_threshold(base, metric, margin, args.p_target,
                                             models, boot_cache,
                                             seed=args.seed)

    if all_models:
        U.save_csv(pd.concat(all_models, ignore_index=True),
                   OUT_DIR / "joint_models.csv")
    if all_thr:
        U.save_csv(pd.concat(all_thr, ignore_index=True),
                   OUT_DIR / "threshold_vs_volume.csv")
    if all_strata:
        U.save_csv(pd.concat(all_strata, ignore_index=True),
                   OUT_DIR / "volume_strata.csv")

    print("\n[OK] Joint quality x prostate-volume analysis complete.")


if __name__ == "__main__":
    main()
