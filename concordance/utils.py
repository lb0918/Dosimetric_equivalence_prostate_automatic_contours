"""
utils.py - shared helpers of the segmentation-concordance pipeline.
==================================================================
This arm asks whether the segmentation source influences the DVH indices, by an
EQUIVALENCE analysis (paired TOST) with the manual segmentation as the
REFERENCE.

Two comparisons per index:
    - 'det'   manual vs the DETERMINISTIC automatic segmentation.
    - 'bayes' manual vs the Bayesian E[index] = mean of the Monte-Carlo passes
              (the <idx>__mean column of the MC summary).

This module centralises everything the numbered scripts share, so that no
definition can drift (panel, equivalence margins, harmonised loading, statistics):
    - constants: paths, confirmatory/exploratory PANEL, pre-declared margin
      table, structure harmonisation maps;
    - harmonised long-format loaders (load_source_long / load_mc_long);
    - statistical primitives (tost_paired, lin_ccc, icc_a1, coverage_stats,
      bland_altman), used identically by the analysis scripts AND by the
      synthetic validation in 00, which therefore tests the code actually run.

Strict conventions of this arm
------------------------------
    - Work happens EXCLUSIVELY on the _pct columns, already expressed as a
      percentage of the prescription and therefore agnostic to the isotope.
      Values are NEVER converted back from the _Gy columns.
    - Sign convention: diff = val_manual - val_auto.
    - No imputation: unmatched patients are dropped and counted.

NB: pingouin is not assumed to be installed, so ICC(A,1) is implemented
explicitly here as a two-way ANOVA decomposition.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# PATHS
# ============================================================
# All inputs live under DATA_ROOT, overridable by environment variable so the
# code runs unchanged on any machine. Outputs go under OUT_DIR.
DATA_DIR = Path(os.environ.get(
    "PROTECTA_DVH_DIR", Path(__file__).resolve().parents[1] / "data" / "dvh"))
PIPE_DIR = Path(os.environ.get(
    "PROTECTA_CONCORDANCE_DIR", Path(__file__).resolve().parent))
OUT_DIR = PIPE_DIR / "results"
FIG_DIR = OUT_DIR / "figures"

MANUAL_CSV = DATA_DIR / "dvh_indices_manual_seg.csv"           # reference
DET_CSV = DATA_DIR / "dvh_indices_auto_seg_det_clin0977.csv"   # deterministic
MC_SUMMARY_CSV = DATA_DIR / "dvh_mc_summary_clin0977.csv"      # Monte-Carlo summary
MC_RAW_CSV = DATA_DIR / "dvh_mc_raw_clin0977.csv"              # raw passes (unused here)

PAIRED_TABLE_CSV = OUT_DIR / "paired_table.csv"

# Per-case geometric metrics (cross-dataset, DETERMINISTIC predictions, not
# Bayesian). Shared by 05 (thresholds) and 06 (metric comparison) so a single
# definition of the Dice circulates.
CROSS_DIR = Path(os.environ.get(
    "PROTECTA_CROSS_DATASET_DIR", DATA_DIR / "cross_dataset"))
GEOM_DATASETS = {
    "Prostate": CROSS_DIR / "Dataset516_Cross_510_513_Prostate__nnUNetTrainer__nnUNetPlans_clin0977__3d_fullres__on__Dataset516_Cross_510_513_Prostate__base501_NotTrained" / "metrics_per_case.csv",
    "Rectum":   CROSS_DIR / "Dataset517_Cross_511_514_Rectum__nnUNetTrainer__nnUNetPlans_clin0977__3d_fullres__on__Dataset517_Cross_511_514_Rectum__base502_NotTrained" / "metrics_per_case.csv",
    "Bladder":  CROSS_DIR / "Dataset518_Cross_512_515_Bladder__nnUNetTrainer__nnUNetPlans_clin0977__3d_fullres__on__Dataset518_Cross_512_515_Bladder__base503_NotTrained" / "metrics_per_case.csv",
}

# CLINICAL prostate volume, a PATIENT-level covariate (script 07). Source: the
# longitudinal IPSS dataset, in which `ldr_post_vol` is a per-patient constant
# repeated across visits. Its `record_id` is the SAME identifier as in the DVH
# files and in metrics_per_case.
IPSS_MINIMAL_CSV = Path(os.environ.get(
    "PROTECTA_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "dataset_minimal_v2.csv"
PROSTATE_VOL_COL = "ldr_post_vol"
PROSTATE_VOL_UNITE = "cc"

# ============================================================
# STRUCTURE HARMONISATION
# ============================================================
# Documented pitfall: the deterministic file names the structure 'Bladder neck'
# (with a space) while the rest of the pipeline uses 'BladderNeck'. Without this
# rename, the merge is silently empty for the bladder neck.
STRUCTURE_RENAME = {
    "Bladder neck": "BladderNeck",
    "Bladder Neck": "BladderNeck",
}

# For the Monte-Carlo source, the structure -> model mapping is 1:1, so
# filtering on 'structure' selects the right rows and the 'model' column is
# redundant. The map is kept for traceability and an optional check.
STRUCTURE_TO_MODEL = {
    "Bladder": "bladder",
    "BladderNeck": "prostate",
    "Prostate": "prostate",
    "Rectum": "rectum",
    "Urethra": "prostate",
}

# ============================================================
# PANEL - organs and indices (_pct columns)
# ============================================================
# Each entry is (structure, index, tier). 'index' is the FULL _pct COLUMN NAME
# (e.g. 'D90_pct'); the Monte-Carlo columns derive from it by suffix (below).
#
# CONFIRMATORY - Prostate (the Holm family).
# EXPLORATORY  - BladderNeck (OUTSIDE Holm, reported uncorrected).
# EXCLUDED     - Urethra: its manual contour is fixed and identical across the
#                three pipelines, and the dose grid is fixed too, so the indices
#                are identical (every urethral __std is 0, and the urethra is
#                absent from the deterministic file). There is no concordance to
#                estimate.
TIER_CONF = "confirmatory"
TIER_EXPL = "exploratory"

PANEL = [
    ("Prostate", "D90_pct", TIER_CONF),
    ("Prostate", "V100_pct", TIER_CONF),
    ("Prostate", "V150_pct", TIER_CONF),
    ("Prostate", "V200_pct", TIER_CONF),
    ("BladderNeck", "D2cc_pct", TIER_EXPL),
    ("BladderNeck", "D1cc_pct", TIER_EXPL),
    ("BladderNeck", "V100_pct", TIER_EXPL),
]

# (structure, index) pairs - convenient for the loaders.
PANEL_PAIRS = [(s, i) for s, i, _ in PANEL]

# Excluded structure and the reason, documented in a CSV by script 01.
EXCLUDED_STRUCTURES = [
    {
        "structure": "Urethra",
        "reason": (
            "Manual contour fixed and identical across the three pipelines "
            "(manual, deterministic, Bayesian) with a fixed dose grid, so the "
            "indices are identical. Verified: every urethral __std is 0, and "
            "the urethra is absent from the deterministic file. There is no "
            "concordance to estimate."
        ),
        "flags": "excluded_by_design",
    }
]

# ============================================================
# PRE-DECLARED EQUIVALENCE MARGINS
# ============================================================
# unite: 'pct' = points of % of the prescription (dose indices, D prefix)
#        'pp'  = percentage points of volume (volume indices, V prefix)
# delta_conf is the decision margin (the strong claim); delta_sens is the
# sensitivity margin. Confirmatory entries carry two distinct margins;
# exploratory ones carry a single, widened quality-assurance margin
# (conf == sens) and are flagged 'no_lit_anchor'.
DELTA = {
    ("Prostate", "D90_pct"): dict(
        tier=TIER_CONF, unite="pct", delta_conf=10.0, delta_sens=23.0,
        ancre="De Brabandere 2012 (CT contouring) = 23%; fine geometric bound 1.1 mm ~ 9%",
        flags="",
    ),
    ("Prostate", "V100_pct"): dict(
        tier=TIER_CONF, unite="pp", delta_conf=3.0, delta_sens=5.0,
        ancre="expert-vs-fusion ~ 2.4 pp",
        flags="",
    ),
    ("Prostate", "V150_pct"): dict(
        tier=TIER_CONF, unite="pp", delta_conf=4.0, delta_sens=6.0,
        ancre="V-high family, widened literature bound",
        flags="",
    ),
    ("Prostate", "V200_pct"): dict(
        tier=TIER_CONF, unite="pp", delta_conf=4.0, delta_sens=6.0,
        ancre="V-high family, widened literature bound",
        flags="",
    ),
    ("BladderNeck", "D2cc_pct"): dict(
        tier=TIER_EXPL, unite="pct", delta_conf=10.0, delta_sens=10.0,
        ancre="", flags="no_lit_anchor;hathout",
    ),
    ("BladderNeck", "D1cc_pct"): dict(
        tier=TIER_EXPL, unite="pct", delta_conf=10.0, delta_sens=10.0,
        ancre="", flags="no_lit_anchor;hathout",
    ),
    ("BladderNeck", "V100_pct"): dict(
        tier=TIER_EXPL, unite="pp", delta_conf=5.0, delta_sens=5.0,
        ancre="", flags="no_lit_anchor;hathout",
    ),
}

# ============================================================
# SEGMENTATION QUALITY METRICS
# ============================================================
# Each entry carries orient, label and unite.
#   orient = +1  a HIGH value means a good segmentation (Dice, recall, ...)
#   orient = -1  a LOW value means a good segmentation (distances, errors)
#
# Every threshold analysis works on the ORIENTED SCORE s = orient * x, so that a
# high score ALWAYS means a better segmentation. Thresholds are converted back
# into the metric's native unit before being reported.
#
# NB: 'jaccard' is deliberately absent. J = D/(2-D) is a strictly monotone
# transform of the Dice, so the AUC, the Spearman correlation and the Youden
# cutpoint are identical to it BY CONSTRUCTION; including it would only
# duplicate a row.
QUALITY_METRICS = {
    "dice":               dict(orient=+1, label="Dice",                  unite=""),
    "assd_mm":            dict(orient=-1, label="ASSD",                  unite="mm"),
    "hausdorff95_mm":     dict(orient=-1, label="HD95",                  unite="mm"),
    "precision":          dict(orient=+1, label="Precision",             unite=""),
    "recall_sensitivity": dict(orient=+1, label="Recall",                unite=""),
    "volume_similarity":  dict(orient=+1, label="Volume similarity",     unite=""),
    "abs_rel_vol_err":    dict(orient=-1, label="|rel. volume error|",   unite=""),
    "pr_asym":            dict(orient=-1, label="|precision − recall|",  unite=""),
}


def index_label(index: str) -> str:
    """Figure label for a DVH index: 'D90_pct' -> 'D90'.

    The columns carry the '_pct' suffix (see PANEL), but it has no place in a
    figure: the unit is already carried by the axis or the legend.
    """
    return str(index).replace("_pct", "")


# Unit labels for the figure axes (see the 'unite' key of DELTA).
UNIT_LABEL = {"pct": "% of prescription", "pp": "percentage points"}


def unit_label(unite: str) -> str:
    return UNIT_LABEL.get(str(unite), str(unite))

# Metrics retained as the THRESHOLD AXIS in 05; see 06 for the empirical
# comparison that motivates the choice.
THRESHOLD_METRICS = ("dice", "assd_mm")

# ORTHOGONAL, SIGNED axis: it does not predict |delta| but the DIRECTION of the
# dosimetric gap, which no unsigned metric can. Convention:
# volume_error_ml = pred - gt, so rel_vol_err > 0 means OVER-segmentation.
SIGNED_AXIS = "rel_vol_err"


# ============================================================
# STATISTICS
# ============================================================
ALPHA = 0.05          # level of each one-sided TOST test
CI_LEVEL = 0.90       # two-sided CI = (1 - 2*ALPHA), consistent with the TOST
COMPARISONS = ("det", "bayes")


def ensure_dir(path) -> Path:
    """Create the directory (with parents) and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, path, dpi: int = 150, **kwargs) -> Path:
    """Save a figure as PNG **and** PDF under the same base name.

    The PNG is for quick review, the PDF is the vector format to submit. Going
    through this helper avoids duplicating (and desynchronising) two savefig
    calls in every figure function.

    Returns the PNG path, for the [figure] message of the calling scripts.
    """
    p = Path(path)
    ensure_dir(p.parent)
    for ext in (".png", ".pdf"):
        fig.savefig(p.with_suffix(ext), dpi=dpi, **kwargs)
    return p.with_suffix(".png")


def save_csv(df: pd.DataFrame, path, overwrite: bool = True) -> None:
    """Write a CSV. Overwrites by default; --no-overwrite refuses to."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(
            f"[STOP] {path} already exists. Rerun without --no-overwrite to overwrite."
        )
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[written] {path}  ({len(df)} rows)")


# ------------------------------------------------------------
# Harmonised loading (long format)
# ------------------------------------------------------------
def _read_harmonized(path: Path) -> pd.DataFrame:
    """Read a DVH CSV, rename the structures and normalise record_id to str."""
    if not path.exists():
        raise SystemExit(f"[STOP] File not found: {path}")
    df = pd.read_csv(path)
    if "structure" in df.columns:
        df["structure"] = df["structure"].replace(STRUCTURE_RENAME)
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype(str)
    return df


def _require_columns(df: pd.DataFrame, cols, path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[STOP] Columns missing from {path.name}: {missing}\n"
            f"       Available columns: {sorted(df.columns.tolist())}"
        )


def load_source_long(path: Path, pairs=PANEL_PAIRS, value_name: str = "val") -> pd.DataFrame:
    """
    Load a (record_id, structure, <idx>_pct...) file in long format, restricted
    to the panel: columns [record_id, structure, index, <value_name>].

    Used for the manual source (val_manual) and the deterministic one (val_auto).
    """
    df = _read_harmonized(path)
    _require_columns(df, ["record_id", "structure"], path)
    out = []
    for structure, index in pairs:
        _require_columns(df, [index], path)
        sub = df[df["structure"] == structure]
        part = sub[["record_id"]].copy()
        part["structure"] = structure
        part["index"] = index
        part[value_name] = pd.to_numeric(sub[index], errors="coerce").to_numpy()
        out.append(part)
    return pd.concat(out, ignore_index=True)


def load_mc_long(pairs=PANEL_PAIRS, check_model: bool = True) -> pd.DataFrame:
    """
    Load the Monte-Carlo summary in long format, restricted to the panel:
    [record_id, structure, index, mc_mean, mc_std, mc_lo, mc_hi].

    mc_mean = <idx>__mean   (= the Bayesian E[index])
    mc_std  = <idx>__std
    mc_lo   = <idx>__p2_5   (lower bound of the 95% predictive interval)
    mc_hi   = <idx>__p97_5  (upper bound)

    The structure -> model mapping being 1:1, filtering happens on structure; if
    check_model is set, the 'model' of the retained rows is checked to be unique
    and consistent with STRUCTURE_TO_MODEL.
    """
    df = _read_harmonized(MC_SUMMARY_CSV)
    _require_columns(df, ["record_id", "structure"], MC_SUMMARY_CSV)
    out = []
    for structure, index in pairs:
        sub = df[df["structure"] == structure]
        cols = {
            "mc_mean": f"{index}__mean",
            "mc_std": f"{index}__std",
            "mc_lo": f"{index}__p2_5",
            "mc_hi": f"{index}__p97_5",
        }
        _require_columns(df, list(cols.values()), MC_SUMMARY_CSV)
        if check_model and "model" in sub.columns and len(sub):
            expected = STRUCTURE_TO_MODEL.get(structure)
            models = set(sub["model"].dropna().unique().tolist())
            if expected is not None and models and models != {expected}:
                raise SystemExit(
                    f"[STOP] {structure}: unexpected MC models {models}, "
                    f"expected {{{expected}}}. Check the structure->model mapping."
                )
        part = sub[["record_id"]].copy()
        part["structure"] = structure
        part["index"] = index
        for out_col, src_col in cols.items():
            part[out_col] = pd.to_numeric(sub[src_col], errors="coerce").to_numpy()
        out.append(part)
    return pd.concat(out, ignore_index=True)


def load_geometry(organs=("Prostate",)) -> pd.DataFrame:
    """
    Per-case geometric metrics, from the `metrics_per_case.csv` files of the
    cross-dataset (deterministic predictions).

    Returns [record_id, geom_organ, <native metrics>, rel_vol_err,
    abs_rel_vol_err, pr_asym]. The `case_id` key of the metrics file is the same
    identifier as `record_id` in the DVH files.

    Derived columns:
        rel_vol_err     = volume_error_ml / gt_volume_ml, SIGNED (>0 means
                          over-segmentation). Normalising by the ground-truth
                          volume makes the error comparable across patients,
                          which volume_error_ml in mL is not.
        abs_rel_vol_err = |rel_vol_err|.
        pr_asym         = |precision - recall_sensitivity|, the asymmetry of the
                          error type (over- versus under-segmentation) that the
                          Dice, being their harmonic mean, aggregates and
                          therefore hides.
    """
    native = ["dice", "jaccard", "precision", "recall_sensitivity", "specificity",
              "volume_similarity", "hausdorff95_mm", "hausdorff_mm", "assd_mm",
              "gt_volume_ml", "pred_volume_ml", "volume_error_ml"]
    rows = []
    for organ in organs:
        path = GEOM_DATASETS.get(organ)
        if path is None:
            raise SystemExit(
                f"[STOP] Unknown organ for the geometry: {organ}. "
                f"Known: {sorted(GEOM_DATASETS)}"
            )
        if not path.exists():
            raise SystemExit(f"[STOP] metrics_per_case not found: {path}")
        d = pd.read_csv(path)
        _require_columns(d, ["case_id"] + native, path)
        out = pd.DataFrame({"record_id": d["case_id"].astype(str)})
        out["geom_organ"] = organ
        for c in native:
            out[c] = pd.to_numeric(d[c], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["rel_vol_err"] = out["volume_error_ml"] / out["gt_volume_ml"].replace(0, np.nan)
        out["abs_rel_vol_err"] = out["rel_vol_err"].abs()
        out["pr_asym"] = (out["precision"] - out["recall_sensitivity"]).abs()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_prostate_volume(path: Path | None = None, col: str | None = None,
                         strict: bool = True) -> pd.DataFrame:
    """
    Clinical prostate volume, ONE row per patient: [record_id, prostate_vol].

    The source file is longitudinal (one row per visit) but `ldr_post_vol` is a
    per-patient constant. That invariance is CHECKED rather than assumed - a
    patient with diverging volumes would signal an extraction problem, not a
    case to average away silently - and the table is then deduplicated.

    NOT TO BE CONFUSED with `gt_volume_ml` from load_geometry():
        - gt_volume_ml  volume of the MANUAL CONTOUR on the post-implant CT. It
                        is a property of the reference SEGMENTATION and is the
                        denominator of rel_vol_err.
        - prostate_vol  clinical prostate volume recorded in the database. It is
                        a PATIENT covariate (their anatomy), available BEFORE any
                        segmentation, and therefore usable as an a-priori triage
                        criterion.
    Both measure the same organ and are correlated, but they play different
    roles: the first is an intermediate of the error chain, the second a patient
    stratification factor.

    Returns [record_id (str), prostate_vol (float)], with rows missing the volume
    dropped (no imputation, following the convention of this arm).

    `path` and `col` are resolved AT CALL TIME from the module constants rather
    than bound at definition, so the synthetic validation can repoint them as it
    does for MANUAL_CSV / DET_CSV.
    """
    path = Path(path) if path is not None else IPSS_MINIMAL_CSV
    col = col if col is not None else PROSTATE_VOL_COL
    if not path.exists():
        raise SystemExit(f"[STOP] File not found: {path}")
    df = pd.read_csv(path, usecols=["record_id", col])
    df["record_id"] = df["record_id"].astype(str)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    nuniq = df.dropna(subset=[col]).groupby("record_id")[col].nunique()
    bad = nuniq[nuniq > 1]
    if len(bad):
        msg = (f"[{'STOP' if strict else 'WARNING'}] {len(bad)} patients have "
               f"several distinct values of {col}, although that variable is "
               f"supposed to be constant per patient.")
        if strict:
            raise SystemExit(msg)
        print(msg + " -> using the median.")

    out = (df.dropna(subset=[col])
             .groupby("record_id", as_index=False)[col].median()
             .rename(columns={col: "prostate_vol"}))
    return out


def load_paired_with_geometry(structure_to_organ, pairs=PANEL_PAIRS) -> pd.DataFrame:
    """
    Paired manual-versus-deterministic long table, enriched with the geometric
    metrics of the organ that DRIVES each structure.

    Columns: [record_id, structure, index, val_manual, val_auto, diff, abs_diff,
              geom_organ, <metrics from load_geometry>].

    `structure_to_organ` makes explicit which organ's geometry governs each DVH
    structure; structures absent from the map are excluded. The sign convention
    is preserved: diff = val_manual - val_auto.
    """
    pairs = [(s, i) for s, i in pairs if s in structure_to_organ]
    if not pairs:
        raise SystemExit("[STOP] No panel pair appears in structure_to_organ.")
    man = load_source_long(MANUAL_CSV, pairs=pairs, value_name="val_manual")
    det = load_source_long(DET_CSV, pairs=pairs, value_name="val_auto")
    key = ["record_id", "structure", "index"]
    long = man.merge(det[key + ["val_auto"]], on=key, how="inner")
    long = long.dropna(subset=["val_manual", "val_auto"])
    long["diff"] = long["val_manual"] - long["val_auto"]
    long["abs_diff"] = long["diff"].abs()

    long["geom_organ"] = long["structure"].map(structure_to_organ)
    geom = load_geometry(organs=sorted(set(structure_to_organ.values())))
    out = long.merge(geom, on=["record_id", "geom_organ"], how="inner")
    return out.dropna(subset=["diff"]).reset_index(drop=True)


# ------------------------------------------------------------
# Equivalence and concordance primitives
# ------------------------------------------------------------
def tost_paired(diff, delta, alpha: float = ALPHA):
    """
    Paired TOST, on the paired differences diff = manual - auto.

    Returns a dict:
        n, mean_diff, sd_diff, se,
        ci_lo, ci_hi      two-sided (1 - 2*alpha) CI, independent of delta,
        p_lower, p_upper  p-values of the two one-sided tests at +/- delta,
        p_tost            max(p_lower, p_upper),
        equivalent        True when the (1-2a) CI lies strictly inside
                          (-delta, +delta), which is equivalent to
                          p_tost < alpha.

    Tests:
        lower  H0: mu <= -delta  vs H1: mu > -delta
        upper  H0: mu >=  delta  vs H1: mu <  delta
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = diff.size
    res = dict(n=n, mean_diff=np.nan, sd_diff=np.nan, se=np.nan,
               ci_lo=np.nan, ci_hi=np.nan, p_lower=np.nan, p_upper=np.nan,
               p_tost=np.nan, equivalent=False)
    if n < 2:
        return res
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    df = n - 1
    res.update(mean_diff=mean, sd_diff=sd)
    if sd == 0.0:
        # Constant differences: the decision is deterministic.
        equivalent = abs(mean) < delta
        res.update(se=0.0, ci_lo=mean, ci_hi=mean,
                   p_lower=0.0 if mean > -delta else 1.0,
                   p_upper=0.0 if mean < delta else 1.0,
                   p_tost=0.0 if equivalent else 1.0,
                   equivalent=bool(equivalent))
        return res
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha, df)          # two-sided (1 - 2*alpha) CI
    ci_lo = mean - tcrit * se
    ci_hi = mean + tcrit * se
    t_lower = (mean + delta) / se               # H0: mu = -delta
    p_lower = stats.t.sf(t_lower, df)           # reject if mean is clearly > -delta
    t_upper = (mean - delta) / se               # H0: mu = +delta
    p_upper = stats.t.cdf(t_upper, df)          # reject if mean is clearly < +delta
    p_tost = max(p_lower, p_upper)
    res.update(se=se, ci_lo=ci_lo, ci_hi=ci_hi, p_lower=p_lower,
               p_upper=p_upper, p_tost=p_tost,
               equivalent=bool((ci_lo > -delta) and (ci_hi < delta)))
    return res


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient (CCC), population moments."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sx2 = np.mean((x - mx) ** 2)
    sy2 = np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return np.nan
    return float(2 * sxy / denom)


def icc_a1(x, y):
    """
    ICC(A,1) - two-way random-effects model, ABSOLUTE agreement, single measure
    (McGraw & Wong 1996; Shrout & Fleiss ICC(2,1)). Here k = 2 raters (manual,
    automatic) and n subjects. Explicit ANOVA decomposition:

        ICC(A,1) = (MSR - MSE) /
                   (MSR + (k-1)*MSE + (k/n)*(MSC - MSE))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = x.size
    if n < 2:
        return np.nan
    k = 2
    M = np.column_stack([x, y])              # n subjects x k raters
    grand = M.mean()
    row_means = M.mean(axis=1)               # per subject
    col_means = M.mean(axis=0)               # per rater
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((M - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    dfe = (n - 1) * (k - 1)
    if dfe <= 0:
        return np.nan
    mse = ss_err / dfe
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def bland_altman(val_manual, val_auto):
    """
    Bland-Altman descriptors.
    Returns: bias, sd_diff, loa_lo, loa_hi, prop_slope, prop_slope_p, n.
    The proportional bias is the OLS slope of diff on the mean (linregress).
    """
    m = np.asarray(val_manual, dtype=float)
    a = np.asarray(val_auto, dtype=float)
    mask = ~(np.isnan(m) | np.isnan(a))
    m, a = m[mask], a[mask]
    n = m.size
    res = dict(n=n, bias=np.nan, sd_diff=np.nan, loa_lo=np.nan, loa_hi=np.nan,
               prop_slope=np.nan, prop_slope_p=np.nan)
    if n < 2:
        return res
    diff = m - a
    means = (m + a) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    res.update(bias=bias, sd_diff=sd,
               loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd)
    if np.ptp(means) > 0 and n >= 3:
        lr = stats.linregress(means, diff)
        res.update(prop_slope=float(lr.slope), prop_slope_p=float(lr.pvalue))
    return res


def coverage_stats(val_manual, mc_lo, mc_hi, mc_mean=None, mc_std=None):
    """
    Empirical coverage of the 95% Monte-Carlo predictive interval
    ([mc_lo, mc_hi]) by the reference manual value.

    Degenerate intervals (mc_lo == mc_hi) are counted separately
    (n_degenerate) and are NOT counted as covered by default; coverage is
    estimated on the non-degenerate intervals.

    Gaussian variant (when mc_mean and mc_std are supplied): the interval
    mean +/- 1.96*std, which works around the coarseness of the p2.5/p97.5
    quantiles estimated from a small number of draws.

    Returns a dict of counts and rates.
    """
    v = np.asarray(val_manual, dtype=float)
    lo = np.asarray(mc_lo, dtype=float)
    hi = np.asarray(mc_hi, dtype=float)
    mask = ~(np.isnan(v) | np.isnan(lo) | np.isnan(hi))
    v, lo, hi = v[mask], lo[mask], hi[mask]
    n_total = v.size
    degen = lo == hi
    n_degen = int(degen.sum())
    nd = ~degen
    n_nd = int(nd.sum())
    covered_nd = int(np.sum((v[nd] >= lo[nd]) & (v[nd] <= hi[nd])))
    res = dict(
        n_total=n_total,
        n_degenerate=n_degen,
        n_nondegenerate=n_nd,
        n_covered_nondegenerate=covered_nd,
        coverage_pctinterval=(covered_nd / n_nd) if n_nd else np.nan,
        coverage_gaussian=np.nan,
        n_covered_gaussian=np.nan,
        n_gaussian=np.nan,
    )
    if mc_mean is not None and mc_std is not None:
        mu = np.asarray(mc_mean, dtype=float)[mask]
        sd = np.asarray(mc_std, dtype=float)[mask]
        gmask = ~(np.isnan(mu) | np.isnan(sd))
        mu, sd, vg = mu[gmask], sd[gmask], v[gmask]
        glo, ghi = mu - 1.96 * sd, mu + 1.96 * sd
        n_g = vg.size
        cov_g = int(np.sum((vg >= glo) & (vg <= ghi)))
        res.update(n_gaussian=n_g,
                   n_covered_gaussian=cov_g,
                   coverage_gaussian=(cov_g / n_g) if n_g else np.nan)
    return res


def delta_row(structure: str, index: str) -> dict:
    """Return the pre-declared margin row for (structure, index)."""
    return DELTA[(structure, index)]
