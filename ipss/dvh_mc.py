"""
dvh_mc.py
=========
Reads the dose-volume (DVH) index sets produced from three different
segmentations of the organs (prostate, urethra, bladder, bladder neck, rectum):

  - "manual"   -> manual segmentation (clinical reference),
                  `dvh_indices_manual_seg.csv`;
  - "auto_det" -> automatic segmentation from a deterministic model,
                  `dvh_indices_auto_seg_det.csv`;
  - "mc_bayes" -> automatic segmentation from a Bayesian model with 20
                  Monte-Carlo passes, `dvh_mc_summary.csv`.

Source formats
--------------
- "manual"/"auto_det": flat file, one row per (record_id, structure), each
  dose-volume index in its own column (`Dmean_Gy`, `V100_pct`, ...), plus
  metadata and QA columns (n_voxels, flags, pred_mode, ...) which are dropped.
  The deterministic segmentation names the bladder neck "Bladder neck" (renamed
  to "BladderNeck") and does NOT contain the urethra: no model segments the
  urethra automatically, so for the "auto_det" source the urethra and its uD*
  indices are taken from the MANUAL segmentation (see load_dvh_indices and
  MANUAL_ONLY_STRUCTURES).
- "mc_bayes": one row per (record_id, structure), each index summarised by seven
  statistics of the Monte-Carlo uncertainty,
  `<index>__mean/__std/__min/__p2_5/__p50/__p97_5/__max`. One of them is kept as
  the central value (see DVH_MC_STAT); the others stay in the file for a
  dedicated uncertainty analysis but do not enter X.

Every function returns the SAME normalised format - one row per (record_id,
structure), columns = dose-volume indices - so it can be consumed
interchangeably by `00_build_ipss_dataset.py`, `dvh_pca.py` and
`dvh_curated.py`.

Structures: Bladder, BladderNeck, Prostate, Rectum, Urethra. The
urethra-specific indices (uD5/uD10/uD30/uD0.1cc/uD*_base) are defined only for
the Urethra structure and are NaN elsewhere.
"""
from pathlib import Path

import pandas as pd

# Monte-Carlo statistic used as the central value of each index ("mc_bayes").
DVH_MC_STAT = "mean"

# Monte-Carlo uncertainty as a predictor ("mc_bayes"): on top of the central
# value, the VARIANCE of the Monte-Carlo passes of each index (= __std^2) can be
# added as a separate feature suffixed DVH_MC_VAR_SUFFIX. That variance encodes
# the segmentation uncertainty of the Bayesian model, since an index that is
# unstable from one pass to the next is less reliable. DVH_MC_VAR_STAT is the
# dispersion statistic read from the summary file (standard deviation), squared
# to obtain the variance.
DVH_MC_VAR_STAT = "std"
DVH_MC_VAR_SUFFIX = "_var"

# Structure metadata (size, geometry, QA) - NOT dose-volume indices. Removed
# before any use as a feature. (prescription_dose_Gy / model / n_passes are flat
# columns with no statistic suffix and are therefore never captured by the
# __<stat> filter; they are listed here for safety.)
META_BASE_METRICS = {
    "n_voxels", "n_planes", "n_planes_interpolated", "thickness_mm",
    "prescription_dose_Gy", "volume_cc", "mask_volume_cc", "coverage_in_dose",
}

# Columns of the flat files (manual/auto_det) that are NOT dose-volume indices:
# model identifiers, prescription sources, QA flags. Removed before any use as a
# feature, in addition to META_BASE_METRICS.
PLAIN_NON_INDEX_COLS = META_BASE_METRICS | {
    "model", "n_passes", "pred_mode", "presc_source", "dose_grid_scaling",
    "n_dose_voxels_clamped", "flags", "urethra_subsegmentable",
}

# Normalisation of structure names that differ between segmentations (the
# deterministic segmentation writes "Bladder neck" with a space).
STRUCTURE_RENAME = {"Bladder neck": "BladderNeck"}

# Structure(s) NEVER segmented automatically: no model contours the urethra. For
# the "auto_det" source its indices (including the urethral uD* indices) are
# therefore taken from the MANUAL segmentation.
MANUAL_ONLY_STRUCTURES = {"Urethra"}

# Dataset variants: a source name may carry a suffix designating an alternative
# set of files (indices recomputed on another contour set) while sharing the
# FORMAT and dispatch logic of its base source. The base source is resolved
# before the format dispatch.
DATASET_VARIANT_SUFFIXES = ("_clin0977",)


def base_source(source: str) -> str:
    """Return the base source of a possibly variant-suffixed source name.

    See DATASET_VARIANT_SUFFIXES, e.g. 'mc_bayes_clin0977' -> 'mc_bayes',
    'auto_det' -> 'auto_det'. Used for the format dispatch, since variants share
    the format and logic of their base source.
    """
    for suf in DATASET_VARIANT_SUFFIXES:
        if source.endswith(suf):
            return source[: -len(suf)]
    return source


_ID_COLS = ["record_id", "structure"]


def load_mc_median(path: str | Path, stat: str = DVH_MC_STAT,
                   drop_meta: bool = True) -> pd.DataFrame:
    """Load the Monte-Carlo summary file, keeping only the `stat` statistic of
    each index, renamed without its suffix.

    Args:
        path: path to the Monte-Carlo summary file.
        stat: Monte-Carlo statistic to extract (defaults to DVH_MC_STAT).
        drop_meta: if True, removes the structure metadata (META_BASE_METRICS)
            so only dose-volume indices remain.

    Returns:
        DataFrame with one row per (record_id, structure); columns are
        record_id, structure, then each dose-volume index (e.g. `Dmean_Gy`,
        `V100_pct`, `uD5_Gy`) at the selected statistic. record_id is a
        stripped string.
    """
    df = pd.read_csv(path, dtype={"record_id": str})
    df["record_id"] = df["record_id"].str.strip()

    suffix = f"__{stat}"
    stat_cols = [c for c in df.columns if c.endswith(suffix)]
    if not stat_cols:
        raise ValueError(
            f"No column ending with '{suffix}' in {path}. "
            f"Expected statistics: __mean/__std/__min/__p2_5/__p50/__p97_5/__max."
        )

    rename = {c: c[: -len(suffix)] for c in stat_cols}
    out = df[_ID_COLS + stat_cols].rename(columns=rename)

    if drop_meta:
        out = out.drop(columns=[m for m in META_BASE_METRICS if m in out.columns])

    return out.reset_index(drop=True)


def load_mc_variance(path: str | Path, var_stat: str = DVH_MC_VAR_STAT,
                     suffix: str = DVH_MC_VAR_SUFFIX,
                     drop_meta: bool = True) -> pd.DataFrame:
    """Load the VARIANCE of the Monte-Carlo passes of each DVH index
    ("mc_bayes" source): variance = (standard deviation `__{var_stat}`)^2.

    Used to supply the segmentation uncertainty of the Bayesian model as a
    predictor, alongside the central value from load_mc_median.

    Args:
        path: path to the Monte-Carlo summary file.
        var_stat: dispersion statistic to read (defaults to the standard
            deviation), squared to obtain the variance.
        suffix: suffix appended to each index name to distinguish the variance
            column from the value column.
        drop_meta: if True, ignores the structure metadata (META_BASE_METRICS)
            so only dose-volume indices remain.

    Returns:
        DataFrame with one row per (record_id, structure); columns are
        record_id, structure, then each dose-volume index suffixed (e.g.
        `Dmean_Gy_var`, `uD5_Gy_var`) at its Monte-Carlo variance. record_id is
        a stripped string.
    """
    df = pd.read_csv(path, dtype={"record_id": str})
    df["record_id"] = df["record_id"].str.strip()

    stat_suffix = f"__{var_stat}"
    std_cols = [c for c in df.columns if c.endswith(stat_suffix)]
    if not std_cols:
        raise ValueError(
            f"No column ending with '{stat_suffix}' in {path}. "
            f"Expected statistics: __mean/__std/__min/__p2_5/__p50/__p97_5/__max."
        )

    out = df[_ID_COLS].copy()
    for c in std_cols:
        base = c[: -len(stat_suffix)]
        if drop_meta and base in META_BASE_METRICS:
            continue
        out[f"{base}{suffix}"] = pd.to_numeric(df[c], errors="coerce") ** 2

    return out.reset_index(drop=True)


def load_plain_indices(path: str | Path, drop_meta: bool = True) -> pd.DataFrame:
    """Load a flat DVH index file (manual or deterministic segmentation): one
    row per (record_id, structure), one dose-volume index per column.

    Normalises the structure names (STRUCTURE_RENAME) and, if drop_meta,
    removes the non dose-volume metadata and QA columns
    (PLAIN_NON_INDEX_COLS) to match the format returned by load_mc_median.

    Returns:
        DataFrame with one row per (record_id, structure); columns are
        record_id, structure, then each dose-volume index. record_id is a
        stripped string. Since the deterministic segmentation has no urethra,
        its uD* indices are simply absent and become NaN downstream.
    """
    df = pd.read_csv(path, dtype={"record_id": str})
    df["record_id"] = df["record_id"].str.strip()
    df["structure"] = df["structure"].replace(STRUCTURE_RENAME)

    if drop_meta:
        df = df.drop(columns=[c for c in PLAIN_NON_INDEX_COLS if c in df.columns])

    # One row per (record_id, structure) - guard against duplicates.
    df = df.drop_duplicates(subset=_ID_COLS, keep="first")

    return df.reset_index(drop=True)


def load_dvh_indices(path: str | Path, source: str,
                     stat: str = DVH_MC_STAT,
                     manual_path: str | Path | None = None,
                     add_variance: bool | None = None) -> pd.DataFrame:
    """Load the DVH indices in the normalised format, whatever the source
    segmentation.

    Args:
        path: path to the index file matching `source`.
        source: "manual" / "auto_det" (flat files) or "mc_bayes" (Monte-Carlo
            summary), possibly carrying a dataset-variant suffix.
        stat: for "mc_bayes", the Monte-Carlo statistic to extract (defaults to
            DVH_MC_STAT). Ignored for the flat sources.
        manual_path: path to the manual segmentation, used only for the
            "auto_det" source to take the urethra from it
            (MANUAL_ONLY_STRUCTURES, never segmented automatically). If None, it
            is read from config.DVH_SOURCES["manual"].
        add_variance: for "mc_bayes" ONLY, if True each index is accompanied by
            its Monte-Carlo VARIANCE (= __std^2, columns suffixed
            DVH_MC_VAR_SUFFIX) as additional features. If None, the value is
            read from config.DVH_MC_USE_VARIANCE. No effect for the
            deterministic segmentations, which have no Monte-Carlo draws.

    Returns:
        DataFrame with one row per (record_id, structure), columns = dose-volume
        indices, record_id as a stripped string. For "mc_bayes" with
        add_variance, each index `X` is accompanied by its variance `X_var`.
    """
    # Dataset variants share the format of their base source: dispatch on it.
    src = base_source(source)
    if src == "mc_bayes":
        med = load_mc_median(path, stat=stat)
        if add_variance is None:
            try:
                from config import DVH_MC_USE_VARIANCE
                add_variance = DVH_MC_USE_VARIANCE
            except Exception:
                add_variance = False
        if add_variance:
            var = load_mc_variance(path)
            med = med.merge(var, on=_ID_COLS, how="left")
        return med
    if src == "manual":
        return load_plain_indices(path)
    if src == "auto_det":
        det = load_plain_indices(path)
        # The urethra has no automatic segmentation: drop any urethra rows on
        # the deterministic side (for safety) and replace them with the
        # urethral indices of the MANUAL segmentation.
        if manual_path is None:
            from config import DVH_SOURCES
            manual_path = DVH_SOURCES["manual"]
        manual = load_plain_indices(manual_path)
        manual_uro = manual[manual["structure"].isin(MANUAL_ONLY_STRUCTURES)]
        det = det[~det["structure"].isin(MANUAL_ONLY_STRUCTURES)]
        out = pd.concat([det, manual_uro], ignore_index=True)
        return out.reset_index(drop=True)
    raise ValueError(
        f"Unknown DVH source: {source!r}. "
        f"Expected 'manual', 'auto_det' or 'mc_bayes'."
    )


def dvh_cohort_record_ids(sources, combine: str = "intersection") -> set:
    """Set of record_ids ELIGIBLE on DVH availability alone, independently of
    any target or feature.

    A patient is "available" for a source if their record_id appears in the raw
    file of that source, with at least one segmented structure. Only the
    record_id column is read (usecols), which stays cheap even on large index
    files. Used to restrict the training cohort IDENTICALLY across scenarios
    (see config.RESTRICT_TO_DVH_COHORT), so that every scenario shares exactly
    the same patients and stays comparable.

    Args:
        sources: list of config.DVH_SOURCES keys.
        combine: "intersection" -> record_ids present in EVERY source (a real
            DVH in each scenario, no imputation); "union" -> present in at least
            one source.

    Returns:
        set[str] of stripped record_ids, following the same conventions as
        load_plain_indices / load_mc_median (dtype str then .str.strip()).
    """
    from config import DVH_SOURCES

    id_sets = []
    for src in sources:
        if src not in DVH_SOURCES:
            raise ValueError(
                f"Unknown DVH source: {src!r} (expected one of: {list(DVH_SOURCES)})."
            )
        ids = pd.read_csv(DVH_SOURCES[src], usecols=["record_id"],
                          dtype={"record_id": str})["record_id"]
        id_sets.append(set(ids.str.strip().dropna().unique()))

    if not id_sets:
        return set()
    if combine == "union":
        return set().union(*id_sets)
    if combine == "intersection":
        out = set(id_sets[0])
        for s in id_sets[1:]:
            out &= s
        return out
    raise ValueError(
        f"Unknown combine mode: {combine!r} (expected 'intersection' or 'union')."
    )
