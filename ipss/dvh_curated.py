"""
dvh_curated.py
==============
"curated" DVH strategy: instead of reducing EVERY index by PCA (dvh_pca.py), a
RESTRICTED PANEL of dose-volume indices clinically motivated for urinary
toxicity is selected (read through dvh_mc.load_dvh_indices from the active
segmentation source config.DVH_SEG_SOURCE; for the Monte-Carlo source the
statistic in config.DVH_MC_STAT is used). Features stay interpretable - one
index is one named feature <Structure>_<metric> - at the price of an a-priori
selection rather than an unsupervised compression.

Note: the urethra is never segmented automatically, so for the auto_det source
the urethral indices of the panel (uD*) are taken from the manual segmentation
(see dvh_mc.load_dvh_indices, MANUAL_ONLY_STRUCTURES).

Default panel (config.DVH_CURATED_PANEL):
  Prostate    : D90_Gy, V100_pct, V150_pct, V200_pct   (target coverage / overdose)
  Urethra     : uD10_Gy, uD30_Gy, uD5_Gy, uD0.1cc_Gy   (urethral hot spots)
  BladderNeck : D2cc_Gy, D1cc_Gy, V100_pct             (bladder neck dose)

compute_dvh_curated() returns a DataFrame indexed by record_id over the
requested cohort, with <Structure>_<metric> columns. Patients without a DVH get
NaN, imputed downstream by each model's preprocessor, exactly as in the PCA
strategy. No target y is used: this step is unsupervised.
"""
from pathlib import Path

import pandas as pd

from config import (
    DVH_INDICES_CSV, DVH_CURATED_PANEL, DVH_SEG_SOURCE, DVH_MC_STAT,
    DVH_MC_VAR_SUFFIX,
)
from dvh_mc import load_dvh_indices


def _curated_wide(med: pd.DataFrame, panel: dict) -> pd.DataFrame:
    """Pivot DVH indices (record_id, structure, metrics) to one row per patient.

    Columns are the <Structure>_<metric> entries of the panel, indexed by
    record_id as a string.

    For a Bayesian source with the Monte-Carlo variance enabled (see
    config.DVH_MC_USE_VARIANCE), each panel index `m` is accompanied by its
    variance `m<DVH_MC_VAR_SUFFIX>` when that column is present in `med`.
    """
    long = med[med["structure"].isin(panel)].set_index(["record_id", "structure"])
    avail_structs = set(long.index.get_level_values("structure"))
    cols = {}
    for struct, metrics in panel.items():
        for m in metrics:
            # Central index (required), then its Monte-Carlo variance if available.
            for metric in (m, f"{m}{DVH_MC_VAR_SUFFIX}"):
                name = f"{struct}_{metric}"
                # A whole STRUCTURE may be missing from a source (some automatic
                # segmentations do not contour BladderNeck). Treat it as a
                # missing index -> NaN imputed downstream, rather than letting
                # .xs raise a KeyError.
                if metric in long.columns and struct in avail_structs:
                    cols[name] = long[metric].xs(struct, level="structure")
                elif metric == m:
                    print(f"  ! curated DVH index absent from the file: {name} (-> NaN)")
                    cols[name] = pd.Series(dtype=float)
                # A missing variance (deterministic source) is simply omitted.
    return pd.DataFrame(cols).sort_index()


def compute_dvh_curated(record_ids, panel: dict = DVH_CURATED_PANEL,
                        out_dir: Path | None = None) -> pd.DataFrame:
    """Extract the curated DVH panel for the requested patients.

    Indices come from the segmentation source in config.DVH_SEG_SOURCE, reduced
    to the central statistic in config.DVH_MC_STAT for the Monte-Carlo source.

    Args:
        record_ids: cohort identifiers.
        panel: dict {structure: [metrics]} (defaults to config.DVH_CURATED_PANEL).
        out_dir: if given, writes dvh_curated_panel.csv there for inspection.

    Returns:
        DataFrame indexed by record_id (the full requested cohort, with its
        original labels), <Structure>_<metric> columns in panel order. NaN for
        patients without a DVH.
    """
    med = load_dvh_indices(DVH_INDICES_CSV, DVH_SEG_SOURCE, DVH_MC_STAT)
    wide = _curated_wide(med, panel)  # index = record_id as string

    # Type robustness (as in dvh_pca): record_id is a string in the DVH file
    # while X.index may be numeric. Match in string space, then restore the
    # cohort's original labels.
    orig_ids = pd.Index(pd.unique(record_ids), name="record_id")
    str_to_orig = pd.Series(orig_ids.values, index=orig_ids.astype(str).str.strip())
    str_to_orig = str_to_orig[~str_to_orig.index.duplicated()]

    # Stable order and guaranteed columns. _curated_wide already built the
    # columns in panel order (central index first, then its _var counterpart in
    # the Bayesian case), so that insertion order is kept as is.
    panel_cols = list(wide.columns)
    wide = wide.reindex(columns=panel_cols)

    scores = wide.rename(index=str_to_orig).reindex(orig_ids)
    scores.index.name = "record_id"

    n_with = int(scores.notna().any(axis=1).sum())
    print(f"  - DVH curated: {scores.shape[1]} indices "
          f"({n_with}/{len(orig_ids)} patients with a DVH; the rest imputed downstream)")

    if out_dir is not None:
        scores.to_csv(Path(out_dir) / "dvh_curated_panel.csv")

    return scores
