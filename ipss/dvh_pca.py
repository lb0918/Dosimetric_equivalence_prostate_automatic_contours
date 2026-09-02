"""
dvh_pca.py
==========
Reduces the DVH indices of the active segmentation source
(config.DVH_SEG_SOURCE, read by dvh_mc.load_dvh_indices as one row per
structure and patient; for the Monte-Carlo source the statistic in
config.DVH_MC_STAT is used) into per-patient principal components, keeping the
smallest number of PCs explaining at least DVH_PCA_VARIANCE of the variance.

For a Bayesian source with the Monte-Carlo variance enabled
(config.DVH_MC_USE_VARIANCE), the `<metric>_var` variance columns returned by
dvh_mc.load_dvh_indices are metrics like any other: they enter the pivot and
then the PCA automatically, the z-score standardisation handling the scale gap
between a value and its variance.

Pipeline (unsupervised, never uses the target y):
  1. pivot: one row per patient, columns = <Structure>_<metric>
     (Bladder_Dmean_Gy, Rectum_V70Gy_pct, ..., plus <metric>_var when enabled);
  2. median imputation -> z-score standardisation -> PCA;
  3. keep the first n PCs whose cumulative variance reaches the threshold.

Returns a DataFrame indexed by record_id with <DVH_PC_PREFIX>1..n columns.

NOTE on leakage: the PCA basis is fitted once on the supplied cohort, without
using the target. This matches the deployment preprocessor of 02_train.py, which
is also fitted on all patients. Since the PCA is unsupervised, the impact on the
out-of-fold evaluation is negligible.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (
    DVH_INDICES_CSV, DVH_PCA_VARIANCE, DVH_PC_PREFIX,
    DVH_SEG_SOURCE, DVH_MC_STAT,
)
from dvh_mc import load_dvh_indices


def _pivot_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot DVH indices (record_id, structure, metrics) to one row per patient.

    `df` comes from dvh_mc.load_mc_median, with metadata already removed.
    Columns that are entirely empty after the pivot - typically the
    urethra-specific indices (uD5/uD10/...) for the other structures - are
    dropped: they carry no information and would break the median imputation
    (median of an all-NaN column).
    """
    metric_cols = [c for c in df.columns if c not in ("record_id", "structure")]
    pivot = df.pivot_table(index="record_id", columns="structure",
                           values=metric_cols, aggfunc="first")
    pivot.columns = [f"{struct}_{metric}" for metric, struct in pivot.columns]
    pivot = pivot.dropna(axis=1, how="all")
    return pivot.sort_index()


def compute_dvh_pca(record_ids, variance_threshold: float = DVH_PCA_VARIANCE,
                    out_dir: Path | None = None) -> pd.DataFrame:
    """Compute the PCA scores of the DVH indices for the requested patients.

    Args:
        record_ids: cohort identifiers. The PCA basis is fitted on the
            intersection of these patients with those having a DVH.
        variance_threshold: minimum cumulative variance fraction to reach.
        out_dir: if given, writes dvh_pca_scores.csv and dvh_pca_variance.csv
            there for inspection and reproducibility.

    Returns:
        DataFrame indexed by record_id (the full requested cohort), with
        <DVH_PC_PREFIX>1..n columns. Patients without a DVH get NaN, imputed
        downstream by each model's preprocessor.
    """
    raw = load_dvh_indices(DVH_INDICES_CSV, DVH_SEG_SOURCE, DVH_MC_STAT)
    pivot = _pivot_per_patient(raw)

    # Type robustness: record_id is a string in the DVH file while X.index may
    # be numeric. Match in string space through a str -> label table, then
    # restore the cohort's original labels.
    orig_ids = pd.Index(pd.unique(record_ids), name="record_id")
    str_to_orig = pd.Series(orig_ids.values, index=orig_ids.astype(str).str.strip())
    str_to_orig = str_to_orig[~str_to_orig.index.duplicated()]
    with_dvh = pivot.index.intersection(str_to_orig.index)
    if len(with_dvh) == 0:
        raise ValueError("No patient in the cohort has DVH indices.")

    X_fit = pivot.loc[with_dvh]
    feature_names = X_fit.columns.tolist()

    # impute -> scale -> PCA (fitted on the cohort patients that have a DVH)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(imputer.fit_transform(X_fit))

    pca = PCA(random_state=0)
    pca.fit(X_sc)

    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cum, variance_threshold)) + 1
    n_pc = min(n_pc, pca.n_components_)

    print(f"  - DVH PCA: {len(feature_names)} indices -> {n_pc} PC "
          f"(cumulative variance {cum[n_pc - 1] * 100:.1f} % >= "
          f"{variance_threshold * 100:.0f} %); fitted on {len(with_dvh)} DVH patients")

    pc_cols = [f"{DVH_PC_PREFIX}{i + 1}" for i in range(n_pc)]

    # Scores of the patients with a DVH (reusing X_sc), remapped onto the
    # original labels; NaN for cohort patients without a DVH.
    scores_str = pd.DataFrame(pca.transform(X_sc)[:, :n_pc],
                              index=with_dvh, columns=pc_cols)
    scores = scores_str.rename(index=str_to_orig).reindex(orig_ids)
    scores.index.name = "record_id"

    if out_dir is not None:
        scores.to_csv(Path(out_dir) / "dvh_pca_scores.csv")
        var_df = pd.DataFrame({
            "PC": [f"{DVH_PC_PREFIX}{i + 1}" for i in range(pca.n_components_)],
            "explained_variance_pct": (pca.explained_variance_ratio_ * 100).round(3),
            "cumulative_variance_pct": (cum * 100).round(3),
            "kept": [i < n_pc for i in range(pca.n_components_)],
        })
        var_df.to_csv(Path(out_dir) / "dvh_pca_variance.csv", index=False)

    return scores
