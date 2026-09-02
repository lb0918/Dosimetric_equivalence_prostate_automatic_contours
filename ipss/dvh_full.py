"""
dvh_full.py
===========
"fulldvh" DVH strategy: hands the models EVERY available dose-volume index,
with no reduction and no selection.

Unlike the two other informed strategies:
  - curated (dvh_curated.py) -> a restricted panel of a-priori chosen indices;
  - pca     (dvh_pca.py)     -> all indices COMPRESSED into principal
                                components (non-interpretable features).
here every index is kept as is, as a named feature <Structure>_<metric>
(Bladder_Dmean_Gy, Prostate_V150_pct, Urethra_uD10_Gy, ...). This is the full
pivot of the index file of the active segmentation source
(config.DVH_SEG_SOURCE; for the Monte-Carlo source, the statistic named in
DVH_MC_STAT, plus the Monte-Carlo variance if config.DVH_MC_USE_VARIANCE).

Not to be confused with the "rawdvh" strategy (no DVH switch active), which uses
only the few raw DVH columns embedded in the patient-level table by
00_build_ipss_dataset.py. Here the full index file is read directly and only
entirely empty columns are dropped (urethra-specific indices for the other
structures).

compute_dvh_full() returns a DataFrame indexed by record_id over the requested
cohort, with <Structure>_<metric> columns. Patients without a DVH get NaN,
imputed downstream by each model's preprocessor, exactly as in the pca and
curated strategies. No target y is used: this step is unsupervised.
"""
from pathlib import Path

import pandas as pd

from config import DVH_INDICES_CSV, DVH_SEG_SOURCE, DVH_MC_STAT
from dvh_mc import load_dvh_indices
from dvh_pca import _pivot_per_patient  # full (record_id, structure) -> wide pivot


def compute_dvh_full(record_ids, out_dir: Path | None = None) -> pd.DataFrame:
    """Extract EVERY DVH index for the requested patients, without reduction.

    Indices come from the segmentation source in config.DVH_SEG_SOURCE, reduced
    to the central statistic in config.DVH_MC_STAT for the Monte-Carlo source.

    Args:
        record_ids: cohort identifiers.
        out_dir: if given, writes dvh_full_indices.csv there for inspection.

    Returns:
        DataFrame indexed by record_id (the full requested cohort, with its
        original labels), one <Structure>_<metric> column per index that is not
        entirely empty. NaN for patients without a DVH.
    """
    raw = load_dvh_indices(DVH_INDICES_CSV, DVH_SEG_SOURCE, DVH_MC_STAT)
    wide = _pivot_per_patient(raw)  # index = record_id as string, columns = all indices

    # Type robustness (as in dvh_pca / dvh_curated): record_id is a string in the
    # DVH file while X.index may be numeric. Match in string space, then restore
    # the cohort's original labels.
    orig_ids = pd.Index(pd.unique(record_ids), name="record_id")
    str_to_orig = pd.Series(orig_ids.values, index=orig_ids.astype(str).str.strip())
    str_to_orig = str_to_orig[~str_to_orig.index.duplicated()]

    scores = wide.rename(index=str_to_orig).reindex(orig_ids)
    scores.index.name = "record_id"

    n_with = int(scores.notna().any(axis=1).sum())
    print(f"  - DVH full: {scores.shape[1]} indices "
          f"({n_with}/{len(orig_ids)} patients with a DVH; the rest imputed downstream)")

    if out_dir is not None:
        scores.to_csv(Path(out_dir) / "dvh_full_indices.csv")

    return scores
