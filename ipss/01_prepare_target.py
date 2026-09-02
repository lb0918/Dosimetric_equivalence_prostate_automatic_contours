"""
01_prepare_target.py
====================
Builds, from the patient-level table (one row per IPSS measurement):

  - X: static per-patient feature matrix
       = baseline columns plus the last IPSS measurement strictly BEFORE
       tx_date. That measurement must carry at least one item (a..g): rows with
       a stored total but no item make the obstructive and irritative subscores
       underivable, and the patient is then EXCLUDED (no earlier measurement is
       substituted, as it would not be comparable).
       The pre-treatment IPSS total (pretx_ipss_score_calc) IS REMOVED from X,
       since it defines the delta target; only the pre-treatment obstructive and
       irritative subscores are kept (individual items and quality of life are
       excluded through config.IGNORE_FEATURES).

  - y: DELTA IPSS = IPSS(endpoint) - IPSS(pre-treatment baseline), at the
       endpoints defined in days in config.ENDPOINTS.

       Estimation of IPSS(endpoint) by the trajectory model:
         - Case 1: a real measurement exists within +/- ENDPOINTS[d] days of the
                   target -> use it (ground truth, nearest measurement).
         - Case 2: no measurement in the window, but the endpoint falls inside
                   the patient's measurement span widened by
                   +/- EXTRAP_FACTOR * window -> prediction of a mixed model of
                   the post-treatment IPSS trajectory (population shape through
                   a time spline, plus a patient effect). This handles large
                   gaps without naive linear interpolation.
         - Case 3: endpoint outside that widened span, i.e. pure extrapolation
                   towards the population mean -> EXCLUSION (y = NaN).

       Patients without a pre-treatment IPSS have no baseline and are excluded
       at EVERY endpoint.

Evaluation is done by patient-level K-fold CV, built in 02_train.py; there is no
fixed train/val/test split here.

Outputs in PREP_DIR:
  - features.csv        X indexed by record_id, without pretx_ipss_score_calc
  - targets.csv         y_<d>d (delta IPSS) per endpoint, indexed by record_id
  - targets_source.csv  provenance of each target in {real, model, NaN}, same
                        shape as targets.csv. "real" is a real measurement in
                        the window (ground truth); "model" is imputed by the
                        trajectory model. 02_train.py EVALUATES only on "real"
                        targets, the "model" ones being used for training only.
  - summary.json        per-endpoint patient counts (real/model/excluded), etc.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATASET_MINIMAL, PREP_DIR, SEED,
    ENDPOINTS, ID_DATE_COLS, IGNORE_FEATURES, IPSS_ITEM_COLS,
    PRE_TX_IPSS_COLS, PRE_TX_PREFIX,
    DVH, USE_DVH_PCA, USE_DVH_CURATED, USE_DVH_FULL, DVH_STRUCTURE_PREFIXES,
    TRAJECTORY_SPLINE_DF, TRAJECTORY_RE_FORMULA, TRAJECTORY_EXTRAP_FACTOR,
    RESTRICT_TO_DVH_COHORT, RESTRICT_DVH_COMBINE, RESTRICT_DVH_SOURCES,
    ABLATE_FEATURES,
)
from utils import ensure_dir


def load_long_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ipss_date"] = pd.to_datetime(df["ipss_date"], errors="coerce")
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df["days_since_tx"] = (df["ipss_date"] - df["tx_date"]).dt.days
    df = df.sort_values(["record_id", "ipss_date"]).reset_index(drop=True)
    return df


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: baseline plus the pre-treatment IPSS (last
    measurement strictly before treatment)."""
    # Baseline: take the first row of each patient, since the baseline columns
    # are identical across all their rows. Baseline columns are every column of
    # the dataset EXCEPT:
    #   - identifiers, dates (ID_DATE_COLS) and derived columns;
    #   - pre-treatment IPSS measurements (PRE_TX_IPSS_COLS, extracted below);
    #   - explicitly ignored variables (IGNORE_FEATURES).
    derived_cols = {"days_since_tx"}
    excluded = set(ID_DATE_COLS) | set(PRE_TX_IPSS_COLS) | set(IGNORE_FEATURES) | derived_cols
    baseline_features = [c for c in df.columns if c not in excluded]

    unknown_ignored = set(IGNORE_FEATURES) - set(df.columns)
    if unknown_ignored:
        print(f"  ! IGNORE_FEATURES not found in the dataset (skipped): {sorted(unknown_ignored)}")
    print(f"  - {len(baseline_features)} baseline columns retained "
          f"({len([c for c in IGNORE_FEATURES if c in df.columns])} explicitly ignored)")

    base_cols = ["record_id"] + baseline_features
    baseline = df.groupby("record_id", as_index=False).first()[base_cols]

    # Baseline = LAST IPSS measurement strictly before treatment, which must
    # carry at least one item (a..g).
    #
    # A row may carry a stored ipss_score_calc (a calculated field) with no item
    # answers at all: the obstructive and irritative subscores are then
    # underivable and would be median-imputed downstream, fabricating a baseline
    # for a patient whose real total is nonetheless known.
    #
    # Such patients are EXCLUDED rather than falling back to an earlier
    # measurement: the baseline must stay the measurement closest to treatment,
    # as it is for every other patient. Going further back would introduce a
    # non-comparable baseline, potentially months before the implant, for a
    # handful of cases.
    #
    # Every column of the selected row is taken, so the total and the subscores
    # come from the SAME visit (otherwise obstructive + irritative != total).
    # Patients dropped here come out with pretx_ipss_score_calc = NaN and are
    # excluded by `has_baseline` in main().
    pre_tx = df[df["days_since_tx"] < 0].copy()
    pre_tx_last = (
        pre_tx.sort_values(["record_id", "ipss_date"])
        .groupby("record_id", as_index=False)
        .tail(1)
    )

    item_cols = [c for c in IPSS_ITEM_COLS if c in pre_tx_last.columns]
    if item_cols:
        has_items = pre_tx_last[item_cols].notna().any(axis=1)
        n_dropped = int((~has_items).sum())
        pre_tx_last = pre_tx_last[has_items]
        print(f"  - {n_dropped} patient(s) dropped: last pre-treatment measurement "
              f"carries no IPSS item (subscores underivable); "
              f"{len(pre_tx_last)} patients with a usable baseline")
    else:
        print("  ! IPSS items (IPSS_ITEM_COLS) absent from the dataset: the "
              "pre-treatment row is selected without a completeness check.")
    keep = ["record_id"] + [c for c in PRE_TX_IPSS_COLS if c in pre_tx_last.columns]
    pre_tx_last = pre_tx_last[keep].rename(
        columns={c: f"{PRE_TX_PREFIX}{c}" for c in keep if c != "record_id"}
    )

    X = baseline.merge(pre_tx_last, on="record_id", how="left")
    n_with_pretx = X.filter(like=PRE_TX_PREFIX).notna().any(axis=1).sum()
    print(f"  - Features: {X.shape[1] - 1} columns for {X.shape[0]} patients "
          f"({n_with_pretx} with a pre-treatment IPSS)")
    return X.set_index("record_id")


def fit_trajectory_model(df: pd.DataFrame):
    """Fit a mixed model of the post-treatment IPSS trajectory on all patients.

    IPSS(t) ~ B_spline(days_since_tx)  [fixed effects = population shape]
              + per-patient random effect(s) (intercept, optionally slope)

    The spline captures the typical post-treatment shape (acute peak then
    recovery or plateau); the patient random effect shifts that shape to each
    patient's own level (BLUP, shrunk towards the population for sparsely
    measured patients). ONLY post-treatment measurements (days_since_tx >= 0)
    are used, since the endpoints are post-treatment; the pre-treatment baseline
    is used separately to compute the delta.

    Evaluation note: the model is fitted on the IPSS trajectory of all patients,
    including those that will land in the test folds of 02_train.py. It is used
    to *construct the target* from a patient's own measurements, not to derive
    features, which makes it a standard outcome imputation. The features X are
    pre-treatment and stay disjoint from it.
    """
    import statsmodels.formula.api as smf

    post = (df[df["days_since_tx"] >= 0]
            .dropna(subset=["ipss_score_calc", "days_since_tx"])
            .copy())
    n_obs = len(post)
    n_pat = post["record_id"].nunique()
    md = smf.mixedlm(
        f"ipss_score_calc ~ bs(days_since_tx, df={TRAJECTORY_SPLINE_DF})",
        data=post,
        groups=post["record_id"],
        re_formula=TRAJECTORY_RE_FORMULA,
    )
    # lbfgs alone converges to the boundary optimum cov_re = 0 (llf = inf) while
    # reporting converged=True, which makes the patient random effects
    # inestimable. A powell warm start finds the real optimum (finite llf),
    # which bfgs and cg then reproduce identically.
    mdf = md.fit(method=["powell", "lbfgs"], maxiter=500)
    var_re = float(np.min(np.linalg.eigvalsh(np.atleast_2d(mdf.cov_re.values))))
    print(f"  - Trajectory model fitted on {n_obs} post-treatment measurements / "
          f"{n_pat} patients (spline df={TRAJECTORY_SPLINE_DF}, "
          f"re_formula={TRAJECTORY_RE_FORMULA!r}, converged={mdf.converged}, "
          f"var(patient effect)={var_re:.3f}, llf={mdf.llf:.1f})")
    if not mdf.converged:
        print("  ! The mixed model did NOT converge - interpret the imputed "
              "targets with caution (try re_formula=None or another df).")
    if not np.isfinite(mdf.llf) or var_re <= 1e-8:
        raise RuntimeError(
            "Singular random-effect covariance structure "
            f"(var={var_re:.3g}, llf={mdf.llf}): the \"model\" targets would be "
            "the population mean for every patient. Check the optimiser or "
            "TRAJECTORY_RE_FORMULA before continuing."
        )
    return mdf


def predict_trajectory(mdf, record_id, day: float) -> float:
    """Predicted IPSS for a patient at `day` days after treatment: the fixed
    (population) effect plus the patient's random effect(s). If the patient has
    no estimated random effect (absent from the fit), fall back to the
    population trajectory.
    """
    fixed = float(mdf.predict(pd.DataFrame({"days_since_tx": [float(day)]})).iloc[0])
    re = mdf.random_effects.get(record_id)
    if re is None:
        return fixed
    re_val = float(re.iloc[0])                       # random intercept
    if "days_since_tx" in re.index:                  # random slope, if present
        re_val += float(re["days_since_tx"]) * float(day)
    return fixed + re_val


def estimate_at_endpoint(group: pd.DataFrame, record_id, target_days: int,
                         window_days: int, mdf):
    """Estimate the IPSS at target_days after treatment for one patient.

    Returns the pair (value, source) where source is in {"real", "model", None}:

    - Case 1 ("real"): a real measurement exists within +/- window_days of the
              target -> use it (ground truth, nearest measurement). A measurement
              genuinely available near the endpoint is never smoothed away.
    - Case 2 ("model"): no measurement in the window, but the endpoint is
              SUPPORTED by the patient's measurement span, i.e. inside
              [min_day - f*window, max_day + f*window] with
              f = TRAJECTORY_EXTRAP_FACTOR -> prediction of the trajectory model
              (population shape realigned on the patient's level). This handles
              large gaps without drawing a naive straight line between two
              distant measurements.
    - Case 3 (None): endpoint outside that widened span -> EXCLUSION (NaN),
              since it would be pure extrapolation towards the population mean.

    The source is used as a leakage diagnostic: 02_train.py EVALUATES only on
    "real" targets (ground truth), while "model" targets serve for training
    only.
    """
    post = (group[group["days_since_tx"] >= 0]
            .dropna(subset=["ipss_score_calc"]))
    if post.empty:
        return np.nan, None
    days = post["days_since_tx"].to_numpy()
    scores = post["ipss_score_calc"].to_numpy()

    # Case 1: real measurement in the window -> ground truth (nearest one).
    within = np.abs(days - target_days) <= window_days
    if within.any():
        idx = np.where(within)[0]
        nearest = idx[int(np.argmin(np.abs(days[idx] - target_days)))]
        return float(scores[nearest]), "real"

    # Case 2: endpoint supported by the observed span (plus margin) -> model.
    tol = TRAJECTORY_EXTRAP_FACTOR * window_days
    if (days.min() - tol) <= target_days <= (days.max() + tol):
        return predict_trajectory(mdf, record_id, target_days), "model"

    # Case 3: pure extrapolation -> exclusion.
    return np.nan, None


def build_targets(df: pd.DataFrame, baseline_ipss: pd.Series, mdf):
    """Build (y, y_source): one row per patient, one y_<d>d column per endpoint.

    - y        : DELTA IPSS = IPSS(endpoint) - IPSS(pre-treatment baseline),
                 where IPSS(endpoint) is estimated by estimate_at_endpoint (near
                 real measurement, else the trajectory model `mdf`, else
                 exclusion).
    - y_source : provenance of each target in {"real", "model", NaN}, NaN
                 meaning excluded OR missing baseline. Used as a leakage
                 diagnostic, since 02_train.py evaluates only on "real" targets.

    Patients without a pre-treatment baseline are excluded at EVERY endpoint
    (y = NaN). Patients unsupported at one endpoint get NaN for that endpoint
    only.
    """
    rows, src_rows = [], []
    for record_id, grp in df.groupby("record_id", sort=False):
        row = {"record_id": record_id}
        src = {"record_id": record_id}
        base = baseline_ipss.get(record_id, np.nan)
        for target_days, window_days in ENDPOINTS.items():
            ipss_at_endpoint, source = estimate_at_endpoint(
                grp, record_id, target_days, window_days, mdf)
            delta = ipss_at_endpoint - base  # NaN if the endpoint OR the baseline is missing
            row[f"y_{target_days}d"] = delta
            # The source only counts when the final delta is defined.
            src[f"y_{target_days}d"] = source if np.isfinite(delta) else np.nan
        rows.append(row)
        src_rows.append(src)
    y = pd.DataFrame(rows).set_index("record_id")
    y_source = pd.DataFrame(src_rows).set_index("record_id")
    return y, y_source


def replace_dvh_with_pca(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw DVH indices from X and replace them with the DVH PCs.

    Raw columns are identified by structure prefix (DVH_STRUCTURE_PREFIXES). The
    PCs are computed on the active source's index file for the cohort of X (see
    dvh_pca.compute_dvh_pca). The import is local so sklearn is not loaded when
    USE_DVH_PCA = False.
    """
    from dvh_pca import compute_dvh_pca

    raw_dvh = [c for c in X.columns
               if any(c.startswith(p) for p in DVH_STRUCTURE_PREFIXES)]
    X = X.drop(columns=raw_dvh)
    print(f"  - {len(raw_dvh)} raw DVH columns dropped from X "
          f"({raw_dvh if raw_dvh else '-'})")

    pcs = compute_dvh_pca(X.index, out_dir=PREP_DIR)
    X = X.join(pcs)  # aligned on record_id; NaN for patients without a DVH
    n_with_pc = pcs.notna().any(axis=1).sum()
    print(f"  - {pcs.shape[1]} DVH PCA components added "
          f"({n_with_pc}/{len(X)} patients with a DVH; the rest imputed downstream)")
    return X


def replace_dvh_with_curated(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw DVH indices from X and replace them with the curated panel.

    Raw columns are identified by structure prefix (DVH_STRUCTURE_PREFIXES). The
    panel (config.DVH_CURATED_PANEL) is extracted from the active source for the
    cohort of X (see dvh_curated.compute_dvh_curated). The import is local so
    the module is not loaded when the curated strategy is inactive.
    """
    from dvh_curated import compute_dvh_curated

    raw_dvh = [c for c in X.columns
               if any(c.startswith(p) for p in DVH_STRUCTURE_PREFIXES)]
    X = X.drop(columns=raw_dvh)
    print(f"  - {len(raw_dvh)} raw DVH columns dropped from X "
          f"({raw_dvh if raw_dvh else '-'})")

    panel = compute_dvh_curated(X.index, out_dir=PREP_DIR)
    X = X.join(panel)  # aligned on record_id; NaN for patients without a DVH
    return X


def replace_dvh_with_full(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw DVH indices from X and replace them with ALL DVH indices.

    Raw columns are identified by structure prefix (DVH_STRUCTURE_PREFIXES).
    Every dose-volume index of the active source file is then pivoted and joined
    as is, without reduction or selection (see dvh_full.compute_dvh_full). The
    import is local so the module is not loaded when the full strategy is
    inactive.
    """
    from dvh_full import compute_dvh_full

    raw_dvh = [c for c in X.columns
               if any(c.startswith(p) for p in DVH_STRUCTURE_PREFIXES)]
    X = X.drop(columns=raw_dvh)
    print(f"  - {len(raw_dvh)} raw DVH columns dropped from X "
          f"({raw_dvh if raw_dvh else '-'})")

    full = compute_dvh_full(X.index, out_dir=PREP_DIR)
    X = X.join(full)  # aligned on record_id; NaN for patients without a DVH
    return X


def main():
    print("=" * 70)
    print("PREPARATION - static features + endpoint targets (trajectory model)")
    print("=" * 70)
    ensure_dir(PREP_DIR)

    df = load_long_dataset(DATASET_MINIMAL)
    print(f"[ok] Loaded: {len(df)} measurements, {df['record_id'].nunique()} patients")

    print("\nBuilding the static features...")
    X = build_static_features(df)

    # Recover the IPSS baseline: it is the reference of the delta and is then
    # removed from X.
    baseline_col = f"{PRE_TX_PREFIX}ipss_score_calc"
    if baseline_col not in X.columns:
        raise KeyError(f"IPSS baseline column '{baseline_col}' absent from X. "
                       "Cannot compute the IPSS delta.")
    baseline_ipss = X[baseline_col].copy()

    print("\nFitting the IPSS trajectory model (mixed model, population + patient)...")
    mdf = fit_trajectory_model(df)

    print("\nBuilding the endpoint targets "
          "(delta IPSS = IPSS(endpoint) - baseline; near real measurement, "
          "else the trajectory model)...")
    y, y_source = build_targets(df, baseline_ipss, mdf)

    # Remove the IPSS baseline from the features to avoid redundancy with the
    # target. The pretx_ipss_obstructive / pretx_ipss_irritative subscores stay.
    X = X.drop(columns=[baseline_col])
    print(f"  - Column '{baseline_col}' dropped from X (it defines y).")

    # Exclude patients without a pre-treatment baseline (no delta computable).
    n_before = len(X)
    has_baseline = baseline_ipss.notna()
    excluded_no_baseline = baseline_ipss.index[~has_baseline].tolist()
    X = X.loc[has_baseline]
    y = y.loc[has_baseline]
    y_source = y_source.loc[has_baseline]
    if excluded_no_baseline:
        print(f"  - {len(excluded_no_baseline)} patients excluded (no pre-treatment IPSS)")
    print(f"  - Remaining patients: {len(X)}/{n_before}")

    # Keep only the patients present in X (safety).
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    y_source = y_source.loc[common]

    # Restriction to the cohort that has a DVH (config.RESTRICT_TO_DVH_COHORT).
    # Applied IDENTICALLY for EVERY scenario, including noDVH: eligibility is
    # defined by the combined DVH availability (RESTRICT_DVH_COMBINE over
    # RESTRICT_DVH_SOURCES), never by the source of the current scenario, so all
    # runs share exactly the same patients. record_id is a string in the DVH
    # files, so matching happens in string space (as in dvh_curated / dvh_pca).
    if RESTRICT_TO_DVH_COHORT:
        from dvh_mc import dvh_cohort_record_ids
        eligible = dvh_cohort_record_ids(RESTRICT_DVH_SOURCES, RESTRICT_DVH_COMBINE)
        keep = X.index.astype(str).str.strip().isin(eligible)
        n_before_dvh = len(X)
        X = X.loc[keep]
        y = y.loc[keep]
        y_source = y_source.loc[keep]
        common = X.index
        quant = ("all of the" if RESTRICT_DVH_COMBINE == "intersection"
                 else "at least one of the")
        print(f"\nDVH cohort restriction ({RESTRICT_DVH_COMBINE} of "
              f"{RESTRICT_DVH_SOURCES}):\n"
              f"  - {len(X)}/{n_before_dvh} eligible patients "
              f"(record_id present in {quant} DVH sources).")
        if len(X) == 0:
            raise SystemExit(
                "Empty DVH cohort after restriction - check that the record_ids "
                "of the DVH files match those of the cohort (type and format)."
            )

    # DVH handling in X (master switch DVH in config.py).
    if not DVH:
        # No DVH information in the models: drop the raw indices and do not
        # compute the PCA.
        raw_dvh = [c for c in X.columns
                   if any(c.startswith(p) for p in DVH_STRUCTURE_PREFIXES)]
        X = X.drop(columns=raw_dvh)
        print(f"\nDVH = False -> {len(raw_dvh)} raw DVH columns dropped from X "
              f"(no DVH variable in the models)")
    elif USE_DVH_CURATED:
        # Replace the raw DVH indices with the curated panel (named indices).
        # Takes precedence over the PCA when both switches are active.
        print("\nCurated DVH panel (selected clinical dose-volume indices)...")
        X = replace_dvh_with_curated(X)
    elif USE_DVH_PCA:
        # Replace the raw DVH indices with the principal components.
        print("\nPCA reduction of the DVH indices...")
        X = replace_dvh_with_pca(X)
    elif USE_DVH_FULL:
        # Replace the raw DVH indices with the full index set (no reduction).
        print("\nAll DVH indices (no reduction, no selection)...")
        X = replace_dvh_with_full(X)

    # Inferential ablation: drop the requested feature(s) from X just before
    # saving, AFTER all DVH logic (config.ABLATE_FEATURES). The ablated run is
    # compared to the full run in 04_contrasts.py, by the same machinery as the
    # DVH block. A warning is raised when a requested column is absent, so an
    # ablation that removed nothing is not mistaken for a real one.
    if ABLATE_FEATURES:
        present = [c for c in ABLATE_FEATURES if c in X.columns]
        missing = [c for c in ABLATE_FEATURES if c not in X.columns]
        if missing:
            print(f"\n  ! ABLATE_FEATURES absent from X (no effect for them): {missing}")
        if present:
            X = X.drop(columns=present)
            print(f"\nInferential ablation -> {len(present)} feature(s) dropped "
                  f"from X: {present}")

    # Summary: number of patients with a non-NaN target per endpoint.
    summary = {
        "n_patients_total": len(common),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
        "target_type": "delta_ipss_vs_pretx_baseline",
        "endpoint_estimation": "real_in_window_else_mixedlm_trajectory",
        "trajectory_spline_df": TRAJECTORY_SPLINE_DF,
        "trajectory_re_formula": str(TRAJECTORY_RE_FORMULA),
        "trajectory_extrap_factor": TRAJECTORY_EXTRAP_FACTOR,
        "endpoints": {str(d): w for d, w in ENDPOINTS.items()},
        "seed": SEED,
        "restrict_to_dvh_cohort": RESTRICT_TO_DVH_COHORT,
        "restrict_dvh_combine": RESTRICT_DVH_COMBINE if RESTRICT_TO_DVH_COHORT else None,
        "restrict_dvh_sources": RESTRICT_DVH_SOURCES if RESTRICT_TO_DVH_COHORT else None,
        "per_endpoint": {},
    }
    print("\nTarget availability per endpoint (real = ground truth, "
          "model = imputed by the trajectory):")
    print(f"  {'endpoint':<10} {'n_valid':>8} {'real':>8} {'model':>8} {'excluded':>9}")
    for col in y.columns:
        n_valid = int(y[col].notna().sum())
        n_real = int((y_source[col] == "real").sum())
        n_model = int((y_source[col] == "model").sum())
        n_excluded = len(common) - n_valid
        summary["per_endpoint"][col] = {
            "n_valid": n_valid, "n_real": n_real, "n_model": n_model,
            "excluded": n_excluded,
        }
        print(f"  {col:<10} {n_valid:>8} {n_real:>8} {n_model:>8} {n_excluded:>9}")

    # Saved as CSV for portability (no pyarrow dependency).
    X.to_csv(PREP_DIR / "features.csv")
    y.to_csv(PREP_DIR / "targets.csv")
    y_source.to_csv(PREP_DIR / "targets_source.csv")
    with open(PREP_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] Written to {PREP_DIR}/:")
    print(f"   features.csv        ({X.shape})")
    print(f"   targets.csv         ({y.shape})")
    print(f"   targets_source.csv  ({y_source.shape})  <- real/model provenance per target")
    print(f"   summary.json")


if __name__ == "__main__":
    main()
