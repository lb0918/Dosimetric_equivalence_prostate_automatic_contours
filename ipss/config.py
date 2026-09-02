"""Central configuration for the IPSS prediction pipeline."""
import os
from pathlib import Path

# ------------------------------------------------------------
# Per-process thread budget
# ------------------------------------------------------------
# When several scenarios run in parallel (one process each), every process
# claiming all cores oversubscribes the machine. PIPE_N_THREADS gives each
# process a fixed share, propagated to XGBoost (n_jobs), RandomForest (n_jobs),
# CatBoost (thread_count) and torch (set_num_threads in 02_train.py).
N_THREADS = int(os.environ.get("PIPE_N_THREADS", "6"))

# ============================================================
# PATHS
# ============================================================
# DATA_ROOT holds the input tables (patient-level data, not distributed here).
# PROJECT_DIR is where the pipeline writes its outputs. Both are overridable by
# environment variable so the code runs unchanged on any machine.
DATA_ROOT = Path(os.environ.get(
    "PROTECTA_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
PROJECT_DIR = Path(os.environ.get(
    "PROTECTA_IPSS_DIR", Path(__file__).resolve().parent))

# Patient-level table produced by 00_build_ipss_dataset.py.
DATASET_MINIMAL = str(DATA_ROOT / "dataset_minimal_v2.csv")

# Output directories are defined at the end of this file, once the DVH switches
# are known: they are suffixed by the active DVH strategy (see dvh_run_tag) so
# that different scenarios do not overwrite each other.

# Reproducibility
SEED = 42

# Endpoints expressed directly in days after treatment.
# Format: {target_day: half_window_days}
#   - a real measurement within +/- half_window is used as ground truth;
#   - otherwise, if the endpoint stays inside the patient's measurement span
#     (plus an extrapolation margin), the trajectory model predicts it;
#   - otherwise the patient is excluded for that endpoint (y = NaN).
ENDPOINTS = {
    30: 20,
    400: 100,
    730:  70,
    1125:  50,
    1460:  100,
    1865: 175,
}

# ------------------------------------------------------------
# Optional endpoint override by environment variable
# ------------------------------------------------------------
# Runs a targeted analysis on one or more endpoints without editing this file.
# Format: "target:half_window[,target:half_window...]" in days.
#   PIPE_ENDPOINTS="90:30"          -> {90: 30}
#   PIPE_ENDPOINTS="90:30,400:100"  -> {90: 30, 400: 100}
# Combine with PIPE_TAG_SUFFIX (below) so a targeted run writes to its own
# output directories.
_ep_env = os.environ.get("PIPE_ENDPOINTS")
if _ep_env:
    ENDPOINTS = {int(k): int(v)
                 for k, v in (pair.split(":") for pair in _ep_env.split(","))}

# ------------------------------------------------------------
# IPSS(endpoint) estimation - longitudinal trajectory model
# ------------------------------------------------------------
# The target IPSS(endpoint) is estimated with a mixed model (population effect
# plus patient effect) of the post-treatment IPSS trajectory rather than by
# linear interpolation. See estimate_at_endpoint() in 01_prepare_target.py.
#   - TRAJECTORY_SPLINE_DF: degrees of freedom of the B-spline (bs) on
#     days_since_tx for the fixed effects, i.e. the shape of the population
#     trajectory. Internal knots are placed at quantiles. A cubic regression
#     spline (cr) is rank-deficient against the intercept, hence bs.
#   - TRAJECTORY_RE_FORMULA: patient random-effect structure.
#       None             -> random intercept only (robust).
#       "~days_since_tx" -> random intercept and slope (may fail to converge).
#   - TRAJECTORY_EXTRAP_FACTOR: maximum extrapolation allowed outside the
#     patient's observed measurement span, in units of the endpoint half-window.
#     The model is used only if the endpoint falls in
#     [min_day - f*window, max_day + f*window]; otherwise the patient is
#     excluded, so that no target is fabricated by extrapolating towards the
#     population mean.
TRAJECTORY_SPLINE_DF = 4
TRAJECTORY_RE_FORMULA = None
TRAJECTORY_EXTRAP_FACTOR = 0.7

# Outer cross-validation: 5 folds, split by patient.
N_FOLDS = 5

# Number of REPETITIONS of the outer CV (independent patient partitions).
# N_REPEATS = 1: a single partition.
# N_REPEATS = r > 1: r x K cross-validation. The Nadeau-Bengio test then has
# r*K per-fold differences instead of K, hence r*K-1 degrees of freedom instead
# of K-1, without changing the geometry of each partition (K and the
# n_test/n_train ratio stay fixed, so the targeted standardised effect is
# unchanged and only precision increases). This is the corrected resampled
# t-test of Bouckaert & Frank, the regime the NB correction was designed for.
# Cost: training time scales with r, since every repetition re-tunes its own
# hyperparameters (otherwise the nested CV is no longer honest).
# Overridable by PIPE_N_REPEATS; combine with PIPE_TAG_SUFFIX to keep repeated
# and single-partition runs side by side.
N_REPEATS = int(os.environ.get("PIPE_N_REPEATS", "1"))

# ------------------------------------------------------------
# Inclusion of DVH variables in the models
# ------------------------------------------------------------
# Master switch. If DVH = False, no DVH information enters the models: raw DVH
# index columns (prefixed by DVH_STRUCTURE_PREFIXES) are dropped from X and the
# DVH PCA is not computed, whatever USE_DVH_PCA says. If DVH = True, behaviour
# depends on the switches below.
DVH = True

# ------------------------------------------------------------
# DVH index reduction by PCA
# ------------------------------------------------------------
# If USE_DVH_PCA = True, the raw DVH index columns present in the dataset
# (prefixed by a structure name, e.g. Bladder_Dmean_Gy) are removed from the
# features and replaced by principal components computed on the full index
# table (all structures x all indices, at the Monte-Carlo median value __p50;
# see dvh_mc.py), keeping the smallest number of PCs explaining at least
# DVH_PCA_VARIANCE of the variance. See dvh_pca.py.
USE_DVH_PCA = False

# ------------------------------------------------------------
# Segmentation source of the DVH indices
# ------------------------------------------------------------
# Several dose-volume index sets are available, derived from different
# segmentations of the organs (prostate, urethra, bladder, bladder neck,
# rectum):
#   - "manual"   -> manual segmentation (clinical reference).
#   - "auto_det" -> automatic segmentation from a deterministic model. The
#                   urethra is never segmented automatically, so its indices
#                   (uD*) are taken from the manual segmentation.
#   - "mc_bayes" -> automatic segmentation from a Bayesian model with 20
#                   Monte-Carlo passes. Monte-Carlo uncertainty is summarised
#                   by the statistic named in DVH_MC_STAT.
# The active source selects both the input file (DVH_INDICES_CSV) and the
# output directory suffix (see dvh_run_tag), so segmentations can be compared
# without their runs colliding. Reading is normalised by
# dvh_mc.load_dvh_indices.
DVH_SEG_SOURCE = "mc_bayes_clin0977"  # see DVH_SOURCES keys below

_DVH_DIR = DATA_ROOT / "dvh"
DVH_SOURCES = {
    "manual":   f"{_DVH_DIR}/dvh_indices_manual_seg.csv",
    "auto_det": f"{_DVH_DIR}/dvh_indices_auto_seg_det.csv",
    "mc_bayes": f"{_DVH_DIR}/dvh_mc_summary.csv",
    # Variants recomputed on an alternative contour set. Same FORMAT as their
    # base source (dvh_mc.base_source resolves the suffix for dispatch), but a
    # distinct output tag so they do not overwrite the base runs.
    "auto_det_clin0977": f"{_DVH_DIR}/dvh_indices_auto_seg_det_clin0977.csv",
    "mc_bayes_clin0977": f"{_DVH_DIR}/dvh_mc_summary_clin0977.csv",
}
# Statistic used to summarise Monte-Carlo uncertainty for the "mc_bayes" source.
DVH_MC_STAT = "mean"

# ------------------------------------------------------------
# Restricting the cohort to patients with a DVH
# ------------------------------------------------------------
# If RESTRICT_TO_DVH_COHORT = True, the training cohort is restricted - FOR ALL
# SCENARIOS, including noDVH - to the patients available in the DVH files listed
# in RESTRICT_DVH_SOURCES, combined according to RESTRICT_DVH_COMBINE:
#   - "intersection" -> record_id present in EVERY listed source. Each DVH
#                       scenario then has a real DVH for every patient (no
#                       imputation), which is the cleanest noDVH vs DVH
#                       comparison.
#   - "union"        -> present in at least one source (wider cohort, but
#                       imputation is possible within a given scenario).
# "Available" means the record_id appears in the raw file of the source, with at
# least one segmented structure (see dvh_mc.dvh_cohort_record_ids). The filter is
# applied in 01_prepare_target.py AFTER excluding patients without a pre-treatment
# IPSS, identically for every scenario: eligibility is defined by the combined
# DVH availability, never by the source of the current scenario. All scenarios
# therefore run on exactly the same patients.
# Set to False to keep the full cohort.
RESTRICT_TO_DVH_COHORT = True
RESTRICT_DVH_COMBINE = "intersection"          # "intersection" | "union"
RESTRICT_DVH_SOURCES = ["manual", "auto_det", "mc_bayes"]

# ------------------------------------------------------------
# Monte-Carlo variance of the DVH indices as a predictor
# ------------------------------------------------------------
# For the Bayesian source, each dose-volume index is summarised by the central
# value selected in DVH_MC_STAT. If DVH_MC_USE_VARIANCE = True, the VARIANCE of
# the Monte-Carlo passes (= __std^2) is ADDED for each index as a separate
# feature suffixed DVH_MC_VAR_SUFFIX (e.g. Urethra_uD10_Gy_var). That variance
# encodes the segmentation uncertainty of the Bayesian model: an index that is
# unstable across passes is less reliable. Propagated to all three DVH
# strategies (curated, pca, rawdvh). Has an effect ONLY for the "mc_bayes"
# source, since the manual and deterministic segmentations have no draws.
# Reflected in DVH_RUN_TAG by a "_var" suffix.
DVH_MC_USE_VARIANCE = False
DVH_MC_VAR_SUFFIX = "_var"

DVH_INDICES_CSV = DVH_SOURCES[DVH_SEG_SOURCE]
DVH_PCA_VARIANCE = 0.7          # cumulative variance fraction to retain
DVH_PC_PREFIX = "dvh_pc"        # prefix of the PC features (dvh_pc1, dvh_pc2, ...)

# Prefixes identifying the raw DVH index columns to drop from X when a reduction
# strategy is active (any column starting with "<structure>_"). Note that
# "Bladder_" does not capture "BladderNeck_", which has no underscore there.
DVH_STRUCTURE_PREFIXES = ["Bladder_", "BladderNeck_", "Prostate_", "Rectum_", "Urethra_"]

# NOTE: filtering of structure metadata (n_voxels, volume_cc, mask_volume_cc,
# coverage_in_dose, ...) is centralised in dvh_mc.META_BASE_METRICS and applied
# by dvh_mc.load_mc_median.

# ------------------------------------------------------------
# Curated DVH panel (alternative to PCA)
# ------------------------------------------------------------
# If USE_DVH_CURATED = True (and DVH = True), the raw DVH indices are removed
# from X and replaced by a RESTRICTED PANEL of dose-volume indices clinically
# motivated for urinary toxicity, at their Monte-Carlo median (__p50) value (see
# dvh_curated.py). Unlike PCA, the features stay interpretable: one index is one
# named feature <Structure>_<metric>. USE_DVH_CURATED takes PRECEDENCE over
# USE_DVH_PCA when both are True.
USE_DVH_CURATED = True
DVH_CURATED_PANEL = {
    "Prostate":    ["D90_Gy", "V100_pct", "V150_pct", "V200_pct"],
    "Urethra":     ["uD10_Gy", "uD30_Gy", "uD5_Gy", "uD0.1cc_Gy"],
    "BladderNeck": ["D2cc_Gy", "D1cc_Gy", "V100_pct"],
}

# ------------------------------------------------------------
# ALL DVH indices, no reduction (alternative to PCA / curated)
# ------------------------------------------------------------
# If USE_DVH_FULL = True (and DVH = True), the raw DVH indices are removed from X
# and replaced by EVERY dose-volume index of the active source file, pivoted into
# one feature per index, named <Structure>_<metric> (see dvh_full.py). Nothing is
# reduced (unlike PCA) and nothing is selected a priori (unlike the curated
# panel): every column that is not entirely empty enters X. Not to be confused
# with the "rawdvh" strategy (no switch active), which uses only the few raw DVH
# columns embedded in the patient-level table by 00_build_ipss_dataset.py.
# PRECEDENCE when several switches are True:
# USE_DVH_CURATED > USE_DVH_PCA > USE_DVH_FULL.
USE_DVH_FULL = False

# ------------------------------------------------------------
# Single-feature ablation (inferential control, same design as the DVH block)
# ------------------------------------------------------------
# ABLATE_FEATURES: feature columns to REMOVE from X just before saving, in
# 01_prepare_target.py, AFTER all DVH logic. It measures the contribution of a
# given variable with exactly the same machinery as the DVH block: an otherwise
# identical run is trained without those features, and 04_contrasts.py compares
# it to the full run (delta RMSE, patient-level bootstrap CI, Nadeau-Bengio).
# ABLATE_LABEL is a short suffix appended to the output tag (see dvh_run_tag) so
# the ablated run does not overwrite the full run. An empty list means no
# ablation.
#   e.g. ABLATE_FEATURES=["pretx_ipss_obstructive"], ABLATE_LABEL="noObstr"
ABLATE_FEATURES: list[str] = []
ABLATE_LABEL = ""


# ------------------------------------------------------------
# Active DVH strategy -> output directory suffix
# ------------------------------------------------------------
# Identifies the active combination of DVH switches and suffixes the output
# directories so that runs do not overwrite each other. The strategy is further
# suffixed by the active segmentation source, e.g. "pca_manual",
# "curated_mc_bayes".
#   - DVH = False ................ "noDVH"           (no DVH information in X)
#   - DVH = True, USE_DVH_CURATED  "curated_<src>"   (restricted index panel)
#   - DVH = True, USE_DVH_PCA .... "pca_<src>"       (DVH indices -> PCA components)
#   - DVH = True, USE_DVH_FULL ... "fulldvh_<src>"   (all indices, no reduction)
#   - DVH = True, otherwise ...... "rawdvh_<src>"    (raw DVH indices in X)
# For a Bayesian source with DVH_MC_USE_VARIANCE, the tag also gets a "_var"
# suffix, since the Monte-Carlo variance then enters X alongside the mean.
def dvh_run_tag() -> str:
    if not DVH:
        tag = "noDVH"
    else:
        if USE_DVH_CURATED:
            strat = "curated"
        elif USE_DVH_PCA:
            strat = "pca"
        elif USE_DVH_FULL:
            strat = "fulldvh"
        else:
            strat = "rawdvh"
        tag = f"{strat}_{DVH_SEG_SOURCE}"
        # Bayesian source with the Monte-Carlo variance added: dedicated suffix
        # so mean-only runs are not overwritten. The BASE source is tested so
        # that dataset variants also receive the "_var" suffix.
        from dvh_mc import base_source
        if base_source(DVH_SEG_SOURCE) == "mc_bayes" and DVH_MC_USE_VARIANCE:
            tag += "_var"
    # Ablation run: dedicated suffix so the full run is not overwritten. Also
    # applies to noDVH. Empty label means no suffix.
    if ABLATE_FEATURES and ABLATE_LABEL:
        tag += f"_{ABLATE_LABEL}"
    return tag


# ------------------------------------------------------------
# Scenario override by environment variable
# ------------------------------------------------------------
# Allows several scenarios to run in parallel, one process each, without editing
# this file between launches (editing and importing would otherwise be a data
# race). If PIPE_SCENARIO is unset, the values hardcoded above apply. The keys
# map to output directory suffixes (see dvh_run_tag).
_CURATED = dict(DVH=True, USE_DVH_CURATED=False, USE_DVH_PCA=False, USE_DVH_FULL=True)
# Curated panel in the strict sense (USE_DVH_CURATED=True): clinically motivated
# indices only, tag "curated_<src>". Distinct from _CURATED above which, despite
# its name, enables the FULL index bank (USE_DVH_FULL) -> tag "fulldvh_<src>".
_CURATED_PANEL = dict(DVH=True, USE_DVH_CURATED=True, USE_DVH_PCA=False, USE_DVH_FULL=False)
_SCENARIOS = {
    "noDVH":    dict(DVH=False),
    "manual":   dict(**_CURATED, DVH_SEG_SOURCE="manual"),
    # Curated panel on the MANUAL segmentation (clinical reference) -> tag
    # "curated_manual", distinct from the fulldvh "manual" run above.
    "manual_panel": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="manual"),
    "auto_det": dict(**_CURATED, DVH_SEG_SOURCE="auto_det"),
    "mc_bayes": dict(**_CURATED, DVH_SEG_SOURCE="mc_bayes"),
    # Alternative contour set, curated panel only.
    "auto_det_clin0977": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="auto_det_clin0977"),
    "mc_bayes_clin0977": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977"),
    # Bayesian variant WITH the Monte-Carlo variance (central value plus __std^2
    # of each index). DVH_MC_USE_VARIANCE is set explicitly so the scenario does
    # not depend on the module default.
    "mc_bayes_clin0977_var": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977",
                                  DVH_MC_USE_VARIANCE=True),
    # Ablation of the pre-treatment OBSTRUCTIVE IPSS subscore. Identical to the
    # full run (same panel, cohort, folds and seed) but without that single
    # feature, to be contrasted with the full run in 04_contrasts.py exactly as
    # the DVH block is tested.
    "mc_bayes_clin0977_noObstr": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977",
                                      ABLATE_FEATURES=["pretx_ipss_obstructive"],
                                      ABLATE_LABEL="noObstr"),
    # Joint ablation of both pre-treatment IPSS subscores. Since obstructive plus
    # irritative equals the IPSS total, removing both deprives the model of the
    # whole pre-treatment IPSS level (the baseline total is otherwise dropped
    # because it defines the delta target). The contrast therefore measures the
    # predictive power of the pre-treatment IPSS total.
    "mc_bayes_clin0977_noIpss": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977",
                                     ABLATE_FEATURES=["pretx_ipss_obstructive",
                                                      "pretx_ipss_irritative"],
                                     ABLATE_LABEL="noIpss"),
    # Ablation of AGE, same design as the ablations above.
    "mc_bayes_clin0977_noAge": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977",
                                    ABLATE_FEATURES=["age"],
                                    ABLATE_LABEL="noAge"),
    # Ablation of the PROSTATE VOLUME / IMPLANT block: whether implant geometry
    # adds anything beyond the initial IPSS level. Prostate volume is the classic
    # predictor of urinary retention and toxicity after LDR brachytherapy, and
    # the needle count is a proxy for implant complexity.
    #
    # psa_density is deliberately part of the block: 00_build_ipss_dataset.py
    # defines it as crude_psa / ldr_post_vol. Leaving it in X while removing the
    # ldr_* columns would let the model RECONSTRUCT the volume by division
    # (volume = crude_psa / psa_density), so the ablation would remove nothing.
    # crude_psa itself STAYS in X: the PSA information is preserved and only the
    # volume pathway is cut.
    "mc_bayes_clin0977_noVol": dict(**_CURATED_PANEL, DVH_SEG_SOURCE="mc_bayes_clin0977",
                                    ABLATE_FEATURES=["ldr_post_vol",
                                                     "ldr_previ_volcont",
                                                     "ldr_live_aiguilles",
                                                     "ldr_previ_sourceprev",
                                                     "ldr_live_dil",
                                                     "psa_density"],
                                    ABLATE_LABEL="noVol"),
}
_scn = os.environ.get("PIPE_SCENARIO")
if _scn:
    if _scn not in _SCENARIOS:
        raise SystemExit(f"unknown PIPE_SCENARIO: {_scn!r} "
                         f"(expected one of: {list(_SCENARIOS)})")
    _cfg = _SCENARIOS[_scn]
    DVH = _cfg.get("DVH", DVH)
    USE_DVH_CURATED = _cfg.get("USE_DVH_CURATED", USE_DVH_CURATED)
    USE_DVH_PCA = _cfg.get("USE_DVH_PCA", USE_DVH_PCA)
    USE_DVH_FULL = _cfg.get("USE_DVH_FULL", USE_DVH_FULL)
    DVH_SEG_SOURCE = _cfg.get("DVH_SEG_SOURCE", DVH_SEG_SOURCE)
    DVH_MC_USE_VARIANCE = _cfg.get("DVH_MC_USE_VARIANCE", DVH_MC_USE_VARIANCE)
    ABLATE_FEATURES = _cfg.get("ABLATE_FEATURES", ABLATE_FEATURES)
    ABLATE_LABEL = _cfg.get("ABLATE_LABEL", ABLATE_LABEL)
    DVH_INDICES_CSV = DVH_SOURCES[DVH_SEG_SOURCE]   # derived -> recompute after override

# ------------------------------------------------------------
# Optional run suffix
# ------------------------------------------------------------
# Appends a free-form suffix to EVERY tag and output directory of a run - prep_,
# models_, results_ and, for cross-run analyses, contrasts_, headroom_, figures_
# - so a targeted run (e.g. PIPE_ENDPOINTS="200:50" with PIPE_TAG_SUFFIX="200d")
# writes to <dir>_200d without overwriting existing runs. Empty means no suffix.
# `with_suffix` is the single entry point shared by scripts 02..08; the suffix is
# never rebuilt by hand.
#
# DEFAULT_TAG_SUFFIX is the suffix used when no environment variable is set, so
# that scripts launched plainly read and write the repeated-CV material.
# PIPE_TAG_SUFFIX always takes precedence, including PIPE_TAG_SUFFIX="" to force
# the unsuffixed run.
# Note: the suffix also applies to DVH_RUN_TAG, so TRAINING launched with this
# default writes into the suffixed directories; set N_REPEATS (PIPE_N_REPEATS)
# accordingly so the content matches the tag.
DEFAULT_TAG_SUFFIX = "r5"

TAG_SUFFIX = os.environ.get("PIPE_TAG_SUFFIX", DEFAULT_TAG_SUFFIX)


def with_suffix(name: str) -> str:
    """Append the active run suffix (PIPE_TAG_SUFFIX) to a tag or directory name.

    An empty suffix returns `name` unchanged.
    """
    return f"{name}_{TAG_SUFFIX}" if TAG_SUFFIX else name


DVH_RUN_TAG = with_suffix(dvh_run_tag())

# Root directories grouping every run by output family. Each run lives in a
# subdirectory of these roots (prep/prep_<tag>, models/models_<tag>,
# results/results_<tag>). Cross-run analyses (04..06) resolve a tag's
# directories through these roots.
PREP_ROOT = PROJECT_DIR / "prep"
MODELS_ROOT = PROJECT_DIR / "models"
RESULTS_ROOT = PROJECT_DIR / "results"

# Output directories suffixed by the active DVH strategy.
PREP_DIR = PREP_ROOT / f"prep_{DVH_RUN_TAG}"          # X, y, splits
MODELS_DIR = MODELS_ROOT / f"models_{DVH_RUN_TAG}"    # trained models + best params
RESULTS_DIR = RESULTS_ROOT / f"results_{DVH_RUN_TAG}"  # metrics, plots, SHAP

# ------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------
# By default every column of DATASET_MINIMAL is used as a baseline feature (one
# value per patient, taken from the first row), EXCEPT:
#   - the identifiers and dates below (ID_DATE_COLS), which are never features;
#   - the pre-treatment IPSS measurements (PRE_TX_IPSS_COLS), extracted
#     separately as "last value strictly before tx_date" and prefixed with
#     PRE_TX_PREFIX;
#   - the columns listed in IGNORE_FEATURES.
#
# To drop a variable from the models, add it to IGNORE_FEATURES.

# Identifiers and dates: used by the pipeline logic, never as features.
ID_DATE_COLS = ["record_id", "tx_date", "ipss_date"]

# Individual IPSS items (questions a..g). These are NOT features, but they
# determine which pre-treatment row is used as reference: a row may carry a
# stored ipss_score_calc without any item, in which case the obstructive and
# irritative subscores cannot be derived. See 01_prepare_target.py.
IPSS_ITEM_COLS = [
    "prostsex_ipss_a", "prostsex_ipss_b", "prostsex_ipss_c",
    "prostsex_ipss_d", "prostsex_ipss_e", "prostsex_ipss_f", "prostsex_ipss_g",
]

# Variables explicitly excluded from the features (leave empty to keep all).
# NB: a column removed from PRE_TX_IPSS_COLS must be added here, otherwise it
# falls back into the baseline features (see `excluded` in 01_prepare_target.py).
IGNORE_FEATURES = [
    "hdr_days_after_ldr",
    # Individual IPSS items: redundant with the total and the
    # obstructive/irritative subscores, and more sparsely filled than either.
    *IPSS_ITEM_COLS,
    # Quality of life: coverage below the threshold enforced by the builder.
    "prostsex_qual_vie_a", "prostsex_qual_vie_b",
    # Hormone therapy duration: unusable and leaky.
    #   - hx_debut/hx_fin are filled for a small minority of patients, and in
    #     most of those cases both dates are identical, giving a zero duration.
    #   - the builder falls back to 0 (not NaN) when a date is missing, so
    #     "no ADT" and "unknown duration" become indistinguishable, while the
    #     adt flag is 1 for some of them.
    #   - most importantly, hx_debut/hx_fin are not bounded to the
    #     pre-treatment period. Salvage hormone therapy can start years AFTER
    #     the implant, so as a baseline feature hx_len would hand the model
    #     information dated after the moment of prediction.
    # The `adt` flag, filled for the whole cohort, captures what is known.
    "hx_len",
]
# Pre-treatment columns: last value strictly before tx_date, prefixed with
# PRE_TX_PREFIX. Includes the IPSS total (ipss_score_calc, which is the baseline
# of the delta target and is dropped from X downstream), the obstructive and
# irritative subscores, and any other measurement for which only the last value
# before treatment is relevant.
PRE_TX_IPSS_COLS = [
    "ipss_score_calc",
    "ipss_obstructive", "ipss_irritative",
    "shim_score",
]
# Prefix marking the pre-treatment IPSS features.
PRE_TX_PREFIX = "pretx_"

# Categorical columns (native CatBoost handling + encoding for MLP/LinReg).
CATEGORICAL_FEATURES = ["stage", "gleason", "adt"]

# Optuna
# Per-algorithm trial budget (keys = the `name` passed to run_nested_cv). Lets
# models with a large hyperparameter space (xgboost) get more trials than simple
# ones. linreg has no hyperparameter and is never tuned, so its value is ignored.
# Any algorithm absent from this dict falls back to OPTUNA_N_TRIALS.
OPTUNA_N_TRIALS_BY_ALGO = {
    "linreg": 1,
    "elasticnet": 50,
    "rf": 50,
    "catboost": 50,
    "xgboost": 40,
    "mlp": 50,
}
OPTUNA_N_TRIALS = 1    # fallback for an algorithm absent from the dict above
OPTUNA_TIMEOUT = None  # seconds; None = no limit

# ------------------------------------------------------------
# Optuna pruner - early stopping of unpromising trials
# ------------------------------------------------------------
# Each objective evaluates a hyperparameter set by an inner CV (N_INNER_FOLDS
# folds) and reports the running mean RMSE to Optuna AFTER EVERY FOLD. The
# MedianPruner compares that intermediate value to the median of previous trials
# AT THE SAME FOLD and prunes the trial if it is worse, so the remaining folds of
# clearly bad hyperparameters are not evaluated. Since all trials of a study
# share exactly the same folds, those intermediate values are directly
# comparable. Selection is not biased: pruning only starts from the
# OPTUNA_PRUNER_STARTUP_TRIALS-th trial and only after
# OPTUNA_PRUNER_WARMUP_STEPS folds have been evaluated.
# NB: this has no effect for an algorithm whose budget is
# <= OPTUNA_PRUNER_STARTUP_TRIALS, since no trial reaches the pruning regime.
OPTUNA_PRUNER_STARTUP_TRIALS = 3   # complete trials before pruning activates
OPTUNA_PRUNER_WARMUP_STEPS = 1     # minimum folds evaluated before a trial can be pruned

# Nested CV: number of folds of the INNER CV (Optuna tuning) run inside each
# training fold of the OUTER CV (N_FOLDS, which produces the OOF predictions).
# The outer test fold never takes part in tuning, which removes hyperparameter
# leakage. Cost is roughly (N_FOLDS + 1) times flat tuning.
N_INNER_FOLDS = 4

# CatBoost - fixed parameters
CATBOOST_FIXED = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "verbose": False,
    "early_stopping_rounds": 50,
    "thread_count": N_THREADS,   # fixed share of the cores
}

# MLP - fixed parameters
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 30
MLP_BATCH_SIZE = 32

# XGBoost - fixed parameters
XGBOOST_FIXED = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": SEED,
    "verbosity": 0,
    "n_jobs": N_THREADS,   # fixed share of the cores
    "early_stopping_rounds": 50,
}

# Random Forest - fixed parameters
RF_FIXED = {
    "random_state": SEED,
    "n_jobs": N_THREADS,   # fixed share of the cores
}

# ElasticNet - fixed parameters (only the seed and iteration cap; the rest is tuned)
ELASTICNET_FIXED = {
    "random_state": SEED,
    "max_iter": 10000,
}

# Evaluation
SHAP_BACKGROUND_SIZE = 100   # background patients for the MLP SHAP KernelExplainer
SHAP_MAX_DISPLAY = 15

# Calibration
CALIBRATION_N_BINS = 10  # number of deciles in the binned calibration plot
