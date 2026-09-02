"""
09_mcid_logreg.py
=================
CLASSIFICATION arm of the IPSS pipeline: the ability to predict staying within
the MCID band at a given endpoint, per DVH scenario, compared with what the
existing continuous regressors already provide.

Binary target (one-sided, threshold MCID_THRESH):
    y_bin = 1  if  delta IPSS(endpoint) <= +MCID   (no clinically significant
            0  otherwise                            worsening; improvement counts
                                                    as a success)
    delta IPSS = y_true in oof_predictions.csv = IPSS(endpoint) - pre-treatment
    baseline.

Two families of scores are compared on EXACTLY the same patients and the same
out-of-fold splits (the `fold` column of oof_predictions.csv, the one the
regressors used):

  - logreg   logistic regression trained from scratch on the same features X
             (prep_<tag>/features.csv), with the same preprocessor as the
             regressors (median imputation plus StandardScaler for numeric
             columns, most-frequent imputation plus one-hot for categorical
             ones). C is selected by an inner CV (LogisticRegressionCV), so the
             outer test fold never enters the fit. Score = OOF probability.

  - <algo>   an already trained continuous regressor. It predicts a delta-hat;
             since the target is one-sided (success = small delta), the score
             monotone with P(success) is -delta-hat, and the AUC is computed on
             that.

Metrics (per tag x endpoint x model x evaluation subset real|all):
  - AUC (roc_auc_score) with a patient-level bootstrap CI.
  - Paired delta AUC (logreg - regressor) with a patient-level bootstrap CI over
    the SAME resamples, which directly answers whether the logistic model beats
    thresholding the continuous predictions.

Outputs (suffixed by the active run, see config.with_suffix):
  - mcid[_suffix]/mcid_auc.csv                 long table of every metric.
  - mcid[_suffix]/oof_logreg_<tag>_y<ep>d.csv  OOF logistic probabilities.
  - figures[_suffix]/F7_mcid_auc_<ep>d.png/.pdf  AUC bars (models x scenarios).

Only the OOF files and features are read; the script trains logistic models
only, so it stays light and needs no torch/xgboost/catboost (the preprocessor is
replicated locally).

Usage:
    python 09_mcid_logreg.py
    PIPE_MCID_ENDPOINTS=400,730 PIPE_MCID_THRESH=3 python 09_mcid_logreg.py
    python 09_mcid_logreg.py --replot   # REDRAWS the figure from mcid_auc.csv:
                                        # no logistic refitted, no bootstrap.
"""
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    PROJECT_DIR, MODELS_ROOT, PREP_ROOT, SEED, N_FOLDS,
    CATEGORICAL_FEATURES, with_suffix, TAG_SUFFIX,
)
from utils import load_cached_results, replot_requested

plt.rc("font", family="serif")

# --replot: read mcid_auc.csv back and run only the figure code (see utils).
REPLOT = replot_requested()

# LogisticRegressionCV emits sklearn FutureWarnings (l1_ratios,
# use_legacy_attributes) that are irrelevant here; silence them for a readable log.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ------------------------------------------------------------
# Parameters (overridable by environment variable, to explore without editing)
# ------------------------------------------------------------
# Endpoints to process, in days. The code loops, so extending to further
# endpoints is a matter of PIPE_MCID_ENDPOINTS="400,730".
_ep_env = os.environ.get("PIPE_MCID_ENDPOINTS", "400")
ENDPOINT_DAYS = [int(x) for x in _ep_env.split(",") if x.strip()]

# MCID threshold (IPSS points) and direction of the target. DIRECTION="le" means
# success = delta <= threshold (one-sided). "abs_le" is the symmetric band
# |delta| <= threshold; "ge" is worsening, delta >= threshold. The aligned score
# of the regressors follows automatically.
MCID_THRESH = float(os.environ.get("PIPE_MCID_THRESH", "3"))
DIRECTION = os.environ.get("PIPE_MCID_DIRECTION", "le")   # "le" | "abs_le" | "ge"

# Continuous regressors present in oof_predictions.csv, compared through their
# aligned score.
REG_ALGOS = ["linreg", "elasticnet", "rf", "xgboost", "catboost", "mlp"]
# Display order of the models (logreg first, being the dedicated classifier).
MODEL_ORDER = ["logreg", "elasticnet", "xgboost", "catboost", "mlp", "rf", "linreg"]
# Subset of models shown in the figure (the CSV keeps them all).
FIG_MODELS = ["logreg", "elasticnet", "linreg", "xgboost", "catboost", "mlp"]

MODEL_LABELS = {
    "logreg": "LogReg", "elasticnet": "ElasticNet", "xgboost": "XGBoost",
    "catboost": "CatBoost", "mlp": "MLP", "rf": "RF", "linreg": "LinReg",
}

# Readable scenario labels for the x axis of the figure (DVH segmentation source
# or ablation). The raw tag is kept when absent from this table.
TAG_LABELS = {
    "curated_auto_det_clin0977":      "Deterministic",
    "curated_manual":                 "Manual",
    "curated_mc_bayes_clin0977":      "Probabilistic mean",
    "curated_mc_bayes_clin0977_noIpss": "No IPSS",
    "curated_mc_bayes_clin0977_var":  "Probabilistic mean + var",
    "noDVH":                          "No DVH",
}

N_BOOT = 2000
EVAL_SUBSETS = ["real", "all"]     # real = measured targets, as for the regressors
PRIMARY_SUBSET = "real"            # subset highlighted in the figure

# Scenarios excluded from the analysis and the figure. An empty set means EVERY
# ablation is included, on the same footing as the DVH variants: the MCID arm
# asks the same question as the regression, so there is no reason to restrict
# the set of contrasts here.
# Caution: discover_tags() globs models_* and therefore picks up any new tag,
# including suffixed runs. This is where one is removed.
EXCLUDE_TAGS: set[str] = set()

OUT_DIR = PROJECT_DIR / with_suffix("mcid")
FIG_DIR = PROJECT_DIR / with_suffix("figures")


# ============================================================
# Binary target and aligned score of the regressors
# ============================================================
def make_label(delta: np.ndarray) -> np.ndarray:
    """Binary "MCID success" label derived from the IPSS delta."""
    if DIRECTION == "le":
        return (delta <= MCID_THRESH).astype(int)
    if DIRECTION == "abs_le":
        return (np.abs(delta) <= MCID_THRESH).astype(int)
    if DIRECTION == "ge":
        return (delta >= MCID_THRESH).astype(int)
    raise ValueError(f"Unknown DIRECTION: {DIRECTION!r}")


def reg_score(delta_hat: np.ndarray) -> np.ndarray:
    """Turn the continuous prediction into a score monotone with P(success).

    The score is such that larger means more likely to succeed, so it is
    directly comparable to a probability (roc_auc_score expects a score
    increasing with the positive class):
      - "le"     success = small delta        -> score = -delta_hat
      - "abs_le" success = delta close to 0   -> score = -|delta_hat|
      - "ge"     success = large delta        -> score = +delta_hat
    """
    if DIRECTION == "le":
        return -delta_hat
    if DIRECTION == "abs_le":
        return -np.abs(delta_hat)
    if DIRECTION == "ge":
        return delta_hat
    raise ValueError(f"Unknown DIRECTION: {DIRECTION!r}")


def label_desc() -> str:
    op = {"le": "≤", "abs_le": "|·| ≤", "ge": "≥"}[DIRECTION]
    return f"Δ IPSS {op} {MCID_THRESH:g}"


# ============================================================
# Preprocessor (replicated from 02_train.build_preprocessor, no heavy deps)
# ============================================================
def build_preprocessor(feature_columns):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    num_cols = [c for c in feature_columns if c not in cat_cols]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def make_logreg():
    """L2 logistic regression with C selected by an inner CV. There is no
    leakage: that inner CV only sees the outer training fold. It optimises the
    AUC, which is the reported metric."""
    return LogisticRegressionCV(
        Cs=10, cv=5, scoring="roc_auc",     # l2 penalty (default), lbfgs
        solver="lbfgs", max_iter=5000, random_state=SEED,
    )


# ============================================================
# Out-of-fold LogReg on the EXACT folds of the regressors
# ============================================================
def logreg_oof(X: pd.DataFrame, y_bin: pd.Series, fold: pd.Series) -> pd.Series:
    """Out-of-fold probability of the positive class, fold by fold.

    For each fold k: train on the patients with fold != k, test on those with
    fold == k. This is the IDENTICAL split to the regressors (the `fold` column
    written by 02_train.py), so the comparison is strictly paired.
    """
    proba = pd.Series(np.nan, index=X.index, dtype=float)
    for k in sorted(fold.unique()):
        te = fold.index[fold == k]
        tr = fold.index[fold != k]
        # A fold whose training part is single-class cannot be fitted, so it
        # is skipped.
        if y_bin.loc[tr].nunique() < 2:
            continue
        pipe = Pipeline([
            ("pre", build_preprocessor(X.columns.tolist())),
            ("clf", make_logreg()),
        ])
        pipe.fit(X.loc[tr], y_bin.loc[tr])
        proba.loc[te] = pipe.predict_proba(X.loc[te])[:, 1]
    return proba


# ============================================================
# AUC with a patient-level bootstrap CI
# ============================================================
def auc_or_nan(label: np.ndarray, score: np.ndarray) -> float:
    """AUC, or NaN when a class is missing or the score is degenerate."""
    m = np.isfinite(score)
    if m.sum() < 2 or len(np.unique(label[m])) < 2:
        return np.nan
    return float(roc_auc_score(label[m], score[m]))


def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) matrix of resampled patient indices, drawn with replacement.
    Shared across models, so the paired delta AUC uses the SAME draws."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def mean_auc(label, score_mat, idx=None) -> float:
    """AUC computed WITHIN each repetition, then averaged. `score_mat` is (r, n).

    This aggregation preserves the estimand "AUC of a model learned on one
    partition", whereas averaging the r per-patient scores first would build a
    bagged predictor, better than what is being evaluated. At r=1 this is
    exactly auc_or_nan()."""
    vals = []
    for row in score_mat:
        a = auc_or_nan(label, row) if idx is None else auc_or_nan(label[idx], row[idx])
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else np.nan


def auc_ci_rep(label, score_mat, idx_mat):
    """(auc, lo, hi) of the AUC averaged over the repetitions.

    The bootstrap resamples PATIENTS once per draw, and the same draw is applied
    to every repetition before averaging, so the CI concerns the averaged
    statistic and the repetitions do not inflate n."""
    label = np.asarray(label)
    point = mean_auc(label, score_mat)
    vals = []
    for idx in idx_mat:
        a = mean_auc(label, score_mat, idx)
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def delta_auc_ci_rep(label, mat_a, mat_b, idx_mat):
    """(delta, lo, hi) for AUC(a) - AUC(b), averaged over repetitions, paired."""
    label = np.asarray(label)
    point = mean_auc(label, mat_a) - mean_auc(label, mat_b)
    vals = []
    for idx in idx_mat:
        if len(np.unique(label[idx])) < 2:
            continue
        d = mean_auc(label, mat_a, idx) - mean_auc(label, mat_b, idx)
        if np.isfinite(d):
            vals.append(d)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def auc_ci(label, score, idx_mat):
    """(auc, lo, hi): point estimate plus a 95% CI."""
    label = np.asarray(label)
    score = np.asarray(score)
    point = auc_or_nan(label, score)
    vals = []
    for idx in idx_mat:
        a = auc_or_nan(label[idx], score[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def delta_auc_ci(label, score_a, score_b, idx_mat):
    """(delta, lo, hi) for AUC(a) - AUC(b), paired bootstrap CI (same draws)."""
    label = np.asarray(label)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    point = auc_or_nan(label, score_a) - auc_or_nan(label, score_b)
    vals = []
    for idx in idx_mat:
        la = label[idx]
        if len(np.unique(la)) < 2:
            continue
        da = auc_or_nan(la, score_a[idx]) - auc_or_nan(la, score_b[idx])
        if np.isfinite(da):
            vals.append(da)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# ============================================================
# Scenario discovery and loading of one (tag, endpoint)
# ============================================================
def discover_tags(endpoint: int):
    """Tags that have an oof_predictions.csv for this endpoint, RESTRICTED to
    the active run.

    The glob over models_* picks up every run present on disk, including those
    of another suffix. Without a filter, a suffixed run would also see the
    unsuffixed tags, and the figure would mix two cohorts and two CV designs
    under indistinguishable labels. Only tags carrying the active suffix are
    kept, and the label is looked up on the BARE name, since TAG_LABELS does not
    know about suffixes.

    With an EMPTY suffix there is no automatic filter: from the name alone one
    cannot tell whether a trailing token is a run suffix or an ablation label,
    since "_r5" and "_noAge" look alike. The glob may therefore pick up the tags
    of a suffixed run, and EXCLUDE_TAGS is the way to drop them. A warning
    points this out.
    """
    tags = []
    for d in sorted(MODELS_ROOT.glob("models_*")):
        tag = d.name[len("models_"):]
        if tag in EXCLUDE_TAGS:
            continue
        if TAG_SUFFIX and not tag.endswith(f"_{TAG_SUFFIX}"):
            continue
        if (d / f"y_{endpoint}d" / "oof_predictions.csv").exists():
            tags.append(tag)
    if not TAG_SUFFIX and len(tags) > 1:
        roots = {t.rsplit("_", 1)[0] for t in tags}
        if len(roots) < len(tags):
            print("  ! Without PIPE_TAG_SUFFIX, tags from different runs may "
                  "coexist in the figure. Check the list above; use "
                  "EXCLUDE_TAGS to drop any of them.")
    return tags


def bare_tag(tag: str) -> str:
    """Tag stripped of the active run suffix (for TAG_LABELS, which ignores it)."""
    return tag[: -len(TAG_SUFFIX) - 1] if TAG_SUFFIX and tag.endswith(f"_{TAG_SUFFIX}") else tag


def load_tag_endpoint(tag: str, endpoint: int):
    """(X, oof) - X indexed by UNIQUE record_id, oof possibly repeated.

    The OOF file of a repeated-CV run carries r rows per patient, keyed by
    (record_id, repeat), each from a different partition. X is therefore not
    realigned row by row on the OOF here; `_repeat_views` does that, one
    repetition at a time. A missing `repeat` column is materialised as 0, which
    reduces to the single-partition case.
    """
    oof = pd.read_csv(
        MODELS_ROOT / f"models_{tag}" / f"y_{endpoint}d" / "oof_predictions.csv"
    )
    if "repeat" not in oof.columns:
        oof = oof.assign(repeat=0)
    X = pd.read_csv(PREP_ROOT / f"prep_{tag}" / "features.csv").set_index("record_id")
    return X, oof


def _repeat_views(X: pd.DataFrame, oof: pd.DataFrame):
    """Split the OOF by repetition, everything aligned on a COMMON PATIENT ORDER.

    Returns (pat, Xp, label, is_real, delta, folds, reg_scores):
      pat        record_id in the row order of the FIRST repetition. At r=1 this
                 is the file order, so the bootstrap draws, which index
                 positions, stay identical to the single-partition case.
      Xp         features reindexed on `pat` (one row per patient).
      label      binary MCID target, per patient (invariant across repetitions,
                 since y_true does not depend on the partition).
      is_real    source == 'real' mask, per patient (likewise invariant).
      folds      dict repeat -> Series of fold ids, indexed like `pat`.
      reg_scores dict algo -> array (n_repeats, n_patients) of aligned scores.

    Each repetition is a COMPLETE OOF prediction of the cohort: metrics are
    computed WITHIN a repetition, then averaged. Stacking the repetitions and
    splitting on `fold` would place the same patient in train and in test at
    once, since its r rows carry different folds - exactly the leak this design
    avoids.
    """
    reps = sorted(oof["repeat"].unique())
    first = oof[oof["repeat"] == reps[0]]
    pat = first["record_id"].to_numpy()
    Xp = X.loc[pat]

    delta = first["y_true"].to_numpy(dtype=float)
    label = make_label(delta)
    is_real = (first["source"].to_numpy() == "real")

    folds, reg_scores = {}, {}
    algos = [a for a in REG_ALGOS if a in oof.columns]
    for a in algos:
        reg_scores[a] = np.empty((len(reps), len(pat)), dtype=float)
    for i, r in enumerate(reps):
        g = oof[oof["repeat"] == r].set_index("record_id").reindex(pat)
        folds[r] = pd.Series(g["fold"].to_numpy(), index=Xp.index)
        for a in algos:
            reg_scores[a][i] = reg_score(g[a].to_numpy(dtype=float))
    return pat, Xp, label, is_real, delta, folds, reg_scores


# ============================================================
# Core: one (tag, endpoint) -> metric rows
# ============================================================
def process(tag: str, endpoint: int) -> list[dict]:
    X, oof = load_tag_endpoint(tag, endpoint)
    pat, Xp, label_all, is_real, delta, folds, reg_scores = _repeat_views(X, oof)
    reps = sorted(folds)
    n_rep = len(reps)
    y_bin = pd.Series(label_all, index=Xp.index)

    # One INDEPENDENT OOF logistic per repetition: each uses only the folds of
    # ITS partition. Stacking the repetitions would leak, since a patient would
    # be in test in its own partition and in train in all the others.
    logreg_mat = np.empty((n_rep, len(pat)), dtype=float)
    for i, r in enumerate(reps):
        logreg_mat[i] = logreg_oof(Xp, y_bin, folds[r]).to_numpy()

    scores = {"logreg": logreg_mat, **reg_scores}

    rows = []
    for subset in EVAL_SUBSETS:
        mask = is_real if subset == "real" else np.ones(len(pat), dtype=bool)
        lab = label_all[mask]
        n = int(mask.sum())
        n_pos = int(lab.sum())
        if len(np.unique(lab)) < 2:
            continue
        # n counts PATIENTS, not rows: repetitions do not inflate the draw.
        idx_mat = boot_indices(n, N_BOOT, SEED + endpoint)   # shared across models
        logreg_sub = scores["logreg"][:, mask]
        for model in MODEL_ORDER:
            if model not in scores:
                continue
            sm = scores[model][:, mask]
            auc, lo, hi = auc_ci_rep(lab, sm, idx_mat)
            if model == "logreg":
                d, dlo, dhi = 0.0, np.nan, np.nan
            else:
                # delta AUC = logreg - regressor (paired, same draws).
                d, dlo, dhi = delta_auc_ci_rep(lab, logreg_sub, sm, idx_mat)
            per_rep = [auc_or_nan(lab, s) for s in sm]
            rows.append({
                "endpoint": f"y_{endpoint}d", "days": endpoint, "tag": tag,
                "model": model, "eval_subset": subset, "n": n, "n_pos": n_pos,
                "prevalence": round(n_pos / n, 4),
                "auc": auc, "auc_lo": lo, "auc_hi": hi,
                "dauc_logreg_minus_model": d, "dauc_lo": dlo, "dauc_hi": dhi,
                "n_repeats": n_rep,
                # Dispersion of the AUC across partitions: a direct read of the
                # instability a single-partition CV leaves invisible.
                "auc_sd_rep": (float(np.nanstd(per_rep, ddof=1))
                               if n_rep > 1 else np.nan),
            })
    # OOF logistic probabilities (traceability and re-analysis), in LONG format:
    # one row per (patient, repetition). At r=1, `repeat` is 0 everywhere and the
    # file is a strict superset of the single-partition format.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i, r in enumerate(reps):
        blocks.append(pd.DataFrame({
            "record_id": pat, "repeat": r, "fold": folds[r].to_numpy(),
            "source": np.where(is_real, "real", "model"),
            "y_delta": delta, "y_bin": label_all, "logreg_proba": logreg_mat[i],
        }))
    pd.concat(blocks, ignore_index=True).to_csv(
        OUT_DIR / f"oof_logreg_{tag}_y{endpoint}d.csv", index=False)
    return rows


# ============================================================
# Figure: AUC (models x scenarios) for one endpoint
# ============================================================
def build_figure(df: pd.DataFrame, endpoint: int):
    sub = df[(df["days"] == endpoint) & (df["eval_subset"] == PRIMARY_SUBSET)]
    if sub.empty:
        return None
    tags = list(dict.fromkeys(sub["tag"]))        # discovery order
    models = [m for m in FIG_MODELS if m in set(sub["model"])]

    n_tags, n_models = len(tags), len(models)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * n_tags + 2), 5.2))
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, n_models - 1)) for i, m in enumerate(models)}

    width = 0.8 / n_models
    x = np.arange(n_tags)
    for j, m in enumerate(models):
        s = sub[sub["model"] == m].set_index("tag").reindex(tags)
        xpos = x + (j - (n_models - 1) / 2) * width
        yerr = np.vstack([
            (s["auc"] - s["auc_lo"]).to_numpy(),
            (s["auc_hi"] - s["auc"]).to_numpy(),
        ])
        edge = "black" if m == "logreg" else "none"
        lw = 1.4 if m == "logreg" else 0.0
        ax.bar(xpos, s["auc"].to_numpy(), width=width, color=colors[m],
               edgecolor=edge, linewidth=lw, label=MODEL_LABELS[m], zorder=2)
        ax.errorbar(xpos, s["auc"].to_numpy(), yerr=yerr, fmt="none",
                    ecolor="black", elinewidth=0.7, capsize=1.8, zorder=3)

    ax.axhline(0.5, color="grey", ls="--", lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TAG_LABELS.get(bare_tag(t), bare_tag(t)) for t in tags],
                       rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("AUC (OOF, 95% patient-level bootstrap CI)")
    ax.set_ylim(0.40, min(1.0, float(sub["auc_hi"].max()) + 0.05))
    ax.legend(ncol=min(len(models), 5), fontsize=9, framealpha=0.9,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================
def compute_all() -> pd.DataFrame:
    """Full computation (fits the logistic models, bootstraps) -> long table."""
    all_rows = []
    for endpoint in ENDPOINT_DAYS:
        tags = discover_tags(endpoint)
        if not tags:
            print(f"  ! No scenario with oof_predictions.csv for {endpoint} d - skipped.")
            continue
        print(f"Endpoint {endpoint} d - {len(tags)} scenario(s): {', '.join(tags)}")
        for tag in tags:
            rows = process(tag, endpoint)
            all_rows.extend(rows)
            # Console summary: logreg vs the best regressor (primary subset).
            prim = [r for r in rows if r["eval_subset"] == PRIMARY_SUBSET]
            lg = next((r for r in prim if r["model"] == "logreg"), None)
            regs = [r for r in prim if r["model"] != "logreg" and np.isfinite(r["auc"])]
            if lg and regs:
                best = max(regs, key=lambda r: r["auc"])
                print(f"  {tag:34s} logreg AUC={lg['auc']:.3f} "
                      f"[{lg['auc_lo']:.3f};{lg['auc_hi']:.3f}]  | "
                      f"best reg {best['model']}={best['auc']:.3f}  "
                      f"dAUC(logreg-{best['model']})="
                      f"{best['dauc_logreg_minus_model']:+.3f} "
                      f"[{best['dauc_lo']:+.3f};{best['dauc_hi']:+.3f}]")

    if not all_rows:
        raise SystemExit("No metric produced - check MODELS_ROOT and the endpoints.")

    return pd.DataFrame(all_rows)


def main():
    csv_path = OUT_DIR / "mcid_auc.csv"
    if REPLOT:
        # The CSV is the INPUT here: it is not rewritten, only redrawn from.
        df = load_cached_results(csv_path, "09_mcid_logreg.py")
    else:
        df = compute_all()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = []
    for endpoint in ENDPOINT_DAYS:
        fig = build_figure(df, endpoint)
        if fig is None:
            continue
        p = FIG_DIR / f"F7_mcid_auc_{endpoint}d_2.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(p)

    print(f"\n[done] Written:")
    for p in fig_paths:
        print(f"   {p} / .pdf")
    if not REPLOT:
        print(f"   {csv_path}")
        print(f"   {OUT_DIR}/oof_logreg_<tag>_y<ep>d.csv (OOF logistic probabilities)")


if __name__ == "__main__":
    main()
