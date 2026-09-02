"""
04_contrasts.py
===============
Paired contrast between two runs (e.g. M1 - M0) from the out-of-fold predictions
already saved by 02_train.py, WITHOUT retraining. It answers the pre-registered
inferential question: does the DVH block improve the prediction of the outcome
beyond the clinical variables, by how much, and with what uncertainty? The SAME
machinery serves the ABLATION contrasts (baseline = run deprived of a single
feature, treatment = full run), each corrected inside its own Holm family (see
confirmatory_family) and never mixed with the DVH question.

For each contrast (baseline -> treatment), endpoint and algorithm, on the REAL
targets only (source == "real"), with patients matched by record_id:

  - delta RMSE = RMSE(baseline) - RMSE(treatment), positive when the treatment
    is better. This is the PRIMARY, effect-size-first metric; the pooled point
    estimate matches the one in metrics.csv.
  - Patient-level bootstrap CI (cluster = patient) on delta RMSE, giving the
    uncertainty of the effect. This is the reference inference: a bound on the
    improvement, not a bare "p > 0.05".
  - Corrected Nadeau-Bengio t test on the per-fold delta RMSE. The between-fold
    variance of a cross-validation is optimistic, and the (1/K + n_test/n_train)
    correction inflates it. The p-value is SECONDARY, supporting the CI; with
    few folds the test is deliberately conservative and should not be the main
    judge.
  - Holm correction over the CONFIRMATORY family (primary learner
    PRIMARY_LEARNER x CONFIRMATORY_CONTRASTS x CONFIRMATORY_ENDPOINTS). The
    other cells form a consistency panel, reported uncorrected with
    is_confirmatory=False.

Sign convention: delta RMSE > 0 with a CI entirely above 0 means the treatment
credibly improves the prediction; a CI overlapping 0 means no detectable
improvement, and its upper bound says how large an effect can be excluded.

Outputs (CONTRAST_OUT_DIR):
  - contrasts.csv          one row per (contrast, endpoint, algo)
  - contrasts_summary.md   readable table, primary learner first
  - forest_<contrast>.png  delta RMSE with bootstrap CI per endpoint (primary
                           learner)

The OOF files are read-only; nothing is written outside CONTRAST_OUT_DIR.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from config import PROJECT_DIR, MODELS_ROOT, SEED, N_FOLDS, with_suffix
from utils import ensure_dir

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'


# ============================================================
# CONFIGURATION (set at the top of the file rather than on the CLI)
# ============================================================
# Contrasts to evaluate: (baseline_tag, treatment_tag, label). The tags are the
# DVH_RUN_TAG suffixes of the models_<tag>/ directories (see config.dvh_run_tag).
CONTRASTS = [
    # Curated DVH panel (USE_DVH_CURATED): clinically motivated indices.
    # Eligible for the confirmatory family (see confirmatory_family).
    ("noDVH", "curated_manual",        "M1_manual − M0"),
    ("noDVH", "curated_auto_det",      "M1_det − M0"),
    ("noDVH", "curated_auto_det_clin0977",      "M1_det_clin0977 − M0"),
    ("noDVH", "curated_mc_bayes_var",  "M1_bayes+var − M0"),
    ("noDVH", "curated_mc_bayes",  "M1_bayes − M0"),
    ("noDVH", "curated_mc_bayes_clin0977",  "M1_bayes_clin0977 − M0"),
    ("noDVH", "curated_mc_bayes_clin0977_var",  "M1_bayes+var_clin0977 − M0"),
    ("curated_mc_bayes", "curated_mc_bayes_var", "MC variance only"),
    ("curated_mc_bayes_clin0977", "curated_mc_bayes_clin0977_var",
     "MC variance only (clin0977)"),
    # Ablation of a single clinical feature: baseline = run WITHOUT the feature,
    # treatment = FULL run. delta RMSE = RMSE(without) - RMSE(with) > 0 means the
    # feature improves the prediction, exactly as for the DVH block. Ablations
    # are detected by the invariant base_tag = treat_tag + suffix (see
    # confirmatory_family), giving each its OWN Holm family, never mixed with
    # the DVH question.
    ("curated_mc_bayes_clin0977_noObstr", "curated_mc_bayes_clin0977",
     "Pre-tx obstructive IPSS (ablation)"),
    # Joint ablation of the obstructive and irritative subscores, i.e. the
    # predictive power of the pre-treatment IPSS TOTAL (the baseline level,
    # otherwise removed because it defines the target).
    ("curated_mc_bayes_clin0977_noIpss", "curated_mc_bayes_clin0977",
     "Pre-tx total IPSS (obstr+irrit ablation)"),
    # Ablation of AGE. Same inferential machinery, own Holm family.
    ("curated_mc_bayes_clin0977_noAge", "curated_mc_bayes_clin0977",
     "Age (ablation)"),
    # Ablation of the PROSTATE VOLUME / IMPLANT block (ldr_* plus psa_density;
    # see config._SCENARIOS["mc_bayes_clin0977_noVol"] for the detail and for
    # why psa_density belongs to the block). Tests whether the implant geometry
    # adds anything beyond the pre-treatment IPSS level. Same machinery and same
    # naming invariant as the other ablations, hence its own Holm family.
    ("curated_mc_bayes_clin0977_noVol", "curated_mc_bayes_clin0977",
     "Prostate volume / implant (ablation)"),
    # Full DVH bank (USE_DVH_FULL): every index, no reduction. Computed for
    # consistency but EXPLORATORY - always outside the confirmatory family
    # (never Holm-corrected; confirmatory_family returns None on the tag).
    ("noDVH", "fulldvh_manual",        "M1_manual (full) − M0"),
    ("noDVH", "fulldvh_auto_det",      "M1_det (full) − M0"),
    ("noDVH", "fulldvh_mc_bayes_var",  "M1_bayes+var (full) − M0"),
    ("noDVH", "fulldvh_mc_bayes",  "M1_bayes (full) − M0"),
    ("fulldvh_mc_bayes", "fulldvh_mc_bayes_var", "MC variance only (full)"),
]

# Display order of the algorithms. The PRIMARY learner is confirmatory; the
# others form the consistency panel, reported without multiplicity correction.
ALGOS = ["linreg", "elasticnet", "rf", "xgboost", "catboost", "mlp"]
PRIMARY_LEARNER = "elasticnet"

# Holm multiplicity correction over the confirmatory family.
#   - USE_HOLM = False: NO multiplicity correction. The RAW Nadeau-Bengio p is
#     reported (it keeps its WITHIN-test variance correction, which is
#     orthogonal to Holm) alongside the bootstrap CI. The nb_p_holm column is
#     neither computed nor displayed.
#   - USE_HOLM = True: restores the per-family Holm correction (the DVH block,
#     and each ablation in its own family) - see confirmatory_family. Reversible
#     and changes nothing else in the pipeline.
USE_HOLM = False

# Confirmatory family for the Holm correction (used only when USE_HOLM = True).
# Restrict these to the cells actually pre-registered in the analysis plan.
# Empty lists mean "all".
#   e.g. strictly pre-specified: CONFIRMATORY_CONTRASTS = ["M1_manual − M0"]
#                                CONFIRMATORY_ENDPOINTS = ["y_90d"]
CONFIRMATORY_CONTRASTS = []   # [] -> every label in CONTRASTS
CONFIRMATORY_ENDPOINTS = []   # [] -> every available endpoint

# Optional DISPLAY labels for the figures, overriding the CONTRASTS labels at
# render time only. The labels above stay the keys written to contrasts.csv and
# read by the downstream scripts, so they are never renamed here. A label absent
# from this table is displayed as is.
CONTRAST_DISPLAY = {}

# Bootstrap
N_BOOT = 5000
BOOT_CI = (2.5, 97.5)         # CI percentiles

# Nadeau-Bengio
NB_METRIC = "rmse"            # "rmse" (spec) or "mse" (variant linear in examples)
# n_test/n_train ratio of the outer CV. For a plain K-fold: 1/(K-1).
NB_TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)

# Output directory, outside the per-run results_<tag>/. Suffixed by the active
# run (PIPE_TAG_SUFFIX) so a targeted run writes to contrasts_<suffix>/ without
# overwriting the multi-endpoint contrasts/.
CONTRAST_OUT_DIR = PROJECT_DIR / with_suffix("contrasts")

OOF_NAME = "oof_predictions.csv"


# ============================================================
# I/O
# ============================================================
def models_dir(tag: str) -> Path:
    # The CONTRASTS tags are given WITHOUT a run suffix; with_suffix targets the
    # active run, so a targeted run contrasts against itself.
    return MODELS_ROOT / f"models_{with_suffix(tag)}"


def load_oof(tag: str, endpoint: str) -> pd.DataFrame | None:
    """Load the OOF file of one run and endpoint. record_id is read as a string
    for robust matching."""
    path = models_dir(tag) / endpoint / OOF_NAME
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["record_id"] = df["record_id"].astype(str).str.strip()
    return df


def list_endpoints(tag: str) -> list[str]:
    d = models_dir(tag)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / OOF_NAME).exists())


# ============================================================
# Baseline/treatment matching on the real targets
# ============================================================
def paired_frame(base: pd.DataFrame, treat: pd.DataFrame, algo: str):
    """Return a matched, real-only DataFrame with columns record_id, repeat,
    y_true, fold, pred_base, pred_treat - or None if nothing is usable.

    Matching happens on the key (record_id, repeat), NOT on record_id alone:
    under repeated CV a patient has one prediction per repetition, and a merge
    on record_id would silently produce a cartesian product (r^2 rows per
    patient). OOF files without a `repeat` column are treated as repeat=0, which
    reproduces the single-partition matching exactly.

    Guards: restricted to source == 'real' on BOTH sides, refuses a duplicated
    key, and checks that y_true and fold agree between runs (targets and folds
    do not depend on the DVH block, so a disagreement signals a pipeline
    problem)."""
    if algo not in base.columns or algo not in treat.columns:
        return None, "algorithm absent"

    def real(df):
        return df[df["source"] == "real"] if "source" in df.columns else df

    def keyed(df):
        d = real(df).copy()
        if "repeat" not in d.columns:
            d["repeat"] = 0          # OOF without repetitions: single partition
        return d.set_index(["record_id", "repeat"])

    b, t = keyed(base), keyed(treat)
    for name, d in (("baseline", b), ("treatment", t)):
        if d.index.has_duplicates:
            return None, f"duplicated (record_id, repeat) key on the {name} side"

    # The baseline order is preserved: row order drives the bootstrap draw, and
    # therefore the reproducibility of the CIs.
    common = b.index[b.index.isin(t.index)]
    if len(common) == 0:
        return None, "no matched patient"

    b, t = b.loc[common], t.loc[common]

    # The real targets must agree between runs (numerical tolerance).
    if not np.allclose(b["y_true"].values, t["y_true"].values,
                       rtol=0, atol=1e-6, equal_nan=True):
        n_bad = int((~np.isclose(b["y_true"].values, t["y_true"].values,
                                 atol=1e-6, equal_nan=True)).sum())
        print(f"      ! y_true differs for {n_bad}/{len(common)} patients "
              f"between runs - check that the target does not depend on the DVH block.")

    fold = b["fold"].copy()
    if "fold" in t.columns and not (b["fold"].values == t["fold"].values).all():
        print("      ! folds differ between runs - using the baseline folds.")

    out = pd.DataFrame({
        "record_id": common.get_level_values("record_id"),
        "repeat": common.get_level_values("repeat"),
        "y_true": b["y_true"].values,
        "fold": fold.values,
        "pred_base": b[algo].values,
        "pred_treat": t[algo].values,
    })
    # Usable rows: target plus both predictions non-NaN.
    out = out[~out[["y_true", "pred_base", "pred_treat"]].isna().any(axis=1)]
    if len(out) == 0:
        return None, "only NaN"
    return out.reset_index(drop=True), None


# ============================================================
# Statistics
# ============================================================
def _rmse(err2):
    return float(np.sqrt(np.mean(err2)))


def delta_rmse(df: pd.DataFrame) -> float:
    """Pooled delta RMSE = RMSE(base) - RMSE(treat). Positive means the
    treatment is better."""
    eb = (df["y_true"] - df["pred_base"]).values ** 2
    et = (df["y_true"] - df["pred_treat"]).values ** 2
    return _rmse(eb) - _rmse(et)


def bootstrap_delta(df: pd.DataFrame, n_boot=N_BOOT, ci=BOOT_CI, seed=SEED):
    """Patient-level bootstrap CI (cluster = patient) on delta RMSE, plus a
    two-sided p.

    PATIENTS are resampled with replacement (a patient's base/treat pair moves
    together, preserving the matching), and delta RMSE is recomputed B times.

    Under repeated CV a patient has r rows, one per repetition: the cluster
    stays the patient and its r rows move together. Resampling ROWS would break
    the cluster and shrink the CI by roughly sqrt(r). At r=1 each cluster is one
    row and the draw reduces to the single-partition case.

    Scope: this CI only covers the PATIENT SAMPLING randomness, with the learned
    models held fixed. It says nothing about the learning randomness, which is
    the role of the Nadeau-Bengio test below.
    """
    rng = np.random.default_rng(seed)
    eb = (df["y_true"].values - df["pred_base"].values) ** 2
    et = (df["y_true"].values - df["pred_treat"].values) ** 2

    # Rows grouped by patient. codes[i] is the cluster index of row i.
    codes, uniq = pd.factorize(df["record_id"].values, sort=False)
    n_clusters = len(uniq)
    rows_of = [np.flatnonzero(codes == c) for c in range(n_clusters)]
    single = all(len(r) == 1 for r in rows_of)   # r=1: one cluster is one row

    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = pick if single else np.concatenate([rows_of[c] for c in pick])
        boots[i] = np.sqrt(eb[idx].mean()) - np.sqrt(et[idx].mean())
    lo, hi = np.percentile(boots, ci)
    point = np.sqrt(eb.mean()) - np.sqrt(et.mean())
    # Two-sided p: twice the smaller mass on one side of 0.
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return point, float(lo), float(hi), float(min(p, 1.0))


def _fold_metric_diff(df: pd.DataFrame, metric=NB_METRIC):
    """Per-fold delta RMSE (or delta MSE): base - treat on the real patients of
    the fold.

    Under repeated CV the group is (repeat, fold), giving r*K differences
    instead of K, which raises the Nadeau-Bengio degrees of freedom from K-1 to
    r*K-1. At r=1 (the `repeat` column constant or absent) the result is
    identical to grouping by fold alone, including the order, since groupby
    sorts on the key and `repeat` is constant."""
    keys = ["repeat", "fold"] if "repeat" in df.columns else ["fold"]
    diffs = []
    for _k, g in df.groupby(keys):
        eb = (g["y_true"].values - g["pred_base"].values) ** 2
        et = (g["y_true"].values - g["pred_treat"].values) ** 2
        if metric == "mse":
            diffs.append(eb.mean() - et.mean())
        else:
            diffs.append(np.sqrt(eb.mean()) - np.sqrt(et.mean()))
    return np.asarray(diffs, dtype=float)


def nadeau_bengio(df: pd.DataFrame, ratio=NB_TEST_TRAIN_RATIO, metric=NB_METRIC):
    """Corrected Nadeau-Bengio t on the per-fold performance differences.

    Corrected variance of the mean = (1/M + n_test/n_train) * S^2_d, with
    df = M-1, where M is the NUMBER OF DIFFERENCES (M = K for a single
    partition, M = r*K under repeated CV). Returns (t, df, two-sided p, mean_d,
    M).

    CAUTION - `ratio` does NOT depend on M. It is n_test/n_train inside one
    partition, i.e. 1/(N_FOLDS-1), and it is unchanged when the CV is repeated:
    repeating does not change the fold sizes, hence not the overlap of the
    training sets that this term corrects for. Replacing it by 1/(M-1) under
    repeated CV would be an error - this is precisely what lets repetitions gain
    degrees of freedom without loosening the correction (the "corrected
    resampled t-test" of Bouckaert & Frank). Corollary: the detection threshold
    never drops below z*sqrt(ratio) whatever r; going lower requires increasing
    K.

    The variable is named K below to match the rest of the module, but it holds
    M = len(d)."""
    d = _fold_metric_diff(df, metric)
    K = len(d)
    if K < 2:
        return np.nan, np.nan, np.nan, float(np.mean(d)) if K else np.nan, K
    mean_d = float(d.mean())
    s2 = float(d.var(ddof=1))
    if s2 == 0:
        # Identical differences across folds: a clean effect (p -> 0) or none
        # at all (mean_d = 0).
        p = 0.0 if mean_d != 0 else 1.0
        return (np.inf if mean_d > 0 else -np.inf if mean_d < 0 else 0.0), K - 1, p, mean_d, K
    se = np.sqrt((1.0 / K + ratio) * s2)
    tstat = mean_d / se
    p = float(2.0 * student_t.sf(abs(tstat), df=K - 1))
    return float(tstat), K - 1, p, mean_d, K


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni. Returns the adjusted p-values in input order, NaN
    ignored."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    m = len(idx)
    adj = [np.nan] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # Holm monotonicity
        adj[i] = min(running, 1.0)
    return adj


# ============================================================
# Pipeline
# ============================================================
def confirmatory_family(label, endpoint, algo, base_tag, treat_tag):
    """Family key for the Holm correction, or None when outside the confirmatory
    family. Each family is corrected SEPARATELY because it answers a distinct
    scientific question:

      - DVH block -> one shared family "dvh": every contrast of the DVH block
        (baseline noDVH or a reduced DVH run, treatment a richer DVH run),
        primary learner, subject to the pre-registered filters
        CONFIRMATORY_CONTRASTS / CONFIRMATORY_ENDPOINTS.
      - Ablation of a single feature -> ONE family per contrast
        ("ablation::<label>"): the baseline is the treatment DEPRIVED of one or
        more features, so its tag has the treatment tag as a prefix. The
        correction then covers the endpoints of the primary learner for THAT
        contrast alone, never mixed with the DVH question.

    The full DVH bank and the non-primary learners stay outside any family
    (exploratory or consistency panel, reported uncorrected).
    """
    if algo != PRIMARY_LEARNER:
        return None
    if treat_tag.startswith("fulldvh"):
        return None
    # Ablation contrast: the baseline is the treatment minus one or more
    # features, so its tag is the treatment tag plus a suffix. Dedicated family.
    if base_tag != treat_tag and base_tag.startswith(treat_tag):
        return f"ablation::{label}"
    # Otherwise: confirmatory family of the DVH block (pre-registered filters).
    if CONFIRMATORY_CONTRASTS and label not in CONFIRMATORY_CONTRASTS:
        return None
    if CONFIRMATORY_ENDPOINTS and endpoint not in CONFIRMATORY_ENDPOINTS:
        return None
    return "dvh"


def run():
    rows = []
    for base_tag, treat_tag, label in CONTRASTS:
        eps_base = set(list_endpoints(base_tag))
        eps_treat = set(list_endpoints(treat_tag))
        endpoints = sorted(eps_base & eps_treat,
                           key=lambda e: int(e.strip("y_").rstrip("d"))
                           if e.startswith("y_") else 0)
        if not endpoints:
            print(f"[{label}] no endpoint shared between "
                  f"models_{base_tag}/ and models_{treat_tag}/ - skipped.")
            continue
        print(f"\n[{label}]  baseline=models_{base_tag}  treatment=models_{treat_tag}")
        for ep in endpoints:
            base = load_oof(base_tag, ep)
            treat = load_oof(treat_tag, ep)
            if base is None or treat is None:
                continue
            for algo in ALGOS:
                pf, err = paired_frame(base, treat, algo)
                if pf is None:
                    continue
                d_point, lo, hi, bp = bootstrap_delta(pf)
                tstat, dfree, nbp, mean_d, K = nadeau_bengio(pf)
                eb = (pf["y_true"] - pf["pred_base"]).values ** 2
                et = (pf["y_true"] - pf["pred_treat"]).values ** 2
                fam = confirmatory_family(label, ep, algo, base_tag, treat_tag)
                rows.append({
                    "contrast": label,
                    "baseline": base_tag, "treatment": treat_tag,
                    "endpoint": ep, "algo": algo,
                    "n_paired": len(pf),
                    "rmse_base": _rmse(eb), "rmse_treat": _rmse(et),
                    "delta_rmse": d_point,
                    "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_p": bp,
                    "nb_t": tstat, "nb_df": dfree, "nb_p": nbp,
                    "nb_mean_fold_diff": mean_d, "nb_K": K,
                    "is_primary": algo == PRIMARY_LEARNER,
                    "holm_family": fam,
                    "is_confirmatory": fam is not None,
                })
            print(f"  - {ep}: {sum(1 for r in rows if r['endpoint']==ep and r['contrast']==label)} algorithms")

    res = pd.DataFrame(rows)
    if res.empty:
        print("\nNo contrast computable (OOF files missing?). Nothing written.")
        return res

    # Holm PER confirmatory family (on nb_p; the bootstrap CI stays primary).
    # Each family (the DVH block "dvh", and each ablation "ablation::<label>")
    # is corrected separately, as they answer distinct questions. groupby skips
    # the None keys (outside any family), which stay uncorrected. A family of a
    # single test gives an adjusted p equal to the raw p. When USE_HOLM = False,
    # only the raw NB p and the bootstrap CI are reported.
    if USE_HOLM:
        res["nb_p_holm"] = np.nan
        for _, grp in res.groupby("holm_family"):
            res.loc[grp.index, "nb_p_holm"] = holm(grp["nb_p"].tolist())
    return res


# ============================================================
# Report
# ============================================================
def write_outputs(res: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    res.to_csv(out_dir / "contrasts.csv", index=False)

    holm_note = (f"Holm over the confirmatory family (primary learner "
                 f"\"{PRIMARY_LEARNER}\")." if USE_HOLM else
                 "Holm multiplicity correction DISABLED (USE_HOLM=False): "
                 "raw NB p reported (the NB variance correction is kept).")
    lines = [
        "# Paired contrasts M1 - M0 (OOF, real targets)\n",
        "delta RMSE = RMSE(baseline) - RMSE(treatment); **> 0 means the "
        "treatment improves**. Patient-level bootstrap CI (primary inference); "
        "between-fold Nadeau-Bengio p (secondary, conservative); "
        + holm_note + "\n",
    ]
    # The Holm column is shown only when it was computed.
    cols = ["endpoint", "algo", "n_paired", "delta_rmse",
            "boot_ci_lo", "boot_ci_hi", "boot_p", "nb_p"]
    if USE_HOLM and "nb_p_holm" in res.columns:
        cols.append("nb_p_holm")
    for label in res["contrast"].unique():
        sub = res[res["contrast"] == label].copy()
        lines.append(f"\n## {label}\n")
        # Primary learner first.
        sub["_ord"] = sub["algo"].map({a: i for i, a in enumerate(ALGOS)})
        sub = sub.sort_values(["endpoint", "_ord"])
        show = sub[cols].copy()
        for c in ["delta_rmse", "boot_ci_lo", "boot_ci_hi"]:
            show[c] = show[c].round(3)
        for c in [c for c in ["boot_p", "nb_p", "nb_p_holm"] if c in show.columns]:
            show[c] = show[c].round(4)
        lines.append(show.to_markdown(index=False))
    (out_dir / "contrasts_summary.md").write_text("\n".join(lines))

    # Forest plot of the primary learner, one per contrast.
    for label in res["contrast"].unique():
        sub = res[(res["contrast"] == label) & (res["algo"] == PRIMARY_LEARNER)]
        if sub.empty:
            continue
        sub = sub.sort_values("endpoint", key=lambda s: s.map(
            lambda e: int(e.strip("y_").rstrip("d")) if e.startswith("y_") else 0))
        y = np.arange(len(sub))[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(sub) + 1.5))
        ax.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.hlines(y, sub["boot_ci_lo"], sub["boot_ci_hi"], color="steelblue", lw=2)
        ax.plot(sub["delta_rmse"], y, "o", color="crimson", zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["endpoint"])
        ax.set_xlabel("ΔRMSE = RMSE(baseline) − RMSE(treatment)   "
                      "(> 0 ⇒ treatment helps)")
        ax.set_title(f"{CONTRAST_DISPLAY.get(label, label)} — {PRIMARY_LEARNER}\n"
                     f"patient-level bootstrap CI ({int(N_BOOT)} resamples)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        safe = label.replace(" ", "_").replace("−", "-").replace("/", "_")
        fig.savefig(out_dir / f"forest_{safe}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def main():
    print("=" * 70)
    print("PAIRED CONTRASTS - delta RMSE, bootstrap CI, Nadeau-Bengio"
          + (", Holm" if USE_HOLM else " (Holm disabled)"))
    print("=" * 70)
    ensure_dir(CONTRAST_OUT_DIR)

    res = run()
    if res.empty:
        return
    write_outputs(res, CONTRAST_OUT_DIR)

    # Console summary: primary learner.
    prim = res[res["algo"] == PRIMARY_LEARNER]
    print(f"\n--- Primary learner ({PRIMARY_LEARNER}) ---")
    cols = ["contrast", "endpoint", "n_paired", "delta_rmse",
            "boot_ci_lo", "boot_ci_hi", "nb_p"]
    if USE_HOLM and "nb_p_holm" in prim.columns:
        cols.append("nb_p_holm")
    print(prim[cols].round(3).to_string(index=False))
    print(f"\n[done] Written to {CONTRAST_OUT_DIR}/ "
          f"(contrasts.csv, contrasts_summary.md, forest_*.png)")


if __name__ == "__main__":
    main()
