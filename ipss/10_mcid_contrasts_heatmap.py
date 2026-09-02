"""
10_mcid_contrasts_heatmap.py
============================
Counterpart of 08_all_models_heatmap.py for the BINARY "within the MCID band"
task: instead of the delta RMSE, the effect of a feature block is measured by
the paired delta AUC, with the SAME inference (patient-level bootstrap CI plus
the corrected Nadeau-Bengio test).

This is not what 09_mcid_logreg.py does: that script compares, at a FIXED
scenario, the dedicated logistic model against the thresholded continuous score.
Here the question is the one of 04 and 08 - does this feature block help? - but
asked on the binary target:

    delta AUC = AUC(treatment) - AUC(baseline)   > 0 means the block improves
    (same patients, same folds, same labels: strictly paired)

The binary target and the scores are taken as is from 09_mcid_logreg (direct
import, no duplication): y_bin = 1 if delta IPSS <= MCID_THRESH (overridable by
PIPE_MCID_THRESH / PIPE_MCID_DIRECTION), a regressor's score is its aligned
negative prediction, and the logistic score is its OOF probability.

Inference, identical to 04_contrasts.py:
  - Patient-level bootstrap CI (cluster = patient, with draws SHARED between
    baseline and treatment so the matching is preserved) on the delta AUC. This
    is the reference inference.
  - Nadeau-Bengio t on the per-fold delta AUC: corrected variance
    (1/K + n_test/n_train)*S^2_d, K-1 df. Secondary and conservative.
  - No multiplicity correction (matching USE_HOLM=False in 04).

Figure: diverging heatmap of the delta AUC (RdBu, centred on 0, shared symmetric
scale), one small multiple per contrast, rows = models, columns = endpoints.
Colour encodes magnitude and sign; significance is a separate glyph:
    o  bootstrap CI excluding 0
    *  and in addition p_NB < 0.05
The "LogReg" row is the dedicated classifier, retrained out-of-fold on the exact
folds of the regressors; the other rows are the thresholded continuous
regressors.

Outputs: mcid[_suffix]/mcid_contrasts.csv, figures[_suffix]/F8_mcid_*.png/.pdf
and mcid[_suffix]/mcid_contrasts_table.tex. The OOF files and features are
read-only.

Usage:
    python 10_mcid_contrasts_heatmap.py
    PIPE_MCID_THRESH=3 python 10_mcid_contrasts_heatmap.py
    MCID_SKIP_LOGREG=1 python 10_mcid_contrasts_heatmap.py   # fast, no LogReg
    python 10_mcid_contrasts_heatmap.py --replot   # REDRAWS the figure from
                                        # mcid_contrasts.csv: no logistic
                                        # refitted, no bootstrap.
"""
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from scipy.stats import t as student_t

from config import (PROJECT_DIR, MODELS_ROOT, PREP_ROOT, SEED, N_FOLDS,
                    ENDPOINTS as CFG_ENDPOINTS, with_suffix)
from utils import load_cached_results, replot_requested

plt.rc("font", family="serif")

# --replot: read mcid_contrasts.csv back and run only the figure/table code.
REPLOT = replot_requested()


# ------------------------------------------------------------
# Reuse of 09_mcid_logreg (its module name is not importable, hence the explicit
# load). This guarantees that the binary target, the aligned regressor score and
# the logistic model
# are EXACTLY those of the existing MCID analysis.
# ------------------------------------------------------------
def _load_mcid09():
    path = Path(__file__).resolve().parent / "09_mcid_logreg.py"
    spec = importlib.util.spec_from_file_location("mcid09", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcid09"] = mod
    spec.loader.exec_module(mod)
    return mod


M09 = _load_mcid09()

# ============================================================
# CONFIGURATION
# ============================================================
# Contrasts: (baseline_tag, treatment_tag, short name). The same pairs as
# 04_contrasts.CONTRASTS, in the display order of 08: the DVH source variants
# first, then the clinical ablations. The same eight contrasts as in 08 (see
# 08_all_models_heatmap.CONTRAST_ORDER). 11_mcid_5y reuses this list through
# M10.CONTRASTS, so the downstream figures stay aligned.
CONTRASTS = [
    ("noDVH", "curated_manual",                    "Manual DVH"),
    ("noDVH", "curated_auto_det_clin0977",         "Auto det. DVH"),
    ("noDVH", "curated_mc_bayes_clin0977",         "Bayesian DVH (mean)"),
    ("noDVH", "curated_mc_bayes_clin0977_var",     "Bayesian DVH (mean+var)"),
    ("curated_mc_bayes_clin0977_noIpss", "curated_mc_bayes_clin0977",
     "Pre-tx total IPSS"),
    ("curated_mc_bayes_clin0977_noObstr", "curated_mc_bayes_clin0977",
     "Pre-tx obstructive IPSS"),
    ("curated_mc_bayes_clin0977_noAge",  "curated_mc_bayes_clin0977",
     "Pre-tx age"),
    ("curated_mc_bayes_clin0977_noVol",  "curated_mc_bayes_clin0977",
     "Volume / implant"),
]

# Models = rows of the heatmap. "logreg" is the dedicated classifier, and it is
# expensive since it is retrained out-of-fold for each tag and endpoint; the
# others are the thresholded continuous regressors. MCID_SKIP_LOGREG=1 drops it
# for fast iteration.
MODEL_ORDER = ["logreg", "elasticnet", "rf", "xgboost", "catboost", "mlp"]
if os.environ.get("MCID_SKIP_LOGREG"):
    MODEL_ORDER = [m for m in MODEL_ORDER if m != "logreg"]
MODEL_LABELS = {"logreg": "LogReg", "elasticnet": "ElasticNet", "rf": "RF",
                "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP"}
PRIMARY = "elasticnet"          # primary learner, shown in bold (as in 08)

ENDPOINTS = [f"y_{d}d" for d in sorted(CFG_ENDPOINTS)]
ENDPOINT_DAYS = [str(d) for d in sorted(CFG_ENDPOINTS)]
N_EP = len(ENDPOINTS)

ALPHA = 0.05
N_BOOT = 2000                   # as in 09 (the cost here is two AUCs per draw)
# Symmetric bound of the map, chosen so that small effects stay in the pale
# tones while large ones saturate; the colour bar carries "extend" arrows that
# signal the clipping.
VMAX = 0.15
CMAP = "RdBu_r"
# Colour of the significance glyphs. A vivid green sits off the red/blue
# diverging axis of the colormap, so it stays legible both on saturated cells
# and on near-white ones, where black was lost. Set to "black" for a plain
# rendering.
GLYPH_COLOR = "#00c853"
NB_TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)
EVAL_SUBSET = "real"            # measured targets, consistent with 04_contrasts

OUT_DIR = PROJECT_DIR / with_suffix("mcid")
FIG_DIR = PROJECT_DIR / with_suffix("figures")


# ============================================================
# Fast rank-based AUC (the bootstrap evaluates two scores per cell per draw)
# ============================================================
def fast_auc(label: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC computed from ranks (ties get average ranks). Returns
    NaN when a class is missing or the score is degenerate."""
    m = np.isfinite(score)
    if m.sum() < 2:
        return np.nan
    lab, sc = label[m], score[m]
    n_pos = int(lab.sum())
    n_neg = len(lab) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(sc)
    return float((r[lab == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ============================================================
# Loading of the OOF files and features (04_contrasts paths, via with_suffix)
# ============================================================
_OOF_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}
_LOGREG_CACHE: dict[tuple[str, str], pd.Series | None] = {}


def load_oof_full(tag: str, endpoint: str) -> pd.DataFrame | None:
    """Full OOF of one run and endpoint (real plus imputed targets)."""
    key = (tag, endpoint)
    if key not in _OOF_CACHE:
        p = (MODELS_ROOT / f"models_{with_suffix(tag)}" / endpoint
             / "oof_predictions.csv")
        if not p.exists():
            _OOF_CACHE[key] = None
        else:
            df = pd.read_csv(p)
            df["record_id"] = df["record_id"].astype(str).str.strip()
            # Key (record_id, repeat): under repeated CV a patient has r rows,
            # one per partition. Indexing on record_id alone would duplicate the
            # index and turn the matching joins into a cartesian product WITHOUT
            # raising. A missing `repeat` column is materialised as 0, giving an
            # index equivalent to the single-partition one.
            if "repeat" not in df.columns:
                df["repeat"] = 0
            _OOF_CACHE[key] = df.set_index(["record_id", "repeat"])
    return _OOF_CACHE[key]


def load_oof(tag: str, endpoint: str) -> pd.DataFrame | None:
    """OOF restricted to the EVALUATION subset (real by default)."""
    df = load_oof_full(tag, endpoint)
    if df is None:
        return None
    if EVAL_SUBSET == "real" and "source" in df.columns:
        df = df[df["source"] == "real"]
    return df


def logreg_proba(tag: str, endpoint: str) -> pd.Series | None:
    """OOF probability of the dedicated logistic model for one (tag, endpoint).

    It is trained on the features of that tag using the EXACT folds of the
    regressors (the `fold` column), through M09.logreg_oof, so there is no
    leakage and the matching is preserved.

    As in 09_mcid_logreg.process(), training uses ALL the patients of the OOF
    (real plus imputed targets) and only the EVALUATION is restricted to `real`:
    restricting the training too would give the classifier less data than the
    regressors it is compared against, and would bias the delta AUC.

    Results are cached, since each (tag, endpoint) serves several contrasts.

    Under repeated CV an INDEPENDENT logistic is trained per repetition, using
    the folds of ITS partition. Stacking the repetitions would place a patient in
    test in its own partition and in train in all the others, a direct leak.
    The returned Series is indexed by (record_id, repeat).
    """
    key = (tag, endpoint)
    if key in _LOGREG_CACHE:
        return _LOGREG_CACHE[key]
    oof = load_oof_full(tag, endpoint)
    fpath = PREP_ROOT / f"prep_{with_suffix(tag)}" / "features.csv"
    if oof is None or not fpath.exists():
        _LOGREG_CACHE[key] = None
        return None
    X = pd.read_csv(fpath)
    X["record_id"] = X["record_id"].astype(str).str.strip()
    X = X.set_index("record_id")

    out = []
    for r in sorted(oof.index.get_level_values("repeat").unique()):
        g = oof.xs(r, level="repeat")
        ids = [i for i in g.index if i in X.index]
        if not ids:
            continue
        Xi = X.loc[ids]
        y_bin = pd.Series(M09.make_label(g.loc[ids, "y_true"].to_numpy(float)),
                          index=Xi.index)
        if y_bin.nunique() < 2:
            continue
        fold = pd.Series(g.loc[ids, "fold"].values, index=Xi.index)
        proba = M09.logreg_oof(Xi, y_bin, fold)
        proba.index = pd.MultiIndex.from_product([proba.index, [r]],
                                                 names=["record_id", "repeat"])
        out.append(proba)
    _LOGREG_CACHE[key] = pd.concat(out) if out else None
    return _LOGREG_CACHE[key]


def paired_scores(base_tag, treat_tag, endpoint, model):
    """(label, SB, ST, FOLD) matched on (record_id, repeat), or None.

    `label` is a vector of length n_patients, since the binary target does not
    depend on the partition. `SB`, `ST` and `FOLD` are (n_repeats, n_patients)
    matrices aligned on a COMMON PATIENT ORDER, that of the first repetition; at
    r=1 this is the file order, so the bootstrap draws are unchanged.

    The label comes from the baseline y_true, and the agreement of y_true between
    runs is checked, the target being independent of the feature block.
    """
    b, t = load_oof(base_tag, endpoint), load_oof(treat_tag, endpoint)
    if b is None or t is None:
        return None
    common = b.index[b.index.isin(t.index)]          # baseline order preserved
    if len(common) == 0:
        return None
    b, t = b.loc[common], t.loc[common]
    if not np.allclose(b["y_true"].values, t["y_true"].values, atol=1e-6,
                       equal_nan=True):
        print(f"      ! y_true differs between runs ({endpoint}) - the target "
              f"is meant to be independent of the feature block.")

    if model == "logreg":
        pb, pt = logreg_proba(base_tag, endpoint), logreg_proba(treat_tag, endpoint)
        if pb is None or pt is None:
            return None
        sb_all = pb.reindex(common).to_numpy(float)
        st_all = pt.reindex(common).to_numpy(float)
    else:
        if model not in b.columns or model not in t.columns:
            return None
        sb_all = M09.reg_score(b[model].to_numpy(float))
        st_all = M09.reg_score(t[model].to_numpy(float))

    reps = sorted(common.get_level_values("repeat").unique())
    pat0 = common[common.get_level_values("repeat") == reps[0]].get_level_values("record_id")
    lab_all = M09.make_label(b["y_true"].to_numpy(float))
    fold_all = b["fold"].to_numpy()
    rid = common.get_level_values("record_id")
    rep = common.get_level_values("repeat")

    SB, ST, FOLD = [], [], []
    for r in reps:
        sel = pd.Series(np.flatnonzero(rep == r), index=rid[rep == r]).reindex(pat0)
        if sel.isna().any():         # patient missing from a repetition -> unusable
            return None
        pos = sel.to_numpy(int)
        SB.append(sb_all[pos]); ST.append(st_all[pos]); FOLD.append(fold_all[pos])
    label = lab_all[pd.Series(np.flatnonzero(rep == reps[0]),
                              index=rid[rep == reps[0]]).reindex(pat0).to_numpy(int)]
    SB, ST, FOLD = np.array(SB), np.array(ST), np.array(FOLD)

    # A patient is kept only if usable in EVERY repetition; otherwise the r AUCs
    # would not concern the same population.
    ok = np.isfinite(SB).all(axis=0) & np.isfinite(ST).all(axis=0)
    if ok.sum() < 10 or len(np.unique(label[ok])) < 2:
        return None
    return label[ok], SB[:, ok], ST[:, ok], FOLD[:, ok]


# ============================================================
# Inference: paired bootstrap plus Nadeau-Bengio
# ============================================================
def _mean_dauc(label, SB, ST, idx=None):
    """Delta AUC computed WITHIN each repetition, then averaged. (r, n) matrices."""
    vals = []
    lab = label if idx is None else label[idx]
    if lab.sum() == 0 or lab.sum() == len(lab):
        return np.nan                     # single class: AUC undefined
    for sb, st in zip(SB, ST):
        b, t = (sb, st) if idx is None else (sb[idx], st[idx])
        a = fast_auc(lab, t) - fast_auc(lab, b)
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_dauc(label, SB, ST, seed):
    """(delta AUC, lo, hi) - PATIENT-level bootstrap, with draws shared between
    the two scores (the matching moves with the patient) AND across repetitions.

    Each draw resamples the patients once, then the delta AUC is computed per
    repetition and averaged: the CI concerns the averaged statistic, and the r
    repetitions do not inflate n."""
    rng = np.random.default_rng(seed)
    n = label.shape[0]
    point = _mean_dauc(label, SB, ST)
    vals = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        vals[i] = _mean_dauc(label, SB, ST, rng.integers(0, n, n))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def nadeau_bengio_dauc(label, SB, ST, FOLD):
    """NB-corrected t on the PER-FOLD delta AUC. Returns (t, df, p, mean_d, M).

    Under repeated CV the differences are taken per (repetition, fold), giving
    M = r*K instead of K and therefore df = M-1. The n_test/n_train ratio stays
    1/(K-1), since repeating does not change the fold sizes, hence not the
    overlap of the training sets that this term corrects for.

    A single-class fold gives an undefined AUC and is discarded, so M is the
    number of genuinely usable differences (and the df follows).
    """
    d = []
    for sb, st, fold in zip(SB, ST, FOLD):
        for k in np.unique(fold):
            m = fold == k
            a = fast_auc(label[m], st[m]) - fast_auc(label[m], sb[m])
            if np.isfinite(a):
                d.append(a)
    d = np.asarray(d, dtype=float)
    K = len(d)
    if K < 2:
        return np.nan, np.nan, np.nan, (float(d.mean()) if K else np.nan), K
    mean_d = float(d.mean())
    s2 = float(d.var(ddof=1))
    if s2 == 0:
        p = 0.0 if mean_d != 0 else 1.0
        return (np.inf if mean_d > 0 else -np.inf if mean_d < 0 else 0.0), K - 1, p, mean_d, K
    se = np.sqrt((1.0 / K + NB_TEST_TRAIN_RATIO) * s2)
    tstat = mean_d / se
    return float(tstat), K - 1, float(2.0 * student_t.sf(abs(tstat), df=K - 1)), mean_d, K


# ============================================================
# Computation of every cell
# ============================================================
def compute() -> pd.DataFrame:
    rows = []
    for ci_, (base_tag, treat_tag, name) in enumerate(CONTRASTS):
        print(f"\n[{name}]  baseline={base_tag}  treatment={treat_tag}")
        for ei, ep in enumerate(ENDPOINTS):
            n_ok = 0
            # Seed depending on the (contrast, endpoint) but NOT on the model,
            # so the models of a column share the draws. Derived from the
            # INDICES rather than hash(), which is randomised per process
            # (PYTHONHASHSEED) and would make the figure irreproducible.
            seed = SEED + 1000 * ci_ + ei
            for model in MODEL_ORDER:
                got = paired_scores(base_tag, treat_tag, ep, model)
                if got is None:
                    continue
                label, SB, ST, FOLD = got
                d, lo, hi = bootstrap_dauc(label, SB, ST, seed)
                tstat, dfree, p, mean_d, K = nadeau_bengio_dauc(label, SB, ST, FOLD)
                # Absolute AUCs: averaged over the repetitions, like the delta.
                auc_b = float(np.mean([fast_auc(label, s) for s in SB]))
                auc_t = float(np.mean([fast_auc(label, s) for s in ST]))
                rows.append({
                    "contrast": name, "baseline": base_tag, "treatment": treat_tag,
                    "endpoint": ep, "model": model,
                    # n counts PATIENTS (repetitions do not multiply it).
                    "n_paired": len(label), "n_pos": int(label.sum()),
                    "prevalence": round(float(label.mean()), 4),
                    "auc_base": auc_b, "auc_treat": auc_t,
                    "delta_auc": d, "boot_ci_lo": lo, "boot_ci_hi": hi,
                    "nb_t": tstat, "nb_df": dfree, "nb_p": p,
                    "nb_mean_fold_diff": mean_d, "nb_K": K,
                    "n_repeats": SB.shape[0],
                    "is_primary": model == PRIMARY,
                })
                n_ok += 1
            prev = next((r["prevalence"] for r in rows
                         if r["contrast"] == name and r["endpoint"] == ep), np.nan)
            print(f"  - {ep}: {n_ok} models   prevalence={prev}")
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r["ci_excl0"] = (r["boot_ci_lo"] > 0) | (r["boot_ci_hi"] < 0)
    r["nb_sig"] = r["nb_p"] < ALPHA
    return r


def matrices(r, contrast):
    s = r[r["contrast"] == contrast]

    def piv(col):
        return (s.pivot(index="model", columns="endpoint", values=col)
                .reindex(index=MODEL_ORDER, columns=ENDPOINTS).values)

    return piv("delta_auc"), piv("ci_excl0"), piv("nb_sig")


# ============================================================
# Figure (mise en page de 08)
# ============================================================
def build_figure(r):
    norm = Normalize(vmin=-VMAX, vmax=VMAX)
    ncol = 4
    nrow = int(np.ceil((len(CONTRASTS) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.3 * nrow))
    axes = axes.ravel()

    im = None
    for k, (_, _, name) in enumerate(CONTRASTS):
        ax = axes[k]
        delta, ci, nb = matrices(r, name)
        im = ax.imshow(np.asarray(delta, dtype=float), cmap=CMAP, norm=norm,
                       aspect="auto")

        ys, xs = np.arange(len(MODEL_ORDER)), np.arange(len(ENDPOINTS))
        for i in ys:
            for j in xs:
                if nb[i, j] is not None and bool(nb[i, j]):
                    ax.plot(j, i, marker="*", ms=8, color=GLYPH_COLOR,
                            mec="white", mew=0.4, zorder=3)
                elif ci[i, j] is not None and bool(ci[i, j]):
                    ax.plot(j, i, marker="o", ms=3.2, color=GLYPH_COLOR,
                            mec="white", mew=0.3, zorder=3)

        ax.set_title(name, fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(ENDPOINT_DAYS, fontsize=12, rotation=45)
        ax.set_yticks(ys)
        if k % ncol == 0:
            ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=12)
            for lab in ax.get_yticklabels():
                if lab.get_text() == MODEL_LABELS[PRIMARY]:
                    lab.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(ENDPOINTS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)

    n = len(CONTRASTS)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    legend_ax = axes[n] if n < len(axes) else axes[-1]
    cbar_host = axes[n + 1] if n + 1 < len(axes) else legend_ax

    handles = [
        Line2D([0], [0], marker="o", color="none", mfc=GLYPH_COLOR, mec="white",
               ms=6, label="bootstrap CI excludes 0"),
        Line2D([0], [0], marker="*", color="none", mfc=GLYPH_COLOR, mec="white",
               ms=11, label=r"+ $p_\mathrm{NB} < 0.05$"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=12, frameon=False,
                     title="Significance", title_fontsize=12)

    cbar_ax = cbar_host.inset_axes([0.08, 0.45, 0.86, 0.12])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label(r"$\Delta$AUC  ($>0$: the block improves)", fontsize=11)
    cb.ax.tick_params(labelsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================
# LaTeX table: cross-model consensus
# ============================================================
def build_latex(r):
    L = [r"% Generated by 10_mcid_contrasts_heatmap.py - \usepackage{booktabs}",
         r"\begin{table}[t]", r"  \centering"]
    L.append(r"  \caption{Binary MCID task (" + M09.label_desc().replace("Δ", r"$\Delta$")
             + r"): cross-model consensus on the paired $\Delta$AUC "
             r"$=$ AUC(treatment) $-$ AUC(baseline). For each contrast and each "
             r"model, the number of endpoints (out of " + str(N_EP) + r") whose "
             r"patient-level bootstrap CI excludes $0$ and, in parentheses, "
             r"whose $p_\text{NB}<0.05$ (Nadeau--Bengio, without multiplicity "
             r"correction). \emph{LogReg} is the dedicated logistic model "
             r"retrained on the exact folds of the regressors; the other rows "
             r"are the thresholded continuous regressors.}")
    L.append(r"  \label{tab:mcid_contrasts}")
    L.append(r"  \begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}")
    L.append(r"    \toprule")
    L.append(r"    Contrast & " + " & ".join(MODEL_LABELS[m] for m in MODEL_ORDER)
             + r" \\")
    L.append(r"    \midrule")
    for _, _, name in CONTRASTS:
        cells = []
        for m in MODEL_ORDER:
            s = r[(r["contrast"] == name) & (r["model"] == m)]
            n_ci, n_nb = int(s["ci_excl0"].sum()), int(s["nb_sig"].sum())
            txt = rf"{n_ci} ({n_nb})"
            if n_ci >= 5:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        L.append(rf"    {name} & " + " & ".join(cells) + r" \\")
    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")
    L.append(r"  \\[2pt] {\footnotesize Each cell: \# endpoints with CI$\neq$0 "
             r"(\# endpoints with $p_\text{NB}<0.05$), out of " + str(N_EP) +
             r" endpoints.}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"MCID CONTRASTS - paired dAUC, bootstrap CI, Nadeau-Bengio")
    print(f"target: {M09.label_desc()}   |   eval: {EVAL_SUBSET}   |   "
          f"B = {N_BOOT}")
    print("=" * 70)

    if REPLOT:
        # The CSV is the INPUT here and is not rewritten. The LaTeX table is an
        # OUTPUT derived from r, like the figure, so it is regenerated in both
        # modes.
        r = load_cached_results(OUT_DIR / "mcid_contrasts.csv",
                                "10_mcid_contrasts_heatmap.py")
    else:
        r = compute()
    if r.empty:
        raise SystemExit("No cell computable - check models_*/ and prep_*/.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not REPLOT:
        r.to_csv(OUT_DIR / "mcid_contrasts.csv", index=False)
    (OUT_DIR / "mcid_contrasts_table.tex").write_text(build_latex(r))

    fig = build_figure(r)
    fig.savefig(FIG_DIR / "F8_mcid_contrasts_dauc_heatmap.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "F8_mcid_contrasts_dauc_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    n_cells = len(MODEL_ORDER) * len(ENDPOINTS)
    print(f"\nConsensus (CI excludes 0 / {n_cells}; p_NB<0.05 / {n_cells}) "
          f"per contrast:")
    for _, _, name in CONTRASTS:
        s = r[r["contrast"] == name]
        print(f"  {name:<26} CI!=0: {int(s['ci_excl0'].sum()):2d}/{n_cells}   "
              f"p_NB<0.05: {int(s['nb_sig'].sum()):2d}/{n_cells}   "
              f"median |dAUC|: {s['delta_auc'].abs().median():.3f}")
    print("\nPrevalence of the positive class per endpoint:")
    for ep in ENDPOINTS:
        s = r[r["endpoint"] == ep]
        if not s.empty:
            print(f"  {ep:>8}: {s['prevalence'].min():.3f}-"
                  f"{s['prevalence'].max():.3f}  (n = {int(s['n_paired'].max())})")
    print(f"\n[done] Written:")
    print(f"   {FIG_DIR}/F8_mcid_contrasts_dauc_heatmap.png / .pdf")
    if not REPLOT:
        print(f"   {OUT_DIR}/mcid_contrasts.csv")
    print(f"   {OUT_DIR}/mcid_contrasts_table.tex")


if __name__ == "__main__":
    main()
