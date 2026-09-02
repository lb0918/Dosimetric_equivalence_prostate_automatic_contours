"""
08b_all_models_heatmap_naive.py
===============================
NAIVE twin of 08_all_models_heatmap.py - a methodological demonstration figure.

Same material (contrasts.csv, same learners, same endpoints, same delta RMSE),
but the inference is the one most commonly met in the applied ML literature:

  - NO patient-level bootstrap CI;
  - NO Nadeau-Bengio variance correction;
  - NO multiplicity correction;
  - significance declared IF AND ONLY IF p < 0.05.

Two naive regimes are available (NAIVE_TEST), both common:

  "patient" (default) - paired Student t on the per-patient squared errors,
    n-1 degrees of freedom. Each patient counts as an independent replication,
    whereas the question asked ("does block X help?") concerns the randomness of
    LEARNING: a single model was learned per fold, and the number of patients
    says nothing about the variability of that learning.

  "fold" - RAW Student t on the per-fold delta RMSE. This is exactly the test
    that NB corrects: the CV reuses the same training data from one fold to the
    next, so S^2_d/K underestimates the variance of the mean. The link is
    analytic, requiring no re-reading of the OOF files:
        se_NB    = sqrt((1/K + n_test/n_train) * S^2_d)
        se_naive = sqrt(S^2_d / K)
        t_naive  = t_NB * sqrt(1 + K * n_test/n_train)

Note on the reference criterion: the bootstrap CI of 04_contrasts resamples
PATIENTS, so it does not correct the learning randomness either. Comparing the
naive test against it would therefore not isolate the effect of the variance
correction. Hence REFERENCE_CRIT = "nb" by default: the comparison is made at a
constant estimand, with only the variance correction changing.

Outputs use the F6b_ prefix, distinct from those of 08. Inputs are read-only
(contrasts.csv, plus the OOF files under models_*/ in "patient" mode); only the
figures and contrasts directories are written.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import t as student_t

from config import (PROJECT_DIR, MODELS_ROOT, ENDPOINTS as CFG_ENDPOINTS,
                    N_FOLDS, with_suffix)

plt.rc("font", family="serif")

CSV = PROJECT_DIR / with_suffix("contrasts") / "contrasts.csv"
FIG_DIR = PROJECT_DIR / with_suffix("figures")
TEX_DIR = PROJECT_DIR / with_suffix("contrasts")
ALPHA = 0.05
VMAX = 1.5                          # symmetric bound of the map (|delta RMSE| clipped)
CMAP = "RdBu_r"                     # CVD-safe diverging: red > 0, blue < 0
# Colour of the significance glyphs. A vivid green sits off the red/blue
# diverging axis of the colormap, so it stays legible both on saturated cells
# and on near-white ones, where black was lost. Set to "black" for a plain
# rendering.
GLYPH_COLOR = "#00c853"

# n_test/n_train ratio of the outer CV (same as 04_contrasts.NB_TEST_TRAIN_RATIO).
NB_TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)

# Which naive test declares significance?
#   "patient" (default) paired Student t on the PER-PATIENT squared errors
#             (df = n-1). This is the most widespread practice, a paired t-test
#             on the test-set errors, and the most optimistic: it treats each
#             patient as an independent replication of TRAINING, whereas the
#             only randomness that matters for "does block X help?" is that of
#             the learned model. It re-reads the OOF files under models_*/, so
#             it is slower, but it is the regime worth illustrating.
#   "fold"    RAW Student t on the per-fold delta RMSE (df = K-1): exactly the
#             test that NB corrects, recomputed from contrasts.csv without I/O.
NAIVE_TEST = "patient"

# Outline the cells declared significant by the naive test but NOT retained by
# the reference analysis: these are the spurious effects that common practice
# produces. Set to False for a purely naive figure, with no trace of the
# corrected inference.
HIGHLIGHT_FRAGILE = True
# The map is a red/blue diverging one, so only an outline OFF that axis stands
# out; a red frame was lost in the dark red cells. Green stays legible on both
# poles and on the neutral centre.
FRAGILE_COLOR = "#00c853"

# Reference criterion used to qualify an effect as spuriously declared:
#   "nb"    (default) p_NB < 0.05, i.e. the SAME test as the naive regime but
#           with the variance correction. This is the like-for-like comparison:
#           only the correction changes, so every outlined star is attributable
#           to its removal rather than to a change of estimand.
#   "ci"    patient-level bootstrap CI excluding 0. CAUTION: the bootstrap CI
#           resamples PATIENTS, so it does not correct the learning randomness
#           either, and comparing against it demonstrates nothing.
#   "ci+nb" CI excluding 0 AND p_NB < 0.05 - the strictest criterion.
REFERENCE_CRIT = "nb"

CONTRAST_ORDER = [
    ("M1_manual − M0",                          "Manual DVH"),
    ("M1_det_clin0977 − M0",                    "Auto det. DVH"),
    ("M1_bayes_clin0977 − M0",                  "Bayesian DVH (mean)"),
    ("M1_bayes+var_clin0977 − M0",              "Bayesian DVH (mean+var)"),
    ("Pre-tx total IPSS (obstr+irrit ablation)", "Pre-tx total IPSS"),
    ("Age (ablation)",                          "Pre-tx age"),
]
ALGO_ORDER = ["elasticnet", "rf", "xgboost", "catboost", "mlp"]
ALGO_LABELS = {"elasticnet": "ElasticNet", "rf": "RF",
               "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP"}
PRIMARY = "elasticnet"
ENDPOINTS = [f"y_{d}d" for d in sorted(CFG_ENDPOINTS)]
ENDPOINT_DAYS = [e.strip("y_").rstrip("d") for e in ENDPOINTS]
N_EP = len(ENDPOINTS)


# ============================================================
# Loading and naive test
# ============================================================
def naive_p(nb_t, nb_K, ratio=NB_TEST_TRAIN_RATIO):
    """Two-sided p of the RAW Student t-test on the per-fold delta RMSE.

    Reconstructed from the stored NB-corrected t: both statistics share the
    numerator (mean_d) and S^2_d, only the standard error differing, hence the
    constant scale factor sqrt(1 + K*ratio).
    """
    if pd.isna(nb_t) or pd.isna(nb_K) or nb_K < 2:
        return np.nan
    K = int(nb_K)
    if not np.isfinite(nb_t):
        # S^2_d = 0 in 04_contrasts (t = +/-inf): the naive test also gives p -> 0.
        return 0.0
    t_naive = float(nb_t) * np.sqrt(1.0 + K * ratio)
    return float(2.0 * student_t.sf(abs(t_naive), df=K - 1))


_OOF_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}


def _load_oof(tag: str, endpoint: str) -> pd.DataFrame | None:
    """Cached OOF file of one run and endpoint (same paths as 04_contrasts)."""
    key = (tag, endpoint)
    if key not in _OOF_CACHE:
        path = (MODELS_ROOT / f"models_{with_suffix(tag)}" / endpoint
                / "oof_predictions.csv")
        if path.exists():
            df = pd.read_csv(path)
            df["record_id"] = df["record_id"].astype(str).str.strip()
            if "source" in df.columns:
                df = df[df["source"] == "real"]
            # This script indexes by record_id alone, so it assumes ONE
            # prediction per patient. On a repeated-CV OOF file (a `repeat`
            # column), that index is duplicated and the downstream joins would
            # silently become a cartesian product. Fail loudly rather than emit
            # wrong numbers; adapt the script to the (record_id, repeat) key
            # before using it on such runs.
            if "repeat" in df.columns and df["repeat"].nunique() > 1:
                raise NotImplementedError(
                    f"Repeated-CV OOF ({df['repeat'].nunique()} repetitions) for "
                    f"{tag}/{endpoint}: 08b only matches on record_id. "
                    f"Adapt it to the (record_id, repeat) key before using it.")
            _OOF_CACHE[key] = df.set_index("record_id")
        else:
            _OOF_CACHE[key] = None
    return _OOF_CACHE[key]


def patient_p(base_tag, treat_tag, endpoint, algo):
    """Paired Student t on the PER-PATIENT squared errors.

    d_i = (y_i - yhat_i^base)^2 - (y_i - yhat_i^treat)^2, tested against 0 with
    n-1 degrees of freedom. No grouping by fold and no correction: the usual
    naive regime. Returns (p, n), or (nan, 0) when the pair cannot be matched.
    """
    b, t = _load_oof(base_tag, endpoint), _load_oof(treat_tag, endpoint)
    if b is None or t is None or algo not in b.columns or algo not in t.columns:
        return np.nan, 0
    common = b.index.intersection(t.index)
    if len(common) == 0:
        return np.nan, 0
    bb, tt = b.loc[common], t.loc[common]
    y = bb["y_true"].values
    d = (y - bb[algo].values) ** 2 - (y - tt[algo].values) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan, n
    s = d.std(ddof=1)
    if s == 0:
        return (0.0 if d.mean() != 0 else 1.0), n
    tstat = d.mean() / (s / np.sqrt(n))
    return float(2.0 * student_t.sf(abs(tstat), df=n - 1)), n


def load():
    r = pd.read_csv(CSV)
    if NAIVE_TEST == "patient":
        out = [patient_p(b, t, e, a) for b, t, e, a in
               zip(r["baseline"], r["treatment"], r["endpoint"], r["algo"])]
        r["naive_p"] = [p for p, _ in out]
        r["naive_n"] = [n for _, n in out]
    else:
        r["naive_p"] = [naive_p(t, k) for t, k in zip(r["nb_t"], r["nb_K"])]
        r["naive_n"] = r["nb_K"]
    r["naive_sig"] = r["naive_p"] < ALPHA
    # Reference criterion (as in 08), kept only to count the spurious effects.
    r["ci_excl0"] = (r["boot_ci_lo"] > 0) | (r["boot_ci_hi"] < 0)
    r["nb_sig"] = r["nb_p"] < ALPHA
    r["rigorous_sig"] = {"ci+nb": r["ci_excl0"] & r["nb_sig"],
                         "ci": r["ci_excl0"]}.get(REFERENCE_CRIT, r["nb_sig"])
    r["fragile"] = r["naive_sig"] & ~r["rigorous_sig"]
    return r


def matrices(r, contrast):
    """(delta, naive_sig, fragile): learners x endpoints matrices."""
    s = r[r["contrast"] == contrast]

    def piv(col):
        return (s.pivot(index="algo", columns="endpoint", values=col)
                .reindex(index=ALGO_ORDER, columns=ENDPOINTS).values)

    return piv("delta_rmse"), piv("naive_sig"), piv("fragile")


# ============================================================
# Figure
# ============================================================
def build_figure(r):
    norm = Normalize(vmin=-VMAX, vmax=VMAX)
    ncol = 4
    nrow = int(np.ceil((len(CONTRAST_ORDER) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.1 * nrow))
    axes = axes.ravel()

    im = None
    for k, (key, name) in enumerate(CONTRAST_ORDER):
        ax = axes[k]
        delta, sig, frag = matrices(r, key)
        im = ax.imshow(delta, cmap=CMAP, norm=norm, aspect="auto")

        ys, xs = np.arange(len(ALGO_ORDER)), np.arange(len(ENDPOINTS))
        for i in ys:
            for j in xs:
                if not bool(sig[i, j]):
                    continue
                ax.plot(j, i, marker="*", ms=8, color=GLYPH_COLOR,
                        mec="white", mew=0.4, zorder=3)
                if HIGHLIGHT_FRAGILE and bool(frag[i, j]):
                    # Slightly inset, so two adjacent outlined cells stay
                    # distinct instead of merging into a single block.
                    w = 0.86
                    ax.add_patch(Rectangle((j - w / 2, i - w / 2), w, w,
                                           fill=False, ec=FRAGILE_COLOR, lw=1.9,
                                           zorder=4))

        ax.set_title(name, fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(ENDPOINT_DAYS, fontsize=12, rotation=45)
        ax.set_yticks(ys)
        if k % ncol == 0:
            ax.set_yticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], fontsize=12)
            for lab in ax.get_yticklabels():
                if lab.get_text() == ALGO_LABELS[PRIMARY]:
                    lab.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(ENDPOINTS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ALGO_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)

    n = len(CONTRAST_ORDER)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    legend_ax = axes[n] if n < len(axes) else axes[-1]
    cbar_host = axes[n + 1] if n + 1 < len(axes) else legend_ax

    test_lbl = ("patient-level paired t" if NAIVE_TEST == "patient"
                else "raw per-fold t")
    handles = [
        Line2D([0], [0], marker="*", color="none", mfc=GLYPH_COLOR, mec="white",
               ms=11, label=rf"$p < 0.05$ ({test_lbl})"),
    ]
    if HIGHLIGHT_FRAGILE:
        ref = {"ci+nb": "CI + $p_\\mathrm{NB}$",
               "ci": "bootstrap CI"}.get(REFERENCE_CRIT,
                                         "$p_\\mathrm{NB} < 0.05$")
        handles.append(
            Line2D([0], [0], marker="s", color="none", mfc="none",
                   mec=FRAGILE_COLOR, mew=1.9, ms=10,
                   label=f"not retained by the\ncorrected test ({ref})"))
    legend_ax.legend(handles=handles, loc="center", fontsize=11,
                     frameon=False, title="Declared significance",
                     title_fontsize=12)

    cbar_ax = cbar_host.inset_axes([0.08, 0.45, 0.86, 0.12])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.ax.tick_params(labelsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================
# LaTeX table: how many effects does the naive regime declare?
# ============================================================
def build_latex(r):
    L = []
    L.append(r"% Generated by 08b_all_models_heatmap_naive.py - \usepackage{booktabs}")
    L.append(r"\begin{table}[t]")
    L.append(r"  \centering")
    L.append(r"  \caption{Naive inference on the same material: " +
             (r"paired Student $t$ on the per-patient squared errors"
              if NAIVE_TEST == "patient" else
              r"raw Student $t$ on the per-fold $\Delta$RMSE") +
             r", with significance declared at "
             r"$p<0.05$, without a bootstrap CI, without the Nadeau--Bengio "
             r"variance correction and without a multiplicity correction. For "
             r"each contrast and each learner: the number of endpoints (out of "
             + str(N_EP) +
             r") declared significant and, in parentheses, those that the "
             r"corrected analysis (" +
             {"ci+nb": r"bootstrap CI $\neq 0$ \emph{and} $p_\text{NB}<0.05$",
              "ci": r"patient-level bootstrap CI excluding $0$"}.get(
                 REFERENCE_CRIT, r"$p_\text{NB}<0.05$, the same test with the "
                                 r"variance correction") +
             r") does not retain --- that is, the spuriously declared effects.}")
    L.append(r"  \label{tab:all_models_naive}")
    algos_tex = " & ".join(ALGO_LABELS[a] for a in ALGO_ORDER)
    L.append(r"  \begin{tabular}{l" + "c" * len(ALGO_ORDER) + "}")
    L.append(r"    \toprule")
    L.append(r"    Contrast & " + algos_tex + r" \\")
    L.append(r"    \midrule")
    for key, name in CONTRAST_ORDER:
        cells = []
        for a in ALGO_ORDER:
            s = r[(r["contrast"] == key) & (r["algo"] == a)]
            n_sig, n_frag = int(s["naive_sig"].sum()), int(s["fragile"].sum())
            txt = rf"{n_sig} ({n_frag})"
            if n_frag >= 1:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        L.append(rf"    {name} & " + " & ".join(cells) + r" \\")
    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")
    note = (r"Each patient is treated as an independent replication "
            r"($n-1$ df, $n$ = matched patients), whereas the relevant "
            r"randomness is that of learning."
            if NAIVE_TEST == "patient" else
            r"With $K=" + str(N_FOLDS) + r"$ folds, $t_\text{naive} = "
            r"\sqrt{1 + K\,n_\text{test}/n_\text{train}}\;t_\text{NB} = "
            + f"{np.sqrt(1 + N_FOLDS * NB_TEST_TRAIN_RATIO):.2f}"
            + r"\,t_\text{NB}$.")
    L.append(r"  \\[2pt] {\footnotesize Each cell: \# endpoints with $p<0.05$ "
             r"(\# not retained by the reference analysis), out of " + str(N_EP) +
             r" endpoints. " + note + r"}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# ============================================================
# MAIN
# ============================================================
def main():
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found - run 04_contrasts.py first.")
    r = load()

    FIG_DIR.mkdir(exist_ok=True)
    fig = build_figure(r)
    fig.savefig(FIG_DIR / "F6b_all_models_delta_heatmap_naive.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "F6b_all_models_delta_heatmap_naive.pdf",
                bbox_inches="tight")
    plt.close(fig)

    (TEX_DIR / "all_models_naive_table.tex").write_text(build_latex(r))

    n_cells = len(ALGO_ORDER) * len(ENDPOINTS)
    desc = ("per-patient paired t" if NAIVE_TEST == "patient" else
            f"raw per-fold t; t_naive = "
            f"{np.sqrt(1 + N_FOLDS * NB_TEST_TRAIN_RATIO):.2f}*t_NB")
    ref = {"ci+nb": "CI+NB", "ci": "CI"}.get(REFERENCE_CRIT, "NB")
    print(f"NAIVE regime (p<0.05 on the {desc})")
    print(f"per contrast, over {n_cells} cells:")
    for key, name in CONTRAST_ORDER:
        s = r[(r["contrast"] == key) & (r["algo"].isin(ALGO_ORDER))]
        print(f"  {name:<26} p<0.05: {int(s['naive_sig'].sum()):2d}/{n_cells}   "
              f"(ref. {ref}: {int(s['rigorous_sig'].sum()):2d}/{n_cells})   "
              f"spurious: {int(s['fragile'].sum()):2d}")
    sub = r[r["algo"].isin(ALGO_ORDER) &
            r["contrast"].isin([k for k, _ in CONTRAST_ORDER])]
    tot = len(sub)
    print(f"\n  TOTAL  p<0.05: {int(sub['naive_sig'].sum())}/{tot}   "
          f"ref.: {int(sub['rigorous_sig'].sum())}/{tot}   "
          f"spuriously declared: {int(sub['fragile'].sum())}")
    print(f"\n[done] Written:")
    print(f"   {FIG_DIR}/F6b_all_models_delta_heatmap_naive.png / .pdf")
    print(f"   {TEX_DIR}/all_models_naive_table.tex")


if __name__ == "__main__":
    main()
