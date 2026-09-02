"""
10b_mcid_contrasts_heatmap_naive.py
===================================
NAIVE twin of 10_mcid_contrasts_heatmap.py - a methodological demonstration
figure on the binary MCID task, exactly as 08b is to 08.

Same material (mcid_contrasts.csv plus the same OOF scores, contrasts, models
and delta AUC), but the inference is the one most commonly met:

  - NO patient-level bootstrap CI;
  - NO Nadeau-Bengio variance correction;
  - NO multiplicity correction;
  - significance declared IF AND ONLY IF p < 0.05.

Two naive regimes are available (NAIVE_TEST):

  "delong" (default) - paired DeLong test between the two correlated AUCs (same
    patients, same labels). This is THE standard AUC comparison test in clinical
    research, and the most optimistic here: its variance describes only the
    PATIENT SAMPLING randomness, treating the two models as fixed. Yet the two
    AUCs compared come from models LEARNED on the same data resampled by the CV,
    and the question "does this feature block help?" concerns that learning
    randomness, which DeLong ignores entirely.

  "fold" - RAW Student t on the per-fold delta AUC. This is exactly the test
    that NB corrects, and it is recomputed from mcid_contrasts.csv without
    re-reading the OOF files:
        t_naive = t_NB * sqrt(1 + K * n_test/n_train)

As in 08b, the cells declared significant that the CORRECTED test does not
retain are outlined (REFERENCE_CRIT = "nb" by default: same test, same estimand,
with only the variance correction changing, so every outline is attributable to
its removal).

Prerequisite: 10_mcid_contrasts_heatmap.py must have been run, since it provides
mcid_contrasts.csv with delta_auc, boot_ci_*, nb_t/nb_p/nb_K. Inputs are
read-only; outputs use the F8b_ prefix.

Usage:
    python 10b_mcid_contrasts_heatmap_naive.py
    MCID_SKIP_LOGREG=1 python 10b_mcid_contrasts_heatmap_naive.py
    python 10b_mcid_contrasts_heatmap_naive.py --replot   # REDRAWS the figure
                                        # from mcid_contrasts_naive.csv. More
                                        # useful here than elsewhere: with
                                        # NAIVE_TEST="delong", load() calls
                                        # M10.paired_scores for EVERY cell and
                                        # therefore refits the logistic models,
                                        # which dominate the runtime.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import norm
from scipy.stats import t as student_t

from config import N_FOLDS
from utils import load_cached_results, replot_requested

# --replot: read mcid_contrasts_naive.csv back and run only the figure and table.
REPLOT = replot_requested()


# ------------------------------------------------------------
# Reuse of 10 (and hence of 09): the contrasts, models, endpoints, paired OOF
# loading, binary target and scores are all taken as is.
# ------------------------------------------------------------
def _load(name: str, mod_name: str):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


M10 = _load("10_mcid_contrasts_heatmap.py", "mcid10")

plt.rc("font", family="serif")

# ============================================================
# CONFIGURATION
# ============================================================
NAIVE_TEST = "delong"           # "delong" | "fold"

# Corrected criterion used to qualify an effect as spuriously declared.
#   "nb"    (default) p_NB < 0.05, the same test as the "fold" regime but with
#           the variance correction. A constant-estimand comparison.
#   "ci"    patient-level bootstrap CI excluding 0. CAUTION: the bootstrap also
#           resamples PATIENTS, so it does not correct the learning randomness
#           either; it is on the same side as DeLong, not the other.
#   "ci+nb" both.
REFERENCE_CRIT = "nb"
HIGHLIGHT_FRAGILE = True
# The map is a red/blue diverging one, so only an outline OFF that axis stands
# out; a red frame was lost in the dark red cells. Green stays legible on both
# poles and on the neutral centre.
FRAGILE_COLOR = "#00c853"

ALPHA = M10.ALPHA
VMAX = M10.VMAX
CMAP = M10.CMAP
# Colour of the significance glyphs. A vivid green sits off the red/blue
# diverging axis of the colormap, so it stays legible both on saturated cells
# and on near-white ones, where black was lost. Set to "black" for a plain
# rendering.
GLYPH_COLOR = "#00c853"
CONTRASTS = M10.CONTRASTS
MODEL_ORDER = M10.MODEL_ORDER
MODEL_LABELS = M10.MODEL_LABELS
PRIMARY = M10.PRIMARY
ENDPOINTS = M10.ENDPOINTS
ENDPOINT_DAYS = M10.ENDPOINT_DAYS
N_EP = M10.N_EP
NB_TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)

CSV = M10.OUT_DIR / "mcid_contrasts.csv"
FIG_DIR = M10.FIG_DIR
OUT_DIR = M10.OUT_DIR


# ============================================================
# Paired DeLong test (Sun & Xu 2014, fast midrank formulation)
# ============================================================
def _midrank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties get their midrank), 1-based."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_repeats(label, A, B):
    """DeLong applied WITHIN each repetition, then aggregated. A, B are (r, n).

    The delta AUC is the mean of the per-repetition delta AUCs, the same
    estimand as the `delta_auc` column of mcid_contrasts.csv, which the guard
    below checks.

    z is the MEAN of the per-repetition z values, not their sqrt(r)
    combination. This is deliberate: the script depicts the NAIVE regime, that
    of an analyst unaware of having r partitions. Granting it the power gain of
    repeated CV would distort the demonstration, so the TYPICAL significance of
    one partition is reported, merely stabilised. The r repetitions also share
    every patient, so combining them in sqrt(r) would assume an independence
    that does not exist.

    At r=1 this is strictly identical to delong_test().
    """
    ds, zs = [], []
    for a, b in zip(A, B):
        d, z, _ = delong_test(label, a, b)
        if np.isfinite(d):
            ds.append(d)
        if np.isfinite(z):
            zs.append(z)
    if not ds:
        return np.nan, np.nan, np.nan
    d = float(np.mean(ds))
    if not zs:
        return d, np.nan, np.nan
    z = float(np.mean(zs))
    return d, z, float(2.0 * norm.sf(abs(z)))


def delong_test(label: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """(delta AUC = AUC(a) - AUC(b), z, two-sided p) by the paired DeLong test.

    The structural components V10/V01 give the covariance matrix of the two
    correlated AUCs; var(delta AUC) = L*S*L^T with L = [1, -1]. The correlation
    induced by BOTH scores concerning the SAME patients is therefore accounted
    for - but the learning randomness is not.
    """
    label = np.asarray(label).astype(int)
    pos = label == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    # Scores ordered with the positives first, as the Sun & Xu formulation expects.
    preds = np.vstack([np.concatenate([score_a[pos], score_a[~pos]]),
                       np.concatenate([score_b[pos], score_b[~pos]])])
    k = preds.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2.0 * m)
    v01 = (tz[:, :m] - tx) / n                 # composantes des positifs
    v10 = 1.0 - (tz[:, m:] - ty) / m           # components of the negatives
    cov = np.cov(v01) / m + np.cov(v10) / n
    L = np.array([1.0, -1.0])
    var = float(L @ cov @ L)
    d = float(aucs[0] - aucs[1])
    if not np.isfinite(var) or var <= 0:
        return d, np.nan, (0.0 if d != 0 else 1.0)
    z = d / np.sqrt(var)
    return d, float(z), float(2.0 * norm.sf(abs(z)))


def fold_naive_p(nb_t, nb_K):
    """p of the RAW Student t on the per-fold delta AUC, rebuilt from t_NB."""
    if pd.isna(nb_t) or pd.isna(nb_K) or nb_K < 2:
        return np.nan
    K = int(nb_K)
    if not np.isfinite(nb_t):
        return 0.0
    t_naive = float(nb_t) * np.sqrt(1.0 + K * NB_TEST_TRAIN_RATIO)
    return float(2.0 * student_t.sf(abs(t_naive), df=K - 1))


# ============================================================
# Loading and naive p per cell
# ============================================================
def load():
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found - run "
                         f"10_mcid_contrasts_heatmap.py first.")
    r = pd.read_csv(CSV)

    if NAIVE_TEST == "delong":
        ps, zs, checks = [], [], []
        base_by_name = {name: (b, t) for b, t, name in CONTRASTS}
        for _, row in r.iterrows():
            btag, ttag = base_by_name.get(row["contrast"], (None, None))
            got = (M10.paired_scores(btag, ttag, row["endpoint"], row["model"])
                   if btag else None)
            if got is None:
                ps.append(np.nan); zs.append(np.nan); checks.append(np.nan)
                continue
            label, SB, ST, _ = got
            d, z, p = delong_repeats(label, ST, SB)   # a = treatment, b = baseline
            ps.append(p); zs.append(z); checks.append(d - row["delta_auc"])
        r["naive_p"], r["naive_z"] = ps, zs
        # Guard: the DeLong delta AUC must reproduce the one in the CSV (same
        # estimand, independent computations). A gap above 1e-6 means the
        # matching diverged.
        mx = np.nanmax(np.abs(checks)) if len(checks) else 0.0
        print(f"  check, DeLong dAUC vs mcid_contrasts.csv: "
              f"max gap = {mx:.2e}")
        if mx > 1e-6:
            print("  ! non-negligible gap - check the matching of the OOF files.")
    else:
        r["naive_p"] = [fold_naive_p(t, k) for t, k in zip(r["nb_t"], r["nb_K"])]
        r["naive_z"] = r["nb_t"] * np.sqrt(1.0 + r["nb_K"] * NB_TEST_TRAIN_RATIO)

    r["naive_sig"] = r["naive_p"] < ALPHA
    r["rigorous_sig"] = {"ci+nb": r["ci_excl0"] & r["nb_sig"],
                         "ci": r["ci_excl0"]}.get(REFERENCE_CRIT, r["nb_sig"])
    r["fragile"] = r["naive_sig"] & ~r["rigorous_sig"]
    return r


def matrices(r, contrast):
    s = r[r["contrast"] == contrast]

    def piv(col):
        return (s.pivot(index="model", columns="endpoint", values=col)
                .reindex(index=MODEL_ORDER, columns=ENDPOINTS).values)

    return piv("delta_auc"), piv("naive_sig"), piv("fragile")


# ============================================================
# Figure
# ============================================================
def build_figure(r):
    norm_ = Normalize(vmin=-VMAX, vmax=VMAX)
    ncol = 4
    nrow = int(np.ceil((len(CONTRASTS) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.3 * nrow))
    axes = axes.ravel()

    im = None
    for k, (_, _, name) in enumerate(CONTRASTS):
        ax = axes[k]
        delta, sig, frag = matrices(r, name)
        im = ax.imshow(np.asarray(delta, dtype=float), cmap=CMAP, norm=norm_,
                       aspect="auto")

        ys, xs = np.arange(len(MODEL_ORDER)), np.arange(len(ENDPOINTS))
        for i in ys:
            for j in xs:
                if sig[i, j] is None or not bool(sig[i, j]):
                    continue
                ax.plot(j, i, marker="*", ms=8, color=GLYPH_COLOR,
                        mec="white", mew=0.4, zorder=3)
                if HIGHLIGHT_FRAGILE and bool(frag[i, j]):
                    w = 0.86       # inset, so adjacent outlines stay distinct
                    ax.add_patch(Rectangle((j - w / 2, i - w / 2), w, w,
                                           fill=False, ec=FRAGILE_COLOR, lw=1.9,
                                           zorder=4))

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

    test_lbl = "paired DeLong" if NAIVE_TEST == "delong" else "raw per-fold t"
    handles = [Line2D([0], [0], marker="*", color="none", mfc=GLYPH_COLOR, mec="white",
                      ms=11, label=rf"$p < 0.05$ ({test_lbl})")]
    if HIGHLIGHT_FRAGILE:
        ref = {"ci+nb": "CI + $p_\\mathrm{NB}$",
               "ci": "bootstrap CI"}.get(REFERENCE_CRIT, "$p_\\mathrm{NB} < 0.05$")
        handles.append(
            Line2D([0], [0], marker="s", color="none", mfc="none",
                   mec=FRAGILE_COLOR, mew=1.9, ms=10,
                   label=f"not retained by the\ncorrected test ({ref})"))
    legend_ax.legend(handles=handles, loc="center", fontsize=11, frameon=False,
                     title="Declared significance", title_fontsize=12)

    cbar_ax = cbar_host.inset_axes([0.08, 0.45, 0.86, 0.12])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label(r"$\Delta$AUC  ($>0$: the block improves)", fontsize=11)
    cb.ax.tick_params(labelsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================
# Table LaTeX
# ============================================================
def build_latex(r):
    test_tex = (r"paired DeLong test" if NAIVE_TEST == "delong"
                else r"raw Student $t$ on the per-fold $\Delta$AUC")
    ref_tex = {"ci+nb": r"bootstrap CI $\neq 0$ \emph{and} $p_\text{NB}<0.05$",
               "ci": r"patient-level bootstrap CI excluding $0$"}.get(
                   REFERENCE_CRIT, r"$p_\text{NB}<0.05$, the same estimand with "
                                   r"the variance correction")
    L = [r"% Generated by 10b_mcid_contrasts_heatmap_naive.py - \usepackage{booktabs}",
         r"\begin{table}[t]", r"  \centering"]
    L.append(r"  \caption{Binary MCID task, naive inference: " + test_tex +
             r", with significance declared at $p<0.05$, without a bootstrap CI, "
             r"without the Nadeau--Bengio variance correction and without a "
             r"multiplicity correction. For each contrast and each model: the "
             r"number of endpoints (out of " + str(N_EP) + r") declared "
             r"significant and, in parentheses, those that the corrected "
             r"analysis (" + ref_tex + r") does not retain --- that is, the "
             r"spuriously declared effects.}")
    L.append(r"  \label{tab:mcid_contrasts_naive}")
    L.append(r"  \begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}")
    L.append(r"    \toprule")
    L.append(r"    Contrast & " + " & ".join(MODEL_LABELS[m] for m in MODEL_ORDER)
             + r" \\")
    L.append(r"    \midrule")
    for _, _, name in CONTRASTS:
        cells = []
        for mdl in MODEL_ORDER:
            s = r[(r["contrast"] == name) & (r["model"] == mdl)]
            n_sig, n_frag = int(s["naive_sig"].sum()), int(s["fragile"].sum())
            txt = rf"{n_sig} ({n_frag})"
            if n_frag >= 1:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        L.append(rf"    {name} & " + " & ".join(cells) + r" \\")
    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")
    L.append(r"  \\[2pt] {\footnotesize Each cell: \# endpoints with $p<0.05$ "
             r"(\# not retained by the corrected analysis), out of " + str(N_EP) +
             r" endpoints." +
             (r" DeLong describes only the patient sampling randomness, "
              r"treating the two models as fixed."
              if NAIVE_TEST == "delong" else
              r" With $K=" + str(N_FOLDS) + r"$ folds, $t_\text{naive} = "
              + f"{np.sqrt(1 + N_FOLDS * NB_TEST_TRAIN_RATIO):.2f}"
              + r"\,t_\text{NB}$.") + r"}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("MCID CONTRASTS - NAIVE regime (dAUC, p<0.05 alone)")
    print(f"test: {NAIVE_TEST}   |   corrected reference: {REFERENCE_CRIT}")
    print("=" * 70)
    if REPLOT:
        # The naive CSV is the INPUT here (it already carries naive_p, naive_z
        # and fragile), so load() is bypassed, and with it the DeLong calls and
        # the logistic fits.
        r = load_cached_results(OUT_DIR / "mcid_contrasts_naive.csv",
                                "10b_mcid_contrasts_heatmap_naive.py")
    else:
        r = load()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not REPLOT:
        r.to_csv(OUT_DIR / "mcid_contrasts_naive.csv", index=False)
    (OUT_DIR / "mcid_contrasts_naive_table.tex").write_text(build_latex(r))

    fig = build_figure(r)
    fig.savefig(FIG_DIR / "F8b_mcid_contrasts_dauc_heatmap_naive.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "F8b_mcid_contrasts_dauc_heatmap_naive.pdf",
                bbox_inches="tight")
    plt.close(fig)

    n_cells = len(MODEL_ORDER) * len(ENDPOINTS)
    ref = {"ci+nb": "CI+NB", "ci": "CI"}.get(REFERENCE_CRIT, "NB")
    print(f"\nper contrast, over {n_cells} cells:")
    for _, _, name in CONTRASTS:
        s = r[r["contrast"] == name]
        print(f"  {name:<26} p<0.05: {int(s['naive_sig'].sum()):2d}/{n_cells}   "
              f"(ref. {ref}: {int(s['rigorous_sig'].sum()):2d}/{n_cells})   "
              f"spurious: {int(s['fragile'].sum()):2d}")
    tot = len(r)
    print(f"\n  TOTAL  p<0.05: {int(r['naive_sig'].sum())}/{tot}   "
          f"ref. {ref}: {int(r['rigorous_sig'].sum())}/{tot}   "
          f"bootstrap CI: {int(r['ci_excl0'].sum())}/{tot}   "
          f"spuriously declared: {int(r['fragile'].sum())}")
    dvh = r[r["contrast"].isin([n for _, _, n in CONTRASTS[:4]])]
    print(f"  DVH BLOCK: p<0.05 {int(dvh['naive_sig'].sum())}/"
          f"{len(dvh)}   vs NB {int(dvh['nb_sig'].sum())}/{len(dvh)}")
    print(f"\n[done] Written:")
    print(f"   {FIG_DIR}/F8b_mcid_contrasts_dauc_heatmap_naive.png / .pdf")
    if not REPLOT:
        print(f"   {OUT_DIR}/mcid_contrasts_naive.csv")
    print(f"   {OUT_DIR}/mcid_contrasts_naive_table.tex")


if __name__ == "__main__":
    main()