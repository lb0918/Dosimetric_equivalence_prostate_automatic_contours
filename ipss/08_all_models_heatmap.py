"""
08_all_models_heatmap.py
========================
Overview of contrasts.csv across the retained learners (ALGO_ORDER): the paired
delta RMSE for each (contrast x endpoint x learner).

  - Figure: DIVERGING heatmap of delta RMSE (RdBu, centred on 0, with a shared
    symmetric scale so panels stay comparable), one small multiple per contrast,
    rows = learners, columns = endpoints. COLOUR encodes magnitude and sign;
    SIGNIFICANCE is a separate glyph, never colour alone:
        o  patient-level bootstrap CI excluding 0 (reference inference)
        *  and in addition Nadeau-Bengio p < 0.05 (variance-corrected, no Holm)
  - LaTeX table: cross-model consensus - for each contrast and learner, the
    number of endpoints whose CI excludes 0 and, in parentheses, whose
    NB p < 0.05.

RdBu is a ColorBrewer diverging scheme that is safe for colour vision
deficiency (two poles plus a neutral centre). The CSV is read-only; only the
figures and contrasts directories are written.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from config import PROJECT_DIR, ENDPOINTS as CFG_ENDPOINTS, with_suffix

plt.rc("font", family="serif")

# The contrasts and figures directories are suffixed by the active run
# (PIPE_TAG_SUFFIX), so a targeted run reads and writes its own directories
# without overwriting the multi-endpoint outputs.
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

# Contrasts in display order (CSV key -> short name), grouped by meaning:
# the DVH source variants first, then the clinical-feature ablations.
# "MC variance only" is deliberately absent: it opposes two DVH variants to each
# other rather than to the clinical model, and 06_figures already covers it.
# A contrast listed here but absent from contrasts.csv would come out as 0/N,
# which reads as "no effect" when it actually means "no data": check that the
# corresponding tag was trained for the active run before adding one.
CONTRAST_ORDER = [
    ("M1_manual − M0",                          "Manual DVH"),
    ("M1_det_clin0977 − M0",                    "Auto det. DVH"),
    ("M1_bayes_clin0977 − M0",                  "Bayesian DVH (mean)"),
    ("M1_bayes+var_clin0977 − M0",              "Bayesian DVH (mean+var)"),
    ("Pre-tx total IPSS (obstr+irrit ablation)", "Pre-tx total IPSS"),
    ("Pre-tx obstructive IPSS (ablation)",      "Pre-tx obstructive IPSS"),
    ("Age (ablation)",                          "Pre-tx age"),
    ("Prostate volume / implant (ablation)",    "Volume / implant"),
]
ALGO_ORDER = ["elasticnet", "rf", "xgboost", "catboost", "mlp"]
ALGO_LABELS = {"elasticnet": "ElasticNet", "rf": "RF",
               "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP"}
PRIMARY = "elasticnet"
# Endpoints derived from config.ENDPOINTS, so they follow the active run.
# Sorted by increasing day.
ENDPOINTS = [f"y_{d}d" for d in sorted(CFG_ENDPOINTS)]
ENDPOINT_DAYS = [e.strip("y_").rstrip("d") for e in ENDPOINTS]
N_EP = len(ENDPOINTS)                          # denominator of the "x/N" counts


# ============================================================
# Loading and pivots
# ============================================================
def load():
    r = pd.read_csv(CSV)
    r["ci_excl0"] = (r["boot_ci_lo"] > 0) | (r["boot_ci_hi"] < 0)
    r["nb_sig"] = r["nb_p"] < ALPHA
    return r


def matrices(r, contrast):
    """(delta, ci, nb): learners x endpoints matrices for one contrast."""
    s = r[r["contrast"] == contrast]
    def piv(col):
        m = (s.pivot(index="algo", columns="endpoint", values=col)
             .reindex(index=ALGO_ORDER, columns=ENDPOINTS))
        return m
    return piv("delta_rmse").values, piv("ci_excl0").values, piv("nb_sig").values


# ============================================================
# Figure
# ============================================================
def build_figure(r):
    norm = Normalize(vmin=-VMAX, vmax=VMAX)
    # Grid sized for the n contrast panels plus two slots (glyph legend and
    # colour bar). ncol is fixed and nrow deduced, which stays robust to the
    # number of contrasts.
    ncol = 4
    nrow = int(np.ceil((len(CONTRAST_ORDER) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.1 * nrow))
    axes = axes.ravel()

    im = None
    for k, (key, name) in enumerate(CONTRAST_ORDER):
        ax = axes[k]
        delta, ci, nb = matrices(r, key)
        im = ax.imshow(delta, cmap=CMAP, norm=norm, aspect="auto")

        # Significance glyphs (colour = magnitude; glyph = state).
        ys, xs = np.arange(len(ALGO_ORDER)), np.arange(len(ENDPOINTS))
        for i in ys:
            for j in xs:
                if bool(nb[i, j]):
                    ax.plot(j, i, marker="*", ms=8, color=GLYPH_COLOR,
                            mec="white", mew=0.4, zorder=3)
                elif bool(ci[i, j]):
                    ax.plot(j, i, marker="o", ms=3.2, color=GLYPH_COLOR,
                            mec="white", mew=0.3, zorder=3)

        ax.set_title(name, fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(ENDPOINT_DAYS, fontsize=12, rotation=45)
        ax.set_yticks(ys)
        if k % ncol == 0:            # learner labels only in the first column
            ax.set_yticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], fontsize=12)
            # Highlight the primary learner.
            for lab in ax.get_yticklabels():
                if lab.get_text() == ALGO_LABELS[PRIMARY]:
                    lab.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # Thin grid between cells.
        ax.set_xticks(np.arange(-.5, len(ENDPOINTS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ALGO_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)

    # Remaining slots after the n panels: glyph legend, then colour bar. Every
    # unoccupied slot is switched off, and the first two free ones are reused,
    # which stays robust to the number of contrasts and learners.
    n = len(CONTRAST_ORDER)
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
    legend_ax.legend(handles=handles, loc="center", fontsize=12,
                     frameon=False, title="Significance", title_fontsize=12)

    cbar_ax = cbar_host.inset_axes([0.08, 0.45, 0.86, 0.12])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.ax.tick_params(labelsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ============================================================
# LaTeX table: cross-model consensus
# ============================================================
def build_latex(r):
    L = []
    L.append(r"% Generated by 08_all_models_heatmap.py - \usepackage{booktabs}")
    L.append(r"\begin{table}[t]")
    L.append(r"  \centering")
    L.append(r"  \caption{Cross-model consensus over contrasts.csv. For each "
             r"contrast and each learner, the number of endpoints (out of " +
             str(N_EP) + r") whose patient-level bootstrap CI excludes $0$ and, "
             r"in parentheses, whose $p_\text{NB}<0.05$ (Nadeau--Bengio, without "
             r"Holm).}")
    L.append(r"  \label{tab:all_models_consensus}")
    algos_tex = " & ".join(ALGO_LABELS[a] for a in ALGO_ORDER)
    L.append(r"  \begin{tabular}{l" + "c" * len(ALGO_ORDER) + "}")
    L.append(r"    \toprule")
    L.append(r"    Contrast & " + algos_tex + r" \\")
    L.append(r"    \midrule")
    for key, name in CONTRAST_ORDER:
        cells = []
        for a in ALGO_ORDER:
            s = r[(r["contrast"] == key) & (r["algo"] == a)]
            n_ci, n_nb = int(s["ci_excl0"].sum()), int(s["nb_sig"].sum())
            txt = rf"{n_ci} ({n_nb})"
            if n_ci >= 5:                     # strong consensus -> bold
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
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found - run 04_contrasts.py first.")
    r = load()

    FIG_DIR.mkdir(exist_ok=True)
    fig = build_figure(r)
    fig.savefig(FIG_DIR / "F6_all_models_delta_heatmap.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "F6_all_models_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    (TEX_DIR / "all_models_consensus_table.tex").write_text(build_latex(r))

    # Console summary, restricted to the retained learners so it matches the
    # figure and the table.
    n_cells = len(ALGO_ORDER) * len(ENDPOINTS)
    print(f"Consensus (CI excludes 0 / {n_cells} cells; p_NB<0.05 / {n_cells}) "
          f"per contrast:")
    for key, name in CONTRAST_ORDER:
        s = r[(r["contrast"] == key) & (r["algo"].isin(ALGO_ORDER))]
        print(f"  {name:<26} CI!=0: {int(s['ci_excl0'].sum()):2d}/{n_cells}   "
              f"p_NB<0.05: {int(s['nb_sig'].sum()):2d}/{n_cells}")
    print(f"\n[done] Written:")
    print(f"   {FIG_DIR}/F6_all_models_delta_heatmap.png / .pdf")
    print(f"   {TEX_DIR}/all_models_consensus_table.tex")


if __name__ == "__main__":
    main()
