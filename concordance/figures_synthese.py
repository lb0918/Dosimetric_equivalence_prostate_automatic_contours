"""
Summary of the DVH concordance analysis.

Produces, from the result CSVs:
  1. summary_table_concordance.csv        (master table)
  2. summary_table_concordance.png/.pdf   (rendering of the table)
  3. summary_table_concordance.tex        (the same table, as LaTeX)
  4. fig1_standardised_equivalence.png/.pdf  (forest normalised by the margin)
  5. fig2_agreement_coverage.png/.pdf     (CCC det vs bayes + MC coverage)

Expected inputs in RES:
  tost_results.csv, concordance_descriptors.csv, mc_coverage.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils as U

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'


# --------------------------------------------------------------------------
# Configurable constants
# --------------------------------------------------------------------------
RES = U.OUT_DIR             # directory of the input CSVs
OUT = U.OUT_DIR / "summary"  # output directory

# Display order shared by every output.
# ORDER = [
#     "Prostate_D90_pct", "Prostate_V100_pct", "Prostate_V150_pct",
#     "Prostate_V200_pct", "BladderNeck_D2cc_pct", "BladderNeck_D1cc_pct",
#     "BladderNeck_V100_pct",
# ]
ORDER = [
    "Prostate_D90_pct", "Prostate_V100_pct", "Prostate_V150_pct",
    "Prostate_V200_pct",
]
COV_ORDER = ["D90_pct", "V100_pct", "V150_pct", "V200_pct"]  # coverage panel
COMP_LABEL = {"det": "Deterministic", "bayes": "Bayesian (mean)"}
NOMINAL = 0.95  # nominal coverage of the Monte-Carlo intervals

# Font sizes of the summary table.
TAB_FS = 12       # table body
TAB_HEAD_FS = 15  # column headers

# Figure font sizes: a legible floor once scaled down to a column.
FS_TICK = 13    # tick labels
FS_LABEL = 15   # axis titles
FS_ANNOT = 13   # in-axes annotations
FS_LEG = 13     # legends

# Default matplotlib palette.
INK = "black"
GREY = "tab:gray"
DET = "tab:blue"    # deterministic source
BAY = "tab:orange"  # source bayesienne (moyenne)
FAIL = "tab:red"    # non-equivalence

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Chargement et fusion
# --------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tost = pd.read_csv(RES / "tost_results.csv")
    desc = pd.read_csv(RES / "concordance_descriptors.csv")
    cov = pd.read_csv(RES / "mc_coverage.csv")

    key = ["structure", "index", "comparison"]
    df = tost.merge(
        desc[key + ["bias", "loa_lo", "loa_hi", "ccc", "icc_a1"]],
        on=key, how="left",
    )
    df["k"] = df["structure"] + "_" + df["index"]
    df["ord"] = df["k"].map({k: i for i, k in enumerate(ORDER)})
    df = df.sort_values(["ord", "comparison"]).reset_index(drop=True)
    return df, cov


# --------------------------------------------------------------------------
# 1. Tableau maitre (CSV)
# --------------------------------------------------------------------------
def build_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.DataFrame({
        "structure": df["structure"],
        "index": df["index"].str.replace("_pct", "", regex=False),
        "comparison": df["comparison"].map(COMP_LABEL),
        "tier": df["tier"],
        "n": df["n"].astype(int),
        # "unit": df["unite"],
        "bias": df["bias"].round(2),
        "CI90_lo": df["ci90_lo"].round(2),
        "CI90_hi": df["ci90_hi"].round(2),
        "delta_margin": df["delta_conf"].round(1),
        "equivalence": np.where(df["verdict_conf"], "yes", "no"),
        "CCC": df["ccc"].round(3),
        "LoA_lo": df["loa_lo"].round(1),
        "LoA_hi": df["loa_hi"].round(1),
    })
    tab.to_csv(OUT / "summary_table_concordance.csv", index=False)
    return tab


# --------------------------------------------------------------------------
# 2. Table rendering (figure)
# --------------------------------------------------------------------------
def table_content(tab: pd.DataFrame) -> tuple[list[str], list[list[str]], pd.DataFrame]:
    """Content shared by the figure and the LaTeX export (same rows, same order)."""
    def fmt_ci(r):
        return f"{r['bias']:+.2f}  [{r['CI90_lo']:+.2f}, {r['CI90_hi']:+.2f}]"

    tab = tab.copy()
    # Keep only the structure/index pairs listed in ORDER.
    keys = tab["structure"] + "_" + tab["index"] + "_pct"
    tab = tab[keys.isin(ORDER)].reset_index(drop=True)
    tab["cmp"] = tab["comparison"].str.split().str[0]
    rows = [[
        f"{r['structure']}  {r['index']}", r["cmp"],
        fmt_ci(r), f"{r['delta_margin']:.0f}", r["equivalence"],
        f"{r['CCC']:.3f}",
    ] for _, r in tab.iterrows()]

    cols = ["Structure / index", "Source",
            "Mean bias  [90 % CI]", r"$\delta$", "Equiv.", "CCC"]
    return cols, rows, tab


def source_color(val: str) -> str:
    """Source colour, consistent with figures 1 and 2."""
    return BAY if val == "Bayesian" else DET


def render_table(tab: pd.DataFrame) -> None:
    cols, rows, tab = table_content(tab)
    # Widths summing to 1.0, wide enough for the headers at TAB_HEAD_FS.
    colw = [0.24, 0.17, 0.29, 0.09, 0.11, 0.10]
    xstart, acc = [], 0.0
    for w in colw:
        xstart.append(acc)
        acc += w
    # Indices of the centred and right-aligned columns.
    centered = {1, 2, 4, 5}
    right = {3}

    def xalign(j, xs, w):
        if j == 0:
            return xs, "left"
        if j in right:
            return xs + w, "right"
        return xs + w / 2, "center"

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * (n + 2.2)))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2)

    yhead = n + 1.1
    for j, c in enumerate(cols):
        xx, ha = xalign(j, xstart[j], colw[j])
        ax.text(xx, yhead, c, fontsize=TAB_HEAD_FS, fontweight="bold", color=INK, ha=ha, va="center")
    ax.plot([0, 1], [yhead - 0.45, yhead - 0.45], color=INK, lw=1.0)

    prev = None
    for i, r in enumerate(rows):
        y = n - i
        struct = tab.iloc[i]["structure"]
        if prev is not None and struct != prev:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color="lightgray", lw=0.6)
        prev = struct
        for j, val in enumerate(r):
            xx, ha = xalign(j, xstart[j], colw[j])
            color, weight = INK, "normal"
            if j == 4 and val == "no":
                color, weight = FAIL, "bold"
            if j == 1:
                color = source_color(val)
            ax.text(xx, y, val, fontsize=TAB_FS, color=color, ha=ha, va="center", fontweight=weight)

    ax.plot([0, 1], [0.5, 0.5], color=INK, lw=1.0)
    fig.savefig(OUT / "summary_table_concordance.png")
    fig.savefig(OUT / "summary_table_concordance.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2b. The same table, as LaTeX
# --------------------------------------------------------------------------
def export_latex(tab: pd.DataFrame) -> None:
    """Write a booktabs LaTeX table identical to the figure rendering."""
    cols, rows, tab = table_content(tab)

    # Headers: LaTeX version of the symbols shown in the figure.
    head_tex = {"Mean bias  [90 % CI]": r"Mean bias~[90\,\% CI]"}
    align = "l c c r c c"  # same alignment as xalign() in the figure

    def esc(s: str) -> str:
        return s.replace("%", r"\%").replace("  ", r"\ \ ")

    def cell(j: int, val: str) -> str:
        if j == 2:  # bias + CI: math mode, for proper minus signs
            point, ci = val.split("  ", 1)
            ci = ci.replace(", ", r",\, ")
            return f"${point}$\\ \\ ${ci}$"
        if j == 1:
            name = "bayorange" if val == "Bayesian" else "detblue"
            return rf"\textcolor{{{name}}}{{{esc(val)}}}"
        if j == 4 and val == "no":
            return rf"\textcolor{{failred}}{{\textbf{{no}}}}"
        return esc(val)

    L = []
    L.append("% Necessite : \\usepackage{booktabs} et \\usepackage{xcolor}")
    for name, col in (("detblue", DET), ("bayorange", BAY), ("failred", FAIL)):
        rgb = mpl.colors.to_hex(col).lstrip("#").upper()
        L.append(rf"\definecolor{{{name}}}{{HTML}}{{{rgb}}}")
    L.append("")
    L.append(r"\begin{table}[htbp]")
    L.append(r"  \centering")
    L.append(r"  \caption{DVH concordance summary: mean bias with 90\,\% confidence "
             r"interval, equivalence margin $\delta$, equivalence verdict and "
             r"concordance correlation coefficient.}")
    L.append(r"  \label{tab:synthese_concordance}")
    L.append(rf"  \begin{{tabular}}{{{align}}}")
    L.append(r"    \toprule")
    header = " & ".join(rf"\large\textbf{{{head_tex.get(c, esc(c))}}}" for c in cols)
    L.append(f"    {header} \\\\")
    L.append(r"    \midrule")

    prev = None
    for i, r in enumerate(rows):
        struct = tab.iloc[i]["structure"]
        if prev is not None and struct != prev:
            L.append(rf"    \cmidrule(lr){{1-{len(cols)}}}")
        prev = struct
        L.append("    " + " & ".join(cell(j, v) for j, v in enumerate(r)) + r" \\")

    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")
    L.append(r"\end{table}")

    (OUT / "summary_table_concordance.tex").write_text("\n".join(L) + "\n",
                                                          encoding="utf-8")


# --------------------------------------------------------------------------
# 3. Standardised forest (bias / margin)
# --------------------------------------------------------------------------
def fig_equivalence(df: pd.DataFrame) -> None:
    def short(row):
        lab = row["index"].replace("_pct", "").replace("_", " ")
        return f"{row['structure']} {lab}"

    fig, ax = plt.subplots(figsize=(7,5.2))
    rows = []
    ylabels = []
    y = 0.0
    prev = None
    for k in ORDER:
        block = df[df["k"] == k]
        if block.empty:
            continue
        struct = block["structure"].iloc[0]
        if prev is not None and struct != prev:
            y -= 0.6  # espace entre structures
        prev = struct
        for _, r in block.iterrows():
            d = r["delta_conf"]
            m, lo, hi = r["mean_diff"] / d, r["ci90_lo"] / d, r["ci90_hi"] / d
            col = BAY if r["comparison"] == "bayes" else DET
            ok = bool(r["verdict_conf"])
            rows.append((y, m, lo, hi, col, ok))
            ylabels.append(short(r) + f"  ({COMP_LABEL[r['comparison']].split()[0]})")
            y -= 1.0
        y -= 0.2

    ymin = min(p for p, *_ in rows) - 0.8
    ymax = max(p for p, *_ in rows) + 0.8

    ax.axvspan(-1, 1, color="#000000", alpha=0.04, lw=0)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    for xv in (-1, 1):
        ax.axvline(xv, color=INK, lw=0.8)

    for yv, m, lo, hi, col, ok in rows:
        ec = col if ok else FAIL
        ax.plot([lo, hi], [yv, yv], color=ec, lw=1.4, solid_capstyle="round", zorder=2)
        for xe in (lo, hi):
            ax.plot([xe, xe], [yv - 0.12, yv + 0.12], color=ec, lw=1.4)
        if ok:
            ax.scatter([m], [yv], s=26, color=col, zorder=3, edgecolor="white", linewidth=0.6)
        else:
            ax.scatter([m], [yv], s=30, facecolor="white", edgecolor=FAIL, linewidth=1.4, zorder=3)

    ax.set_yticks([p for p, *_ in rows])
    ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel(r"Mean difference, normalized by the equivalence margin  ($\overline{d}/\delta$)",
                  fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.text(0, ymax + 0.15, "Equivalence zone", ha="center", va="bottom",
            fontsize=FS_LABEL + 2, color=INK)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DET, markersize=6, label="Deterministic"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BAY, markersize=6, label="Bayesian (mean)"),
        # Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
            #    markeredgecolor=FAIL, markeredgewidth=1.4, markersize=6, label="not equivalent"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=FS_LEG,
              handletextpad=0.4, borderaxespad=0.3)

    fig.savefig(OUT / "fig1_standardised_equivalence.png")
    fig.savefig(OUT / "fig1_standardised_equivalence.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Per-patient agreement (CCC) + MC coverage
# --------------------------------------------------------------------------
def fig_agreement_coverage(df: pd.DataFrame, cov: pd.DataFrame) -> None:
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) CCC det vs bayes
    labels_a = []
    yv = np.arange(len(ORDER))[::-1]
    for i, k in enumerate(ORDER):
        d = df[(df["k"] == k) & (df["comparison"] == "det")]["ccc"].values
        b = df[(df["k"] == k) & (df["comparison"] == "bayes")]["ccc"].values
        d = d[0] if len(d) else np.nan
        b = b[0] if len(b) else np.nan
        yy = yv[i]
        axa.plot([d, b], [yy, yy], color="lightgray", lw=1.2, zorder=1)
        axa.scatter([d], [yy], s=32, color=DET, zorder=3, label="Deterministic" if i == 0 else None)
        axa.scatter([b], [yy], s=32, color=BAY, zorder=3, label="Bayesian" if i == 0 else None)
        st, ix = k.split("_", 1)
        labels_a.append(f"{st} {ix.replace('_pct', '')}")

    for xv in (0.5, 0.9):
        axa.axvline(xv, color=GREY, lw=0.6, ls=(0, (2, 3)))
    # x in data coordinates, y in axes fraction: otherwise the label floats
    # above the frame as soon as the font grows.
    axa.text(0.9, 0.99, "0.90", transform=axa.get_xaxis_transform(),
             fontsize=FS_ANNOT, color=GREY, ha="center", va="top")
    axa.set_yticks(yv)
    axa.set_yticklabels(labels_a, fontsize=FS_TICK)
    axa.set_xlim(0.4, 1.0)
    axa.set_xlabel("CCC (per-patient agreement)", fontsize=FS_LABEL)
    axa.tick_params(axis="x", labelsize=FS_TICK)
    for s in ("top", "right", "left"):
        axa.spines[s].set_visible(False)
    axa.tick_params(axis="y", length=0)
    axa.legend(loc="lower right", frameon=False, fontsize=FS_LEG, handletextpad=0.3)
    axa.set_title("(a)", loc="left", fontsize=FS_LABEL, color=INK)

    # (b) MC coverage, confirmatory prostate
    covp = cov[cov["tier"] == "confirmatory"].copy()
    covp["ord"] = covp["index"].map({k: i for i, k in enumerate(COV_ORDER)})
    covp = covp.sort_values("ord")
    xb = np.arange(len(covp))
    axb.axhline(NOMINAL, color=INK, lw=0.9, ls=(0, (4, 3)))
    axb.text(len(covp) - 0.5, NOMINAL + 0.005, f"nominal {NOMINAL*100:.0f}%",
             fontsize=FS_ANNOT, color=INK, ha="right", va="bottom")
    axb.bar(xb, covp["coverage_pctinterval"], width=0.6,
            color=mpl.colors.to_rgba(BAY, 0.35), edgecolor=BAY, linewidth=0.8)
    for x, v in zip(xb, covp["coverage_pctinterval"]):
        axb.text(x, v + 0.012, f"{v*100:.0f}", ha="center", va="bottom",
                 fontsize=FS_ANNOT, color=INK)
    axb.set_xticks(xb)
    axb.set_xticklabels([c.replace("_pct", "") for c in covp["index"]], fontsize=FS_TICK)
    axb.set_ylim(0, 1.0)
    axb.set_ylabel("Monte-Carlo CI coverage", fontsize=FS_LABEL)
    axb.tick_params(axis="y", labelsize=FS_TICK)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.tick_params(length=3)
    axb.set_title("(b)  Prostate", loc="left", fontsize=FS_LABEL, color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_agreement_coverage.png")
    fig.savefig(OUT / "fig2_agreement_coverage.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    ensure_dir(OUT)
    df, cov = load()
    tab = build_table(df)
    render_table(tab)
    export_latex(tab)
    fig_equivalence(df)
    fig_agreement_coverage(df, cov)
    print("Outputs written to", OUT.resolve())


if __name__ == "__main__":
    main()