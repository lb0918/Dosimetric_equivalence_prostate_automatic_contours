"""Summary figures for the IPSS prediction results.

Nothing is retrained and no statistic is recomputed: this script reads the
outputs of `03_evaluate.py` (results_<tag>/metrics.csv) and of `04_contrasts.py`
(contrasts/contrasts.csv) and turns them into figures.

  F1  Contrast forest (primary learner): delta RMSE and bootstrap CI of each DVH
      block against the clinical model, at every endpoint.
  F2  Absolute performance (primary learner): OOF RMSE and R2 of every run,
      i.e. the same comparison seen from raw performance.
  F3  Exclusion bounds: the upper bound of the bootstrap CI on the gain side,
      per endpoint, against the IPSS MCID. This is what makes a null result
      quantitative rather than merely negative.
  F4  Consistency panel: delta RMSE of every learner, endpoint and contrast.
  F5  Monte-Carlo variance alone (primary learner), highlighting any cell whose
      CI excludes zero.
  F13 Composite of F1 and F3, sharing one colour key.

Sign convention, inherited from `04_contrasts.py`: delta RMSE = RMSE(baseline) -
RMSE(treatment), so delta RMSE > 0 means the DVH block improves the prediction.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ENDPOINTS as CFG_ENDPOINTS, PROJECT_DIR, RESULTS_ROOT,
                    with_suffix)
from utils import ensure_dir

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

# ============================================================
# CONSTANTS
# ============================================================

# Root of the runs. Each run lives in <BASE_DIR>/results_<tag>/metrics.csv, and
# the contrasts in <BASE_DIR>/contrasts/contrasts.csv.
BASE_DIR = PROJECT_DIR
# contrasts/ and figures/ are suffixed by the active run (PIPE_TAG_SUFFIX), so a
# targeted run reads contrasts_<suffix>/ and writes to figures_<suffix>/ without
# touching the multi-endpoint outputs.
CONTRASTS_CSV = BASE_DIR / with_suffix("contrasts") / "contrasts.csv"
FIG_DIR = BASE_DIR / with_suffix("figures")
OVERWRITE = True  # when False, an existing figure is not rewritten

# ------------------------------------------------------------
# Registry of the DVH segmentation sources
# ------------------------------------------------------------
# Each source ties together its contrast (the key in contrasts.csv), its
# performance run (results_<run>/metrics.csv), its labels and its style. The
# ACTIVE_SOURCES selection (CLI --sources; all by default) decides which enter
# F1 to F4.
DVH_SOURCE_REGISTRY = {
    # Colour coding ALIGNED with the DVH concordance figures: deterministic =
    # blue (C0), Bayesian = orange (C1). The manual segmentation therefore takes
    # green (C2); it is not comparable to an automatic source, being the input
    # reference.
    "manual": dict(
        contrast="M1_manual − M0", run="curated_manual",
        contrast_label="Manual DVH", run_label="+ manual DVH",
        color="C2", marker="s"),
    "auto_det": dict(
        contrast="M1_det_clin0977 − M0", run="curated_auto_det_clin0977",
        contrast_label="Deterministic auto DVH", run_label="+ deterministic auto DVH",
        color="C0", marker="^"),
    "mc_bayes": dict(
        contrast="M1_bayes_clin0977 − M0", run="curated_mc_bayes_clin0977",
        contrast_label="Bayesian DVH (mean)", run_label="+ Bayesian DVH (MC mean)",
        color="C1", marker="D"),
    "mc_bayes_var": dict(
        contrast="M1_bayes+var_clin0977 − M0", run="curated_mc_bayes_clin0977_var",
        contrast_label="Bayesian DVH (mean + variance)",
        run_label="+ Bayesian DVH (MC mean + variance)",
        color="C4", marker="v"),
}
ALL_SOURCES = list(DVH_SOURCE_REGISTRY)
ALL_FIGURES = ["F1", "F2", "F3", "F4", "F5", "F13"]


def _cli_config():
    """Parse --sources a,b,... and --figures F1,F2,... in any order. Returns
    (sources|None, figures|None); None means the default (all)."""
    srcs = figs = None
    argv, i = sys.argv[1:], 0
    while i < len(argv):
        if argv[i] == "--sources" and i + 1 < len(argv):
            srcs = [s.strip() for s in argv[i + 1].split(",") if s.strip()]; i += 2
        elif argv[i] == "--figures" and i + 1 < len(argv):
            figs = [s.strip().upper() for s in argv[i + 1].split(",") if s.strip()]
            i += 2
        else:
            i += 1
    return srcs, figs


_cli_sources, _cli_figures = _cli_config()
ACTIVE_SOURCES = _cli_sources if _cli_sources else ALL_SOURCES
for _s in ACTIVE_SOURCES:
    if _s not in DVH_SOURCE_REGISTRY:
        raise SystemExit(f"Unknown DVH source: {_s!r} (expected one of: {ALL_SOURCES})")
FIGURES = _cli_figures if _cli_figures else ALL_FIGURES

# File suffix: empty when every source is active, otherwise derived from the
# active sources (e.g. "_manual") so an all-sources figure is not overwritten.
FILE_SUFFIX = "" if ACTIVE_SOURCES == ALL_SOURCES else "_" + "-".join(ACTIVE_SOURCES)

# Derived from the selection (order = ACTIVE_SOURCES). The clinical run M0
# (noDVH) is always the reference of F2.
RUN_LABELS = {"noDVH": "Clinical only (M0)",
              **{DVH_SOURCE_REGISTRY[s]["run"]: DVH_SOURCE_REGISTRY[s]["run_label"]
                 for s in ACTIVE_SOURCES}}
DVH_CONTRASTS = {DVH_SOURCE_REGISTRY[s]["contrast"]: DVH_SOURCE_REGISTRY[s]["contrast_label"]
                 for s in ACTIVE_SOURCES}
VARIANCE_CONTRAST = "MC variance only (clin0977)"

PRIMARY_LEARNER = "elasticnet"
LEARNER_LABELS = {
    "elasticnet": "ElasticNet",
    "linreg": "Linear regression",
    "rf": "Random forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "mlp": "Multilayer perceptron",
}

# Endpoints, in days after the implant. Derived from config.ENDPOINTS so the
# figures follow the active run automatically. Sorted by increasing day.
ENDPOINT_DAYS = sorted(CFG_ENDPOINTS)

# Minimal clinically important difference of the IPSS, in score points. Used as
# the reference when reading the exclusion bounds (F3).
MCID_IPSS = 3.0

# ------------------------------------------------------------
# Palette: the standard matplotlib cycle (tab10), addressed by its C0..C9
# aliases. The aliases follow the active prop_cycle, so the figures stay
# consistent with any matplotlib style applied upstream. The four segmentation
# sources take C0, C1, C2 and C4 (four well-separated hues); red C3 is reserved
# for flagged elements (MCID, highlighted cells) so it is never confused with a
# source; grey C7 serves the consistency panel and incidental text; black marks
# the reference clinical model.
# NB: literal names ("blue", "teal") remain valid, but a non-existent name such
# as "darkteal" fails at render time. The CN aliases avoid that pitfall.
# ------------------------------------------------------------
C_C0 = "C0"       # blue
C_C1 = "C1"       # orange
C_C2 = "C2"       # green
C_C4 = "C4"       # purple
C_GREY = "C7"     # grey
C_INK = "black"   # axes, text, clinical reference
C_ACCENT = "C3"   # red, reserved for flagged cells and thresholds

# Colours and markers of the ACTIVE sources, in display order. Each source keeps
# its own hue from the registry whatever the selection, so a given source is
# always drawn in the same colour.
SOURCE_COLORS = [DVH_SOURCE_REGISTRY[s]["color"] for s in ACTIVE_SOURCES]
SOURCE_MARKERS = [DVH_SOURCE_REGISTRY[s]["marker"] for s in ACTIVE_SOURCES]
# Colour per CONTRAST: lets F3 paint each bar in the hue of the source that
# produced it, so it reads with the same key as the F1 points.
CONTRAST_COLORS = {DVH_SOURCE_REGISTRY[s]["contrast"]: DVH_SOURCE_REGISTRY[s]["color"]
                   for s in ACTIVE_SOURCES}
C_PRIMARY = C_C0    # primary learner (F3, F4, F5)
C_PANEL = C_GREY    # consistency-panel learners (F4)
DPI = 300

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C_INK,
    "axes.linewidth": 0.8,
    "xtick.color": C_INK,
    "ytick.color": C_INK,
    "text.color": C_INK,
    "legend.frameon": False,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
})


# ============================================================
# LOADING
# ============================================================

def endpoint_days(endpoint: str) -> int:
    """`y_400d` -> 400."""
    return int(endpoint.removeprefix("y_").removesuffix("d"))


def load_contrasts() -> pd.DataFrame:
    df = pd.read_csv(CONTRASTS_CSV)
    df["days"] = df["endpoint"].map(endpoint_days)
    df = df[df["days"].isin(ENDPOINT_DAYS)]
    return df.sort_values(["contrast", "days", "algo"]).reset_index(drop=True)


def load_metrics() -> pd.DataFrame:
    frames = []
    for tag, label in RUN_LABELS.items():
        path = RESULTS_ROOT / f"results_{with_suffix(tag)}" / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"metrics.csv missing for run {tag!r}: {path}")
        m = pd.read_csv(path)
        m["run"] = tag
        m["run_label"] = label
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    df["days"] = df["endpoint"].map(endpoint_days)
    df = df[df["days"].isin(ENDPOINT_DAYS)]
    return df.reset_index(drop=True)


def save(fig, name: str) -> None:
    ensure_dir(FIG_DIR)
    # The selection suffix is inserted before the extension (empty when every
    # source is active).
    p = Path(name)
    path = FIG_DIR / f"{p.stem}{FILE_SUFFIX}{p.suffix}"
    if path.exists() and not OVERWRITE:
        print(f"  [skip] {path.name} already exists (OVERWRITE=False)")
        plt.close(fig)
        return
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok]   {path}")


# ============================================================
# F1 - Contrast forest, primary learner
# ============================================================

def draw_forest_primary(ax, contrasts: pd.DataFrame) -> None:
    """Draw the contrast forest into a supplied axis (used by F1 and F13)."""
    d = contrasts[(contrasts["algo"] == PRIMARY_LEARNER)
                  & (contrasts["contrast"].isin(DVH_CONTRASTS))]

    n_src = len(DVH_CONTRASTS)
    # A single source needs no vertical offset (the series is centred on the
    # endpoint).
    offsets = np.array([0.0]) if n_src == 1 else np.linspace(0.28, -0.28, n_src)

    for k, (contrast, label) in enumerate(DVH_CONTRASTS.items()):
        sub = d[d["contrast"] == contrast].set_index("days").reindex(ENDPOINT_DAYS)
        y = np.arange(len(ENDPOINT_DAYS)) + offsets[k]
        color = SOURCE_COLORS[k]
        ax.hlines(y, sub["boot_ci_lo"], sub["boot_ci_hi"], color=color, lw=1.4)
        ax.plot(sub["delta_rmse"], y, "o", ms=4.5, color=color,
                mec="white", mew=0.6, label=label, zorder=3)

    ax.axvline(0, color=C_INK, lw=0.9, zorder=1)
    ax.set_yticks(np.arange(len(ENDPOINT_DAYS)))
    ax.set_yticklabels([f"{d_} d" for d_ in ENDPOINT_DAYS])
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE = RMSE(clinical) − RMSE(clinical + DVH), IPSS points")
    ax.set_ylabel("Post-implant endpoint")
    ax.grid(axis="x", zorder=0)

    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.36), ncol=1, fontsize=10)


def fig_forest_primary(contrasts: pd.DataFrame) -> None:
    """Delta RMSE and bootstrap CI of every DVH block, at every endpoint."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    draw_forest_primary(ax, contrasts)
    save(fig, "F1_forest_dvh_elasticnet.png")


# ============================================================
# F2 - Absolute performance of every run, primary learner
# ============================================================

def fig_absolute_performance(metrics: pd.DataFrame) -> None:
    """OOF RMSE and R2 (real targets) of every run, primary learner."""
    d = metrics[metrics["algo"] == PRIMARY_LEARNER]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    # noDVH is the reference (solid line); the active sources are dashed and
    # styled from the registry, in the order of RUN_LABELS.
    colors = [C_INK] + SOURCE_COLORS
    styles = ["-"] + ["--"] * len(ACTIVE_SOURCES)
    markers = ["o"] + SOURCE_MARKERS

    for k, (tag, label) in enumerate(RUN_LABELS.items()):
        sub = d[d["run"] == tag].set_index("days").reindex(ENDPOINT_DAYS)
        for ax, metric in zip(axes, ["rmse", "r2"]):
            ax.plot(ENDPOINT_DAYS, sub[metric], styles[k], marker=markers[k],
                    ms=4, lw=1.2, color=colors[k], label=label, alpha=0.9)

    axes[0].set_ylabel("OOF RMSE (IPSS points)")
    axes[0].set_title("Prediction error")
    axes[1].set_ylabel("OOF R²")
    axes[1].set_title("Explained variance")
    for ax in axes:
        ax.set_xlabel("Post-implant endpoint (days)")
        ax.set_xticks(ENDPOINT_DAYS)
        ax.grid(axis="y")

    # Real-target sample sizes, printed under the x axis.
    n_real = (d[d["run"] == "noDVH"].set_index("days")
              .reindex(ENDPOINT_DAYS)["n"].astype(int))
    axes[0].set_xticklabels([f"{d_}\n(n={n})" for d_, n in zip(ENDPOINT_DAYS, n_real)],
                            fontsize=7.5)
    axes[1].set_xticklabels([f"{d_}\n(n={n})" for d_, n in zip(ENDPOINT_DAYS, n_real)],
                            fontsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.16))
    save(fig, "F2_absolute_performance.png")


# ============================================================
# F3 - Exclusion bounds
# ============================================================

def draw_exclusion_bounds(ax, contrasts: pd.DataFrame) -> None:
    """Draw the exclusion bounds into a supplied axis (used by F3 and F13).

    For each endpoint the MOST PERMISSIVE bound across the active sources is
    kept (the worst case for the null), and reported against the IPSS MCID.
    """
    d = contrasts[(contrasts["algo"] == PRIMARY_LEARNER)
                  & (contrasts["contrast"].isin(DVH_CONTRASTS))]
    worst = (d.groupby("days")["boot_ci_hi"].max().reindex(ENDPOINT_DAYS))
    argworst = (d.loc[d.groupby("days")["boot_ci_hi"].idxmax()]
                .set_index("days")["contrast"].reindex(ENDPOINT_DAYS))

    y = np.arange(len(ENDPOINT_DAYS))
    # Each bar takes the hue of the source that produced it, so the bar and the
    # matching F1 point read with the same colour key.
    bar_colors = [CONTRAST_COLORS.get(src, C_PRIMARY) for src in argworst.values]
    ax.barh(y, worst.values, height=0.55, color=bar_colors, alpha=0.85,
            edgecolor="white", zorder=3)

    for yi, (val, src) in enumerate(zip(worst.values, argworst.values)):
        ax.text(val + 0.06, yi, f"{val:.2f}  ({DVH_CONTRASTS[src]})",
                va="center", fontsize=10, color=C_INK)

    ax.axvline(MCID_IPSS, color=C_ACCENT, lw=1.2, ls="--", zorder=4)
    ax.text(MCID_IPSS + 0.13, len(ENDPOINT_DAYS) - 4,
            f"IPSS MCID ≈ {MCID_IPSS:.0f} points", rotation=90, ha="right",
            va="top", fontsize=8, color=C_ACCENT)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{d_} d" for d_ in ENDPOINT_DAYS])
    ax.invert_yaxis()
    ax.set_xlim(0, MCID_IPSS * 1.12)
    ax.set_xlabel("Maximum RMSE gain from the 95% bootstrap CI, IPSS points")
    ax.grid(axis="x", zorder=0)


def fig_exclusion_bounds(contrasts: pd.DataFrame) -> None:
    """Largest gain compatible with the data: upper bound of the bootstrap CI."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    draw_exclusion_bounds(ax, contrasts)
    save(fig, "F3_exclusion_bounds.png")


# ============================================================
# F13 - Composite: (a) contrast forest + (b) exclusion bounds
# ============================================================

def fig_forest_and_bounds(contrasts: pd.DataFrame) -> None:
    """Composite PDF figure: (a) F1 (delta RMSE per source), (b) F3 (bounds).

    Both panels read with the SAME colour key: the bar in (b) carries the hue of
    the source that sets the bound, hence of the matching point in (a).
    """
    fig = plt.figure(figsize=(13.4, 4.8))
    blocks = fig.subfigures(1, 2, wspace=0.04)

    draw_forest_primary(blocks[0].subplots(), contrasts)
    draw_exclusion_bounds(blocks[1].subplots(), contrasts)

    for sub, tag in zip(blocks, ("(a)", "(b)")):
        sub.text(0.005, 0.995, tag, fontsize=12, fontweight="bold", color=C_INK,
                 ha="left", va="top")
        sub.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.14)

    save(fig, "F13_forest_bounds.pdf")


# ============================================================
# F4 - Consistency panel, every learner
# ============================================================

def fig_consistency_panel(contrasts: pd.DataFrame) -> None:
    """Delta RMSE of every learner, every DVH contrast and every endpoint.

    Cells whose bootstrap CI excludes zero are filled, the others hollow. Under
    the null one expects scatter with no dominant direction and a few isolated
    exclusions with no echo across learners.
    """
    d = contrasts[contrasts["contrast"].isin(DVH_CONTRASTS)].copy()
    d["excl0"] = (d["boot_ci_lo"] > 0) | (d["boot_ci_hi"] < 0)

    order = [PRIMARY_LEARNER] + [a for a in LEARNER_LABELS if a != PRIMARY_LEARNER]
    order = [a for a in order if a in set(d["algo"])]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    for i, algo in enumerate(order):
        sub = d[d["algo"] == algo]
        jitter = rng.uniform(-0.16, 0.16, len(sub))
        color = C_PRIMARY if algo == PRIMARY_LEARNER else C_PANEL
        filled, hollow = sub["excl0"].values, ~sub["excl0"].values
        ax.scatter(sub["delta_rmse"][hollow], (i + jitter)[hollow], s=22,
                   facecolors="none", edgecolors=color, linewidths=0.9, alpha=0.75)
        ax.scatter(sub["delta_rmse"][filled], (i + jitter)[filled], s=26,
                   color=color, edgecolors="white", linewidths=0.5, zorder=3)

    ax.axvline(0, color=C_INK, lw=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LEARNER_LABELS[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE (IPSS points); > 0: DVH improves")
    ax.set_title(f"Consistency panel: {len(DVH_CONTRASTS)} DVH blocks × "
                 f"{len(ENDPOINT_DAYS)} endpoints × {len(order)} learners")
    ax.grid(axis="x")

    ax.scatter([], [], s=26, color=C_GREY, edgecolors="white",
               label="Bootstrap CI excluding 0")
    ax.scatter([], [], s=22, facecolors="none", edgecolors=C_GREY,
               label="Bootstrap CI covering 0")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.34), ncol=2, fontsize=8)

    n_excl = int(d["excl0"].sum())
    ax.text(0.995, 0.03, f"{n_excl} / {len(d)} cells exclude 0",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=C_GREY)
    save(fig, "F4_consistency_panel.png")


# ============================================================
# F5 - Monte-Carlo variance alone
# ============================================================

def fig_variance_only(contrasts: pd.DataFrame) -> None:
    """The "MC variance only" contrast, primary learner, flagging any cell whose
    CI excludes zero."""
    d = (contrasts[(contrasts["algo"] == PRIMARY_LEARNER)
                   & (contrasts["contrast"] == VARIANCE_CONTRAST)]
         .set_index("days").reindex(ENDPOINT_DAYS))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(ENDPOINT_DAYS))
    excl0 = (d["boot_ci_lo"] > 0) | (d["boot_ci_hi"] < 0)
    colors = np.where(excl0, C_ACCENT, C_PRIMARY)

    for yi in y:
        ax.hlines(yi, d["boot_ci_lo"].iloc[yi], d["boot_ci_hi"].iloc[yi],
                  color=colors[yi], lw=1.5)
        ax.plot(d["delta_rmse"].iloc[yi], yi, "o", ms=5, color=colors[yi],
                mec="white", mew=0.6, zorder=3)

    ax.axvline(0, color=C_INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d_} d" for d_ in ENDPOINT_DAYS])
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE (IPSS points); > 0: MC variance improves")
    ax.set_title("Monte-Carlo variance of the indices as the only added block (ElasticNet)")
    ax.grid(axis="x")

    if bool(excl0.any()):
        k = int(np.argmax(excl0.values))
        # nb_p_holm exists only when Holm is enabled (USE_HOLM in
        # 04_contrasts.py); otherwise annotate with the raw NB p.
        if "nb_p_holm" in d.columns:
            note = f"(Holm p = {d['nb_p_holm'].iloc[k]:.2f}; very wide CI)"
        else:
            note = f"(NB p = {d['nb_p'].iloc[k]:.2f}; very wide CI)"
        ax.annotate("CI excluding 0, not confirmed\n" + note,
                    xy=(d["delta_rmse"].iloc[k], k),
                    xytext=(0.55, 0.82), textcoords="axes fraction",
                    fontsize=7.5, color=C_ACCENT, ha="left",
                    arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.8))
    save(fig, "F5_variance_only.png")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(f"Active DVH sources: {ACTIVE_SOURCES}  |  figures: {FIGURES}"
          + (f"  |  suffix: {FILE_SUFFIX}" if FILE_SUFFIX else ""))
    contrasts = load_contrasts()

    # Each figure maps to one function; only F2 reads the metrics.
    if "F1" in FIGURES:
        fig_forest_primary(contrasts)
    if "F2" in FIGURES:
        fig_absolute_performance(load_metrics())
    if "F3" in FIGURES:
        fig_exclusion_bounds(contrasts)
    if "F4" in FIGURES:
        fig_consistency_panel(contrasts)
    if "F5" in FIGURES:
        fig_variance_only(contrasts)
    if "F13" in FIGURES:
        fig_forest_and_bounds(contrasts)


if __name__ == "__main__":
    main()
