"""
"Where the endpoints sit" figure - a static figure for slide presentations.

Same material as `ipss_longitudinal_viewer.py` (the patient-level table in long
format), sized for a 16:9 slide and exported to PDF (with a PNG fallback).

What it shows: the endpoint windows of `config.ENDPOINTS` (target +/- half-width,
in days after the implant) laid over the cohort.

  Panel A  Cohort IPSS trajectory (median plus IQR band per time bin) and the
           median pre-treatment IPSS level, which is the baseline of the IPSS
           delta. The endpoint windows are the shaded bands; the dotted line at
           the centre of each is the exact target.
  Panel B  Density of the actual follow-up: number of IPSS measurements per bin.
           This panel is what justifies the placement of the windows, which fall
           on the peaks of the clinical follow-up schedule. The label above each
           window gives the share of the cohort with AT LEAST one real
           measurement inside it, i.e. case 1 ("real") of
           `01_prepare_target.py`.

A VARIANT of the main figure is always written to a separate file
(`F0_endpoint_windows_log`): it is identical except that the y axis of panel B
is logarithmic, which makes the sparse bins between follow-up peaks legible
again. Disable it with `--no-log`.

Option `--coverage`: a complementary figure (separate file) breaking the cohort
down per endpoint into real / model / excluded, following exactly the rule of
`estimate_at_endpoint()` (a measurement inside the window gives case 1;
otherwise, an endpoint inside the patient's measurement span widened by
+/- TRAJECTORY_EXTRAP_FACTOR * window gives the model case; otherwise the
patient is excluded).

Usage:
    python3 endpoint_windows_figure.py
    python3 endpoint_windows_figure.py --lang fr      # labels in French
    python3 endpoint_windows_figure.py --coverage     # extra coverage figure
    python3 endpoint_windows_figure.py --no-log       # skip the log variant

Outputs:
    F0_endpoint_windows.pdf / .png
    F0_endpoint_windows_log.pdf / .png    (panel B on a log scale; unless --no-log)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from config import (
    DATASET_MINIMAL, ENDPOINTS, IPSS_ITEM_COLS, PROJECT_DIR,
    RESTRICT_DVH_COMBINE, RESTRICT_DVH_SOURCES, RESTRICT_TO_DVH_COHORT,
    TRAJECTORY_EXTRAP_FACTOR, with_suffix,
)
from utils import ensure_dir

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'
# ============================================================
# FIGURE SETTINGS
# ============================================================

FIG_DIR = PROJECT_DIR / with_suffix("figures")
STEM = "F0_endpoint_windows"
STEM_LOG = "F0_endpoint_windows_log"
STEM_COVERAGE = "F0_endpoint_coverage"

# Taille FINALE sur le slide (pouces). Un slide Beamer 16:9 fait ~6.3 in de
# wide in the text area: drawing at this size gives correct font sizes with no
# \includegraphics rescaling, which is the main driver of legibility for a
# projected figure.
FIG_W, FIG_H = 6.9, 4.3

# Palette: a single accent for every window, since they are instances of the
# SAME object and the x axis already carries the time order, so colour has
# nothing further to encode. It is set against the blue of the trajectory;
# blue/amber is the safest pair under colour vision deficiency.
INK = "#1F2933"          # texte principal
MUTED = "#7B8794"        # texte secondaire, axes
TRAJ = "#25476A"         # trajectoire IPSS
TRAJ_FILL = "#25476A"    # IQR band (alpha applied)
BARS = "#C3CCD5"         # histogramme du suivi
ACCENT = "#D97706"       # window fill
ACCENT_DARK = "#9A5400"  # target line and labels

WIN_ALPHA = 0.16
IQR_ALPHA = 0.16

# Binning of the trajectory (panel A) and of the histogram (panel B), in days
TRAJ_BIN_DAYS = 90
TRAJ_MIN_N = 15          # bins with fewer observations are not drawn
HIST_BIN_DAYS = 30

# Log variant of panel B. The axis floor (a log axis is never 0) is the decade
# just below the sparsest non-empty bin; going lower would waste an empty decade
# of useful height. `HEADROOM` multiplies the peak to reserve room for the n (%)
# label block above the bars.
HIST_LOG_FLOOR_MIN = 0.5
HIST_LOG_HEADROOM = 5.0

# Margin to the right of the last window, in days. Measurements beyond it are
# not displayed; their number is reported in the log.
X_PAD_RIGHT = 90
X_PAD_LEFT = 200         # pre-treatment area shown left of x = 0

SHOW_TITLE = False       # a slide already carries its own title

# Readable label of each target. Endpoints absent from this dict fall back to an
# automatic months/years conversion.
TARGET_LABELS = {30: "1 mo", 400: "13 mo", 730: "2 y",
                 1125: "3 y", 1460: "4 y", 1865: "5 y"}

TXT = {
    "en": {
        "x": "Days since implant",
        "x2": "Years since implant",
        "y_top": "IPSS",
        "y_bot": f"IPSS measurements\nper {HIST_BIN_DAYS} d",
        "y_bot_log": f"IPSS measurements\nper {HIST_BIN_DAYS} d (log)",
        "median": "Cohort median IPSS",
        "baseline": "Pre-tx baseline (median)",
        "window": "Endpoint window (target $\\pm$ half-width)",
        "d": "d",
        "target": "Endpoint target",
        "pretx": "pre-tx",
        "title": "Prediction endpoints on the follow-up of the cohort",
        "cov_y": "Patients (%)",
        "cov_title": "Target availability per endpoint",
        "real": "Real measurement in window",
        "model": "Trajectory model (supported)",
        "excl": "Excluded (unsupported)",
        "cohort": "cohort",
        "obs": "IPSS measurements",
        "patients": "patients",
        "beyond": "further measurements beyond",
    },
    "fr": {
        "x": "Jours depuis l'implant",
        "x2": "Années depuis l'implant",
        "y_top": "IPSS",
        "y_bot": f"Mesures IPSS\npar {HIST_BIN_DAYS} j",
        "y_bot_log": f"Mesures IPSS\npar {HIST_BIN_DAYS} j (log)",
        "median": "IPSS médian (IQR)",
        "baseline": "Ligne de base pré-tx (médiane)",
        "window": "Fenêtre d'endpoint (cible $\\pm$ demi-fenêtre)",
        "d": "j",
        "target": "Cible de l'endpoint",
        "pretx": "pré-tx",
        "title": "Endpoints de prédiction sur le suivi de la cohorte",
        "cov_y": "Patients (%)",
        "cov_title": "Disponibilité de la cible par endpoint",
        "real": "Mesure réelle dans la fenêtre",
        "model": "Modèle de trajectoire (supporté)",
        "excl": "Exclu (non supporté)",
        "cohort": "cohorte",
        "obs": "mesures IPSS",
        "patients": "patients",
        "beyond": "mesures supplémentaires au-delà de",
    },
}


def set_style() -> None:
    """Shared style for the article figures (see 06_figures.py): serif, quiet axes."""
    plt.rc("font", family="serif")
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        # "pdf.fonttype": 42,   # embed TrueType fonts for a crisp PDF
        "ps.fonttype": 42,
    })


# ============================================================
# DATA
# ============================================================

def load_long(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df["ipss_date"] = pd.to_datetime(df["ipss_date"], errors="coerce")
    df["days_since_tx"] = (df["ipss_date"] - df["tx_date"]).dt.days
    df["ipss"] = pd.to_numeric(df["ipss_score_calc"], errors="coerce")
    df = df.dropna(subset=["days_since_tx", "ipss"]).copy()
    df["days_since_tx"] = df["days_since_tx"].astype(float)
    return df.sort_values(["record_id", "days_since_tx"]).reset_index(drop=True)


def apply_pipeline_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the cohort TO THE SAME patients as the pipeline (01_prepare_target).

    Deux filtres, dans l'ordre du pipeline :
      1. usable pre-treatment baseline = a measurement exists strictly before
         tx_date, and the LAST of them carries at least one IPSS item (a..g);
         otherwise the subscores are underivable and the patient is excluded;
      2. DVH availability (config.RESTRICT_TO_DVH_COHORT), through the same
         function as the pipeline. If the DVH module cannot be imported, a
         warning is printed and the filter is skipped rather than failing.
    """
    n0 = df["record_id"].nunique()

    pre = df[df["days_since_tx"] < 0]
    last_pre = pre.groupby("record_id", as_index=False).tail(1)
    items = [c for c in IPSS_ITEM_COLS if c in last_pre.columns]
    if items:
        last_pre = last_pre[last_pre[items].notna().any(axis=1)]
    keep_ids = set(last_pre["record_id"])
    df = df[df["record_id"].isin(keep_ids)]
    print(f"  - usable pre-treatment baseline: {df['record_id'].nunique()}/{n0} patients")

    if RESTRICT_TO_DVH_COHORT:
        try:
            from dvh_mc import dvh_cohort_record_ids
            eligible = dvh_cohort_record_ids(RESTRICT_DVH_SOURCES, RESTRICT_DVH_COMBINE)
            mask = df["record_id"].astype(str).str.strip().isin(eligible)
            n_before = df["record_id"].nunique()
            df = df[mask]
            print(f"  • restriction cohorte DVH ({RESTRICT_DVH_COMBINE} de "
                  f"{RESTRICT_DVH_SOURCES}) : {df['record_id'].nunique()}/{n_before} patients")
        except Exception as exc:  # noqa: BLE001 - optional filter, warn only
            print(f"  ! DVH filter not applied ({type(exc).__name__}: {exc})")

    return df.copy()


def binned_median(days: np.ndarray, vals: np.ndarray, bin_days: float,
                  t_max: float, min_n: int):
    """Median and quartiles of the IPSS per post-treatment time bin (>= 0)."""
    edges = np.arange(0.0, t_max + bin_days, bin_days)
    idx = np.digitize(days, edges) - 1
    centres, med, q1, q3 = [], [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if int(m.sum()) < min_n:
            continue
        v = vals[m]
        centres.append((edges[i] + edges[i + 1]) / 2)
        med.append(np.median(v))
        a, b = np.percentile(v, [25, 75])
        q1.append(a)
        q3.append(b)
    return (np.array(centres), np.array(med), np.array(q1), np.array(q3))


def target_label(target: int) -> str:
    if target in TARGET_LABELS:
        return TARGET_LABELS[target]
    years = target / 365.25
    if abs(years - round(years)) < 0.05:
        return f"{round(years)} y"
    return f"{round(target / 30.44)} mo"


def endpoint_status(df: pd.DataFrame) -> pd.DataFrame:
    """real / model / excluded status of each patient at each endpoint.

    Replicates `estimate_at_endpoint()` from 01_prepare_target.py exactly,
    without fitting the mixed model: the classification depends only on the
    measurement days.
    """
    post = df[df["days_since_tx"] >= 0]
    spans = post.groupby("record_id")["days_since_tx"].agg(["min", "max"])
    ids = sorted(df["record_id"].unique())
    days_by_id = {rid: g.to_numpy() for rid, g in post.groupby("record_id")["days_since_tx"]}

    rows = {}
    for target, window in ENDPOINTS.items():
        tol = TRAJECTORY_EXTRAP_FACTOR * window
        col = []
        for rid in ids:
            d = days_by_id.get(rid)
            if d is None or len(d) == 0:
                col.append("excluded")
            elif np.any(np.abs(d - target) <= window):
                col.append("real")
            elif (spans.loc[rid, "min"] - tol) <= target <= (spans.loc[rid, "max"] + tol):
                col.append("model")
            else:
                col.append("excluded")
        rows[target] = col
    return pd.DataFrame(rows, index=pd.Index(ids, name="record_id"))


# ============================================================
# FIGURE PRINCIPALE
# ============================================================

def draw_windows(ax, targets, windows, y_label_frac=None, line_ymax=1.0,
                 unit="d", tag_fmt="E{i}"):
    """Shaded windows plus the target line; optional E1..En labels.

    `line_ymax` stops the target line below the label block (otherwise it
    traverse et rend le texte illisible).
    """
    for i, (t, w) in enumerate(zip(targets, windows), start=1):
        ax.axvspan(t - w, t + w, color=ACCENT, alpha=WIN_ALPHA, lw=0, zorder=0)
        ax.axvline(t, ymax=line_ymax, color=ACCENT_DARK, lw=0.8, ls=(0, (3, 2)),
                   alpha=0.85, zorder=1)
        if y_label_frac is not None:
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(t, y_label_frac, tag_fmt.format(i=i), transform=tr,
                    ha="center", va="top", fontsize=7.5, fontweight="bold",
                    color=ACCENT_DARK, zorder=5)
            ax.text(t, y_label_frac - 0.085, f"{t} {unit} $\\pm$ {w}",
                    transform=tr, ha="center", va="top", fontsize=6.5,
                    color=ACCENT_DARK, zorder=5)


def make_main_figure(df: pd.DataFrame, lang: str, out_dir: Path,
                     log_hist: bool = False, stem: str = STEM) -> Path:
    """Main figure. `log_hist` switches panel B to a logarithmic scale."""
    txt = TXT[lang]
    targets = sorted(ENDPOINTS)
    windows = [ENDPOINTS[t] for t in targets]

    x_max = max(t + w for t, w in ENDPOINTS.items()) + X_PAD_RIGHT
    x_min = -X_PAD_LEFT

    post = df[df["days_since_tx"] >= 0]
    n_pat = df["record_id"].nunique()
    n_obs = len(df)
    n_beyond = int((post["days_since_tx"] > x_max).sum())

    # Baseline: the last strictly pre-treatment measurement of each patient.
    pre_last = df[df["days_since_tx"] < 0].groupby("record_id").tail(1)
    baseline_med = float(pre_last["ipss"].median())

    centres, med, q1, q3 = binned_median(
        post["days_since_tx"].to_numpy(), post["ipss"].to_numpy(),
        TRAJ_BIN_DAYS, x_max, TRAJ_MIN_N)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw=dict(height_ratios=[2.15, 1.0], hspace=0.12))

    # ---------------- Panneau A — trajectoire IPSS ----------------
    y_top = float(np.nanmax(q3)) if len(q3) else 20.0
    ax1.set_ylim(0, y_top * 1.34)
    draw_windows(ax1, targets, windows, y_label_frac=0.995, line_ymax=0.83,
                 unit=txt["d"])

    ax1.axvline(0, color=MUTED, lw=0.8, ls="-", alpha=0.6, zorder=1)
    ax1.axhline(baseline_med, color=INK, lw=0.9, ls=(0, (5, 2)), alpha=0.75,
                zorder=2, label=txt["baseline"])
    ax1.fill_between(centres, q1, q3, color=TRAJ_FILL, alpha=IQR_ALPHA, lw=0, zorder=2)
    ax1.plot(centres, med, color=TRAJ, lw=1.8, solid_capstyle="round",
             zorder=4, label=txt["median"])
    ax1.plot([-60], [baseline_med], marker="o", ms=5, color=INK, zorder=5,
             clip_on=False)
    ax1.annotate(txt["pretx"], xy=(-60, baseline_med), xytext=(0, -11),
                 textcoords="offset points", ha="center", va="top",
                 fontsize=6.5, color=MUTED)

    ax1.set_ylabel(txt["y_top"])
    ax1.grid(axis="y", color=MUTED, alpha=0.18, lw=0.6)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    sec = ax1.secondary_xaxis(
        "top", functions=(lambda d: d / 365.25, lambda y: y * 365.25))
    sec.set_xlabel(txt["x2"], labelpad=3)
    sec.set_xticks(np.arange(0, 6))
    sec.spines["top"].set_color(MUTED)

    # Legend outside the axes (one row below the figure), so it cannot collide
    # with the trajectory or with the window labels.
    handles = [
        Line2D([], [], color=TRAJ, lw=1.8, label=txt["median"]),
        Line2D([], [], color=INK, lw=0.9, ls=(0, (5, 2)), label=txt["baseline"]),
        Patch(facecolor=ACCENT, alpha=WIN_ALPHA + 0.06, edgecolor="none",
              label=txt["window"]),
    ]

    # ---------------- Panel B - follow-up density ----------------
    bins = np.arange(0, x_max + HIST_BIN_DAYS, HIST_BIN_DAYS)
    counts, _ = np.histogram(post["days_since_tx"].to_numpy(), bins=bins)
    c_max = float(counts.max()) if counts.size else 1.0
    if log_hist:
        pos = counts[counts > 0]
        floor = max(HIST_LOG_FLOOR_MIN,
                    10.0 ** np.floor(np.log10(pos.min())) if pos.size else 1.0)
        # Bars drawn from the axis floor: on a log axis a base of 0 would be
        # at -inf and matplotlib would clip the rectangles unpredictably. Empty
        # bins have no representable height, so they get no bar.
        heights = np.where(counts > 0, np.maximum(counts, floor) - floor, 0.0)
        ax2.bar(bins[:-1], heights, bottom=floor, width=HIST_BIN_DAYS * 0.92,
                align="edge", color=TRAJ, lw=0, zorder=2)
        ax2.set_yscale("log")
        ax2.set_ylim(floor, max(c_max * HIST_LOG_HEADROOM, floor * 100))
        ax2.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax2.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
        ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax2.grid(axis="y", which="minor", color=MUTED, alpha=0.10, lw=0.4)
    else:
        ax2.bar(bins[:-1], counts, width=HIST_BIN_DAYS * 0.92, align="edge",
                color=TRAJ, lw=0, zorder=2)
        ax2.set_ylim(0, c_max * 1.42)
    draw_windows(ax2, targets, windows, line_ymax=0.72)
    ax2.axvline(0, color=MUTED, lw=0.8, alpha=0.6, zorder=1)

    # n (and %) of patients with a REAL measurement inside each window
    tr2 = blended_transform_factory(ax2.transData, ax2.transAxes)
    for t, w in zip(targets, windows):
        inw = post[(post["days_since_tx"] - t).abs() <= w]
        n_real = inw["record_id"].nunique()
        ax2.text(t, 0.97, f"{n_real}\n({100 * n_real / n_pat:.0f}\\%)"
                 if plt.rcParams["text.usetex"] else
                 f"{n_real}\n({100 * n_real / n_pat:.0f}%)",
                 transform=tr2, ha="center", va="top", fontsize=6.5,
                 color=ACCENT_DARK, linespacing=1.2, zorder=5)

    ax2.set_ylabel(txt["y_bot_log"] if log_hist else txt["y_bot"], linespacing=1.3)
    ax2.set_xlabel(txt["x"])
    ax2.set_xlim(x_min, x_max)
    ax2.set_xticks(np.arange(0, x_max, 365))
    ax2.grid(axis="y", which="major", color=MUTED, alpha=0.18, lw=0.6)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4,
               fontsize=7.2, borderaxespad=0.0)

    # Framing note: sample sizes, and measurements outside the displayed window
    note = f"n = {n_pat} {txt['patients']}, {n_obs} {txt['obs']}."
    if n_beyond:
        note += f" {n_beyond} {txt['beyond']} {x_max:.0f} {txt['d']}."
    # fig.text(0.5, 0.008, note, fontsize=6.3, color=MUTED,
            #  ha="center", va="bottom")
# 
    if SHOW_TITLE:
        fig.suptitle(txt["title"], fontsize=10, y=0.995)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.865, bottom=0.215)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    return pdf


# ============================================================
# COMPLEMENTARY FIGURE - target coverage
# ============================================================

def make_coverage_figure(df: pd.DataFrame, lang: str, out_dir: Path) -> Path:
    txt = TXT[lang]
    status = endpoint_status(df)
    targets = sorted(ENDPOINTS)
    n_pat = len(status)

    frac = {k: [] for k in ("real", "model", "excluded")}
    for t in targets:
        col = status[t]
        for k in frac:
            frac[k].append(100 * float((col == k).sum()) / n_pat)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.62))
    x = np.arange(len(targets))
    colors = {"real": TRAJ, "model": "#8FA8C2", "excluded": "#DDE3E9"}
    labels = {"real": txt["real"], "model": txt["model"], "excluded": txt["excl"]}

    bottom = np.zeros(len(targets))
    for k in ("real", "model", "excluded"):
        vals = np.array(frac[k])
        # White hairline between stacked segments, separating the blocks
        ax.bar(x, vals, bottom=bottom, width=0.62, color=colors[k],
               edgecolor="white", lw=0.8, label=labels[k], zorder=2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:  # label only the segments large enough to read
                ax.text(xi, b + v / 2, f"{v:.0f}",
                        ha="center", va="center", fontsize=6.8,
                        color="white" if k == "real" else INK, zorder=3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}\n{t} {txt['d']}\n{target_label(t)}"
                        for i, t in enumerate(targets, start=1)],
                       fontsize=7, linespacing=1.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel(txt["cov_y"])
    ax.grid(axis="y", color=MUTED, alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, handlelength=1.2, columnspacing=1.6)
    fig.text(0.012, 0.012, f"n = {n_pat} {txt['patients']}",
             fontsize=6.3, color=MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{STEM_COVERAGE}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{STEM_COVERAGE}.png")
    plt.close(fig)

    print("\nCoverage per endpoint (% of the cohort):")
    print(f"  {'endpoint':<12} {'real':>7} {'model':>7} {'excluded':>9}")
    for i, t in enumerate(targets):
        print(f"  E{i+1} {t:>5} d  {frac['real'][i]:>6.1f} "
              f"{frac['model'][i]:>7.1f} {frac['excluded'][i]:>9.1f}")
    return pdf


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--lang", choices=("en", "fr"), default="en",
                   help="label language (default: en, as in 06_figures.py)")
    p.add_argument("--cohort", choices=("pipeline", "all"), default="pipeline",
                   help="pipeline = the same filters as 01_prepare_target.py")
    p.add_argument("--coverage", action="store_true",
                   help="produit aussi la figure real/model/exclu par endpoint")
    p.add_argument("--no-log", dest="log", action="store_false",
                   help="do not produce the logarithmic panel B variant")
    p.add_argument("--out", type=Path, default=FIG_DIR, help="dossier de sortie")
    args = p.parse_args()

    set_style()
    ensure_dir(args.out)

    print("=" * 70)
    print("FIGURE - endpoint windows over the follow-up of the cohort")
    print("=" * 70)
    df = load_long(DATASET_MINIMAL)
    print(f"[ok] Loaded: {len(df)} IPSS measurements, {df['record_id'].nunique()} patients")

    if args.cohort == "pipeline":
        print("Restriction to the pipeline cohort:")
        df = apply_pipeline_cohort(df)
    print(f"-> Cohort plotted: {df['record_id'].nunique()} patients, {len(df)} measurements")

    print("\nEndpoints (config.ENDPOINTS) :")
    for i, t in enumerate(sorted(ENDPOINTS), start=1):
        w = ENDPOINTS[t]
        print(f"  E{i}  {t:>5} d ± {w:>3} d  → [{t - w:>5} ; {t + w:>5}]  "
              f"({target_label(t)})")

    pdf = make_main_figure(df, args.lang, args.out)
    print(f"\n✓ {pdf}")
    print(f"✓ {pdf.with_suffix('.png')}")

    if args.log:
        pdf_log = make_main_figure(df, args.lang, args.out,
                                   log_hist=True, stem=STEM_LOG)
        print(f"\n[ok] {pdf_log}  (panel B on a log scale)")
        print(f"✓ {pdf_log.with_suffix('.png')}")

    if args.coverage:
        pdf2 = make_coverage_figure(df, args.lang, args.out)
        print(f"\n✓ {pdf2}")
        print(f"✓ {pdf2.with_suffix('.png')}")


if __name__ == "__main__":
    main()
