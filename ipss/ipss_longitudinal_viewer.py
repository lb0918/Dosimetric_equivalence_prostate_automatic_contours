"""
IPSS Longitudinal Viewer
------------------------
Interactive visualisation of the IPSS over time for a cohort.

Expected CSV (long format, one IPSS measurement per row):
    record_id, tx_date, ipss_date, ipss_score_calc, [stage, gleason, adt, ...]

The delay since treatment (ipss_days) is computed here as
(ipss_date - tx_date) in days.

Controls:
  - "Bin (days)" slider: bin width in days
  - "T min" / "T max" sliders: displayed time window
  - "n min / bin" slider: minimum number of observations per bin
  - "Winsorize %" slider: cap values at the percentiles before aggregating
  - "Stratification" radio buttons: none | adt | gleason | stage | ...
  - Check buttons: log Y scale, median/IQR, IQR band, n per bin
  - "Reset" button

Usage:
    Set CSV_PATH below (or PROTECTA_DATA_ROOT), then run:
        python ipss_longitudinal_viewer.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

# ---------------------------------------------------------------------------
# CONFIGURATION - set the input path here
# ---------------------------------------------------------------------------

from config import DATASET_MINIMAL

CSV_PATH = Path(DATASET_MINIMAL)

# Column names in the CSV.
DATE_TX_COL = "tx_date"        # treatment date
DATE_IPSS_COL = "ipss_date"    # IPSS measurement date
SCORE_COL = "ipss_score_calc"  # total IPSS value

# Candidate stratification columns (used when present).
STRAT_COLS = ("adt", "isup_grade", "gleason", "stage")

# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def load_ipss(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"record_id", DATE_TX_COL, DATE_IPSS_COL, SCORE_COL}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Colonnes manquantes dans le CSV : {missing}")

    # Delay since treatment, in days.
    df[DATE_TX_COL] = pd.to_datetime(df[DATE_TX_COL], errors="coerce")
    df[DATE_IPSS_COL] = pd.to_datetime(df[DATE_IPSS_COL], errors="coerce")
    df["ipss_days"] = (df[DATE_IPSS_COL] - df[DATE_TX_COL]).dt.days

    # Numeric IPSS value.
    df["ipss_val"] = pd.to_numeric(df[SCORE_COL], errors="coerce")

    # Keep only the rows with a valid IPSS and a valid delay.
    df = df.dropna(subset=["ipss_days", "ipss_val"]).copy()
    df["ipss_days"] = pd.to_numeric(df["ipss_days"], errors="coerce")
    df = df.dropna(subset=["ipss_days", "ipss_val"])

    # Optional stratification columns, normalised when present.
    for col in STRAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def compute_bins(days: np.ndarray, values: np.ndarray, bin_width: float,
                 t_min: float, t_max: float, use_median: bool,
                 min_n: int = 1, winsorize_pct: float = 0.0):
    """Return a dict of the per-bin statistics.

    Keys: centres, central, low, high, q1, q3, counts.
    - central: mean or median, depending on use_median
    - low, high: SD (symmetric) or distance to Q1/Q3 (asymmetric)
    - q1, q3: always returned, for the shaded IQR band
    - winsorize_pct : si > 0, cap les valeurs aux percentiles
      [winsorize_pct, 100 - winsorize_pct] AVANT de calculer (ex. 1.0 → [1, 99]).
    """
    empty = {k: np.array([]) for k in
             ("centres", "central", "low", "high", "q1", "q3", "counts")}
    if bin_width <= 0 or t_max <= t_min:
        return empty

    edges = np.arange(t_min, t_max + bin_width, bin_width)
    if len(edges) < 2:
        return empty

    idx = np.digitize(days, edges) - 1
    centres, central, low, high, q1s, q3s, counts = [], [], [], [], [], [], []

    for i in range(len(edges) - 1):
        mask = idx == i
        n = int(mask.sum())
        if n < min_n:
            continue
        vals = values[mask]

        if winsorize_pct > 0 and n >= 4:
            lo_cut, hi_cut = np.percentile(
                vals, [winsorize_pct, 100 - winsorize_pct]
            )
            vals = np.clip(vals, lo_cut, hi_cut)

        c = (edges[i] + edges[i + 1]) / 2
        q1, q3 = np.percentile(vals, [25, 75])

        if use_median:
            med = np.median(vals)
            central.append(med)
            low.append(med - q1)
            high.append(q3 - med)
        else:
            m = np.mean(vals)
            sd = np.std(vals, ddof=1) if n > 1 else 0.0
            central.append(m)
            low.append(sd)
            high.append(sd)

        centres.append(c)
        q1s.append(q1)
        q3s.append(q3)
        counts.append(n)

    return {
        "centres": np.array(centres),
        "central": np.array(central),
        "low": np.array(low),
        "high": np.array(high),
        "q1": np.array(q1s),
        "q3": np.array(q3s),
        "counts": np.array(counts),
    }


# ---------------------------------------------------------------------------
# Groups for the stratification
# ---------------------------------------------------------------------------

def build_groups(df: pd.DataFrame, strat: str):
    """Return a list of (label, sub_df) according to the chosen stratification."""
    if strat == "None" or strat not in df.columns:
        return [("Cohort", df)]

    groups = []
    for value, sub in df.groupby(strat, dropna=False):
        if pd.isna(value):
            label = f"{strat}=NA"
        else:
            label = f"{strat}={value}"
        groups.append((label, sub))
    # Stable sort by label, for consistent colours.
    groups.sort(key=lambda x: str(x[0]))
    return groups


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class IPSSViewer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Default bounds derived from the data.
        d_min = float(df["ipss_days"].min())
        d_max = float(df["ipss_days"].max())
        self.default_tmin = d_min
        self.default_tmax = d_max
        self.default_bin = max(30.0, round((d_max - d_min) / 30, 0))

        # Determine the available stratifications.
        self.strat_options = ["None"]
        for col in STRAT_COLS:
            if col in df.columns:
                self.strat_options.append(col)

        # Figure et axes
        self.fig = plt.figure(figsize=(13, 8.5))
        self.fig.canvas.manager.set_window_title("Longitudinal IPSS — cohort")
        self.ax = self.fig.add_axes([0.30, 0.36, 0.66, 0.58])

        # --- Sliders ---
        ax_bin = self.fig.add_axes([0.30, 0.26, 0.55, 0.025])
        ax_tmin = self.fig.add_axes([0.30, 0.22, 0.55, 0.025])
        ax_tmax = self.fig.add_axes([0.30, 0.18, 0.55, 0.025])
        ax_nmin = self.fig.add_axes([0.30, 0.14, 0.55, 0.025])
        ax_wins = self.fig.add_axes([0.30, 0.10, 0.55, 0.025])

        bin_max = max(self.default_bin * 6, 730.0)
        self.s_bin = Slider(ax_bin, "Bin (days)", 1.0, bin_max,
                            valinit=self.default_bin, valstep=1.0)
        self.s_tmin = Slider(ax_tmin, "T min (d)", d_min, d_max,
                             valinit=d_min, valstep=1.0)
        self.s_tmax = Slider(ax_tmax, "T max (d)", d_min, d_max,
                             valinit=d_max, valstep=1.0)
        self.s_nmin = Slider(ax_nmin, "n min / bin", 1, 100,
                             valinit=10, valstep=1)
        self.s_wins = Slider(ax_wins, "Winsorize %", 0.0, 10.0,
                             valinit=0.0, valstep=0.5)

        for s in (self.s_bin, self.s_tmin, self.s_tmax,
                  self.s_nmin, self.s_wins):
            s.on_changed(self._on_change)

        # --- Radio stratification ---
        ax_radio = self.fig.add_axes([0.02, 0.62, 0.22, 0.30])
        ax_radio.set_title("Stratification", fontsize=10)
        self.radio = RadioButtons(ax_radio, self.strat_options, active=0)
        self.radio.on_clicked(lambda _label: self._on_change(None))

        # --- Checkbuttons ---
        ax_check = self.fig.add_axes([0.02, 0.30, 0.22, 0.28])
        ax_check.set_title("Options", fontsize=10)
        self.check_labels = [
            "Log Y scale",
            "Median + IQR",
            "Shaded IQR band",
            "Show n per bin",
        ]
        # The IPSS is a bounded score, often zero, so the log Y scale is off by default.
        self.check = CheckButtons(
            ax_check, self.check_labels,
            [False, True, True, False],
        )
        self.check.on_clicked(lambda _label: self._on_change(None))

        # --- Bouton reset ---
        ax_btn = self.fig.add_axes([0.02, 0.22, 0.22, 0.05])
        self.btn_reset = Button(ax_btn, "Reset")
        self.btn_reset.on_clicked(self._reset)

        self._draw()
        plt.show()

    # ------------------------------------------------------------------
    def _states(self):
        labels_active = dict(zip(self.check_labels, self.check.get_status()))
        return {
            "bin": float(self.s_bin.val),
            "tmin": float(self.s_tmin.val),
            "tmax": float(self.s_tmax.val),
            "nmin": int(self.s_nmin.val),
            "wins": float(self.s_wins.val),
            "strat": self.radio.value_selected,
            "log_y": labels_active["Log Y scale"],
            "median": labels_active["Median + IQR"],
            "iqr_band": labels_active["Shaded IQR band"],
            "show_n": labels_active["Show n per bin"],
        }

    # ------------------------------------------------------------------
    def _on_change(self, _val):
        # Garantit tmin < tmax
        if self.s_tmin.val >= self.s_tmax.val:
            new_tmin = min(self.s_tmin.val, self.s_tmax.val - 1.0)
            self.s_tmin.eventson = False
            self.s_tmin.set_val(new_tmin)
            self.s_tmin.eventson = True
        self._draw()

    # ------------------------------------------------------------------
    def _reset(self, _evt):
        self.s_bin.reset()
        self.s_tmin.reset()
        self.s_tmax.reset()
        self.s_nmin.reset()
        self.s_wins.reset()

    # ------------------------------------------------------------------
    def _draw(self):
        st = self._states()
        self.ax.clear()

        # In log mode, drop the values <= 0 that would break the axis
        base_df = self.df
        if st["log_y"]:
            base_df = base_df[base_df["ipss_val"] > 0]

        groups = build_groups(base_df, st["strat"])
        cmap = plt.get_cmap("tab10")

        any_data = False
        for i, (label, sub) in enumerate(groups):
            days = sub["ipss_days"].to_numpy()
            vals = sub["ipss_val"].to_numpy()

            res = compute_bins(
                days, vals, st["bin"], st["tmin"], st["tmax"],
                st["median"], min_n=st["nmin"], winsorize_pct=st["wins"],
            )
            centres = res["centres"]
            if len(centres) == 0:
                continue
            any_data = True

            color = cmap(i % 10)
            n_patients = sub["record_id"].nunique()
            n_obs = len(sub)
            full_label = f"{label}  (n_pat={n_patients}, n_obs={n_obs})"

            # Shaded IQR band, always the true quartiles regardless of the
            # mean/median choice.
            if st["iqr_band"]:
                self.ax.fill_between(
                    centres, res["q1"], res["q3"],
                    color=color, alpha=0.18, linewidth=0,
                )

            self.ax.errorbar(
                centres, res["central"],
                yerr=[res["low"], res["high"]],
                fmt="o-", color=color, ecolor=color,
                capsize=3, markersize=5, linewidth=1.4,
                alpha=0.9, label=full_label,
            )

            if st["show_n"]:
                # Annotation placed at the upper end of the error bar.
                y_top = res["central"] + res["high"]
                for x, y, n in zip(centres, y_top, res["counts"]):
                    self.ax.annotate(
                        str(n), (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        color=color,
                    )

        # Mise en forme
        if st["median"]:
            center_label = "Median (± IQR)"
        else:
            center_label = "Mean (± SD)"

        extras = []
        if st["wins"] > 0:
            extras.append(f"winsor {st['wins']:.1f}%")
        if st["nmin"] > 1:
            extras.append(f"n≥{st['nmin']}")
        suffix = f"  [{', '.join(extras)}]" if extras else ""

        self.ax.set_xlabel("Time since treatment (days)")
        self.ax.set_ylabel(f"IPSS — {center_label}")
        self.ax.set_title(
            f"IPSS trajectory — bin = {st['bin']:.0f} d, "
            f"window [{st['tmin']:.0f}; {st['tmax']:.0f}] d{suffix}"
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.axvline(0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)

        if st["log_y"]:
            self.ax.set_yscale("log")

        if any_data:
            self.ax.legend(loc="best", fontsize=9, framealpha=0.9)
        else:
            self.ax.text(0.5, 0.5,
                         "No bin satisfies n ≥ {} within the window".format(st["nmin"]),
                         transform=self.ax.transAxes,
                         ha="center", va="center", fontsize=12, color="grey")

        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        sys.exit(f"Fichier introuvable : {CSV_PATH}\n"
                 f"Modifier la variable CSV_PATH en haut du script.")

    df = load_ipss(CSV_PATH)
    if df.empty:
        sys.exit("No usable IPSS data after cleaning.")

    print(f"Loaded: {len(df)} IPSS observations for "
          f"{df['record_id'].nunique()} patients.")
    print(f"Plage ipss_days : [{df['ipss_days'].min():.0f} ; "
          f"{df['ipss_days'].max():.0f}] jours.")

    IPSSViewer(df)


if __name__ == "__main__":
    main()
