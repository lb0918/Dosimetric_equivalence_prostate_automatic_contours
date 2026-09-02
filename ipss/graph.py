#!/usr/bin/env python3
"""
Aggregate per-endpoint SHAP value files (MLP) into a single beeswarm-style plot.

Context
-------
Each input CSV contains one row per patient and one column per feature, where the
*values are SHAP contributions* (not the raw feature values). Files are named
``shap_values_mlp_y_<ENDPOINT>.csv`` (e.g. y_1500d, y_1862d, ...).

Important data facts (verified on the provided files):
  * The patient_id sets DIFFER across endpoints (no patient is shared by all 6),
    so SHAP values cannot be averaged dot-by-dot (patient x feature). They can
    only be POOLED across endpoints.
  * The files contain SHAP values only -> no raw feature values are available,
    so the beeswarm cannot be colored by feature value (the red/blue of a
    classic shap.summary_plot). We color by ENDPOINT instead, which also reveals
    cross-endpoint consistency.

What "averaged over endpoints" means here:
  * IMPORTANCE / ordering  -> averaged: mean over endpoints of mean(|SHAP|).
    Each endpoint contributes equally regardless of its number of features.
  * DISTRIBUTION / beeswarm -> pooled: all per-patient SHAP dots concatenated.

Outputs
-------
  * shap_beeswarm_endpoints_mean.png : pooled beeswarm, features ordered by
    endpoint-averaged mean(|SHAP|), colored by endpoint.
  * shap_importance_endpoints_mean.png : companion bar chart of the
    endpoint-averaged mean(|SHAP|) (the rigorous "SHAP averaged over endpoints").
  * shap_importance_endpoints_mean.csv : the importance table behind the plots.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import PROJECT_DIR, RESULTS_DIR, DVH_RUN_TAG

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Follows the active DVH strategy automatically (config.RESULTS_DIR =
# results_<tag>). Replace with an explicit path to pin a specific run.
INPUT_DIR = str(RESULTS_DIR)
INPUT_GLOB = "shap_values_mlp_y_*.csv"
ID_COL = "patient_id"

TOP_N = 15            # number of features to display (like the reference figure)
MODEL_NAME = "MLP"
RANDOM_SEED = 0       # for reproducible beeswarm jitter

# Order endpoints by their numeric day value rather than alphabetically.
def endpoint_sort_key(label: str) -> float:
    m = re.search(r"(\d+)", label)
    return float(m.group(1)) if m else float("inf")
# Display names for the y-axis. Any feature not listed falls back to its raw name.
# NOTE: usetex=True is active, so escape underscores ("\_") or avoid them entirely.
FEATURE_LABELS = {
    "num__pretx_prostsex_ipss_a": "IPSS a (pre-tx)",
    "num__pretx_prostsex_ipss_b": "IPSS b (pre-tx)",
    "num__pretx_prostsex_ipss_c": "IPSS c (pre-tx)",
    "num__pretx_prostsex_ipss_d": "IPSS d (pre-tx)",
    "num__pretx_prostsex_ipss_e": "IPSS e (pre-tx)",
    "num__pretx_prostsex_ipss_f": "IPSS f (pre-tx)",
    "num__pretx_prostsex_ipss_g": "IPSS g (pre-tx)",
    "num__pretx_prostsex_qual_vie_a": "Quality of life a (pre-tx)",
    "num__pretx_prostsex_qual_vie_b": "Quality of life b (pre-tx)",
    "num__pretx_shim_score": "SHIM score (pre-tx)",
    "num__age": "Age",
    "num__isup_grade": "ISUP grade",
    "num__crude_psa": "Crude PSA",
    "num__ldr_post_vol": "Prostate volume, post-implant",
    "num__ldr_live_aiguilles": "Number of needles",
    "num__dvh_urethra_available": "Urethra DVH available",
    "num__dvh_bladder_available": "Bladder DVH available",
    "num__dvh_bladderneck_available": "Bladder neck DVH available",
    "num__ipss_days": "IPSS delay (days)",
    "cat__gleason_6.0": "Gleason",
    # Extend as needed; unlisted features keep their raw name.
}


def relabel(features: list[str]) -> list[str]:
    """Map raw feature names to display names, falling back to the raw name."""
    return [FEATURE_LABELS.get(f, f) for f in features]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_endpoints(input_dir: str, pattern: str) -> dict[str, pd.DataFrame]:
    """Load every endpoint file into {endpoint_label: dataframe_of_shap_values}."""
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {input_dir}")

    endpoints: dict[str, pd.DataFrame] = {}
    for path in paths:
        base = os.path.basename(path)
        # shap_values_mlp_y_1500d.csv -> y_1500d
        label = base.replace("shap_values_mlp_", "").replace(".csv", "")
        df = pd.read_csv(path)
        if ID_COL in df.columns:
            df = df.set_index(ID_COL)
        endpoints[label] = df
    return dict(sorted(endpoints.items(), key=lambda kv: endpoint_sort_key(kv[0])))


# --------------------------------------------------------------------------- #
# Importance averaged over endpoints
# --------------------------------------------------------------------------- #
def endpoint_averaged_importance(endpoints: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Mean over endpoints of mean(|SHAP|) per feature.

    A feature absent from an endpoint simply does not contribute to that
    endpoint's mean; the across-endpoint average is taken over the endpoints in
    which the feature appears. This keeps each endpoint on an equal footing.
    """
    per_endpoint_mean_abs = {}
    for label, df in endpoints.items():
        per_endpoint_mean_abs[label] = df.abs().mean(axis=0)  # mean over patients
    imp = pd.DataFrame(per_endpoint_mean_abs)  # features x endpoints
    # mean across endpoints, ignoring endpoints where the feature is absent
    return imp.mean(axis=1, skipna=True).sort_values(ascending=False)




# --------------------------------------------------------------------------- #
# Plot: companion bar chart of endpoint-averaged mean(|SHAP|)
# --------------------------------------------------------------------------- #
def plot_importance_bar(endpoints: dict[str, pd.DataFrame], importance: pd.Series,
                        top_n: int) -> None:
    labels = list(endpoints.keys())
    feats = importance.head(top_n).index.tolist()[::-1]

    # per-endpoint mean(|SHAP|) to show the spread behind the average
    per_ep = {lab: endpoints[lab].abs().mean(axis=0) for lab in labels}
    per_ep = pd.DataFrame(per_ep)

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(feats) + 2.0))
    ax.tick_params(axis='both', labelsize=14)
    y = np.arange(len(feats))
    ax.barh(y, importance.loc[feats].to_numpy(), color="#5078b4",
            alpha=0.85, zorder=2, label="mean over endpoints")

    # overlay individual endpoint values as dots
    cmap = plt.get_cmap("turbo")
    colors = {lab: cmap(t) for lab, t in zip(labels, np.linspace(0.08, 0.92, len(labels)))}
    for lab in labels:
        xs = per_ep[lab].reindex(feats).to_numpy()
        ax.scatter(xs, y, s=18, color=colors[lab], zorder=3, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(relabel(feats), fontsize=15)
    ax.set_xlabel("Moyenne(|SHAP|)",fontsize=15)
    ax.set_title(f"SHAP importance — {MODEL_NAME} — averaged over endpoints (top {top_n})", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", color="0.85", zorder=0)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[lab],
                      markersize=7, label=lab) for lab in labels]
    ax.legend(handles=handles, title="Endpoint", loc="lower right",
              frameon=False, fontsize=15, title_fontsize=15)

    fig.tight_layout()
    fig.savefig(PROJECT_DIR / f"SHAP_fig_{DVH_RUN_TAG}.pdf",
                dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    endpoints = load_endpoints(INPUT_DIR, INPUT_GLOB)
    print(f"Loaded {len(endpoints)} endpoints: {list(endpoints.keys())}")

    importance = endpoint_averaged_importance(endpoints)
    # imp_path = os.path.join(OUTPUT_DIR, "shap_importance_endpoints_mean.csv")
    # importance.rename("mean_abs_shap_over_endpoints").to_csv(imp_path)
    # print(f"Wrote importance table -> {imp_path}")

    # bee_path = os.path.join(OUTPUT_DIR, "shap_beeswarm_endpoints_mean.png")
    # plot_beeswarm(endpoints, importance, TOP_N)
    # print(f"Wrote beeswarm -> {bee_path}")

    # bar_path = os.path.join(OUTPUT_DIR, "shap_importance_endpoints_mean.png")
    plot_importance_bar(endpoints, importance, TOP_N)
    # print(f"Wrote importance bar -> {bar_path}")


if __name__ == "__main__":
    main()