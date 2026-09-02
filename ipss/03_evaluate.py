"""
03_evaluate.py
==============
Reads the out-of-fold predictions saved by 02_train.py and produces:
  - results/metrics.csv             RMSE/MAE/R2 plus calibration slope and
                                    intercept, per endpoint and algorithm
  - results/metrics_summary.md      markdown version of the same table
  - results/scatter_<endpoint>.png  OOF subplots for one endpoint, one panel
                                    per algorithm
  - results/scatter_all.png         aggregated grid (endpoint x algorithm)
  - results/calibration_<endpoint>.png  calibration curves for one endpoint
  - results/calibration_all.png     aggregated calibration grid
  - results/shap_<algo>_<endpoint>.png  SHAP summary of each final model
  - results/shap_values_<algo>_<endpoint>.csv  raw SHAP values
                                    (rows = patients, columns = features)
  - results/shap_summary.csv        mean |SHAP| per (endpoint, algo, feature)
"""
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'


from config import (
    PREP_DIR, MODELS_DIR, RESULTS_DIR, SEED,
    CATEGORICAL_FEATURES, SHAP_BACKGROUND_SIZE, SHAP_MAX_DISPLAY,
    CALIBRATION_N_BINS,
)
from utils import ensure_dir, regression_metrics

warnings.filterwarnings("ignore")
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Central list of the algorithms, in display order.
ALGOS = ["linreg", "elasticnet", "rf", "xgboost", "catboost", "mlp"]


# MLP definition, identical to 02_train.py, needed to reload the checkpoint.
class MLP(nn.Module):
    def __init__(self, n_in, hidden_sizes, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# OOF metrics (RMSE/MAE/R2 plus calibration slope and intercept)
# ============================================================
def calibration_stats(y_true, y_pred):
    """Calibration slope and intercept, from a linear regression of y_true on
    y_pred. A perfectly calibrated model has slope=1 and intercept=0. NaN are
    ignored."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"calib_slope": np.nan, "calib_intercept": np.nan}
    res = linregress(y_pred[mask], y_true[mask])
    return {"calib_slope": float(res.slope),
            "calib_intercept": float(res.intercept)}


def _rep_suffix(df) -> str:
    """Title suffix spelling out the repeated CV (empty for a single partition)."""
    if "repeat" in df.columns and df["repeat"].nunique() > 1:
        return f" \u00d7 {df['repeat'].nunique()} rep."
    return ""


def _n_patients(df) -> int:
    """Number of distinct PATIENTS, which differs from the number of rows under
    repeated CV."""
    return int(df["record_id"].nunique()) if "record_id" in df.columns else len(df)


def _agg_over_repeats(df, algo, fn):
    """Apply `fn(y_true, y_pred)` PER REPETITION, then average the scalars.

    With a single partition (the `repeat` column absent or constant), this is
    exactly `fn()` over the whole table.

    Under repeated CV, each repetition is a COMPLETE OOF prediction of the
    cohort. Averaging the metrics per repetition, rather than pooling the r*n
    rows, matters for two reasons:
      - the returned `n` stays a number of PATIENTS. Pooled, it would be r*n,
        and downstream figures print it as is under their axes, publishing a
        wrong sample size;
      - the calibration statistics (slope, intercept) would otherwise mix
        different partitions in a single fit.
    Adds `rmse_sd` (between-repetition dispersion) and `n_repeats`.
    """
    reps = sorted(df["repeat"].unique()) if "repeat" in df.columns else []
    if len(reps) <= 1:
        out = dict(fn(df["y_true"], df[algo]))
        out["n_repeats"] = 1
        return out
    per = [fn(g["y_true"], g[algo]) for _, g in df.groupby("repeat")]
    out = {}
    for k in per[0]:
        vals = [p[k] for p in per]
        out[k] = int(np.rint(np.nanmean(vals))) if k == "n" else float(np.nanmean(vals))
    if "rmse" in out:
        out["rmse_sd"] = float(np.nanstd([p["rmse"] for p in per], ddof=1))
    out["n_repeats"] = len(reps)
    return out


def compute_all_metrics():
    """OOF metrics per endpoint and algorithm, computed twice:
      - bare columns (rmse, mae, ...) on the REAL targets only
        (source == "real", ground truth); this is the reference evaluation;
      - columns suffixed _all on ALL targets, including the imputed ("model")
        ones.
    If the `source` column is absent, real and all coincide."""
    rows = []
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        endpoint = endpoint_dir.name
        oof_file = endpoint_dir / "oof_predictions.csv"
        if not oof_file.exists():
            continue
        df = pd.read_csv(oof_file)
        df_real = df[df["source"] == "real"] if "source" in df.columns else df
        for algo in ALGOS:
            if algo not in df.columns:
                continue
            m_real = _agg_over_repeats(df_real, algo, regression_metrics)
            c_real = _agg_over_repeats(df_real, algo, calibration_stats)
            m_all = _agg_over_repeats(df, algo, regression_metrics)
            c_all = _agg_over_repeats(df, algo, calibration_stats)
            row = {"endpoint": endpoint, "algo": algo, **m_real, **c_real}
            row.update({f"{k}_all": v for k, v in {**m_all, **c_all}.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def write_metrics_md(metrics_df: pd.DataFrame, out_path: Path):
    lines = [
        "# Model performance (patient-level K-fold CV, OOF predictions)\n",
        "Each cell: **score on the real targets** "
        "(*score on all targets, imputed \"model\" ones included*).\n",
    ]
    metric_cols = ["rmse", "mae", "r2"]
    fmt = metrics_df[["endpoint", "algo"]].copy()
    for col in metric_cols:
        real = metrics_df[col].round(3).astype(str)
        alld = metrics_df[f"{col}_all"].round(3).astype(str)
        fmt[col] = real + " (" + alld + ")"
    pivot = fmt.pivot(index="endpoint", columns="algo", values=metric_cols)
    lines.append(pivot.to_markdown())

    # OOF sample sizes per endpoint (n_real = real targets, n_all = all evaluated)
    counts = (
        metrics_df.groupby("endpoint")
        .agg(n_real=("n", "first"), n_all=("n_all", "first"))
    )
    lines.append("\n\n## OOF sample size per endpoint\n")
    lines.append(counts.to_markdown())
    out_path.write_text("\n".join(lines))


# ============================================================
# OOF scatter plots
# ============================================================
def plot_scatter_per_endpoint(out_dir):
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        endpoint = endpoint_dir.name
        oof_file = endpoint_dir / "oof_predictions.csv"
        if not oof_file.exists():
            continue
        df = pd.read_csv(oof_file)
        algos_present = [a for a in ALGOS if a in df.columns]

        # Grid laid out over the algorithms present.
        ncols = 3
        nrows = int(np.ceil(len(algos_present) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4.5 * nrows),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes).flatten()

        for ax, algo in zip(axes, algos_present):
            y_true, y_pred = df["y_true"].values, df[algo].values
            m = regression_metrics(y_true, y_pred)
            ax.scatter(y_true, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
            lo = min(np.nanmin(y_true), np.nanmin(y_pred))
            hi = max(np.nanmax(y_true), np.nanmax(y_pred))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("Observed Δ IPSS")
            ax.set_ylabel("Predicted Δ IPSS (OOF)")
            ax.set_title(f"{algo}\nRMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.2f}")

        # Hide the unused cells.
        for ax in axes[len(algos_present):]:
            ax.set_visible(False)

        fig.suptitle(f"OOF predicted vs observed — {endpoint} (n={_n_patients(df)}{_rep_suffix(df)})")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{endpoint}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_aggregated(out_dir):
    """One figure: an (endpoint x algorithm) grid, one cell per combination."""
    endpoints = []
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if endpoint_dir.is_dir() and (endpoint_dir / "oof_predictions.csv").exists():
            endpoints.append(endpoint_dir.name)
    if not endpoints:
        return

    # Restrict to the algorithms actually present at the first endpoint.
    df0 = pd.read_csv(MODELS_DIR / endpoints[0] / "oof_predictions.csv")
    algos = [a for a in ALGOS if a in df0.columns]

    fig, axes = plt.subplots(len(endpoints), len(algos),
                             figsize=(4 * len(algos), 4 * len(endpoints)),
                             sharex="row", sharey="row")
    if len(endpoints) == 1:
        axes = axes.reshape(1, -1)
    if len(algos) == 1:
        axes = axes.reshape(-1, 1)

    for i, endpoint in enumerate(endpoints):
        df = pd.read_csv(MODELS_DIR / endpoint / "oof_predictions.csv")
        for j, algo in enumerate(algos):
            ax = axes[i, j]
            if algo not in df.columns:
                ax.set_visible(False)
                continue
            y_true, y_pred = df["y_true"].values, df[algo].values
            m = regression_metrics(y_true, y_pred)
            ax.scatter(y_true, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3, s=20)
            lo = min(np.nanmin(y_true), np.nanmin(y_pred))
            hi = max(np.nanmax(y_true), np.nanmax(y_pred))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax.set_title(f"{endpoint} — {algo}\nRMSE={m['rmse']:.2f} R²={m['r2']:.2f}",
                         fontsize=10)
            if i == len(endpoints) - 1:
                ax.set_xlabel("Observed Δ IPSS")
            if j == 0:
                ax.set_ylabel("Predicted Δ IPSS (OOF)")

    fig.suptitle("OOF predicted vs observed — aggregated view", fontsize=14, y=1.001)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_all.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Calibration plots (regression)
# ============================================================
# For a regression model, calibration is assessed by comparing the observed
# mean to the predicted mean within each bin (deciles of the prediction). A
# lowess curve is overlaid on the full scatter to show the continuous trend.
# Perfect calibration follows the y=x diagonal.
def _calibration_bin_data(y_true, y_pred, n_bins=CALIBRATION_N_BINS):
    """Return (bin_pred_means, bin_true_means, bin_counts) over n_bins deciles
    of y_pred. NaN are ignored."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_pred) == 0:
        return np.array([]), np.array([]), np.array([])
    # Quantile-based bins (deciles); duplicated edges are removed.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_pred, quantiles))
    if len(edges) < 2:
        return np.array([]), np.array([]), np.array([])
    bin_idx = np.clip(np.digitize(y_pred, edges[1:-1], right=False),
                      0, len(edges) - 2)
    pred_means, true_means, counts = [], [], []
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        if sel.sum() == 0:
            continue
        pred_means.append(y_pred[sel].mean())
        true_means.append(y_true[sel].mean())
        counts.append(int(sel.sum()))
    return np.array(pred_means), np.array(true_means), np.array(counts)


def _plot_calibration_ax(ax, y_true, y_pred, title):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_pred) == 0:
        ax.set_visible(False)
        return

    # Background scatter, heavily transparent.
    ax.scatter(y_pred, y_true, alpha=0.15, s=10, color="steelblue",
               edgecolor="none", label="patients (OOF)")

    # Lowess
    try:
        sm = lowess(y_true, y_pred, frac=0.5, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="darkorange", lw=2, label="lowess")
    except Exception:
        pass

    # Decile bins, plotted at their per-bin means.
    pred_m, true_m, counts = _calibration_bin_data(y_true, y_pred)
    if len(pred_m) > 0:
        ax.plot(pred_m, true_m, "o", color="crimson", markersize=7,
                label=f"{len(pred_m)} bins (deciles)", zorder=5)

    # Perfect-calibration diagonal.
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="perfect calibration")

    # Calibration slope and intercept.
    c = calibration_stats(y_true, y_pred)
    ax.set_title(f"{title}\nslope={c['calib_slope']:.2f}  "
                 f"intercept={c['calib_intercept']:.2f}", fontsize=10)
    ax.set_xlabel("Predicted Δ IPSS")
    ax.set_ylabel("Observed Δ IPSS")
    ax.grid(alpha=0.3)


def plot_calibration_per_endpoint(out_dir):
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        endpoint = endpoint_dir.name
        oof_file = endpoint_dir / "oof_predictions.csv"
        if not oof_file.exists():
            continue
        df = pd.read_csv(oof_file)
        algos_present = [a for a in ALGOS if a in df.columns]

        ncols = 3
        nrows = int(np.ceil(len(algos_present) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4.5 * nrows),
                                 sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        for ax, algo in zip(axes, algos_present):
            _plot_calibration_ax(ax, df["y_true"].values, df[algo].values, algo)

        for ax in axes[len(algos_present):]:
            ax.set_visible(False)

        # Single legend at the top of the figure.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)
        fig.suptitle(f"OOF calibration — {endpoint} (n={_n_patients(df)}{_rep_suffix(df)})", y=1.06)
        fig.tight_layout()
        fig.savefig(out_dir / f"calibration_{endpoint}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_calibration_aggregated(out_dir):
    """(endpoint x algorithm) grid of every calibration curve."""
    endpoints = []
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if endpoint_dir.is_dir() and (endpoint_dir / "oof_predictions.csv").exists():
            endpoints.append(endpoint_dir.name)
    if not endpoints:
        return

    df0 = pd.read_csv(MODELS_DIR / endpoints[0] / "oof_predictions.csv")
    algos = [a for a in ALGOS if a in df0.columns]

    fig, axes = plt.subplots(len(endpoints), len(algos),
                             figsize=(4 * len(algos), 4 * len(endpoints)),
                             sharex="row", sharey="row")
    if len(endpoints) == 1:
        axes = axes.reshape(1, -1)
    if len(algos) == 1:
        axes = axes.reshape(-1, 1)

    for i, endpoint in enumerate(endpoints):
        df = pd.read_csv(MODELS_DIR / endpoint / "oof_predictions.csv")
        for j, algo in enumerate(algos):
            ax = axes[i, j]
            if algo not in df.columns:
                ax.set_visible(False)
                continue
            _plot_calibration_ax(ax, df["y_true"].values, df[algo].values,
                                 f"{endpoint} — {algo}")

    fig.suptitle("OOF calibration — aggregated view", fontsize=14, y=1.001)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_all.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SHAP on the final models (trained on all patients)
# ============================================================
def _save_shap_csv(shap_values, feature_names, index, out_dir, algo, endpoint):
    """Save the raw SHAP values to CSV (rows = patients, columns = features)."""
    df_shap = pd.DataFrame(shap_values, columns=feature_names, index=index)
    df_shap.index.name = "patient_id"
    df_shap.to_csv(out_dir / f"shap_values_{algo}_{endpoint}.csv")
    return df_shap


def shap_catboost(endpoint_dir, X, out_dir, endpoint):
    model = CatBoostRegressor()
    model.load_model(str(endpoint_dir / "catboost.cbm"))
    X_t = X.copy()
    for c in CATEGORICAL_FEATURES:
        if c in X_t.columns:
            X_t[c] = X_t[c].fillna("missing").astype(str)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)

    plt.figure()
    shap.summary_plot(shap_values, X_t, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP — CatBoost — {endpoint}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_catboost_{endpoint}.png", dpi=130, bbox_inches="tight")
    plt.close()

    return _save_shap_csv(shap_values, list(X_t.columns), X_t.index, out_dir, "catboost", endpoint)


def shap_xgboost(endpoint_dir, X, out_dir, endpoint):
    """XGBoost: uses the dedicated preprocessor saved during the final training."""
    pre_path = endpoint_dir / "preprocessor_xgboost.joblib"
    model_path = endpoint_dir / "xgboost.json"
    if not (pre_path.exists() and model_path.exists()):
        return
    pre = joblib.load(pre_path)
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)

    plt.figure()
    shap.summary_plot(shap_values, X_t, feature_names=feature_names,
                      max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP — XGBoost — {endpoint}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_xgboost_{endpoint}.png", dpi=130, bbox_inches="tight")
    plt.close()

    return _save_shap_csv(shap_values, list(feature_names), X.index, out_dir, "xgboost", endpoint)


def shap_rf(endpoint_dir, X, out_dir, endpoint):
    """Random Forest: the whole pipeline (preprocessor + estimator) was saved."""
    pipe_path = endpoint_dir / "rf.joblib"
    if not pipe_path.exists():
        return
    pipe = joblib.load(pipe_path)
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["rf"]

    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)

    plt.figure()
    shap.summary_plot(shap_values, X_t, feature_names=feature_names,
                      max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP — Random Forest — {endpoint}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_rf_{endpoint}.png", dpi=130, bbox_inches="tight")
    plt.close()

    return _save_shap_csv(shap_values, list(feature_names), X.index, out_dir, "rf", endpoint)


def shap_mlp(endpoint_dir, X, out_dir, endpoint):
    pre = joblib.load(endpoint_dir / "preprocessor.joblib")
    ckpt = torch.load(endpoint_dir / "mlp.pt", map_location=DEVICE, weights_only=False)
    params = ckpt["params"]
    model = MLP(ckpt["n_in"], params["hidden_sizes"], params["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_t = pre.transform(X).astype(np.float32)
    bg_size = min(SHAP_BACKGROUND_SIZE, len(X_t))
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_t), size=bg_size, replace=False)
    background = X_t[bg_idx]

    def predict_fn(arr):
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
            return model(t).cpu().numpy()

    sample_size = min(100, len(X_t))
    sample_idx = rng.choice(len(X_t), size=sample_size, replace=False)
    sample = X_t[sample_idx]

    explainer = shap.KernelExplainer(
        predict_fn, shap.sample(background, min(50, bg_size), random_state=SEED)
    )
    shap_values = explainer.shap_values(sample, nsamples=100)

    feature_names = pre.get_feature_names_out()
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP — MLP — {endpoint} (n={sample_size})")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_mlp_{endpoint}.png", dpi=130, bbox_inches="tight")
    plt.close()

    sample_patient_ids = X.index[sample_idx]
    return _save_shap_csv(shap_values, list(feature_names), sample_patient_ids, out_dir, "mlp", endpoint)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("EVALUATION - OOF metrics, scatter, calibration, SHAP")
    print("=" * 70)
    ensure_dir(RESULTS_DIR)

    X = pd.read_csv(PREP_DIR / "features.csv", index_col=0)
    y_full = pd.read_csv(PREP_DIR / "targets.csv", index_col=0)

    print("\n[1/5] Computing the OOF metrics (RMSE/MAE/R2 + calibration)...")
    metrics = compute_all_metrics()
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    write_metrics_md(metrics, RESULTS_DIR / "metrics_summary.md")
    print(metrics.to_string(index=False))

    print("\n[2/5] Scatter plots per endpoint...")
    plot_scatter_per_endpoint(RESULTS_DIR)
    plot_scatter_aggregated(RESULTS_DIR)

    print("\n[3/5] Calibration curves per endpoint...")
    plot_calibration_per_endpoint(RESULTS_DIR)
    plot_calibration_aggregated(RESULTS_DIR)

    shap_summary_rows = []

    print("\n[4/5] SHAP plots (tree-based models: CatBoost, XGBoost, RF)...")
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        endpoint = endpoint_dir.name
        if endpoint not in y_full.columns:
            continue
        y_ep = y_full[endpoint].dropna()
        common = X.index.intersection(y_ep.index)
        X_ep = X.loc[common]
        if X_ep.empty:
            continue
        for algo_key, name, fn in [("catboost", "CatBoost", shap_catboost),
                                    ("xgboost", "XGBoost", shap_xgboost),
                                    ("rf", "Random Forest", shap_rf)]:
            try:
                print(f"  - {name} SHAP - {endpoint}")
                df_shap = fn(endpoint_dir, X_ep, RESULTS_DIR, endpoint)
                if df_shap is not None:
                    for feat in df_shap.columns:
                        shap_summary_rows.append({
                            "endpoint": endpoint,
                            "algo": algo_key,
                            "feature": feat,
                            "mean_abs_shap": float(df_shap[feat].abs().mean()),
                        })
            except Exception as e:
                print(f"    ! {name} SHAP failed: {e}")

    print("\n[5/5] MLP SHAP (KernelExplainer, slower)...")
    for endpoint_dir in sorted(MODELS_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        endpoint = endpoint_dir.name
        if endpoint not in y_full.columns:
            continue
        y_ep = y_full[endpoint].dropna()
        common = X.index.intersection(y_ep.index)
        X_ep = X.loc[common]
        if X_ep.empty:
            continue
        try:
            print(f"  - MLP SHAP - {endpoint}")
            df_shap = shap_mlp(endpoint_dir, X_ep, RESULTS_DIR, endpoint)
            if df_shap is not None:
                for feat in df_shap.columns:
                    shap_summary_rows.append({
                        "endpoint": endpoint,
                        "algo": "mlp",
                        "feature": feat,
                        "mean_abs_shap": float(df_shap[feat].abs().mean()),
                    })
        except Exception as e:
            print(f"    ! MLP SHAP failed: {e}")

    if shap_summary_rows:
        summary_df = (
            pd.DataFrame(shap_summary_rows)
            .sort_values(["endpoint", "algo", "mean_abs_shap"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
        summary_df.to_csv(RESULTS_DIR / "shap_summary.csv", index=False)
        print(f"  -> shap_summary.csv written ({len(summary_df)} rows)")

    print(f"\n[done] Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()