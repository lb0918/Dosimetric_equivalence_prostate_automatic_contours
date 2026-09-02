"""Shared helpers: patient-level K-fold, metrics, output directories."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from config import SEED, N_FOLDS, N_REPEATS


# ============================================================
# Replot mode - redraw a figure without recomputing
# ============================================================
# Every analysis script has the same shape: compute -> results CSV -> figure
# built FROM that CSV. Adjusting a figure detail (colour, label, size, row
# order) therefore has no reason to go through the computation again, which for
# the MCID scripts refits logistic regressions and redoes thousands of bootstrap
# resamples.
#
# --replot (or PIPE_REPLOT=1) skips the computation step: the script reads back
# its own results CSV and runs only the figure code. The numbers are exactly
# those of the last computation (nothing is refitted, so no resampling drift),
# and the CSV is not rewritten since it is the INPUT in this mode. Corollary:
# after any change that affects the NUMBERS (threshold, endpoints, cohort, new
# run), the script must be run once without --replot.
def replot_requested(argv: list[str] | None = None) -> bool:
    argv = sys.argv[1:] if argv is None else argv
    return "--replot" in argv or bool(os.environ.get("PIPE_REPLOT"))


def load_cached_results(path, produced_by: str) -> pd.DataFrame:
    """Read back a script's results CSV, for its --replot mode."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"--replot: {path} not found.\n"
            f"  This mode only redraws from an existing computation:\n"
            f"  run `python3 {produced_by}` first (without --replot)."
        )
    df = pd.read_csv(path)
    print(f"--replot: {len(df)} row(s) read back from {path} "
          f"(nothing computed, no model fitted).")
    return df


def kfold_patients(record_ids, seed: int = SEED, n_folds: int = N_FOLDS):
    """Build folds at the patient level, so no patient leaks across folds.

    Returns a list of (train_ids, test_ids) tuples of length n_folds.
    """
    ids = np.asarray(sorted(pd.unique(record_ids)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in kf.split(ids):
        folds.append((ids[tr_idx].tolist(), ids[te_idx].tolist()))
    return folds


def repeated_kfold_patients(record_ids, seed: int = SEED, n_folds: int = N_FOLDS,
                            n_repeats: int = N_REPEATS):
    """Repeated patient-level CV: `n_repeats` independent K-fold partitions.

    Returns a list of (repeat, fold, train_ids, test_ids) of length
    n_repeats * n_folds.

    Repetition r uses seed `seed + 1000*r`, so r=0 reproduces exactly
    kfold_patients(record_ids, seed, n_folds): with n_repeats=1 the output is
    bit-for-bit identical to the single-partition case. The gap of 1000 between
    seeds avoids any collision with the derived seeds used elsewhere (the inner
    CV uses SEED + 101 + k in 02_train.py).
    """
    out = []
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(
                kfold_patients(record_ids, seed=seed + 1000 * r, n_folds=n_folds)):
            out.append((r, k, tr, te))
    return out


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE and R2, ignoring NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
