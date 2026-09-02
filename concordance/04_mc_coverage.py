"""
04_mc_coverage.py  (Bayesian source only)
=========================================
Calibration of the Bayesian uncertainty: does the reference MANUAL value fall
inside the 95% Monte-Carlo predictive interval
([mc_lo, mc_hi] = <idx>__p2_5/__p97_5)?

Per (structure, index) over the 'bayes' pairs:
    - empirical coverage rate = fraction of patients whose val_manual lies in
      [mc_lo, mc_hi];
    - degenerate intervals (mc_lo == mc_hi) are counted separately
      (n_degenerate) and are NOT counted as covered by default: coverage is
      estimated on the non-degenerate intervals only;
    - Gaussian variant: the interval mean +/- 1.96*std (<idx>__mean / __std),
      which works around the coarseness of the empirical p2.5/p97.5 quantiles.

Limitation: with a small number of Monte-Carlo passes the empirical p2.5/p97.5
quantiles are coarse. The Gaussian variant is provided as a sensitivity
analysis.

Output: mc_coverage.csv.

Usage: python 04_mc_coverage.py [--no-overwrite]
"""

import argparse

import pandas as pd

import utils as U


def load_paired_bayes() -> pd.DataFrame:
    if not U.PAIRED_TABLE_CSV.exists():
        raise SystemExit(
            f"[STOP] {U.PAIRED_TABLE_CSV} missing. Run 01_build_paired_table.py first."
        )
    df = pd.read_csv(U.PAIRED_TABLE_CSV)
    df["record_id"] = df["record_id"].astype(str)
    return df[df["comparison"] == "bayes"].copy()


def compute(paired_bayes: pd.DataFrame) -> pd.DataFrame:
    # mc_std is not in paired_table: fetch it from the MC summary (Gaussian variant).
    mc = U.load_mc_long()[["record_id", "structure", "index", "mc_mean", "mc_std"]]
    key = ["record_id", "structure", "index"]
    df = paired_bayes.merge(mc, on=key, how="left")

    rows = []
    for structure, index, tier in U.PANEL:
        g = df[(df["structure"] == structure) & (df["index"] == index)]
        cov = U.coverage_stats(
            g["val_manual"].to_numpy(), g["mc_lo"].to_numpy(), g["mc_hi"].to_numpy(),
            mc_mean=g["mc_mean"].to_numpy(), mc_std=g["mc_std"].to_numpy())
        rows.append(dict(structure=structure, index=index, tier=tier, **cov))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cov = compute(load_paired_bayes())

    show = ["structure", "index", "tier", "n_total", "n_degenerate",
            "n_nondegenerate", "coverage_pctinterval", "coverage_gaussian"]
    print("\n=== MC coverage (Bayesian) - nominal target 0.95 ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(cov[show].to_string(index=False))

    U.save_csv(cov, U.OUT_DIR / "mc_coverage.csv", args.overwrite)


if __name__ == "__main__":
    main()
