"""
01_build_paired_table.py
=========================
Builds the LONG paired table that feeds the whole concordance arm.

For each (structure, index) of the panel and each comparison ('det' | 'bayes'):
    - val_manual   the _pct index from the MANUAL segmentation (reference);
    - val_auto     the _pct index from the automatic segmentation,
                     * 'det'   -> deterministic;
                     * 'bayes' -> E[index] = mean of the Monte-Carlo passes
                                  (<idx>__mean);
    - mc_lo/mc_hi  bounds of the 95% Monte-Carlo predictive interval
                   (<idx>__p2_5 / __p97_5), filled for 'bayes', NaN for 'det'.

Matching is an inner join on record_id, PER structure. There is no imputation:
unmatched patients (or those with a missing value) are dropped and counted. The
number of matched and dropped patients is reported per (structure, index,
comparison).

Outputs (in the results directory):
    - paired_table.csv         the long paired table (main deliverable);
    - paired_merge_report.csv  per-group matching diagnostic;
    - sap_delta_table.csv      the pre-declared equivalence-margin table;
    - excluded_structures.csv  excluded structures and the reason.

Usage: python 01_build_paired_table.py [--no-overwrite]
"""

import argparse

import numpy as np
import pandas as pd

import utils as U

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PAIRED_COLS = ["record_id", "structure", "index", "comparison",
               "val_manual", "val_auto", "mc_lo", "mc_hi"]


def build_paired() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (paired_long, merge_report)."""
    manual = U.load_source_long(U.MANUAL_CSV, value_name="val_manual")
    det = U.load_source_long(U.DET_CSV, value_name="val_auto")
    mc = U.load_mc_long()

    key = ["record_id", "structure", "index"]
    paired_parts = []
    report_rows = []

    for structure, index, tier in U.PANEL:
        man_g = manual[(manual["structure"] == structure) & (manual["index"] == index)]
        n_manual = int(man_g["val_manual"].notna().sum())

        # ---- 'det' comparison ----
        det_g = det[(det["structure"] == structure) & (det["index"] == index)]
        merged = man_g.merge(det_g[key + ["val_auto"]], on=key, how="inner")
        n_merged = len(merged)
        merged = merged.dropna(subset=["val_manual", "val_auto"])
        merged["comparison"] = "det"
        merged["mc_lo"] = np.nan
        merged["mc_hi"] = np.nan
        paired_parts.append(merged[PAIRED_COLS])
        report_rows.append(dict(
            structure=structure, index=index, tier=tier, comparison="det",
            n_manual=n_manual, n_auto=int(det_g["val_auto"].notna().sum()),
            n_matched=n_merged, n_dropped_na=n_merged - len(merged),
            n_paired=len(merged)))

        # ---- 'bayes' comparison ----
        mc_g = mc[(mc["structure"] == structure) & (mc["index"] == index)]
        mb = man_g.merge(mc_g[key + ["mc_mean", "mc_lo", "mc_hi"]], on=key, how="inner")
        n_merged_b = len(mb)
        mb = mb.rename(columns={"mc_mean": "val_auto"})
        mb = mb.dropna(subset=["val_manual", "val_auto"])
        mb["comparison"] = "bayes"
        paired_parts.append(mb[PAIRED_COLS])
        report_rows.append(dict(
            structure=structure, index=index, tier=tier, comparison="bayes",
            n_manual=n_manual, n_auto=int(mc_g["mc_mean"].notna().sum()),
            n_matched=n_merged_b, n_dropped_na=n_merged_b - len(mb),
            n_paired=len(mb)))

    paired = pd.concat(paired_parts, ignore_index=True)
    report = pd.DataFrame(report_rows)
    return paired, report


def sap_delta_table() -> pd.DataFrame:
    """Pre-declared equivalence-margin table."""
    rows = []
    for structure, index, _ in U.PANEL:
        d = U.delta_row(structure, index)
        rows.append(dict(
            structure=structure, index=index, tier=d["tier"], unite=d["unite"],
            delta_conf=d["delta_conf"], delta_sens=d["delta_sens"],
            ancre=d["ancre"], flags=d["flags"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    U.ensure_dir(U.OUT_DIR)
    paired, report = build_paired()

    print("\n=== Matching per group ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(report.to_string(index=False))

    U.save_csv(paired, U.PAIRED_TABLE_CSV, args.overwrite)
    U.save_csv(report, U.OUT_DIR / "paired_merge_report.csv", args.overwrite)
    U.save_csv(sap_delta_table(), U.OUT_DIR / "sap_delta_table.csv", args.overwrite)
    U.save_csv(pd.DataFrame(U.EXCLUDED_STRUCTURES),
               U.OUT_DIR / "excluded_structures.csv", args.overwrite)


if __name__ == "__main__":
    main()
