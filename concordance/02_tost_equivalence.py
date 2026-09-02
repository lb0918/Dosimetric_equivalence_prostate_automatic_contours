"""
02_tost_equivalence.py
======================
Paired EQUIVALENCE test (TOST) per (structure, index, comparison).

For each group:
    diff = val_manual - val_auto
    mean_diff, sd_diff, n, se
    90% CI (Student, n-1 df) - independent of the margin and consistent with a
    TOST at ALPHA = 0.05, since a 90% CI inside [-delta, +delta] is equivalent
    to p_TOST < 0.05.
    p_TOST = the larger of the two one-sided tests, evaluated at both the
    decision margin (delta_conf) and the sensitivity margin (delta_sens).

Equivalence verdicts (90% CI strictly inside [-delta, +delta]):
    - CONFIRMATORY (Prostate): the Holm family is every confirmatory index
      crossed with the two comparisons. The Holm correction is applied to the
      p_TOST values at delta_conf, and verdict_conf = p_holm < ALPHA.
      verdict_sens uses the sensitivity margin WITHOUT a multiplicity
      correction, as a robustness analysis.
    - EXPLORATORY (BladderNeck): OUTSIDE Holm. Here delta_conf == delta_sens (a
      single margin) and the verdict is reported uncorrected, labelled
      exploratory.

Output: tost_results.csv, one row per structure x index x comparison.

Usage: python 02_tost_equivalence.py [--no-overwrite]
"""

import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import utils as U


def load_paired() -> pd.DataFrame:
    if not U.PAIRED_TABLE_CSV.exists():
        raise SystemExit(
            f"[STOP] {U.PAIRED_TABLE_CSV} missing. Run 01_build_paired_table.py first."
        )
    df = pd.read_csv(U.PAIRED_TABLE_CSV)
    df["record_id"] = df["record_id"].astype(str)
    return df


def run_tost(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for structure, index, tier in U.PANEL:
        d = U.delta_row(structure, index)
        dc, ds = d["delta_conf"], d["delta_sens"]
        for comparison in U.COMPARISONS:
            g = paired[(paired["structure"] == structure)
                       & (paired["index"] == index)
                       & (paired["comparison"] == comparison)]
            diff = (g["val_manual"] - g["val_auto"]).to_numpy()
            base = U.tost_paired(diff, dc)          # CI + p_TOST at delta_conf
            sens = U.tost_paired(diff, ds)          # p_TOST at delta_sens
            rows.append(dict(
                structure=structure, index=index, comparison=comparison, tier=tier,
                unite=d["unite"], n=base["n"],
                mean_diff=base["mean_diff"], sd_diff=base["sd_diff"], se=base["se"],
                ci90_lo=base["ci_lo"], ci90_hi=base["ci_hi"],
                delta_conf=dc, delta_sens=ds,
                p_tost_conf=base["p_tost"], p_tost_sens=sens["p_tost"],
                p_holm=np.nan,
                equiv_conf_raw=base["equivalent"],   # 90% CI inside +/-delta_conf, no Holm
                verdict_sens=sens["equivalent"],     # 90% CI inside +/-delta_sens
                verdict_conf=False,                  # filled in below
            ))
    res = pd.DataFrame(rows)

    # ---- Holm correction on the confirmatory family ONLY ----
    conf_mask = res["tier"] == U.TIER_CONF
    conf_idx = res.index[conf_mask]
    pvals = res.loc[conf_idx, "p_tost_conf"].to_numpy()
    if len(pvals):
        reject, p_adj, _, _ = multipletests(pvals, alpha=U.ALPHA, method="holm")
        res.loc[conf_idx, "p_holm"] = p_adj
        res.loc[conf_idx, "verdict_conf"] = np.asarray(reject, dtype=bool)  # p_holm < ALPHA

    # ---- Exploratory: uncorrected verdict (no Holm) ----
    expl_mask = res["tier"] == U.TIER_EXPL
    res.loc[expl_mask, "verdict_conf"] = res.loc[expl_mask, "equiv_conf_raw"]

    res["verdict_conf"] = res["verdict_conf"].astype(bool)
    res["verdict_sens"] = res["verdict_sens"].astype(bool)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    paired = load_paired()
    res = run_tost(paired)

    show = ["structure", "index", "comparison", "tier", "n", "mean_diff",
            "ci90_lo", "ci90_hi", "delta_conf", "p_tost_conf", "p_holm",
            "verdict_conf", "verdict_sens"]
    print("\n=== TOST - equivalence verdicts ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(res[show].to_string(index=False))

    U.save_csv(res, U.OUT_DIR / "tost_results.csv", args.overwrite)


if __name__ == "__main__":
    main()
