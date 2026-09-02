"""
03_concordance_descriptors.py
=============================
Agreement descriptors per (structure, index, comparison), complementing the TOST.

Per group (diff = val_manual - val_auto):
    - Bland-Altman: bias = mean(diff), LoA = bias +/- 1.96*SD(diff),
      proportional bias = OLS slope of diff on the mean (slope and p-value),
      and the flag 'LoA_within_delta', True when the LoA lies entirely inside
      [-delta, +delta] (reported at both the decision and sensitivity margins).
    - Lin's concordance correlation coefficient (CCC).
    - ICC(A,1): two-way random effects, absolute agreement, single measure
      (explicit ANOVA implementation, no pingouin dependency).

Output: concordance_descriptors.csv (main deliverable).
OPTIONAL figures under --figures, in PNG and PDF (the CSVs remain the
deliverable):
    - Bland-Altman per group, with the +/- decision margin band overlaid;
    - forest plot of the mean differences with their 90% CI per comparison.

Usage: python 03_concordance_descriptors.py [--no-overwrite] [--figures]
"""

import argparse

import numpy as np
import pandas as pd

import utils as U

# Figure font sizes: a legible floor once the figure is scaled down to a column.
FS_TICK = 12
FS_LABEL = 13
FS_TITLE = 14
FS_LEG = 11


def load_paired() -> pd.DataFrame:
    if not U.PAIRED_TABLE_CSV.exists():
        raise SystemExit(
            f"[STOP] {U.PAIRED_TABLE_CSV} missing. Run 01_build_paired_table.py first."
        )
    df = pd.read_csv(U.PAIRED_TABLE_CSV)
    df["record_id"] = df["record_id"].astype(str)
    return df


def compute(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for structure, index, tier in U.PANEL:
        d = U.delta_row(structure, index)
        dc, ds = d["delta_conf"], d["delta_sens"]
        for comparison in U.COMPARISONS:
            g = paired[(paired["structure"] == structure)
                       & (paired["index"] == index)
                       & (paired["comparison"] == comparison)]
            m = g["val_manual"].to_numpy()
            a = g["val_auto"].to_numpy()
            ba = U.bland_altman(m, a)
            loa_lo, loa_hi = ba["loa_lo"], ba["loa_hi"]
            within_conf = bool(loa_lo >= -dc and loa_hi <= dc) if ba["n"] >= 2 else False
            within_sens = bool(loa_lo >= -ds and loa_hi <= ds) if ba["n"] >= 2 else False
            rows.append(dict(
                structure=structure, index=index, comparison=comparison, tier=tier,
                unite=d["unite"], n=ba["n"],
                bias=ba["bias"], sd_diff=ba["sd_diff"],
                loa_lo=loa_lo, loa_hi=loa_hi,
                prop_slope=ba["prop_slope"], prop_slope_p=ba["prop_slope_p"],
                delta_conf=dc, delta_sens=ds,
                loa_within_delta_conf=within_conf,
                loa_within_delta_sens=within_sens,
                ccc=U.lin_ccc(m, a),
                icc_a1=U.icc_a1(m, a),
            ))
    return pd.DataFrame(rows)


def make_figures(paired: pd.DataFrame, desc: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U.ensure_dir(U.FIG_DIR)

    # ---- Bland-Altman per group ----
    for structure, index, _ in U.PANEL:
        d = U.delta_row(structure, index)
        dc = d["delta_conf"]
        for comparison in U.COMPARISONS:
            g = paired[(paired["structure"] == structure)
                       & (paired["index"] == index)
                       & (paired["comparison"] == comparison)]
            if len(g) < 2:
                continue
            m = g["val_manual"].to_numpy()
            a = g["val_auto"].to_numpy()
            diff = m - a
            means = (m + a) / 2.0
            ba = U.bland_altman(m, a)
            lab = U.index_label(index)
            fig, ax = plt.subplots(figsize=(6.6, 5.0))
            ax.axhspan(-dc, dc, color="tab:green", alpha=0.12,
                       label=rf"equivalence band $\pm\delta_{{conf}}$ = {dc}")
            ax.scatter(means, diff, s=14, alpha=0.6, color="tab:blue")
            ax.axhline(ba["bias"], color="tab:red", lw=1.5,
                       label=f"bias = {ba['bias']:.2f}")
            ax.axhline(ba["loa_lo"], color="tab:red", ls="--", lw=1,
                       label=f"LoA = [{ba['loa_lo']:.2f}, {ba['loa_hi']:.2f}]")
            ax.axhline(ba["loa_hi"], color="tab:red", ls="--", lw=1)
            ax.axhline(0, color="grey", lw=0.6)
            unit = U.unit_label(d["unite"])
            ax.set_xlabel(f"Mean of (manual, auto) [{unit}]", fontsize=FS_LABEL)
            ax.set_ylabel(f"Difference = manual − auto [{unit}]", fontsize=FS_LABEL)
            ax.set_title(f"Bland-Altman: {structure} {lab} ({comparison})",
                         fontsize=FS_TITLE)
            ax.tick_params(labelsize=FS_TICK)
            ax.legend(fontsize=FS_LEG, loc="best")
            fig.tight_layout()
            U.save_figure(fig, U.FIG_DIR / f"ba_{structure}_{index}_{comparison}.png",
                          dpi=130)
            plt.close(fig)

    # ---- Forest plot of the mean differences with 90% CI, per comparison ----
    for comparison in U.COMPARISONS:
        sub = []
        for structure, index, tier in U.PANEL:
            g = paired[(paired["structure"] == structure)
                       & (paired["index"] == index)
                       & (paired["comparison"] == comparison)]
            diff = (g["val_manual"] - g["val_auto"]).to_numpy()
            d = U.delta_row(structure, index)
            t = U.tost_paired(diff, d["delta_conf"])
            sub.append((f"{structure} {U.index_label(index)}", t["mean_diff"],
                        t["ci_lo"], t["ci_hi"], d["delta_conf"]))
        if not sub:
            continue
        labels = [s[0] for s in sub]
        y = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(sub) + 1.8))
        for yi, (_, mean, lo, hi, dc) in zip(y, sub):
            ax.plot([lo, hi], [yi, yi], color="tab:blue", lw=2)
            ax.plot(mean, yi, "o", color="tab:blue")
            ax.plot([-dc, -dc], [yi - 0.3, yi + 0.3], color="tab:green", lw=1)
            ax.plot([dc, dc], [yi - 0.3, yi + 0.3], color="tab:green", lw=1)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FS_TICK)
        ax.set_xlabel("Mean difference (manual − auto) with 90 % CI\n"
                      r"green bars = $\pm\delta_{conf}$", fontsize=FS_LABEL)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.set_title(f"Forest: equivalence ({comparison})", fontsize=FS_TITLE)
        fig.tight_layout()
        U.save_figure(fig, U.FIG_DIR / f"forest_meandiff_{comparison}.png", dpi=130)
        plt.close(fig)

    print(f"[figures] PNG + PDF written to {U.FIG_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    paired = load_paired()
    desc = compute(paired)

    show = ["structure", "index", "comparison", "n", "bias", "loa_lo", "loa_hi",
            "loa_within_delta_conf", "ccc", "icc_a1"]
    print("\n=== Concordance descriptors ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(desc[show].to_string(index=False))

    U.save_csv(desc, U.OUT_DIR / "concordance_descriptors.csv", args.overwrite)
    if args.figures:
        make_figures(paired, desc)


if __name__ == "__main__":
    main()
