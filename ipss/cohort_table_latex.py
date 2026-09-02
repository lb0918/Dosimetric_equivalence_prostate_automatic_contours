#!/usr/bin/env python3
"""
cohort_table_latex.py
=====================
Generates the LaTeX code of a descriptive table of the cohort variables used in
the IPSS analysis pipeline.

The table reconstructs EXACTLY the cohort that enters the models (see
01_prepare_target.py):
  1. one row per patient = baseline features plus the last IPSS measurement
     strictly BEFORE tx_date (prefix pretx_);
  2. exclusion of patients without a pre-treatment IPSS (no baseline for the
     delta);
  3. restriction to the cohort that has a DVH (config.RESTRICT_TO_DVH_COHORT).

For each variable the table reports:
  - numeric variables      -> n (non-missing), mean +/- standard deviation;
  - categorical variables (config.CATEGORICAL_FEATURES, or a non-numeric dtype)
    -> count (%) per level.

EXCEPTION: the DVH INDICES (columns prefixed by a structure name, see
config.DVH_STRUCTURE_PREFIXES) get NO mean or standard deviation, because their
values depend on the SEGMENTATION SOURCE. The pipeline in any case replaces them
by a panel selected through config.DVH_SEG_SOURCE. They are listed separately,
without statistics, with a note.

The script READS the patient-level table (and the DVH files, for the cohort
restriction) ONLY at run time, to compute the statistics; it prints no patient
data, only aggregates.

Usage:
    python cohort_table_latex.py                 # writes cohort_table.tex + stdout
    python cohort_table_latex.py -o my.tex       # custom output file
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402


# ----------------------------------------------------------------------------
# Import the pipeline functions (a module whose name starts with a digit)
# ----------------------------------------------------------------------------
def _load_prepare_module():
    """Charge 01_prepare_target.py (nom de module non importable directement)."""
    spec = importlib.util.spec_from_file_location(
        "prepare_target", HERE / "01_prepare_target.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Readable labels (adjust freely). A variable absent from this table is shown
# under its raw name. The DVH indices are handled separately.
# ----------------------------------------------------------------------------
LABELS = {
    "age":                 ("Age", "years"),
    "stage":               ("Clinical T stage", ""),
    "gleason":             ("Gleason score", ""),
    "isup_grade":          ("ISUP grade", ""),
    "crude_psa":           ("Pre-treatment PSA", "ng/mL"),
    "psa_density":         ("PSA density", "ng/mL/cc"),
    "adt":                 ("Androgen deprivation therapy (ADT)", ""),
    "hx_len":              ("ADT duration", "days"),
    "ldr_post_vol":        ("Prostate volume, post-implant", "cc"),
    "ldr_previ_volcont":   ("Planned contoured volume", "cc"),
    "ldr_live_aiguilles":  ("Number of needles (implant)", ""),
    "ldr_previ_sourceprev":("Number of planned sources", ""),
    "ldr_live_dil":        ("ldr_live_dil", ""),
    "ipss_days":           ("Delay of the first IPSS measurement", "days"),
    "n_pretx_ipss":        ("Number of pre-treatment IPSS measurements", ""),
    "dvh_urethra_available": ("Urethra DVH available", "flag 0/1"),
    "dvh_bladder_available": ("Bladder DVH available", "flag 0/1"),
    # Pre-treatment IPSS subscores (pretx_ prefix)
    "pretx_prostsex_ipss_a": ("Pre-tx IPSS — Q1 (incomplete emptying)", ""),
    "pretx_prostsex_ipss_b": ("Pre-tx IPSS — Q2 (frequency)", ""),
    "pretx_prostsex_ipss_c": ("Pre-tx IPSS — Q3 (intermittency)", ""),
    "pretx_prostsex_ipss_d": ("Pre-tx IPSS — Q4 (urgency)", ""),
    "pretx_prostsex_ipss_e": ("Pre-tx IPSS — Q5 (weak stream)", ""),
    "pretx_prostsex_ipss_f": ("Pre-tx IPSS — Q6 (straining)", ""),
    "pretx_prostsex_ipss_g": ("Pre-tx IPSS — Q7 (nocturia)", ""),
    "pretx_prostsex_qual_vie_a": ("Pre-tx urinary quality of life (a)", ""),
    "pretx_prostsex_qual_vie_b": ("Pre-tx urinary quality of life (b)", ""),
    "pretx_ipss_obstructive": ("Pre-tx obstructive IPSS subscore", ""),
    "pretx_ipss_irritative":  ("Pre-tx irritative IPSS subscore", ""),
    "pretx_shim_score":       ("Pre-tx SHIM score", ""),
}


def label_of(col: str) -> tuple[str, str]:
    return LABELS.get(col, (col.replace("_", r"\_"), ""))


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


# ----------------------------------------------------------------------------
# Cohort reconstruction (identical to the pipeline)
# ----------------------------------------------------------------------------
def build_cohort() -> pd.DataFrame:
    prep = _load_prepare_module()

    df = prep.load_long_dataset(Path(config.DATASET_MINIMAL))
    X = prep.build_static_features(df)

    # The IPSS baseline defines the target, so it is removed from X, but it
    # also determines the exclusion of patients without a pre-treatment IPSS.
    baseline_col = f"{config.PRE_TX_PREFIX}ipss_score_calc"
    baseline_ipss = X[baseline_col].copy() if baseline_col in X.columns else pd.Series(dtype=float)
    if baseline_col in X.columns:
        X = X.drop(columns=[baseline_col])

    has_baseline = baseline_ipss.notna()
    X = X.loc[has_baseline]

    if getattr(config, "RESTRICT_TO_DVH_COHORT", False):
        from dvh_mc import dvh_cohort_record_ids
        eligible = dvh_cohort_record_ids(
            config.RESTRICT_DVH_SOURCES, config.RESTRICT_DVH_COMBINE)
        keep = X.index.astype(str).str.strip().isin(eligible)
        X = X.loc[keep]

    return X


def is_dvh_index(col: str) -> bool:
    return any(col.startswith(p) for p in config.DVH_STRUCTURE_PREFIXES)


def dvh_metric_unit(metric: str) -> str:
    """Unit inferred from the suffix of a DVH index (D90_Gy, V100_pct, ...)."""
    if metric.endswith("_Gy"):
        return "Gy"
    if metric.endswith("_pct"):
        return r"\%"
    if metric.endswith("_cc"):
        return "cc"
    return ""


def curated_dvh_indices() -> list[tuple[str, str, str]]:
    """DVH indices actually used by the models, i.e. the curated panel
    (config.DVH_CURATED_PANEL), as (name, structure, metric) triples."""
    out = []
    for struct, metrics in getattr(config, "DVH_CURATED_PANEL", {}).items():
        for m in metrics:
            out.append((f"{struct}_{m}", struct, m))
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def numeric_row(col: str, s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    label, unit = label_of(col)
    return {
        "kind": "numeric",
        "label": label,
        "unit": unit,
        "n": n,
        "mean": float(s.mean()) if n else np.nan,
        "std": float(s.std(ddof=1)) if n > 1 else np.nan,
    }


def _pretty_level(k) -> str:
    """Print 2 rather than 2.0 for levels coded as whole-valued floats."""
    try:
        f = float(k)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(k)


def categorical_rows(col: str, s: pd.Series, n_total: int) -> dict:
    counts = s.astype("string").fillna("(manquant)").value_counts(dropna=False)
    label, unit = label_of(col)
    levels = [(_pretty_level(k), int(v), 100.0 * v / n_total)
              for k, v in counts.items()]
    return {"kind": "categorical", "label": label, "unit": unit,
            "n": int(s.notna().sum()), "levels": levels}


def is_binary(s: pd.Series) -> bool:
    """True when the non-missing values are a subset of {0, 1}
    (boolean flags coded numerically: adt, dvh_*_available, ldr_live_dil, ...)."""
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0.0, 1.0})


def classify(col: str, s: pd.Series) -> str:
    if col in getattr(config, "CATEGORICAL_FEATURES", []):
        return "categorical"
    if not pd.api.types.is_numeric_dtype(pd.to_numeric(s, errors="coerce")):
        return "categorical"
    if is_binary(s):
        # 0/1 flags: shown as counts (%) like the other categorical variables,
        # so as not to report a "mean" that is only a proportion.
        return "categorical"
    return "numeric"


# ----------------------------------------------------------------------------
# LaTeX generation
# ----------------------------------------------------------------------------
def fmt(x: float, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "--"
    return f"{x:.{dec}f}"


def build_latex(X: pd.DataFrame) -> str:
    n_total = len(X)

    non_dvh = [c for c in X.columns if not is_dvh_index(c)]

    num_rows, cat_rows = [], []
    for col in non_dvh:
        s = X[col]
        if classify(col, s) == "numeric":
            num_rows.append((col, numeric_row(col, s)))
        else:
            cat_rows.append((col, categorical_rows(col, s, n_total)))

    L = []
    ap = L.append
    ap(r"% Generated by cohort_table_latex.py - requires \usepackage{booktabs}")
    ap(r"\begin{table}[htbp]")
    ap(r"  \centering")
    ap(r"  \caption{Characteristics of the cohort used in the "
       r"d'analyse IPSS ($N = " + str(n_total) + r"$ patients). Variables "
       r"numeric variables: mean $\pm$ standard deviation. Categorical "
       r"variables: count (\%). The dose-volume (DVH) indices are not "
       r"summarised, because their values depend on the segmentation source "
       r"(manual / auto\_det / mc\_bayes).}")
    ap(r"  \label{tab:cohort_ipss}")
    ap(r"  \begin{tabular}{@{}llrr@{}}")
    ap(r"    \toprule")
    ap(r"    Variable & Unit & $n$ & Mean $\pm$ SD \\")
    ap(r"    \midrule")

    # --- Numeric variables ---
    ap(r"    \multicolumn{4}{@{}l}{\textit{Variables continues / ordinales}} \\")
    for _col, r in num_rows:
        unit = tex_escape(r["unit"]) if r["unit"] else ""
        val = f"${fmt(r['mean'])} \\pm {fmt(r['std'])}$"
        ap(f"    {tex_escape(r['label'])} & {unit} & {r['n']} & {val} \\\\")

    # --- Categorical variables ---
    if cat_rows:
        ap(r"    \midrule")
        ap(r"    \multicolumn{4}{@{}l}{\textit{Categorical variables "
           r"— effectif (\%)}} \\")
        for _col, r in cat_rows:
            ap(f"    {tex_escape(r['label'])} & & & \\\\")
            for lv, cnt, pct in r["levels"]:
                ap(f"    \\quad {tex_escape(lv)} & & {cnt} & "
                   f"{fmt(pct, 1)}\\% \\\\")

    # --- DVH indices actually used = the curated panel, without statistics ---
    # (config.DVH_CURATED_PANEL; values depend on the segmentation source)
    dvh_indices = curated_dvh_indices()
    if dvh_indices:
        ap(r"    \midrule")
        ap(r"    \multicolumn{4}{@{}l}{\textit{DVH indices used by the models "
           r"— curated panel (depend on the segmentation source)}} \\")
        for name, _struct, metric in dvh_indices:
            unit = dvh_metric_unit(metric)
            ap(f"    {tex_escape(name)} & {unit} & -- & "
               r"\emph{selon segmentation} \\")

    ap(r"    \bottomrule")
    ap(r"  \end{tabular}")
    ap(r"\end{table}")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(HERE / "cohort_table.tex"),
                    help="Output .tex file (default: cohort_table.tex)")
    args = ap.parse_args()

    X = build_cohort()
    latex = build_latex(X)

    Path(args.output).write_text(latex, encoding="utf-8")
    print(latex)
    print(f"\n% [done] Written to {args.output}  (N = {len(X)} patients, "
          f"{X.shape[1]} variables de cohorte)", file=sys.stderr)


if __name__ == "__main__":
    main()
