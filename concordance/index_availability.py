"""
index_availability.py
=====================
AVAILABILITY audit of the DVH indices of a given panel, across the three
segmentation sources (manual, deterministic, Bayesian). A preliminary
diagnostic: which columns actually exist, and for how many patients they are
filled.

Unlike the strict loaders of utils.py (which RAISE when a column is missing),
this script is TOLERANT: a missing column or structure is a legitimate result to
report (the urethra, for instance, is expected to be absent from the
deterministic source).

For each (structure, index) x source:
    - value_column      the column name looked up in that source
                          * manual / deterministic: the index itself;
                          * Bayesian (MC summary):  '<idx>__mean' (E[index]);
    - structure_present is the structure present in the source?
    - column_present    does the column exist in the file?
    - n_patients_struct number of distinct record_ids for the structure;
    - n_nonNA           number of patients with a filled value;
    - coverage          n_nonNA / n_patients_struct.

Harmonisation: 'Bladder neck' -> 'BladderNeck' (the deterministic file pitfall).

Output: index_availability.csv, plus a coverage matrix printed to stdout.

Usage: python index_availability.py [--no-overwrite]
"""

import argparse

import numpy as np
import pandas as pd

import utils as U

# ------------------------------------------------------------
# Audited panel (mirrors the curated DVH panel of the prediction pipeline)
# ------------------------------------------------------------
DVH_CURATED_PANEL = {
    "Prostate":    ["D90_Gy", "V100_pct", "V150_pct", "V200_pct"],
    "Urethra":     ["uD10_Gy", "uD30_Gy", "uD5_Gy", "uD0.1cc_Gy"],
    "BladderNeck": ["D2cc_Gy", "D1cc_Gy", "V100_pct"],
}

# Sources: (name, path, function mapping an index to its value column name).
SOURCES = [
    ("manual", U.MANUAL_CSV, lambda idx: idx),
    ("det", U.DET_CSV, lambda idx: idx),
    ("bayes", U.MC_SUMMARY_CSV, lambda idx: f"{idx}__mean"),
]

PANEL_PAIRS = [(s, i) for s, idxs in DVH_CURATED_PANEL.items() for i in idxs]


def _read_tolerant(path, wanted_cols):
    """Read only the useful columns that are present; return (df, all_columns)."""
    if not path.exists():
        raise SystemExit(f"[STOP] File not found: {path}")
    all_cols = pd.read_csv(path, nrows=0).columns
    meta = [c for c in ("record_id", "structure", "model", "pred_mode") if c in all_cols]
    present_vals = [c for c in wanted_cols if c in all_cols]
    df = pd.read_csv(path, usecols=meta + present_vals)
    if "structure" in df.columns:
        df["structure"] = df["structure"].replace(U.STRUCTURE_RENAME)
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype(str)
    return df, set(all_cols)


def audit() -> pd.DataFrame:
    rows = []
    for source, path, col_of in SOURCES:
        wanted = [col_of(i) for _, i in PANEL_PAIRS]
        df, all_cols = _read_tolerant(path, wanted)
        structures_in_file = set(df["structure"].unique()) if "structure" in df.columns else set()
        for structure, index in PANEL_PAIRS:
            value_col = col_of(index)
            struct_present = structure in structures_in_file
            col_present = value_col in all_cols
            sub = df[df["structure"] == structure] if struct_present else df.iloc[0:0]
            n_patients = int(sub["record_id"].nunique()) if "record_id" in df.columns else len(sub)
            if col_present and struct_present:
                n_nonna = int(pd.to_numeric(sub[value_col], errors="coerce").notna().sum())
            else:
                n_nonna = 0
            rows.append(dict(
                source=source, structure=structure, index=index,
                value_column=value_col,
                structure_present=struct_present,
                column_present=col_present,
                n_patients_structure=n_patients,
                n_nonNA=n_nonna,
                coverage=(n_nonna / n_patients) if n_patients else np.nan,
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    res = audit()

    print("\n=== Availability per (structure, index) x source ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(res.to_string(index=False))

    # Compact matrix: coverage (n_nonNA) per source.
    res["cell"] = res.apply(
        lambda r: (f"{r.n_nonNA}/{r.n_patients_structure}"
                   if r.column_present and r.structure_present
                   else ("no_struct" if not r.structure_present else "no_col")),
        axis=1)
    pivot = res.pivot(index=["structure", "index"], columns="source", values="cell")
    pivot = pivot.reindex(columns=[s for s, _, _ in SOURCES])
    print("\n=== n_nonNA / n_patients matrix (no_col = column absent, "
          "no_struct = structure absent) ===")
    print(pivot.to_string())

    U.ensure_dir(U.OUT_DIR)
    U.save_csv(res.drop(columns="cell"), U.OUT_DIR / "index_availability.csv", args.overwrite)


if __name__ == "__main__":
    main()
