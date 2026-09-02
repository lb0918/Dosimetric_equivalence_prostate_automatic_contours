"""
00_build_ipss_dataset.py
========================
Builds the patient-level table feeding 01_prepare_target.py: one row per IPSS
measurement, with the baseline columns repeated, the raw DVH indices joined, and
the post-salvage measurements filtered out.

Inputs (see config.DATA_ROOT):
  - one directory per patient, holding the per-form exports;
  - the salvage-HDR delay table;
  - the Monte-Carlo DVH summary.

Outputs (no existing file is overwritten):
  - <patient>/baseline_ipss.csv       one row per patient, enriched baseline
  - ipss_longitudinal_enriched.csv    IPSS measurements plus subscores
  - dataset_minimal_v2.csv            final table, read by config.DATASET_MINIMAL

Variable selection
------------------
Only variables with solid coverage over the full cohort are included. Variables
that are clinically relevant but recorded only for a small curated subset
(alphablock, base_pde5, ari_psa, ultrasound prostate volume, atcd_*
comorbidities, pos_cores, ADT dates) are deliberately EXCLUDED: after the
downstream median imputation they become nearly constant, and they conflate
"not documented" with "absent". They are better explored in an analysis
dedicated to the curated subset.

Retained urinary predictors: pre-treatment IPSS total, obstructive and
irritative subscores (which follow distinct time courses, obstructive tracking
oedema and irritative mucosal injury), QoL bother, prostate volume at implant,
PSA density, age, stage, gleason, isup_grade, crude_psa, adt, hx_len, ldr_*,
and the urethral and bladder-neck dose indices.

Known collinearities, flagged rather than removed here:
  - seeds (ldr_previ_sourceprev), needles (ldr_live_aiguilles) and volume are
    mutually collinear, all driven by prostate volume; keep one or reduce them;
  - a D'Amico composite must not be mixed with its own components
    (gleason, PSA, stage).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dvh_mc import load_mc_median, load_mc_variance
from config import DATA_ROOT, DVH_MC_USE_VARIANCE, DVH_MC_VAR_SUFFIX

# ============================================================
# PATHS
# ============================================================
PARENT_DIR = DATA_ROOT / "patients_LDR"
HDR_DAYS_FILE = DATA_ROOT / "hdr_days_after_ldr.csv"
# Monte-Carlo DVH dataset, one row per (record_id, structure).
DVH_FILE = DATA_ROOT / "dvh" / "dvh_mc_summary.csv"

IPSS_LONG_FILE = DATA_ROOT / "ipss_longitudinal_enriched.csv"
OUTPUT_FILE = DATA_ROOT / "dataset_minimal_v2.csv"

BASELINE_OUT_NAME = "baseline_ipss.csv"   # written into each patient directory
IPSS_FILENAME = "ipss_international_prostate_symptoms_score.csv"
BASELINE_SRC = "baseline.csv"
LDR_SRC = "ldr_details.csv"

# ============================================================
# IPSS CONSTANTS
# ============================================================
# Standard IPSS questions (a..g) mapped to the AUA subscores:
#   Obstructive/voiding = incomplete emptying (a), intermittency (c),
#                         weak stream (e), straining (f)
#   Irritative/storage  = frequency (b), urgency (d), nocturia (g)
IPSS_QUESTIONS = ["prostsex_ipss_a", "prostsex_ipss_b", "prostsex_ipss_c",
                  "prostsex_ipss_d", "prostsex_ipss_e", "prostsex_ipss_f",
                  "prostsex_ipss_g"]
IPSS_OBSTRUCTIVE_Q = ["prostsex_ipss_a", "prostsex_ipss_c",
                      "prostsex_ipss_e", "prostsex_ipss_f"]
IPSS_IRRITATIVE_Q = ["prostsex_ipss_b", "prostsex_ipss_d", "prostsex_ipss_g"]

COLS_TO_DROP_FROM_IPSS = ["redcap_repeat_instrument", "redcap_repeat_instance",
                          "ipss_international_prostate_symptoms_score_complete",
                          "ipss_score", "patient_id", "record_id_file"]

# DVH structures and metrics embedded as raw indices in the table. The DVH
# reduction strategies of the pipeline read the complete summary file through
# dvh_mc.py instead. These are the urinary targets: urethral and bladder-neck
# dose.
DVH_STRUCTURES = ["Urethra", "Bladder", "BladderNeck"]
DVH_METRICS = ["Dmin_Gy", "Dmax_Gy", "Dmean_Gy", "D95_Gy", "D50_Gy", "D2_Gy"]


# ============================================================
# Helpers
# ============================================================
def clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_id"] = df["record_id"].astype(str).str.strip()
    return df


def _get(series: pd.Series, col: str):
    return series[col] if (not series.empty and col in series.index) else np.nan


def compute_age_decimal(dob, tx_date) -> float:
    dob = pd.to_datetime(dob, errors="coerce")
    tx_date = pd.to_datetime(tx_date, errors="coerce")
    if pd.isna(dob) or pd.isna(tx_date):
        return np.nan
    return round((tx_date - dob).days / 365.25, 2)


# ============================================================
# PART A - enriched baseline (one row per patient)
# ============================================================
def build_patient_baseline(patient_dir: Path) -> dict | None:
    """Build the enriched baseline row of one patient, or None if empty."""
    bpath = patient_dir / BASELINE_SRC
    lpath = patient_dir / LDR_SRC
    if not bpath.exists():
        return None
    try:
        bdf = pd.read_csv(bpath)
    except Exception as e:
        print(f"  [WARN] {patient_dir.name}: could not read {BASELINE_SRC} ({e})")
        return None
    if bdf.empty:
        return None
    b = bdf.iloc[0]

    l = pd.Series(dtype=object)
    if lpath.exists():
        try:
            ldf = pd.read_csv(lpath)
            if not ldf.empty:
                l = ldf.iloc[0]
        except Exception as e:
            print(f"  [WARN] {patient_dir.name}: could not read {LDR_SRC} ({e})")

    tx_date = pd.to_datetime(_get(b, "tx_date"), errors="coerce")

    # ADT duration in days. The start and end dates are recorded only for a
    # minority of patients, so hx_len falls back to 0 elsewhere; the `adt` flag
    # carries the main signal. See config.IGNORE_FEATURES, which drops hx_len.
    hx_debut = pd.to_datetime(_get(b, "hx_debut"), errors="coerce")
    hx_fin = pd.to_datetime(_get(b, "hx_fin"), errors="coerce")
    hx_len = (hx_fin - hx_debut).days if (pd.notna(hx_debut) and pd.notna(hx_fin)) else 0.0

    crude_psa = pd.to_numeric(_get(b, "crude_psa"), errors="coerce")

    # PSA density (PSA over the volume at implant), more cancer-specific than
    # raw PSA.
    vol_for_density = pd.to_numeric(_get(l, "ldr_post_vol"), errors="coerce")
    psa_density = (
        crude_psa / vol_for_density
        if (pd.notna(crude_psa) and pd.notna(vol_for_density) and vol_for_density > 0)
        else np.nan
    )

    return {
        # --- identifiers / dates ---
        "record_id": str(_get(b, "record_id")).strip(),
        "tx_date": _get(b, "tx_date"),
        # --- demographics ---
        "age": compute_age_decimal(_get(b, "dob"), tx_date),
        # --- oncology (components only; adding a D'Amico composite on top
        #     would be collinear with them) ---
        "stage": _get(b, "stage"),
        "gleason": _get(b, "gleason"),
        "isup_grade": _get(b, "isup_grade"),
        "crude_psa": crude_psa,
        "psa_density": psa_density,
        # --- ADT (neoadjuvant confounder) ---
        "adt": _get(b, "adt"),
        "hx_len": hx_len,                         # ADT duration in days
        # --- implant volume and dosimetry (volume, needles and seeds are
        #     mutually collinear) ---
        "ldr_post_vol": _get(l, "ldr_post_vol"),          # volume at implant (primary)
        "ldr_previ_volcont": _get(l, "ldr_previ_volcont"),
        "ldr_live_aiguilles": _get(l, "ldr_live_aiguilles"),      # needles
        "ldr_previ_sourceprev": _get(l, "ldr_previ_sourceprev"),  # seeds (collinear)
        "ldr_live_dil": _get(l, "ldr_live_dil"),
    }


def generate_all_baselines(parent_dir: Path) -> pd.DataFrame:
    rows = []
    patient_dirs = sorted(p for p in parent_dir.iterdir() if p.is_dir())
    for pdir in patient_dirs:
        row = build_patient_baseline(pdir)
        if row is None:
            continue
        # Also write the per-patient version next to the source forms.
        pd.DataFrame([row]).to_csv(pdir / BASELINE_OUT_NAME, index=False)
        rows.append(row)
    baseline = clean_ids(pd.DataFrame(rows))
    print(f"[ok] Enriched baseline: {len(baseline)} patients, "
          f"{baseline.shape[1] - 1} columns")
    return baseline


# ============================================================
# PART B - longitudinal IPSS plus subscores
# ============================================================
def build_ipss_longitudinal(parent_dir: Path, out_file: Path) -> pd.DataFrame:
    frames = []
    for pdir in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        f = pdir / IPSS_FILENAME
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] {pdir.name}: could not read the IPSS form ({e})")
            continue
        if df.empty:
            continue
        df = df.drop(columns=[c for c in COLS_TO_DROP_FROM_IPSS if c in df.columns],
                     errors="ignore")
        frames.append(df)

    long = pd.concat(frames, ignore_index=True)
    long = clean_ids(long)

    # Obstructive and irritative subscores (sum of the matching questions).
    # min_count=1 keeps the result NaN when every question of a subscore is
    # missing.
    for q in IPSS_QUESTIONS:
        if q not in long.columns:
            long[q] = np.nan
    long["ipss_obstructive"] = long[IPSS_OBSTRUCTIVE_Q].sum(axis=1, min_count=1)
    long["ipss_irritative"] = long[IPSS_IRRITATIVE_Q].sum(axis=1, min_count=1)

    # Numeric-aware sort.
    long["_rid_num"] = pd.to_numeric(long["record_id"], errors="coerce")
    long = (long.sort_values(["_rid_num", "record_id", "ipss_date"], na_position="last")
                .drop(columns="_rid_num").reset_index(drop=True))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_file, index=False)
    print(f"[ok] Enriched longitudinal IPSS: {len(long)} measurements, "
          f"{long['record_id'].nunique()} patients -> {out_file.name}")
    return long


# ============================================================
# PART C - side inputs (salvage HDR, DVH)
# ============================================================
def load_hdr(path: Path) -> pd.DataFrame:
    df = clean_ids(pd.read_csv(path, dtype={"record_id": str}))
    return df[["record_id", "hdr_days_after_ldr"]]


def _pivot_dvh(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Pivot (record_id, structure, metrics) to one row per patient, with
    <Structure>_<metric> columns, restricted to DVH_STRUCTURES."""
    sub = df[df["structure"].isin(DVH_STRUCTURES)].copy()
    wide = sub.set_index(["record_id", "structure"])[metrics].unstack("structure")
    wide.columns = [f"{structure}_{metric}" for metric, structure in wide.columns]
    return wide


def load_dvh(path: Path) -> pd.DataFrame:
    """Embedded raw DVH indices: central Monte-Carlo value of the selected
    structures and metrics.

    Read through dvh_mc.load_mc_median (one row per (record_id, structure),
    metadata already removed), then pivoted to one row per patient
    (<Structure>_<metric>).

    With config.DVH_MC_USE_VARIANCE, the Monte-Carlo VARIANCE (= __std^2) of
    each metric is embedded as well, in <Structure>_<metric>_var columns. These
    raw columns feed the "rawdvh" strategy; the curated and pca strategies drop
    them by structure prefix and recompute their own variance through
    dvh_mc.load_dvh_indices.
    """
    med = load_mc_median(path, stat="mean")
    wide = _pivot_dvh(med, DVH_METRICS)

    if DVH_MC_USE_VARIANCE:
        var = load_mc_variance(path)
        var_metrics = [f"{m}{DVH_MC_VAR_SUFFIX}" for m in DVH_METRICS]
        vwide = _pivot_dvh(var, var_metrics)
        wide = wide.join(vwide)

    return wide.reset_index()


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    print("=" * 70)
    print("IPSS dataset - enriched baseline + IPSS subscores + DVH + salvage filter")
    print("=" * 70)

    if not PARENT_DIR.is_dir():
        print(f"Error: {PARENT_DIR} not found.", file=sys.stderr)
        return 1

    # A/B/C
    baseline = generate_all_baselines(PARENT_DIR)
    ipss = build_ipss_longitudinal(PARENT_DIR, IPSS_LONG_FILE)
    hdr = load_hdr(HDR_DAYS_FILE)
    dvh = load_dvh(DVH_FILE)

    # ------------------------------------------------------------
    # MERGE baseline + IPSS (left join, so no baseline patient is lost)
    # ------------------------------------------------------------
    merged = baseline.merge(ipss, on="record_id", how="left", validate="one_to_many")
    merged = merged.merge(hdr, on="record_id", how="left")
    merged = merged.merge(dvh, on="record_id", how="left")

    # ------------------------------------------------------------
    # Per-structure DVH availability flags.
    # ------------------------------------------------------------
    for struct in DVH_STRUCTURES:
        cols = [c for c in dvh.columns if c.startswith(f"{struct}_")]
        ids = set(dvh.loc[dvh[cols].notna().any(axis=1), "record_id"])
        merged[f"dvh_{struct.lower()}_available"] = merged["record_id"].isin(ids).astype(int)

    # ------------------------------------------------------------
    # Dates and delay between measurement and treatment.
    # ------------------------------------------------------------
    merged["ipss_date"] = pd.to_datetime(merged["ipss_date"], errors="coerce")
    merged["tx_date"] = pd.to_datetime(merged["tx_date"], errors="coerce")
    merged["ipss_days"] = (merged["ipss_date"] - merged["tx_date"]).dt.days

    # ------------------------------------------------------------
    # Informative missingness, leak-safe: number of PRE-treatment IPSS
    # measurements. The TOTAL number of visits is avoided, since it leaks
    # post-treatment follow-up.
    # ------------------------------------------------------------
    pretx_counts = (merged[merged["ipss_days"] < 0]
                    .groupby("record_id")["ipss_days"].size()
                    .rename("n_pretx_ipss"))
    merged = merged.merge(pretx_counts, on="record_id", how="left")
    merged["n_pretx_ipss"] = merged["n_pretx_ipss"].fillna(0).astype(int)

    # ------------------------------------------------------------
    # Salvage filter: keep only the measurements taken before the salvage HDR
    # cutoff, since salvage treatment resets the trajectory.
    # ------------------------------------------------------------
    before = len(merged)
    merged = merged[
        merged["hdr_days_after_ldr"].isna()
        | (merged["ipss_days"] <= merged["hdr_days_after_ldr"])
    ].copy()
    print(f"[ok] Measurements dropped after the salvage cutoff: {before - len(merged)}")

    # ------------------------------------------------------------
    # SORT + SAVE
    # ------------------------------------------------------------
    merged = merged.sort_values(["record_id", "ipss_date"]).reset_index(drop=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------
    # DIAGNOSTICS (aggregated)
    # ------------------------------------------------------------
    n_total = merged["record_id"].nunique()
    print("\n===== DIAGNOSTICS =====")
    print(f"Baseline patients      : {baseline['record_id'].nunique()}")
    print(f"IPSS patients          : {ipss['record_id'].nunique()}")
    print(f"Final patients         : {n_total}")
    for struct in DVH_STRUCTURES:
        flag = f"dvh_{struct.lower()}_available"
        n_avail = merged.loc[merged[flag] == 1, "record_id"].nunique()
        print(f"DVH {struct:<11} available: {n_avail}/{n_total}")
    print("\nBaseline feature coverage (patients with at least one non-NaN value):")
    feat_cols = [c for c in baseline.columns if c not in ("record_id", "tx_date")]
    cov = (merged.groupby("record_id")[feat_cols].first().notna().sum()
           .sort_values(ascending=False))
    for c, n in cov.items():
        print(f"   {c:<24} {n:>4}/{n_total}  ({100 * n / n_total:.0f}%)")

    print(f"\n[done] {OUTPUT_FILE}")
    print(f"  Shape: {merged.shape}")
    print("  Read downstream through config.DATASET_MINIMAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
