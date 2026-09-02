"""
diagnostic_effectif_prostate.py
===============================
Why does the prostate TOST rest on fewer pairs than the number of segmented
patients in the cohort?

From the index CSVs alone, this script establishes WHERE patients are lost and
WHICH DICOM export variable separates the retained cases from the lost ones.

Confidentiality convention: this script prints ONLY aggregates (counts, medians,
cross-tabulations). No record_id and no patient-level value is ever printed.

Usage: python diagnostic_effectif_prostate.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

import utils as U

PANEL_IDX = ["D90_pct", "V100_pct", "V150_pct", "V200_pct"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    man = U._read_harmonized(U.MANUAL_CSV)
    det = U._read_harmonized(U.DET_CSV)
    mc = U._read_harmonized(U.MC_SUMMARY_CSV)

    # ------------------------------------------------------------------
    section("1. Sample size per structure and per source")
    # ------------------------------------------------------------------
    for name, df in (("manual", man), ("deterministic", det), ("bayesian (MC)", mc)):
        n = df.groupby("structure").record_id.nunique()
        print(f"  {name:14s} {df.record_id.nunique():4d} patients | " +
              " ".join(f"{k}={v}" for k, v in n.items()))

    prost_man = set(man[man.structure == "Prostate"].record_id)
    prost_det = set(det[det.structure == "Prostate"].record_id)
    all_man, all_det = set(man.record_id), set(det.record_id)
    missing_prostate = all_man - prost_man

    # ------------------------------------------------------------------
    section("2. Arithmetic decomposition of the TOST n")
    # ------------------------------------------------------------------
    print(f"  patients in the manual file                   : {len(all_man)}")
    print(f"  ... with a Prostate row                       : {len(prost_man)}")
    print(f"  ... WITHOUT a Prostate row                    : {len(missing_prostate)}")
    print(f"  patients in the deterministic file            : {len(all_det)}")
    print(f"  ... with a Prostate row                       : {len(prost_det)}")
    print(f"  INTERSECTION = n of the prostate TOST         : {len(prost_man & prost_det)}")
    print(f"  lost because the manual Prostate is absent    : {len(prost_det - prost_man)}")
    print(f"  lost because absent from the deterministic    : {len(prost_man - prost_det)}")

    print("\n  Which OTHER structures do the patients without a manual Prostate have?")
    sub = man[man.record_id.isin(missing_prostate)]
    print("   ", sub.groupby("structure").record_id.nunique().to_dict())
    print("  -> the RTSTRUCT was therefore read; only the prostate ROI is missing.")

    # ------------------------------------------------------------------
    section("3. The discriminant: the source of the prescription dose")
    # ------------------------------------------------------------------
    ps = man.groupby("record_id").presc_source.agg(
        lambda s: ";".join(sorted(set(s.dropna()))))
    tab = pd.DataFrame({"presc_source": ps})
    tab["has_prostate"] = tab.index.isin(prost_man)
    print(pd.crosstab(tab.presc_source, tab.has_prostate, margins=True).to_string())
    print("\n  Prescription dose associated with each source:")
    print(man.groupby("presc_source").prescription_dose_Gy
             .agg(["size", "nunique", "median"]).to_string())

    # ------------------------------------------------------------------
    section("4. Temporal signature: loss rate per identifier range")
    # ------------------------------------------------------------------
    ids = pd.Series(sorted(all_man))
    d = pd.DataFrame({"rid": ids,
                      "id_range": pd.to_numeric(ids, errors="coerce").floordiv(100).astype("Int64")})
    d["has_prostate"] = d.rid.isin(prost_man)
    g = d.groupby("id_range").agg(total=("rid", "size"), present=("has_prostate", "sum"))
    g["missing"] = g.total - g.present
    g["loss_rate_pct"] = (g.missing / g.total * 100).round(1)
    print(g.to_string())

    # ------------------------------------------------------------------
    section("5. Geometric signature: CT slice thickness")
    # ------------------------------------------------------------------
    dp = det[det.structure == "Prostate"].copy()
    dp["included"] = dp.record_id.isin(prost_man)
    th = pd.to_numeric(dp.thickness_mm, errors="coerce")
    regular = th.round(2).isin([1.00, 2.00])
    print(pd.crosstab(regular.map({True: "regular (1.00 or 2.00 mm)",
                                   False: "irregular"}),
                      dp.included.map({True: "included", False: "excluded"}),
                      margins=True).to_string())
    print(f"\n  share of irregular thickness: included {(~regular)[dp.included].mean() * 100:.1f} %"
          f"  |  excluded {(~regular)[~dp.included].mean() * 100:.1f} %")

    # ------------------------------------------------------------------
    section("6. Representativeness: do the excluded differ from the included?")
    # ------------------------------------------------------------------
    print("  Comparison on the AUTOMATIC side, the only source available for both groups.")
    rows = []
    for c in PANEL_IDX + ["volume_cc", "thickness_mm", "n_planes"]:
        if c not in dp.columns:
            continue
        a = pd.to_numeric(dp.loc[dp.included, c], errors="coerce").dropna()
        b = pd.to_numeric(dp.loc[~dp.included, c], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        rows.append(dict(variable=c, n_included=len(a), med_included=a.median(),
                         n_excluded=len(b), med_excluded=b.median(),
                         d_standardise=(b.mean() - a.mean()) / sp if sp else np.nan,
                         p_MannWhitney=p))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    # ------------------------------------------------------------------
    section("7. Secondary anomalies")
    # ------------------------------------------------------------------
    print("  Fill rate of the quality-control columns (manual file):")
    for c in ("mask_volume_cc", "coverage_in_dose", "n_planes_interpolated"):
        if c in man.columns:
            s = man.groupby("structure")[c].apply(
                lambda v: pd.to_numeric(v, errors="coerce").notna().mean())
            print(f"    {c:22s}", {k: round(float(x), 2) for k, x in s.items()})

    mp = man[man.structure == "Prostate"]
    npi = pd.to_numeric(mp.n_planes_interpolated, errors="coerce")
    print(f"\n  Manual prostate contours that required plane interpolation: "
          f"{int((npi > 0).sum())} / {len(mp)}")

    bn_pairs = set(man[man.structure == "BladderNeck"].record_id) & \
               set(det[det.structure == "BladderNeck"].record_id)
    assumed_dose = set(ps[ps != "rtdose"].index)
    print(f"\n  BladderNeck pairs: {len(bn_pairs)}, of which {len(bn_pairs & assumed_dose)} "
          f"a dose de prescription SUPPOSEE (repli isotope).")


if __name__ == "__main__":
    sys.exit(main())
