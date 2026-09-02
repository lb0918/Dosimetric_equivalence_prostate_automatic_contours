"""
00_validate_synthetic.py
========================
End-to-end SYNTHETIC validation of the concordance pipeline. Run it BEFORE any
real run.

A synthetic cohort is built in which the manual-versus-automatic offset is KNOWN
per index:
    - offset well below the margin -> the TOST MUST conclude equivalence;
    - offset above the margin      -> the TOST MUST reject equivalence;
    - auto == manual               -> CCC and ICC must be close to 1;
    - manual drawn from the MC distribution -> empirical coverage close to 95%.
The deterministic file deliberately names the structure 'Bladder neck' (with a
space), to verify that the harmonisation ('Bladder neck' -> 'BladderNeck')
prevents a silently empty merge.

The test repoints the paths of `utils` to synthetic CSVs, then runs the REAL
FUNCTIONS of scripts 01 to 04, duplicating no logic. Output is PASS / FAIL per
check, with a non-zero exit code if any check fails.

Usage: python 00_validate_synthetic.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import utils as U

SEED = 20260703
N_AUTO = 400          # patients present in the automatic sources (det + mc)
N_MANUAL_EXTRA = 20   # manual-only patients (which must be dropped)

# Per (structure, index): population mu/sigma, deterministic offset,
# deterministic noise, and MC bias.
# det_offset=None -> the automatic value equals the manual one (CCC/ICC test).
# degenerate_frac -> fraction of patients with a degenerate MC interval (std=0).
SIGMA_WITHIN = 3.0    # measurement / MC dispersion (drives the coverage check)
SPEC = {
    ("Prostate", "D90_pct"):   dict(mu=95, sigma=6, det_offset=2.0,  det_sd=3.0, mc_bias=0.0),
    ("Prostate", "V100_pct"):  dict(mu=97, sigma=2, det_offset=8.0,  det_sd=2.0, mc_bias=0.0),
    ("Prostate", "V150_pct"):  dict(mu=60, sigma=8, det_offset=None, det_sd=0.0, mc_bias=0.0),
    ("Prostate", "V200_pct"):  dict(mu=25, sigma=5, det_offset=1.0,  det_sd=2.0, mc_bias=0.0),
    ("BladderNeck", "D2cc_pct"): dict(mu=90, sigma=8, det_offset=2.0,  det_sd=3.0, mc_bias=0.0),
    ("BladderNeck", "D1cc_pct"): dict(mu=95, sigma=8, det_offset=20.0, det_sd=3.0, mc_bias=20.0),
    ("BladderNeck", "V100_pct"): dict(mu=80, sigma=6, det_offset=1.0,  det_sd=2.0, mc_bias=0.0,
                                      degenerate_frac=0.30),
}

# Deliberately mismatched structure name in the deterministic file.
DET_STRUCT_NAME = {"Prostate": "Prostate", "BladderNeck": "Bladder neck"}


def _structure_indices(structure):
    return [i for s, i, _ in U.PANEL if s == structure]


def generate_csvs(outdir: Path):
    rng = np.random.default_rng(SEED)
    manual_ids = np.arange(N_AUTO + N_MANUAL_EXTRA)     # 0..419
    auto_ids = np.arange(N_AUTO)                         # 0..399

    manual_parts, det_parts, mc_parts = [], [], []

    for structure in ("Prostate", "BladderNeck"):
        indices = _structure_indices(structure)
        man = pd.DataFrame({"record_id": manual_ids, "structure": structure})
        det = pd.DataFrame({"record_id": auto_ids,
                            "structure": DET_STRUCT_NAME[structure],
                            "pred_mode": "deterministic"})
        mc = pd.DataFrame({"record_id": auto_ids, "structure": structure,
                           "model": U.STRUCTURE_TO_MODEL[structure]})

        for index in indices:
            sp = SPEC[(structure, index)]
            mu_all = sp["mu"] + rng.normal(0, sp["sigma"], size=manual_ids.size)  # latent/patient
            manual_all = mu_all + rng.normal(0, SIGMA_WITHIN, size=manual_ids.size)
            man[index] = manual_all
            man[index.replace("_pct", "_Gy")] = manual_all * 1.45   # colonne _Gy factice

            mu_auto = mu_all[:N_AUTO]
            manual_auto = manual_all[:N_AUTO]

            # --- deterministic: auto = manual - offset (+ noise), or identical ---
            if sp["det_offset"] is None:
                det_val = manual_auto.copy()
            else:
                det_val = manual_auto - sp["det_offset"] + rng.normal(0, sp["det_sd"], size=N_AUTO)
            det[index] = det_val

            # --- MC : 20 passes ~ Normal(mu + biais, SIGMA_WITHIN) ---
            draws = mu_auto[:, None] + sp["mc_bias"] + rng.normal(0, SIGMA_WITHIN, size=(N_AUTO, 20))
            frac = sp.get("degenerate_frac", 0.0)
            if frac > 0:
                n_deg = int(round(frac * N_AUTO))
                deg_idx = rng.choice(N_AUTO, size=n_deg, replace=False)
                draws[deg_idx, :] = (mu_auto[deg_idx] + sp["mc_bias"])[:, None]  # std=0
            mc[f"{index}__mean"] = draws.mean(axis=1)
            mc[f"{index}__std"] = draws.std(axis=1, ddof=1)
            mc[f"{index}__min"] = draws.min(axis=1)
            mc[f"{index}__p2_5"] = np.percentile(draws, 2.5, axis=1)
            mc[f"{index}__p50"] = np.percentile(draws, 50, axis=1)
            mc[f"{index}__p97_5"] = np.percentile(draws, 97.5, axis=1)
            mc[f"{index}__max"] = draws.max(axis=1)

        manual_parts.append(man)
        det_parts.append(det)
        mc_parts.append(mc)

    manual_csv = outdir / "syn_manual.csv"
    det_csv = outdir / "syn_det.csv"
    mc_csv = outdir / "syn_mc_summary.csv"
    pd.concat(manual_parts, ignore_index=True).to_csv(manual_csv, index=False)
    pd.concat(det_parts, ignore_index=True).to_csv(det_csv, index=False)
    pd.concat(mc_parts, ignore_index=True).to_csv(mc_csv, index=False)
    return manual_csv, det_csv, mc_csv


def _load_module(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Checker:
    def __init__(self):
        self.fails = 0

    def check(self, label, cond):
        status = "PASS" if cond else "FAIL"
        if not cond:
            self.fails += 1
        print(f"  [{status}] {label}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="concordance_syn_"))
    manual_csv, det_csv, mc_csv = generate_csvs(tmp)

    # Repoint utils to the synthetic CSVs (the scripts read these attributes).
    U.MANUAL_CSV = manual_csv
    U.DET_CSV = det_csv
    U.MC_SUMMARY_CSV = mc_csv
    U.OUT_DIR = tmp
    U.FIG_DIR = tmp / "figures"
    U.PAIRED_TABLE_CSV = tmp / "paired_table.csv"

    m01 = _load_module("01_build_paired_table.py", "build01")
    m02 = _load_module("02_tost_equivalence.py", "tost02")
    m03 = _load_module("03_concordance_descriptors.py", "desc03")
    m04 = _load_module("04_mc_coverage.py", "cov04")

    paired, report = m01.build_paired()
    tost = m02.run_tost(paired)
    desc = m03.compute(paired)
    cov = m04.compute(paired[paired["comparison"] == "bayes"].copy())

    def trow(df, s, i, c):
        r = df[(df.structure == s) & (df["index"] == i) & (df.comparison == c)]
        return r.iloc[0]

    def crow(df, s, i):
        r = df[(df.structure == s) & (df["index"] == i)]
        return r.iloc[0]

    ck = Checker()

    print("\n--- Harmonisation and matching ---")
    bn_det = paired[(paired.structure == "BladderNeck") & (paired.comparison == "det")]
    ck.check("'Bladder neck'->'BladderNeck' harmonisation (det not empty)", len(bn_det) > 0)
    r = report[(report.structure == "Prostate") & (report["index"] == "D90_pct")
               & (report.comparison == "det")].iloc[0]
    ck.check(f"inner match = {N_AUTO} (drops the {N_MANUAL_EXTRA} extra manual patients)",
             int(r["n_paired"]) == N_AUTO)
    ck.check("no imputation (n_dropped_na == 0)", int(r["n_dropped_na"]) == 0)

    print("\n--- TOST: equivalence expected (offset well below the margin) ---")
    ck.check("Prostate D90 (det) equivalent", bool(trow(tost, "Prostate", "D90_pct", "det").verdict_conf))
    ck.check("Prostate V200 (det) equivalent", bool(trow(tost, "Prostate", "V200_pct", "det").verdict_conf))
    ck.check("Prostate V150 (det) equivalent (identical)", bool(trow(tost, "Prostate", "V150_pct", "det").verdict_conf))
    ck.check("BladderNeck D2cc (det) equivalent", bool(trow(tost, "BladderNeck", "D2cc_pct", "det").verdict_conf))

    print("\n--- TOST: rejection expected (offset above the margin) ---")
    pv = trow(tost, "Prostate", "V100_pct", "det")
    ck.check("Prostate V100 (det) NOT equivalent (decision margin)", not bool(pv.verdict_conf))
    ck.check("Prostate V100 (det) NOT equivalent (sensitivity margin)", not bool(pv.verdict_sens))
    ck.check("BladderNeck D1cc (det) NOT equivalent", not bool(trow(tost, "BladderNeck", "D1cc_pct", "det").verdict_conf))
    ck.check("BladderNeck D1cc (bayes) NOT equivalent", not bool(trow(tost, "BladderNeck", "D1cc_pct", "bayes").verdict_conf))

    print("\n--- Holm: applied to the confirmatory family only ---")
    ck.check("p_holm filled for the Prostate", tost[tost.tier == U.TIER_CONF]["p_holm"].notna().all())
    ck.check("p_holm absent for BladderNeck", tost[tost.tier == U.TIER_EXPL]["p_holm"].isna().all())

    print("\n--- CCC / ICC ≈ 1 quand auto == manuel ---")
    dv = crow(desc, "Prostate", "V150_pct")   # V150 det is identical to manual
    ck.check("Prostate V150 CCC ~ 1", dv.ccc > 0.999)
    ck.check("Prostate V150 ICC(A,1) ~ 1", dv.icc_a1 > 0.999)
    ck.check("Prostate V150 LoA inside the decision margin", bool(dv.loa_within_delta_conf))
    dv2 = crow(desc, "Prostate", "V100_pct")
    ck.check("Prostate V100 CCC < 0.99 (offset present)", dv2.ccc < 0.99)

    print("\n--- MC coverage close to 0.95, and degenerate intervals ---")
    cd90 = crow(cov, "Prostate", "D90_pct")
    # NB : l'intervalle p2.5/p97.5 issu de 20 tirages sous-couvre par construction
    # (coarse resolution) -> a floor of 0.85 is tolerated; see the documented limitation.
    ck.check(f"Prostate D90 percentile-interval coverage ~0.95 (obs={cd90.coverage_pctinterval:.3f})",
             0.85 <= cd90.coverage_pctinterval <= 1.0)
    ck.check(f"Prostate D90 Gaussian coverage ~0.95 (obs={cd90.coverage_gaussian:.3f})",
             0.88 <= cd90.coverage_gaussian <= 1.0)
    cbn = crow(cov, "BladderNeck", "V100_pct")
    ck.check(f"BladderNeck V100 degenerate intervals counted (n={int(cbn.n_degenerate)})",
             int(cbn.n_degenerate) > 0)
    ck.check("degenerate ones excluded from the denominator (n_nondeg + n_deg == n_total)",
             int(cbn.n_nondegenerate) + int(cbn.n_degenerate) == int(cbn.n_total))

    print("\n--- Primitive TOST (test unitaire direct) ---")
    ck.check("tost identical (diff=0) -> equivalent", U.tost_paired(np.zeros(50), 5.0)["equivalent"])
    big = U.tost_paired(np.full(200, 12.0) + np.random.default_rng(1).normal(0, 1, 200), 5.0)
    ck.check("tost offset 12 vs margin 5 -> NOT equivalent", not big["equivalent"])

    print(f"\n=== {'TOUS LES TESTS PASSENT' if ck.fails == 0 else str(ck.fails) + ' ECHEC(S)'} ===")
    print(f"(CSV synthetiques : {tmp})")
    sys.exit(1 if ck.fails else 0)


if __name__ == "__main__":
    main()
