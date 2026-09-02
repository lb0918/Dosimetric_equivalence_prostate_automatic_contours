"""
07_validate_joint_synthetic.py
==============================
End-to-end SYNTHETIC validation of the joint quality x volume analysis (script
`07_joint_dice_volume.py`), to the same standard as `00_validate_synthetic.py`.
Run it BEFORE any real run.

Principle: a cohort is built in which the dependence of equivalence on BOTH the
Dice and the prostate volume is KNOWN ANALYTICALLY; the paths of `utils` are
repointed to those synthetic CSVs; then the REAL FUNCTIONS of 07 are run (no
logic is duplicated) and compared against the closed-form truths.

Three regimes, one per index, chosen to cover the three ways the script can go
wrong:

  D90_pct  - LOGISTIC TRUTH WITH INTERACTION.
        equiv ~ Bernoulli(sigmoid(b0 + b1*dice + b2*v + b3*dice*v)),
        with v = (V - VOL_CENTER)/10.
        True threshold: dice*(v) = (logit(p*) - b0 - b2*v) / (b1 + b3*v).
        The script MUST detect the volume effect, detect the interaction, and
        recover the threshold curve.

  V150_pct - TRUTH WITH NO VOLUME EFFECT.
        |delta| ~ half-normal with standard deviation sigma(dice), with no
        volume term at all.
        The script MUST NOT invent a volume effect, and its threshold curve must
        be FLAT. This is the check that catches an overfitting model.

  V100_pct - QUANTILE TRUTH (the dual estimator).
        |delta| ~ half-normal with standard deviation
        sigma = s0 + s1*dice + s2*v, so Q95(|delta|) = k*sigma is EXACTLY linear
        in (dice, v), with k the standard-normal 0.975 quantile.
        True threshold: dice*(v) = (margin/k - s0 - s2*v) / s1.
        This exercises the quantile-regression branch, independently of the
        logistic one.

To these are added the matching checks (a single volume row per patient although
the source file is longitudinal, dropping of patients without a volume,
detection of an inconsistent volume) and the support mask of the figures.

Output: PASS / FAIL per check, with a non-zero exit code if any check fails.

Usage: python 07_validate_joint_synthetic.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import utils as U

SEED = 20260730
N_PATIENTS = 900          # patients with a DVH and geometry
N_NO_VOLUME = 40          # of those, without ldr_post_vol -> must be dropped
N_VISITS_MAX = 9          # rows per patient in the longitudinal IPSS file
BOOT_TEST = 60            # reduced bootstrap: the logic is validated, not the precision

# --- D90 truth: logistic with interaction ----------------------------------
# The calibration is constrained by TWO opposing requirements, which must hold
# together for the test to mean anything:
#   (1) the p* threshold must cross the observed Dice range over the whole
#       tabulated volume range, otherwise an extrapolated threshold would be
#       validated;
#   (2) the interaction must be DETECTABLE. The dice*v column is nearly
#       collinear with v when the Dice is tightly spread, the residual signal
#       being |B3|*sd(dice)*sd(v). With a narrow Dice spread and a small |B3| it
#       weighs a fraction of a logit and is undetectable, so such a calibration
#       would in fact be testing a non-existent interaction.
# Hence a wider Dice spread and a large |B3|, with B1 chosen so that the
# denominator B1 + B3*v only vanishes beyond the tabulated range.
B0, B1, B2, B3 = -51.80, 60.0, 23.34, -25.0
DICE_SD = 0.060

# --- V150 truth: no volume effect ------------------------------------------
C0, C1 = 10.16, -8.0      # sigma(dice) = C0 + C1*dice

# --- V100 truth: quantile linear in (dice, v) ------------------------------
S0, S1, S2 = 7.60, -7.0, -0.15    # sigma(dice, v) = S0 + S1*dice + S2*v

# --- V200: filler, no assertion --------------------------------------------
E0, E1 = 9.0, -7.0

K95 = float(stats.norm.ppf(0.975))       # Q95(|N(0,s)|) = K95 * s
VOL_CENTER = 35.0                        # centre of the truth (cc)

P_TARGET = 0.90
TOL_THR_LOGIT = 0.020     # tolerance on the recovered Dice threshold (Dice points)
TOL_THR_QUANT = 0.030
TOL_FLAT = 0.015          # max amplitude of a threshold curve that should be flat


def _load_module(fname: str, modname: str):
    spec = importlib.util.spec_from_file_location(
        modname, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# ANALYTIC TRUTHS
# ============================================================
def truth_thr_d90(V, p_target=P_TARGET):
    """True Dice threshold for D90 (logistic with interaction), native volume."""
    v = (np.asarray(V, float) - VOL_CENTER) / 10.0
    return (np.log(p_target / (1 - p_target)) - B0 - B2 * v) / (B1 + B3 * v)


def truth_thr_v100(V, delta):
    """True Dice threshold for V100 (Q95 tolerance bound), native volume."""
    v = (np.asarray(V, float) - VOL_CENTER) / 10.0
    return (delta / K95 - S0 - S2 * v) / S1


# ============================================================
# GENERATION
# ============================================================
def generate_csvs(outdir: Path):
    rng = np.random.default_rng(SEED)
    ids = np.arange(N_PATIENTS)

    # --- anatomy and segmentation quality ---
    # The Dice increases slightly with the volume: this is the real mechanical
    # coupling (a small organ has a penalised Dice), and it is precisely what
    # confounds volume with quality, hence what the joint model must disentangle.
    vol = np.clip(rng.normal(35.0, 10.0, N_PATIENTS), 14.0, 72.0)
    dice = np.clip(0.880 + 0.0015 * (vol - 35.0) + rng.normal(0, DICE_SD, N_PATIENTS),
                   0.55, 0.995)
    v = (vol - VOL_CENTER) / 10.0

    # --- manual indices (the level matters little: only the delta is analysed) ---
    manual = {
        "D90_pct": rng.normal(95, 6, N_PATIENTS),
        "V100_pct": rng.normal(97, 2, N_PATIENTS),
        "V150_pct": rng.normal(60, 8, N_PATIENTS),
        "V200_pct": rng.normal(25, 5, N_PATIENTS),
    }
    delta = {i: U.DELTA[("Prostate", i)]["delta_conf"] for i in manual}

    def _half_normal(sigma):
        sigma = np.clip(sigma, 0.02, None)
        return np.abs(rng.normal(0.0, sigma))

    absd = {}
    # D90: equiv drawn from the true logistic, then |delta| placed on the right side of the margin.
    eta = B0 + B1 * dice + B2 * v + B3 * dice * v
    p_eq = 1.0 / (1.0 + np.exp(-eta))
    eq = rng.random(N_PATIENTS) < p_eq
    absd["D90_pct"] = np.where(
        eq,
        delta["D90_pct"] * rng.uniform(0.05, 0.95, N_PATIENTS),
        delta["D90_pct"] * rng.uniform(1.05, 2.50, N_PATIENTS))
    # V150: |delta| depends ONLY on the Dice.
    absd["V150_pct"] = _half_normal(C0 + C1 * dice)
    # V100: |delta| half-normal with a standard deviation linear in (dice, v).
    absd["V100_pct"] = _half_normal(S0 + S1 * dice + S2 * v)
    # V200 : remplissage.
    absd["V200_pct"] = _half_normal(E0 + E1 * dice)

    sign = np.where(rng.random(N_PATIENTS) < 0.5, -1.0, 1.0)
    man = pd.DataFrame({"record_id": ids, "structure": "Prostate"})
    det = pd.DataFrame({"record_id": ids, "structure": "Prostate",
                        "pred_mode": "deterministic"})
    for idx, vals in manual.items():
        man[idx] = vals
        # diff = manual - auto, hence auto = manual - diff.
        det[idx] = vals - sign * absd[idx]

    manual_csv = outdir / "syn_manual.csv"
    det_csv = outdir / "syn_det.csv"
    man.to_csv(manual_csv, index=False)
    det.to_csv(det_csv, index=False)

    # --- geometry (metrics_per_case format of the cross-dataset) ---
    gt_vol = vol * rng.uniform(0.92, 1.08, N_PATIENTS)
    rel_err = rng.normal(0.0, 0.08, N_PATIENTS)
    geom = pd.DataFrame({
        "case_id": ids,
        "dice": dice,
        "jaccard": dice / (2 - dice),
        "precision": np.clip(dice + rng.normal(0, 0.02, N_PATIENTS), 0.4, 1.0),
        "recall_sensitivity": np.clip(dice + rng.normal(0, 0.02, N_PATIENTS), 0.4, 1.0),
        "specificity": 0.999,
        "volume_similarity": 1 - np.abs(rel_err),
        # ASSD anti-correlated with the Dice: reproduces the collinearity of 06.
        "hausdorff95_mm": 6.0 - 5.0 * dice + np.abs(rng.normal(0, 0.4, N_PATIENTS)),
        "hausdorff_mm": 9.0 - 6.0 * dice + np.abs(rng.normal(0, 0.6, N_PATIENTS)),
        "assd_mm": np.clip(4.0 - 3.0 * dice + rng.normal(0, 0.10, N_PATIENTS), 0.1, None),
        "gt_volume_ml": gt_vol,
        "pred_volume_ml": gt_vol * (1 + rel_err),
        "volume_error_ml": gt_vol * rel_err,
    })
    geom_csv = outdir / "syn_metrics_per_case.csv"
    geom.to_csv(geom_csv, index=False)

    # --- longitudinal IPSS file: volume REPEATED across several visits ---
    # This is the pitfall load_prostate_volume must defuse: without
    # deduplication, each patient would weigh as much as their number of visits.
    no_vol = set(ids[-N_NO_VOLUME:].tolist())
    rows = []
    for pid, vv in zip(ids, vol):
        for visit in range(int(rng.integers(2, N_VISITS_MAX + 1))):
            rows.append(dict(record_id=pid, redcap_event_name=f"visite_{visit}",
                             ldr_post_vol=(np.nan if pid in no_vol else vv)))
    ipss_csv = outdir / "syn_ipss_minimal.csv"
    pd.DataFrame(rows).to_csv(ipss_csv, index=False)

    truth = dict(vol=vol, dice=dice, absd=absd, delta=delta, ids=ids,
                 no_vol=no_vol, n_with_vol=N_PATIENTS - N_NO_VOLUME)
    return manual_csv, det_csv, geom_csv, ipss_csv, truth


class Checker:
    def __init__(self):
        self.fails = 0

    def check(self, label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        if not cond:
            self.fails += 1
        print(f"  [{status}] {label}" + (f"   {detail}" if detail else ""))

    def near(self, label, got, want, tol):
        ok = np.isfinite(got) and abs(got - want) <= tol
        self.check(label, ok, f"got={got:.4f} expected={want:.4f} tol={tol}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="joint_vol_syn_"))
    print(f"[setup] repertoire temporaire : {tmp}")
    manual_csv, det_csv, geom_csv, ipss_csv, truth = generate_csvs(tmp)

    # Repoint the paths of utils to the synthetic CSVs.
    U.MANUAL_CSV = manual_csv
    U.DET_CSV = det_csv
    U.GEOM_DATASETS["Prostate"] = geom_csv
    U.IPSS_MINIMAL_CSV = ipss_csv
    U.OUT_DIR = tmp / "results"
    U.ensure_dir(U.OUT_DIR)

    m07 = _load_module("07_joint_dice_volume.py", "joint_dice_volume_07")
    m07.OUT_DIR = U.OUT_DIR / "joint_dice_volume"
    U.ensure_dir(m07.OUT_DIR)

    c = Checker()
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------
    print("\n[1] Loading of the prostate volume")
    # ------------------------------------------------------------
    vol_tbl = U.load_prostate_volume()
    c.check("a single row per patient (longitudinal file deduplicated)",
            len(vol_tbl) == vol_tbl["record_id"].nunique())
    c.check("patients without a volume excluded from the table",
            len(vol_tbl) == truth["n_with_vol"],
            f"n={len(vol_tbl)} expected={truth['n_with_vol']}")
    ref = pd.DataFrame({"record_id": truth["ids"].astype(str),
                        "truth_vol": truth["vol"]})
    chk = vol_tbl.merge(ref, on="record_id")
    c.check("volume values identical to the truth",
            bool(np.allclose(chk["prostate_vol"], chk["truth_vol"])))

    # An inconsistent volume must be refused explicitly, not silently averaged.
    bad = pd.read_csv(ipss_csv)
    bad.loc[bad.index[0], "ldr_post_vol"] = 999.0
    bad_csv = tmp / "syn_ipss_incoherent.csv"
    bad.to_csv(bad_csv, index=False)
    try:
        U.load_prostate_volume(path=bad_csv)
        raised = False
    except SystemExit:
        raised = True
    c.check("non-constant volume per patient -> SystemExit in strict mode", raised)

    # ------------------------------------------------------------
    print("\n[2] Table de base conjointe")
    # ------------------------------------------------------------
    base = m07.load_base()
    n_pat = base["record_id"].nunique()
    c.check("patients without a volume dropped from the base table",
            n_pat == truth["n_with_vol"], f"n_patients={n_pat}")
    c.check("4 prostate indices x n patients",
            len(base) == 4 * truth["n_with_vol"], f"n_lignes={len(base)}")
    c.check("no missing volume after the join",
            bool(base["prostate_vol"].notna().all()))

    desc = m07.volume_quality_descriptive(base)
    rho_dice = float(desc.loc[desc["variable"] == "dice", "spearman_vs_volume"].iloc[0])
    c.check("volume->Dice coupling recovered (positive, as simulated)",
            rho_dice > 0.05, f"rho={rho_dice:+.3f}")
    c.check("descriptive block computed on unique patients, not on rows",
            int(desc.loc[desc["variable"] == "dice", "n"].iloc[0]) == truth["n_with_vol"])

    # ------------------------------------------------------------
    print("\n[3] Joint models - effect detection")
    # ------------------------------------------------------------
    models = m07.joint_models(base, "dice", seed=0)
    mc = models[models["margin"] == "conf"].set_index("index")

    c.check("D90 (truth: volume effect) -> volume LRT significant",
            mc.loc["D90_pct", "lrt_volume_p"] < 0.05,
            f"p={mc.loc['D90_pct', 'lrt_volume_p']:.2e}")
    c.check("D90 (truth: interaction) -> interaction LRT significant",
            mc.loc["D90_pct", "lrt_interaction_p"] < m07.INTERACTION_ALPHA,
            f"p={mc.loc['D90_pct', 'lrt_interaction_p']:.2e}")
    c.check("V150 (truth: NO volume effect) -> volume LRT not significant",
            mc.loc["V150_pct", "lrt_volume_p"] > 0.05,
            f"p={mc.loc['V150_pct', 'lrt_volume_p']:.3f}")
    # The DECISION is tested, not the raw p-value: on a null truth, a p at
    # a p just under 0.05 is an ordinary false positive, and it is precisely against
    # it is INTERACTION_ALPHA that protects the threshold curve.
    c.check("V150 -> no interaction RETAINED (null truth)",
            not bool(mc.loc["V150_pct", "threshold_shifts"]),
            f"p={mc.loc['V150_pct', 'lrt_interaction_p']:.3f} "
            f"(alpha={m07.INTERACTION_ALPHA})")
    c.check("sign of the volume effect matches the simulation (OR > 1)",
            mc.loc["D90_pct", "m2_or_volume_per_10cc"] > 1.0,
            f"OR={mc.loc['D90_pct', 'm2_or_volume_per_10cc']:.3f}")

    # ------------------------------------------------------------
    print("\n[4] Threshold curve - is the analytic truth recovered?")
    # ------------------------------------------------------------
    thr, boot_cache = m07.threshold_vs_volume(base, "dice", P_TARGET, BOOT_TEST,
                                              rng, models)
    t90 = thr[(thr["index"] == "D90_pct") & (thr["margin"] == "conf")]
    c.check("D90: interaction model retained automatically",
            bool(t90["model"].iloc[0].startswith("M3")), t90["model"].iloc[0])
    for _, r in t90.iterrows():
        c.near(f"D90 seuil Dice a V={r['volume_cc']:.1f} cc (q{r['vol_quantile']:.2f})",
               r["thr_logistic"], float(truth_thr_d90(r["volume_cc"])),
               TOL_THR_LOGIT)
    inside = (t90["thr_logistic"] >= t90["thr_logistic_lo"]) & \
             (t90["thr_logistic"] <= t90["thr_logistic_hi"])
    c.check("bootstrap CI brackets the point threshold", bool(inside.all()))

    t150 = thr[(thr["index"] == "V150_pct") & (thr["margin"] == "conf")]
    amp = float(np.nanmax(t150["thr_logistic"]) - np.nanmin(t150["thr_logistic"]))
    c.check("V150: FLAT threshold curve (no volume effect simulated)",
            amp < TOL_FLAT, f"amplitude={amp:.4f} sur q10-q90 du volume")
    c.check("V150: additive model retained (no interaction)",
            bool(t150["model"].iloc[0].startswith("M2")), t150["model"].iloc[0])

    # ------------------------------------------------------------
    print("\n[5] Dual estimator (quantile) - linear Q95 truth")
    # ------------------------------------------------------------
    # The V100 truth has NO interaction: the additive model is forced, to isolate
    # the quantile branch independently of the selection made by the logistic
    # likelihood-ratio test.
    g = m07.cell(base, "Prostate", "V100_pct", "dice")
    d_v100 = U.DELTA[("Prostate", "V100_pct")]["delta_conf"]
    fit = m07.joint_fit(g, "dice", d_v100, P_TARGET, interaction=False)
    V_test = np.percentile(g["prostate_vol"], [10, 50, 90])
    _, thr_q = m07.thresholds_at(fit, V_test)
    for Vk, got in zip(V_test, thr_q):
        c.near(f"V100 seuil de tolerance Q95 a V={Vk:.1f} cc",
               float(got), float(truth_thr_v100(Vk, d_v100)), TOL_THR_QUANT)

    # ------------------------------------------------------------
    print("\n[6] Controle non parametrique par tertile")
    # ------------------------------------------------------------
    strata = m07.volume_strata(base, "dice", P_TARGET)
    s90 = strata[(strata["index"] == "D90_pct") &
                 (strata["margin"] == "conf")].sort_values("vol_median")
    c.check("3 ordered volume tertiles",
            len(s90) == 3 and s90["vol_median"].is_monotonic_increasing)
    # The D90 truth makes the threshold DECREASE as the volume increases: the
    # tertiles
    # must reproduce that direction WITHOUT any functional form in volume being
    # imposed on them. The logistic p* threshold is read, not the Youden
    # cutpoint: Youden targets
    # max(sens+spec), a different target, and its variance at this sample size
    # exceeds the shift to be measured, so it would detect the direction only by
    # chance half the time.
    got_dir = float(s90["thr_logistic"].iloc[-1] - s90["thr_logistic"].iloc[0])
    want_dir = float(truth_thr_d90(s90["vol_median"].iloc[-1])
                     - truth_thr_d90(s90["vol_median"].iloc[0]))
    c.check("direction of the threshold shift reproduced without a volume model",
            np.sign(got_dir) == np.sign(want_dir),
            f"observed={got_dir:+.3f} truth={want_dir:+.3f}")
    # V150 never reaches the target equivalence rate WITHIN the observed Dice
    # range, since its rate plateaus below it. The logistic p* threshold there is
    # therefore a pure extrapolation of the sigmoid, with enormous variance, and
    # comparing it across tertiles would only test that noise. Instead the checks
    # are (a) that the extrapolation flag reports it, and (b) the flatness of the
    # dual estimator, which does stay inside the range.
    s150 = strata[(strata["index"] == "V150_pct") & (strata["margin"] == "conf")]
    flag_ok = True
    for _, r in strata.iterrows():
        thr, want = r["thr_logistic"], "ok"
        if np.isfinite(thr):
            if thr > r["metric_max"]:
                want = "extrapolated_high"
            elif thr < r["metric_min"]:
                want = "extrapolated_low"
        else:
            want = ""
        flag_ok &= (r["thr_logistic_extrap"] == want)
    c.check("extrapolation flag consistent with the observed range, in every "
            "tertiles", bool(flag_ok))
    n_extrap = int((s150["thr_logistic_extrap"] != "ok").sum())
    c.check("V150: p* threshold outside the observed range correctly flagged (not quotable)",
            n_extrap > 0, f"{n_extrap}/3 tertiles flagges")
    amp150 = float(s150["thr_tolerance"].max() - s150["thr_tolerance"].min())
    c.check("V150: per-tertile tolerance bound nearly identical (flat truth)",
            amp150 < 0.06, f"amplitude={amp150:.4f}")

    # ------------------------------------------------------------
    print("\n[7] Support mask of the figures")
    # ------------------------------------------------------------
    x = np.array([0.80, 0.85, 0.90, 0.95])
    Vv = np.array([20.0, 30.0, 40.0, 50.0])
    XX, VV = np.meshgrid(np.linspace(0.80, 0.95, 12), np.linspace(20, 50, 12))
    mask = m07._support_mask(XX, VV, x, Vv)
    c.check("the empty corner (high Dice / small volume) is masked",
            bool(mask[0, -1]))
    c.check("an observed point is not masked", not bool(mask[0, 0]))

    # ------------------------------------------------------------
    print("\n[8] Figures - error-free generation")
    # ------------------------------------------------------------
    try:
        m07.figure_volume_vs_quality(base, desc)
        m07.figure_joint_surface(base, "dice", "conf", P_TARGET, models)
        m07.figure_threshold_vs_volume(base, "dice", "conf", P_TARGET, models,
                                       boot_cache)
        figs_ok = all((m07.OUT_DIR / f).exists() for f in
                      ["volume_vs_quality.png", "joint_pequiv_dice_conf.png",
                       "threshold_vs_volume_dice_conf.png"])
    except Exception as exc:                     # noqa: BLE001 - on veut le message
        figs_ok = False
        print(f"    exception : {type(exc).__name__}: {exc}")
    c.check("the 3 figures are produced", figs_ok)

    print(f"\n[resultat] {c.fails} echec(s).")
    print(f"[note] synthetic outputs kept in {tmp}")
    sys.exit(1 if c.fails else 0)


if __name__ == "__main__":
    main()
