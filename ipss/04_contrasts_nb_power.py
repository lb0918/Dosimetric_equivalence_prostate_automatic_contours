"""
04_contrasts_nb_power.py
========================
SYNTHETIC power analysis of the Nadeau-Bengio (NB) test as used in
04_contrasts.py. It answers: would a REAL signal pass the NB test given the
geometry of the study (M differences, ratio n_test/n_train = 1/(K-1))? In other
words, when NB fails to reject, is that an absence of effect or a lack of power?

NO multiplicity correction is applied here: the NB test is evaluated ALONE,
endpoint by endpoint, at alpha = 0.05.

The KEY fact exploited here is that the NB statistic
        t = mean_d / sqrt((1/M + ratio) * S^2_d),   df = M-1
is SCALE-INVARIANT. Its distribution, and therefore its power, depends ONLY on
the STANDARDISED per-fold effect
        delta = E[d_k] / SD(d_k)
(the mean per-fold delta RMSE over its standard deviation) and on (M, ratio). A
SINGLE universal power curve therefore answers the question, and each endpoint
places its observed delta on that curve: the RMSE unit disappears and sigma
cancels. The distribution is simulated under d_k ~ N(delta, 1).

M = N_FOLDS * N_REPEATS: under repeated CV the test has r*K differences instead
of K. The ratio stays 1/(K-1), since repeating does not change the fold sizes.
Raising the number of repetitions therefore lowers the minimum detectable effect
without loosening the correction, which is the whole point of the "corrected
resampled t-test".

Decision regime: alpha = 0.05 per endpoint (NB test in isolation, uncorrected).

Outputs (in the contrasts directory):
  - nb_power_<contrast>.md   minimum detectable effect at the target power, the
                             rejection boundary delta*, and the observed delta
                             with its power.
  - nb_power_<contrast>.png  universal power curve with the observed deltas.

The OOF files are read-only; the matching and per-fold differences come from
04_contrasts.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from config import N_FOLDS, N_REPEATS, SEED
from utils import ensure_dir

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'


# Reuse the matching and per-fold differences of 04_contrasts, so the folds and
# the metric are exactly those of the reported inference. Loaded by path because
# the module name starts with a digit.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "contrasts04", str(Path(__file__).with_name("04_contrasts.py")))
_c04 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_c04)

# ============================================================
# CONFIGURATION
# ============================================================
# Ablation contrast to diagnose (baseline = run deprived of the feature(s),
# treatment = full run). It must match a pair of trained tags.
# Overridable on the command line:
#   python3 04_contrasts_nb_power.py BASE_TAG TREAT_TAG "label"
import sys
BASE_TAG = "curated_mc_bayes_clin0977_noObstr"
TREAT_TAG = "curated_mc_bayes_clin0977"
CONTRAST_LABEL = "Pre-tx obstructive IPSS (ablation)"
if len(sys.argv) >= 4:
    BASE_TAG, TREAT_TAG, CONTRAST_LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
# Label rendered in the figure. CONTRAST_LABEL stays the key (file slug, match
# against contrasts.csv); only the displayed version may be overridden.
CONTRAST_DISPLAY = _c04.CONTRAST_DISPLAY.get(CONTRAST_LABEL, CONTRAST_LABEL)
LEARNER = _c04.PRIMARY_LEARNER


def _slug(label: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in label]
    return "".join(keep).strip("_")

RATIO = _c04.NB_TEST_TRAIN_RATIO          # 1/(K-1) - does NOT depend on r
K = N_FOLDS                                # folds per partition
M = N_FOLDS * N_REPEATS                    # NUMBER OF DIFFERENCES = df + 1

# The asymmetry that makes repeated CV worthwhile:
#   - M = r*K grows with r, gaining degrees of freedom (df = M-1) and shrinking
#     the 1/M term;
#   - RATIO = 1/(K-1) is FIXED, since repeating does not change the fold sizes,
#     hence not the overlap of the training sets that this term corrects for.
# Consequence: the detection threshold tends to z*sqrt(RATIO) as r grows and
# never falls below it; going lower requires increasing K, not r.

ALPHA_RAW = 0.05
POWER_TARGET = 0.80
N_SIM = 40000
DELTA_GRID = np.linspace(0.0, 5.0, 101)   # standardised effect delta = mean_d / sd_d
OUT_DIR = _c04.CONTRAST_OUT_DIR


# ============================================================
# Vectorised NB test on SYNTHETIC per-fold differences
# ============================================================
def nb_pvalue_from_diffs(d):
    """Two-sided NB p for an array of shape (..., M) of per-fold differences,
    the last axis being the folds. Reproduces 04_contrasts.nadeau_bengio exactly
    (corrected, df = M-1).

    M is inferred from the array shape; RATIO is a constant of the fold geometry
    and does not depend on it (see the module header)."""
    d = np.asarray(d, dtype=float)
    Kk = d.shape[-1]
    mean_d = d.mean(axis=-1)
    s2 = d.var(axis=-1, ddof=1)
    se = np.sqrt((1.0 / Kk + RATIO) * s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = mean_d / se
    # s2 == 0: identical differences -> p=0 for a clean effect, p=1 for none,
    # matching the module.
    p = 2.0 * student_t.sf(np.abs(tstat), df=Kk - 1)
    zero = s2 == 0
    p = np.where(zero, np.where(mean_d != 0, 0.0, 1.0), p)
    return p


def nb_power_curve(deltas, n_sim=N_SIM, alpha=ALPHA_RAW, seed=SEED):
    """Per-endpoint NB power (test in isolation) at each delta: under
    d_k ~ N(delta, 1), P(p_NB < alpha). sigma = 1 without loss of generality,
    by scale invariance."""
    rng = np.random.default_rng(seed)
    power = np.empty(len(deltas))
    for i, delta in enumerate(deltas):
        d = rng.normal(delta, 1.0, size=(n_sim, M))
        power[i] = (nb_pvalue_from_diffs(d) < alpha).mean()
    return power


def min_detectable_delta(deltas, power, target=POWER_TARGET):
    """Smallest delta reaching the `target` power, by linear interpolation."""
    above = np.where(power >= target)[0]
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return float(deltas[0])
    x0, x1 = deltas[j - 1], deltas[j]
    y0, y1 = power[j - 1], power[j]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def power_at(delta, deltas, power):
    return float(np.interp(abs(delta), deltas, power))


# ============================================================
# Observed effects (real data) of the ablation contrast
# ============================================================
def observed_deltas():
    """For each shared endpoint: (endpoint, mean_fold_diff, sd_fold_diff,
    delta_std, nb_p_obs, n_paired). Uses the SAME folds and differences as
    04_contrasts."""
    eps = sorted(set(_c04.list_endpoints(BASE_TAG)) & set(_c04.list_endpoints(TREAT_TAG)),
                 key=lambda e: int(e.strip("y_").rstrip("d")) if e.startswith("y_") else 0)
    out = []
    for ep in eps:
        base = _c04.load_oof(BASE_TAG, ep)
        treat = _c04.load_oof(TREAT_TAG, ep)
        if base is None or treat is None:
            continue
        pf, err = _c04.paired_frame(base, treat, LEARNER)
        if pf is None:
            continue
        d = _c04._fold_metric_diff(pf)
        mean_d = float(d.mean())
        sd_d = float(d.std(ddof=1))
        delta = mean_d / sd_d if sd_d > 0 else np.inf
        _, _, nbp, _, _ = _c04.nadeau_bengio(pf)
        out.append({
            "endpoint": ep, "mean_fold_diff": mean_d, "sd_fold_diff": sd_d,
            "delta_std": delta, "nb_p": nbp, "n_paired": len(pf),
        })
    return out


# ============================================================
# Report
# ============================================================
def main():
    print("=" * 70)
    print("NADEAU-BENGIO POWER - would a real signal pass the test?")
    print(f"  K={K} folds x {N_REPEATS} repetition(s) = M={M} differences "
          f"(df={M-1}), ratio n_test/n_train={RATIO:.3g}, learner={LEARNER}")
    print("=" * 70)
    ensure_dir(OUT_DIR)

    # 1) Universal power curve (test in isolation), uncorrected regime.
    power_raw = nb_power_curve(DELTA_GRID, alpha=ALPHA_RAW)
    mde_raw = min_detectable_delta(DELTA_GRID, power_raw)

    # Analytic threshold delta*: at the boundary |t| = t_crit, and on average
    # S^2_d is close to sigma^2, giving delta* ~ t_crit * sqrt(1/K + ratio).
    # This is a landmark; the simulation additionally accounts for the
    # variability of S^2_d, and the two agree closely.
    tcrit = student_t.ppf(1 - ALPHA_RAW / 2, df=M - 1)
    delta_star = tcrit * np.sqrt(1.0 / M + RATIO)
    # Irreducible floor: even as r grows, the RATIO term remains.
    delta_floor = 1.959963985 * np.sqrt(RATIO)

    # 2) Observed effects.
    obs = observed_deltas()

    # ---- Console and Markdown ----
    L = []
    L.append(f"# Nadeau-Bengio power - {CONTRAST_LABEL}\n")
    L.append(f"_Contrast: baseline `{BASE_TAG}` vs treatment `{TREAT_TAG}` "
             f"(learner {LEARNER}). Would a real signal pass the NB test at this "
             f"CV geometry?_\n")
    L.append(
        f"NB test as used in 04_contrasts.py: `t = mean_d / "
        f"sqrt((1/M + ratio)*S^2_d)`, with **M={M}** differences "
        f"({K} folds x {N_REPEATS} repetition(s)), **ratio={RATIO:.3g}**, "
        f"df={M-1}. The statistic is SCALE-INVARIANT: its power depends only on "
        f"the standardised per-fold effect **delta = mean(per-fold delta RMSE) "
        f"/ SD(per-fold delta RMSE)**. Simulated under d_k ~ N(delta, 1), "
        f"{N_SIM:,} draws.\n")

    L.append("## Sensitivity of the test (single endpoint, alpha = 0.05)\n")
    L.append(f"- Detection threshold delta\\* (rejection boundary) is about "
             f"**{delta_star:.2f}** (analytic `t_crit*sqrt(1/M+ratio)`, "
             f"t_crit={tcrit:.3f}).")
    L.append(f"- Minimum detectable effect at {int(POWER_TARGET*100)} % power: "
             f"**delta ~ {mde_raw:.2f}**.")
    L.append(f"  In practice, for NB to flag an endpoint, the mean per-fold "
             f"delta RMSE must be about **{mde_raw:.1f}x** its between-fold "
             f"standard deviation.")
    if N_REPEATS > 1:
        L.append(f"- The {N_REPEATS} repetitions raise the degrees of freedom "
                 f"from {K-1} to {M-1}. The ratio does not move: repeating does "
                 f"not change the fold sizes, hence not the overlap of the "
                 f"training sets that this term corrects for. Hence an "
                 f"irreducible floor delta\\* -> **{delta_floor:.2f}** as r "
                 f"grows; going lower requires increasing K, not r.")
    else:
        L.append(f"- With a single partition ({M-1} df) this is a high bar. "
                 f"Repeating the CV (N_REPEATS > 1) would lower the threshold "
                 f"without changing the estimand or loosening the correction.")
    L.append("")
    L.append("_No multiplicity correction is applied: each endpoint is judged "
             "on its own by NB at alpha = 0.05._\n")
    L.append("## Observed effects (ablation contrast)\n")
    L.append("| endpoint | mean per-fold dRMSE | between-fold SD | observed delta | NB power (raw) | observed NB p |")
    L.append("|---|---|---|---|---|---|")
    for r in obs:
        pw = power_at(r["delta_std"], DELTA_GRID, power_raw)
        L.append(f"| {r['endpoint']} | {r['mean_fold_diff']:+.3f} | "
                 f"{r['sd_fold_diff']:.3f} | {r['delta_std']:.2f} | "
                 f"{pw:.0%} | {r['nb_p']:.3f} |")
    # ---- Verdict ----
    # Two statuses only. It is tempting to add a third, "underpowered", by
    # comparing the OBSERVED delta to the minimum detectable effect - that is a
    # logical trap: under the null the observed delta is small BY CONSTRUCTION,
    # so a true null would always land there and an "informative null" category
    # would never fire. The power argument runs the other way: the minimum
    # detectable effect is a property of the DESIGN (M, ratio), not of the
    # observation. A non-rejection at M differences bounds the true delta below
    # roughly that value, whatever the observed delta.
    deltas_obs = np.array([r["delta_std"] for r in obs])
    med_delta = float(np.median(deltas_obs))
    sig = [r["endpoint"] for r in obs if r["nb_p"] < ALPHA_RAW]
    nul = [r["endpoint"] for r in obs if r["nb_p"] >= ALPHA_RAW]

    L.append("")
    L.append("## Verdict\n")
    L.append(
        f"- Geometry: **M={M}** differences (df={M-1}), rejection threshold "
        f"delta\\* ~ {delta_star:.2f}, {int(POWER_TARGET*100)} %-power threshold "
        f"delta ~ {mde_raw:.2f}. Observed deltas: median **{med_delta:.2f}**, "
        f"range {deltas_obs.min():.2f} to {deltas_obs.max():.2f}.")
    L.append(
        f"- **{len(sig)}/{len(obs)}** endpoints significant (p_NB < "
        f"{ALPHA_RAW})" + (f": {', '.join(sig)}." if sig else "."))
    if nul:
        L.append(
            f"- **{len(nul)}/{len(obs)}** not rejected: {', '.join(nul)}. "
            f"Since the design has {int(POWER_TARGET*100)} % power at "
            f"delta = {mde_raw:.2f}, these non-rejections constitute a BOUND: a "
            f"standardised effect of at least {mde_raw:.2f} would have been "
            f"detected {int(POWER_TARGET*100)} times out of 100. The true delta "
            f"is therefore likely below that threshold.")
        L.append(
            f"  This is a power argument, NOT an equivalence test: it makes the "
            f"absence of an effect credible, it does not demonstrate it. A "
            f"formal proof would require a TOST against an equivalence margin "
            f"chosen a priori. The bound also concerns delta, which is "
            f"unitless; for the magnitude in RMSE points, only the bootstrap CI "
            f"in `contrasts.csv` answers.")
    if N_REPEATS == 1:
        L.append(
            f"  With a single partition the minimum detectable effect is high "
            f"enough that the bound excludes very little. Repeated CV is what "
            f"makes such non-rejections interpretable.")
    L.append("")
    L.append(
        f"**What a non-rejection does NOT license.** The patient-level bootstrap "
        f"CI of 04_contrasts is not a fallback: it resamples PATIENTS with the "
        f"learned models held fixed, so it estimates the sampling randomness of "
        f"the test set, not the LEARNING randomness that NB corrects for. These "
        f"are two distinct variance components, not two severity levels on the "
        f"same one. Empirically it behaves like an UNCORRECTED test, so invoking "
        f"its exclusion of 0 to rescue a contrast that NB does not retain amounts "
        f"to preferring the uncorrected test over the corrected one. The two are "
        f"read together: the bootstrap gives the MAGNITUDE of the effect, NB says "
        f"whether it survives the learning randomness.")

    slug = _slug(CONTRAST_LABEL)
    (OUT_DIR / f"nb_power_{slug}.md").write_text("\n".join(L))
    print("\n".join(L))

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(DELTA_GRID, power_raw, lw=2.2, color="steelblue",
            label=f"NB, M={M} diffs (α=0.05, no multiplicity correction)")
    ax.axhline(POWER_TARGET, color="grey", ls=":", lw=1)
    ax.text(DELTA_GRID[-1], POWER_TARGET + 0.01, "80%", ha="right", va="bottom",
            color="grey", fontsize=9)
    ax.axvline(delta_star, color="crimson", ls="--", lw=1, alpha=0.7)
    ax.text(delta_star, 0.02, " δ*  (α=0.05 boundary)", color="crimson",
            fontsize=8, rotation=90, va="bottom", ha="left")
    # Observed deltas, plotted on the isolated-test curve.
    for r in obs:
        dl = min(r["delta_std"], DELTA_GRID[-1])
        pw = power_at(r["delta_std"], DELTA_GRID, power_raw)
        ax.plot(dl, pw, "o", color="black", zorder=6, ms=5)
        ax.annotate(r["endpoint"].replace("y_", ""), (dl, pw),
                    textcoords="offset points", xytext=(4, -2), fontsize=7)
    ax.set_xlabel("Standardised per-fold effect  δ = mean(ΔRMSE/fold) / between-fold SD")
    ax.set_ylabel("Power = P(reject)")
    ax.set_title(f"Nadeau–Bengio power (M={M} diffs = {K}×{N_REPEATS}, "
                 f"df={M-1}, ratio={RATIO:.3g})\n{CONTRAST_DISPLAY}"
                 "  —  black dots = observed endpoints of this contrast",
                 fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, DELTA_GRID[-1])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"nb_power_{slug}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] Written to {OUT_DIR}/ (nb_power_{slug}.md, nb_power_{slug}.png)")


if __name__ == "__main__":
    main()
