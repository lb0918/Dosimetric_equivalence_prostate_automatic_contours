"""
05_headroom.py
==============
HEADROOM analysis of the clinical model M0 against the MEASUREMENT FLOOR of the
IPSS change. It addresses the objection that a null DVH result is trivial
because nothing predicts the IPSS change well: it shows WHERE the null is
informative (M0 leaves room that the DVH block could have filled) and where it
is limited by instrument noise (M0 already saturates the reliability of the
IPSS change).

Principle
---------
The IPSS change is IPSS(endpoint) - IPSS(baseline), a difference of two noisy
measurements, so its measurement error is SEM*sqrt(2). No predictor can go below
that floor, where the residual is questionnaire noise.
Per endpoint (primary learner, real targets):
  RMSE_M0, R2_M0; SD_y = RMSE/sqrt(1-R2); floor = SEM*sqrt(2);
  headroom = RMSE_M0 - floor; R2 ceiling = 1 - floor^2/SD_y^2.
  The class is "headroom" if headroom > HEADROOM_MARGIN, otherwise "floor".

Routes for the SEM of a single IPSS (SEM = SD_baseline*sqrt(1-reliability)):
  Route 1 - cohort      closely spaced pre-treatment IPSS pairs from the cohort
                        itself. This assumes the true symptom level is stable
                        between the two measurements; with symptomatic drift
                        (for instance under alpha-blockers) or a small n, the
                        SEM is OVERESTIMATED, since it measures real change
                        rather than noise.
  Route 2 - alpha       SD_baseline*sqrt(1-0.82). Internal consistency,
                        conservative.
  Route 3 - test-retest SD_baseline*sqrt(1-0.90). Test-retest of the total
                        index, same instrument and same population as the
                        cohort. Preferred anchor.
  Route 4 - test-retest SD_baseline*sqrt(1-0.92). Original English IPSS, used as
                        a sensitivity analysis.
  Route 5 - mixed model within-subject residual around a smoothed trajectory
                        (see estimate_mixedmodel_sem).

Plausibility guard: a route whose SEM implies an R2 ceiling below the observed
R2 of M0 (frac_ceiling_used > 1, i.e. r2_m0 > r2_ceiling) is REFUTED BY THE DATA
and excluded from the verdict (scenario_compatible = False).

References: Gregoire et al. Prog Urol 1996, 6:240-9 (French-Canadian IPSS;
alpha = 0.82; total-index test-retest rho = 0.90; per-item kappa 0.41-0.66;
interval about 10.5 days); Barry et al. J Urol 1992, 148:1549-57 (r about 0.92);
SEM and minimal detectable change from classical test theory (COSMIN; Beninato
& Portney 2011).

Outputs (HEADROOM_OUT_DIR): headroom.csv, headroom_summary.md,
                            headroom_plot.png and headroom_plot.pdf
Inputs are read-only; nothing is written outside HEADROOM_OUT_DIR.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROJECT_DIR, RESULTS_ROOT, SEED, DATASET_MINIMAL, with_suffix
from utils import ensure_dir
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

# ============================================================
# CONFIGURATION (set at the top of the file rather than on the CLI)
# ============================================================
M0_TAG = "noDVH"
# results_<tag> suffixed by the active run (PIPE_TAG_SUFFIX).
M0_METRICS = RESULTS_ROOT / f"results_{with_suffix(M0_TAG)}" / "metrics.csv"
PRIMARY_LEARNER = "elasticnet"

RMSE_COL, R2_COL, N_COL = "rmse", "r2", "n"

# --- Schema of the longitudinal dataset (one row per IPSS measurement) ------
LONG_DATASET = Path(DATASET_MINIMAL)
ID_COL = "record_id"
IPSS_TOTAL_COL = "ipss_score_calc"
DAYS_COL = "ipss_days"          # days between the IPSS measurement and treatment
IPSS_DATE_COL = "ipss_date"     # fallback when DAYS_COL is absent
TX_DATE_COL = "tx_date"         # fallback when DAYS_COL is absent

# --- SEM estimation routes --------------------------------------------------
USE_DATA_SEM = True
ALPHA_GREGOIRE = 0.82           # Route 2 - internal consistency
R_GREGOIRE = 0.90               # Route 3 - test-retest of the total index
R_BARRY = 0.92                  # Route 4 - test-retest of the total index
SEM_PRETX_MAX_GAP_DAYS = 90     # Route 1 - maximum gap within a pre-treatment pair
SEM_MIN_PAIRS = 20              # Route 1 - minimum number of pairs to estimate
# Route 5 - mixed model: within-subject residual around a smoothed trajectory.
# SEM = sqrt(residual variance). It uses EVERY measurement and is not
# contaminated by symptom drift (absorbed by the spline and the random
# intercept) the way Route 1 is. It is an UPPER bound on the measurement SEM,
# since the residual also contains spline misfit and fast within-subject
# fluctuation.
USE_MIXED_SEM = True
MIXED_SPLINE_DF = 4             # bs(ipss_days, df=...), consistent with 01_prepare_target
MIXED_MIN_PATIENTS = 30
MIXED_DAYS_RANGE = None         # (lo, hi) to restrict the range; None = everything

SEM_LIT_SCENARIOS = {"lit_optimistic": 2.0, "lit_central": 3.0, "lit_conservative": 3.6}
INCLUDE_LIT_BAND = True

CHANGE_FACTOR = np.sqrt(2.0)
HEADROOM_MARGIN = 0.5

HEADROOM_OUT_DIR = PROJECT_DIR / with_suffix("headroom")
OVERWRITE = True


# ============================================================
# READING THE M0 METRICS
# ============================================================
def load_m0_metrics(path: Path = M0_METRICS) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"M0 metrics not found: {path}")
    df = pd.read_csv(path)
    df = df[df["algo"] == PRIMARY_LEARNER].copy()
    if df.empty:
        raise ValueError(f"No '{PRIMARY_LEARNER}' row in {path}")
    df = df.rename(columns={RMSE_COL: "rmse", R2_COL: "r2", N_COL: "n"})
    df["rmse"] = df["rmse"].astype(float)
    df["r2"] = df["r2"].astype(float)
    df["sd_y"] = df["rmse"] / np.sqrt(np.clip(1.0 - df["r2"], 1e-6, None))
    return df[["endpoint", "n", "rmse", "r2", "sd_y"]].reset_index(drop=True)


# ============================================================
# LONGITUDINAL DATASET: days, baseline SD, cohort SEM
# ============================================================
def _load_long(long_path: Path = LONG_DATASET):
    if not long_path.exists():
        return None
    df = pd.read_csv(long_path)
    if not {ID_COL, IPSS_TOTAL_COL}.issubset(df.columns):
        return None
    if DAYS_COL not in df.columns:
        if {IPSS_DATE_COL, TX_DATE_COL}.issubset(df.columns):
            d = (pd.to_datetime(df[IPSS_DATE_COL], errors="coerce")
                 - pd.to_datetime(df[TX_DATE_COL], errors="coerce")).dt.days
            df = df.assign(**{DAYS_COL: d})
        else:
            return None
    return df.dropna(subset=[IPSS_TOTAL_COL, DAYS_COL]).copy()


def baseline_ipss_sd(long_df: pd.DataFrame):
    """SD of the baseline IPSS (the pre-treatment measurement closest to
    treatment), over the cohort."""
    pre = long_df[long_df[DAYS_COL] < 0]
    if pre.empty:
        return None, 0
    idx = pre.groupby(ID_COL)[DAYS_COL].idxmax()
    base = pre.loc[idx, IPSS_TOTAL_COL].astype(float)
    if base.shape[0] < 3:
        return None, int(base.shape[0])
    return float(base.std(ddof=1)), int(base.shape[0])


def estimate_cohort_sem(long_df: pd.DataFrame):
    """Route 1 - SEM from closely spaced pre-treatment pairs.
    SEM = SD(differences)/sqrt(2)."""
    pre = long_df[long_df[DAYS_COL] < 0].sort_values([ID_COL, DAYS_COL])
    diffs = []
    for _, g in pre.groupby(ID_COL):
        if len(g) < 2:
            continue
        vals = g[IPSS_TOTAL_COL].to_numpy(dtype=float)
        days = g[DAYS_COL].to_numpy(dtype=float)
        d = np.diff(vals); gaps = np.abs(np.diff(days))
        diffs.extend(d[gaps <= SEM_PRETX_MAX_GAP_DAYS].tolist())
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < SEM_MIN_PAIRS:
        return None, int(diffs.size)
    return float(np.std(diffs, ddof=1) / CHANGE_FACTOR), int(diffs.size)


def estimate_mixedmodel_sem(long_df: pd.DataFrame):
    """Route 5 - SEM as the residual SD of a mixed model with a smoothed
    trajectory.

    ipss ~ bs(ipss_days, df=MIXED_SPLINE_DF) + (1 | record_id).
    The spline (fixed effects) absorbs the population mean trajectory, the
    random intercept absorbs each patient's own level, and the residual is
    approximately measurement noise plus spline misfit plus fast fluctuation.
    SEM = sqrt(scale).
    ICC = between-subject variance / (between + residual), the implied
    reliability, to be compared with the literature values above.
    Returns a dict {sem, icc, n_obs, n_pat}, or None when statsmodels is
    unavailable, the fit fails, or there are too few patients.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = long_df.dropna(subset=[IPSS_TOTAL_COL, DAYS_COL, ID_COL]).copy()
    if MIXED_DAYS_RANGE is not None:
        lo, hi = MIXED_DAYS_RANGE
        d = d[(d[DAYS_COL] >= lo) & (d[DAYS_COL] <= hi)]
    if d[ID_COL].nunique() < MIXED_MIN_PATIENTS:
        return None
    try:
        md = smf.mixedlm(f"{IPSS_TOTAL_COL} ~ bs({DAYS_COL}, df={MIXED_SPLINE_DF})",
                         d, groups=d[ID_COL])
        mf = md.fit(reml=True, method="lbfgs")
        resid_var = float(mf.scale)
        between = float(mf.cov_re.iloc[0, 0]) if mf.cov_re.size else 0.0
        sem = float(np.sqrt(resid_var))
        icc = between / (between + resid_var) if (between + resid_var) > 0 else np.nan
        return {"sem": sem, "icc": icc,
                "n_obs": int(len(d)), "n_pat": int(d[ID_COL].nunique())}
    except Exception:
        return None


def build_sem_scenarios(long_df):
    scenarios, info = {}, {"sd_baseline": None, "n_baseline": 0,
                           "sem_route1": None, "n_pairs": 0, "rel_route1": None,
                           "sem_route5": None, "icc_route5": None, "n_obs_route5": 0}
    if long_df is not None and USE_DATA_SEM:
        sd_base, n_base = baseline_ipss_sd(long_df)
        info["sd_baseline"], info["n_baseline"] = sd_base, n_base
        sem1, npairs = estimate_cohort_sem(long_df)
        info["sem_route1"], info["n_pairs"] = sem1, npairs
        if sem1 is not None:
            scenarios["route1_cohort"] = (sem1, "route")
            if sd_base and sd_base > 0:
                info["rel_route1"] = 1.0 - (sem1 / sd_base) ** 2
        if sd_base is not None:
            scenarios["route2_alpha"] = (sd_base * np.sqrt(1 - ALPHA_GREGOIRE), "route")
            scenarios["route3_gregoire"] = (sd_base * np.sqrt(1 - R_GREGOIRE), "route")
            scenarios["route4_barry"] = (sd_base * np.sqrt(1 - R_BARRY), "route")
        if USE_MIXED_SEM:
            mm = estimate_mixedmodel_sem(long_df)
            if mm is not None:
                scenarios["route5_mixed"] = (mm["sem"], "route")
                info["sem_route5"] = mm["sem"]
                info["icc_route5"] = mm["icc"]
                info["n_obs_route5"] = mm["n_obs"]
    if INCLUDE_LIT_BAND or not scenarios:
        for name, sem in SEM_LIT_SCENARIOS.items():
            scenarios[name] = (sem, "literature")
    return scenarios, info


# ============================================================
# HEADROOM COMPUTATION
# ============================================================
def compute_headroom(m0, sem_single, scenario, family):
    sem_delta = sem_single * CHANGE_FACTOR
    rows = []
    for _, r in m0.iterrows():
        sd_y = r["sd_y"]
        headroom_rmse = r["rmse"] - sem_delta
        r2_ceiling = float(np.clip(1.0 - (sem_delta ** 2) / (sd_y ** 2), 0.0, 1.0))
        frac = (r["r2"] / r2_ceiling) if r2_ceiling > 0 else np.inf
        rows.append({
            "endpoint": r["endpoint"], "n": int(r["n"]),
            "scenario_sem": scenario, "family": family,
            "sem_single": round(sem_single, 3), "sem_delta": round(sem_delta, 3),
            "rmse_m0": round(r["rmse"], 3), "r2_m0": round(r["r2"], 3),
            "sd_y": round(sd_y, 3), "floor_rmse": round(sem_delta, 3),
            "headroom_rmse": round(headroom_rmse, 3),
            "r2_ceiling": round(r2_ceiling, 3),
            "r2_headroom": round(r2_ceiling - r["r2"], 3),
            "frac_ceiling_used": round(frac, 3) if np.isfinite(frac) else np.nan,
            # Is this SEM compatible with the observed R2?
            "row_compatible": bool(r["r2"] <= r2_ceiling + 1e-9),
            "class": "headroom" if headroom_rmse > HEADROOM_MARGIN else "floor",
        })
    return pd.DataFrame(rows)


def build_table(m0, scenarios):
    parts = [compute_headroom(m0, sem, name, fam)
             for name, (sem, fam) in scenarios.items()]
    out = pd.concat(parts, ignore_index=True)
    # A route is data-compatible if and only if NO endpoint refutes it.
    compat = out.groupby("scenario_sem")["row_compatible"].transform("all")
    out["scenario_compatible"] = compat
    out["_ord"] = out["endpoint"].str.extract(r"(\d+)").astype(float)
    out["_fam"] = (out["family"] != "route").astype(int)
    out = out.sort_values(["_ord", "_fam", "sem_single"]).drop(columns=["_ord", "_fam"])
    return out.reset_index(drop=True)


# ============================================================
# OUTPUTS
# ============================================================
def _verdict_per_endpoint(tab):
    """Verdict over the data-compatible ROUTES, falling back to any compatible
    scenario when no route qualifies."""
    base = tab[(tab["family"] == "route") & (tab["scenario_compatible"])]
    if base.empty:
        base = tab[tab["scenario_compatible"]]
    if base.empty:
        base = tab
    rows = []
    for ep, g in base.groupby("endpoint", sort=False):
        classes = set(g["class"])
        if classes == {"headroom"}:
            verdict = "HEADROOM (robust) - informative null"
        elif classes == {"floor"}:
            verdict = "FLOOR (robust) - null limited by the instrument"
        else:
            verdict = "AMBIGUOUS - depends on the SEM route"
        rows.append({"endpoint": ep, "rmse_m0": g["rmse_m0"].iloc[0],
                     "headroom_min": round(g["headroom_rmse"].min(), 3),
                     "headroom_max": round(g["headroom_rmse"].max(), 3),
                     "verdict": verdict})
    out = pd.DataFrame(rows)
    out["_ord"] = out["endpoint"].str.extract(r"(\d+)").astype(float)
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def write_outputs(tab, info, out_dir):
    ensure_dir(out_dir)
    csv_path = out_dir / "headroom.csv"
    if csv_path.exists() and not OVERWRITE:
        raise FileExistsError(f"{csv_path} already exists (OVERWRITE=False).")
    tab.to_csv(csv_path, index=False)

    excluded = sorted(set(tab.loc[(tab["family"] == "route")
                                  & (~tab["scenario_compatible"]), "scenario_sem"]))
    verdict = _verdict_per_endpoint(tab)
    lines = ["# Headroom - M0 versus the IPSS-change measurement floor", "",
             "## SEM estimation routes", ""]
    if info["sd_baseline"] is not None:
        lines.append(f"- Baseline IPSS SD (cohort, {info['n_baseline']} patients): "
                     f"**{info['sd_baseline']:.2f} pts**")
    if info["sem_route1"] is not None:
        rel = info["rel_route1"]
        lines.append(f"- Route 1 (cohort, {info['n_pairs']} pre-treatment pairs): SEM = "
                     f"**{info['sem_route1']:.2f}**"
                     + (f" => implied reliability {rel:.2f}" if rel is not None else ""))
    if info["sd_baseline"] is not None:
        sd = info["sd_baseline"]
        lines += [f"- Route 2 (internal consistency {ALPHA_GREGOIRE}): SEM = "
                  f"**{sd*np.sqrt(1-ALPHA_GREGOIRE):.2f}** (conservative)",
                  f"- Route 3 (test-retest {R_GREGOIRE}): SEM = "
                  f"**{sd*np.sqrt(1-R_GREGOIRE):.2f}** (preferred)",
                  f"- Route 4 (test-retest {R_BARRY}): SEM = "
                  f"**{sd*np.sqrt(1-R_BARRY):.2f}** (sensitivity)"]
    if info["sem_route5"] is not None:
        lines.append(f"- Route 5 (mixed model, {info['n_obs_route5']} measurements): SEM = "
                     f"**{info['sem_route5']:.2f}** => ICC {info['icc_route5']:.2f} "
                     f"(UPPER bound; the residual mixes noise, misfit and fluctuation)")
    if excluded:
        lines.append(f"\nRoutes **refuted by the data** (frac_ceiling_used > 1, "
                     f"excluded from the verdict): {', '.join(excluded)}.")
    if info["sd_baseline"] is None:
        lines.append("- Cohort routes not estimated - literature band only.")
    lines += ["", "## Verdict per endpoint (data-compatible routes)", "",
              verdict.to_markdown(index=False), "",
              "## Detail per SEM scenario", "",
              tab.drop(columns="row_compatible").to_markdown(index=False), "",
              "## How to read this", "",
              "- **headroom** together with a tightly bounded DVH null means the "
              "null is INFORMATIVE: the DVH block had room and did not fill it.",
              "- **floor**: M0 already saturates the reliability of the IPSS "
              "change, so a null is expected.",
              "- `scenario_compatible` = False: the SEM is refuted by the data "
              "(M0 explains more than the reliability ceiling that SEM would "
              "imply), so the route is excluded from the verdict.",
              "- Route 1 is often refuted when the two pre-treatment IPSS "
              "measurements bracket a real symptom change, since it then "
              "measures signal rather than noise."]
    (out_dir / "headroom_summary.md").write_text("\n".join(lines), encoding="utf-8")
    _plot(tab, out_dir / "headroom_plot.png")


def _plot(tab, path):
    band = tab[(tab["family"] == "route") & (tab["scenario_compatible"])]
    if band.empty:
        band = tab[tab["scenario_compatible"]] if tab["scenario_compatible"].any() else tab
    eps = list(dict.fromkeys(tab["endpoint"]))
    endpoints_clean = [e.replace("y_", "").replace("d", " ") for e in eps]
    x = np.arange(len(eps))
    rmse = [tab.loc[tab["endpoint"] == e, "rmse_m0"].iloc[0] for e in eps]
    lo = [band.loc[band["endpoint"] == e, "floor_rmse"].min() for e in eps]
    hi = [band.loc[band["endpoint"] == e, "floor_rmse"].max() for e in eps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(x, lo, hi, alpha=0.25, color="tab:red",
                    label="ΔIPSS floor")
    ax.plot(x, lo, color="tab:red", lw=0.8, ls="--")
    ax.plot(x, hi, color="tab:red", lw=0.8, ls="--")
    ax.plot(x, rmse, "o-", color="tab:blue", label="$M_0$ RMSE (clinical)")
    for xi, ri, fhi in zip(x, rmse, hi):
        ax.annotate("headroom" if ri - fhi > HEADROOM_MARGIN else "floor",
                    (xi, ri), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8,
                    color="tab:green" if ri - fhi > HEADROOM_MARGIN else "gray")
    ax.set_xticks(x); ax.set_xticklabels(endpoints_clean)
    ax.set_ylabel("RMSE (IPSS points)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ============================================================
# SYNTHETIC SELF-TEST
# ============================================================
def synthetic_check():
    rng = np.random.default_rng(SEED)
    true_sem, sd_latent, n_pat = 2.5, 6.0, 300
    recs = []
    for pid in range(n_pat):
        latent = rng.normal(12, sd_latent)
        # Two pre-treatment measurements (Route 1) plus a post-treatment
        # trajectory (Route 5).
        for day in [-60, -30, 30, 90, 180, 365, 730]:
            traj = 4.0 * np.exp(-((day - 45) / 120) ** 2) if day > 0 else 0.0
            recs.append({ID_COL: pid,
                         IPSS_TOTAL_COL: latent + traj + rng.normal(0, true_sem),
                         DAYS_COL: day})
    long_df = pd.DataFrame(recs)
    sd_base, _ = baseline_ipss_sd(long_df)
    sem1, npairs = estimate_cohort_sem(long_df)
    assert sem1 is not None and abs(sem1 - true_sem) < 0.5, sem1
    mm = estimate_mixedmodel_sem(long_df)
    if mm is not None:  # statsmodels available
        assert abs(mm["sem"] - true_sem) < 0.8, mm  # upper bound -> wide tolerance
    m0 = pd.DataFrame({"endpoint": ["y_30d", "y_2600d"], "n": [300, 120],
                       "rmse": [6.7, 4.6], "r2": [0.16, 0.39]})
    m0["sd_y"] = m0["rmse"] / np.sqrt(1 - m0["r2"])
    scen, _ = build_sem_scenarios(long_df)
    assert {"route2_alpha", "route3_gregoire", "route4_barry"} <= set(scen)
    tab = build_table(m0, scen)
    # A grossly oversized SEM route must be flagged as incompatible.
    big = build_table(m0, {"huge": (8.0, "route")})
    assert not big["scenario_compatible"].all(), "compatibility gate inactive"
    print(f"[self-test] OK - Route 1 SEM={sem1:.2f} (true {true_sem}), "
          f"SD_base={sd_base:.2f}, {npairs} pairs; compatibility gate active.")


# ============================================================
# MAIN
# ============================================================
def main():
    synthetic_check()
    m0 = load_m0_metrics()
    long_df = _load_long()
    scenarios, info = build_sem_scenarios(long_df)
    if info["sem_route1"] is not None:
        print(f"[ok] Route 1: SEM={info['sem_route1']:.2f} ({info['n_pairs']} pairs); "
              f"SD_baseline={info['sd_baseline']:.2f}")
    elif info["sd_baseline"] is not None:
        print(f"- SD_baseline={info['sd_baseline']:.2f} (routes 2/3/4 available)")
    else:
        print("- Longitudinal dataset absent - literature band only.")
    tab = build_table(m0, scenarios)
    ensure_dir(HEADROOM_OUT_DIR)
    write_outputs(tab, info, HEADROOM_OUT_DIR)
    print(f"\n[done] Written to {HEADROOM_OUT_DIR}/")


if __name__ == "__main__":
    main()
