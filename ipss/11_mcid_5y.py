"""
11_mcid_5y.py
=============
Binary MCID task over the FIVE-YEAR HORIZON, rather than at a single endpoint.

09_mcid_logreg.py restricts itself to one endpoint; 10_mcid_contrasts_heatmap.py
treats each endpoint as an independent column. Here the target is ONE label per
patient, defined over the whole window covered by EPS_5Y. The earliest endpoint
is excluded, since it describes the acute post-treatment phase rather than the
return to baseline.

Target - "DURABLE RETURN" (the ALL rule):

    y_bin = 1  if delta IPSS <= MCID at EVERY real measurement of the patient
               inside the window
            0  if the patient leaves the band at least once

    equivalently, y_bin = 1 if max_ep delta IPSS(ep) <= MCID over the measured
    endpoints.

Inclusion: any patient with at least MIN_MEAS real measurement in the window.
Patients with a single measurement therefore carry a label resting on one point,
which is accepted by design.

Aligned regressor score - the SAME aggregation as the target:
    score = -max_ep delta_hat(ep), over the SAME endpoints that define the
    patient's label.
This is the only coherent choice: the label depends on the window actually
observed, so the score must concern the same one. (The three DIRECTION modes of
09 compose correctly: "le" -> max delta, "abs_le" -> max |delta|, "ge" ->
min delta.)

Folds - METHODOLOGICAL POINT. The folds of 02_train.py are NOT consistent from
one endpoint to the next, since each endpoint received its own KFold. A
patient-level target requires a single partition, so one is created here:
StratifiedKFold(N_FOLDS, seed=SEED) on the five-year label. Consequences:
  - LogReg is strictly out-of-fold on that partition (no leakage).
  - The regressor scores remain free of a patient leaking onto itself (each
    delta_hat(ep) is already out-of-fold at its endpoint), but this partition is
    not the one under which those models were retrained. The Nadeau-Bengio test
    therefore measures variability between SUBGROUPS of patients rather than
    between refits: on the regressor rows it is indicative, and the
    patient-level bootstrap CI remains the reference inference. On the LogReg
    row it keeps its usual meaning.

Two arms, written in a single pass since the label is shared:

  A. Counterpart of 09 on the five-year target - AUC per scenario and model with
     a bootstrap CI, and the paired delta AUC (logreg minus regressor) over the
     same draws.
     -> figures/F9_mcid5y_auc.png

  B. Counterpart of 10 on the five-year target - delta AUC per feature-block
     CONTRAST and model, with a bootstrap CI and Nadeau-Bengio. A single target
     means a single heatmap (rows = models, columns = contrasts) instead of
     small multiples.
     -> figures/F10_mcid5y_contrasts_heatmap.png

The OOF files and features are read-only; only logistic models are trained.

Usage:
    python 11_mcid_5y.py
    PIPE_MCID_THRESH=3 python 11_mcid_5y.py
    MCID_SKIP_LOGREG=1 python 11_mcid_5y.py      # fast, without the LogReg row
    python 11_mcid_5y.py --replot     # REDRAWS the figures from mcid5y_auc.csv
                                      # and mcid5y_contrasts.csv: no logistic
                                      # refitted, no bootstrap.
"""
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold

from config import PROJECT_DIR, PREP_ROOT, SEED, N_FOLDS, with_suffix
from utils import load_cached_results, replot_requested

# --replot: read mcid5y_auc.csv / mcid5y_contrasts.csv back and run only the
# figure and table code (see utils.replot_requested).
REPLOT = replot_requested()


def _load(name: str, mod_name: str):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Script 10 provides fast_auc, load_oof_full, CONTRASTS and the plotting
# conventions; it loads 09 itself (binary target, aligned score, OOF logistic),
# exposed here as M09.
M10 = _load("10_mcid_contrasts_heatmap.py", "mcid10")
M09 = M10.M09

plt.rc("font", family="serif")

# ============================================================
# CONFIGURATION
# ============================================================
# Five-year window. The earliest endpoint is deliberately excluded (acute phase).
EPS_5Y = [400, 730, 1125, 1460, 1865]
MIN_MEAS = 1                    # minimum number of real measurements in the window
AGG_RULE = "all"                # "all" = durable return (see the module docstring)

MODEL_ORDER = M10.MODEL_ORDER   # ["logreg", "elasticnet", "rf", ...]
MODEL_LABELS = M10.MODEL_LABELS
PRIMARY = M10.PRIMARY
CONTRASTS = M10.CONTRASTS

# Scenarios of arm A, in display order. This must cover every tag referenced by
# M10.CONTRASTS (arm B), otherwise a contrast would come out without a matching
# row. The ablations are included as signal controls.
TAGS = ["noDVH", "curated_manual", "curated_auto_det_clin0977",
        "curated_mc_bayes_clin0977", "curated_mc_bayes_clin0977_var",
        "curated_mc_bayes_clin0977_noIpss", "curated_mc_bayes_clin0977_noObstr",
        "curated_mc_bayes_clin0977_noAge", "curated_mc_bayes_clin0977_noVol"]
TAG_LABELS = dict(M09.TAG_LABELS)
TAG_LABELS.setdefault("curated_mc_bayes_clin0977_noAge", "No age")
TAG_LABELS.setdefault("curated_mc_bayes_clin0977_noObstr", "No obstructive IPSS")
TAG_LABELS.setdefault("curated_mc_bayes_clin0977_noVol", "No volume / implant")

ALPHA = 0.05
N_BOOT = 2000
VMAX = 0.15                     # symmetric bound of the delta AUC heatmap
CMAP = M10.CMAP
GLYPH_COLOR = M10.GLYPH_COLOR
NB_TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)

OUT_DIR = PROJECT_DIR / with_suffix("mcid")
FIG_DIR = PROJECT_DIR / with_suffix("figures")


# ============================================================
# Five-year target and aggregated scores
# ============================================================
def _aggregate(mat: pd.DataFrame) -> pd.Series:
    """Aggregate a patients x endpoints matrix into the statistic that makes the
    ALL rule equivalent to a single comparison against the threshold.

    "le"     inside the band everywhere        <=> max delta <= threshold
    "abs_le" same with the absolute value     <=> max |delta| <= threshold
    "ge"     above the threshold everywhere    <=> min delta >= threshold
    """
    d = M09.DIRECTION
    if d == "le":
        return mat.max(axis=1)
    if d == "abs_le":
        return mat.abs().max(axis=1)
    if d == "ge":
        return mat.min(axis=1)
    raise ValueError(f"DIRECTION inconnue : {d!r}")


_TAG_CACHE: dict[str, tuple | None] = {}


def build_tag(tag: str):
    """(y_bin, scores, n_meas) for one scenario, over the five-year horizon.

    y_bin  Series indexed by record_id (the "durable return" label).
    scores dict model -> Series of aligned scores (logreg excluded, since it is
           added later: it depends on the fold partition, shared by all tags).
    n_meas number of real measurements retained per patient.

    For each patient, BOTH the label and the score use ONLY the endpoints where
    that patient has a real measurement - the same window on both sides.
    """
    if tag in _TAG_CACHE:
        return _TAG_CACHE[tag]

    # OOF files are indexed by (record_id, repeat). The TARGET and the
    # measurement window are INVARIANT across repetitions (y_true and `source`
    # do not depend on the partition), so they are built once, on the first
    # repetition. Only the PREDICTIONS vary, so `scores[algo]` becomes a
    # (patients x repetitions) DataFrame, aggregated repetition by repetition.
    oofs = {ep: M10.load_oof_full(tag, f"y_{ep}d") for ep in EPS_5Y}
    oofs = {ep: o for ep, o in oofs.items() if o is not None}
    if not oofs:
        _TAG_CACHE[tag] = None
        return None
    reps = sorted(set().union(*(set(o.index.get_level_values("repeat"))
                                for o in oofs.values())))

    def _real(o, r):
        g = o.xs(r, level="repeat")
        return g[g["source"] == "real"] if "source" in g.columns else g

    true_cols = {ep: _real(o, reps[0])["y_true"].astype(float)
                 for ep, o in oofs.items()}
    D = pd.DataFrame(true_cols)                     # NaN = no real measurement
    mask = D.notna()
    n_meas = mask.sum(axis=1)
    keep = n_meas >= MIN_MEAS
    D, mask, n_meas = D[keep], mask[keep], n_meas[keep]

    y_bin = pd.Series(M09.make_label(_aggregate(D).to_numpy(float)), index=D.index)

    scores = {}
    for algo in M09.REG_ALGOS:
        cols_by_rep = {}
        for r in reps:
            cols = {ep: _real(o, r)[algo].astype(float)
                    for ep, o in oofs.items() if algo in o.columns}
            if not cols:
                continue
            H = pd.DataFrame(cols).reindex(index=D.index, columns=D.columns)
            # Patient window: aggregate the prediction only where the target is observed.
            cols_by_rep[r] = M09.reg_score(_aggregate(H.where(mask)).to_numpy(float))
        if cols_by_rep:
            scores[algo] = pd.DataFrame(cols_by_rep, index=D.index)

    _TAG_CACHE[tag] = (y_bin, scores, n_meas)
    return _TAG_CACHE[tag]


def make_folds(y_bin: pd.Series) -> pd.Series:
    """SINGLE patient-level partition, stratified on the five-year label.

    This is required: the folds of 02_train differ from one endpoint to the
    next, so no inherited partition is valid at the patient level (see the
    module docstring).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=y_bin.index, dtype=int)
    for k, (_, te) in enumerate(skf.split(y_bin.index.values, y_bin.values)):
        fold.iloc[te] = k
    return fold


_LOGREG_CACHE: dict[str, pd.Series | None] = {}


def logreg_5y(tag: str) -> pd.Series | None:
    """OOF probability of the dedicated logistic on the five-year target, using
    the folds from make_folds()."""
    if tag in _LOGREG_CACHE:
        return _LOGREG_CACHE[tag]
    got = build_tag(tag)
    fpath = PREP_ROOT / f"prep_{with_suffix(tag)}" / "features.csv"
    if got is None or not fpath.exists():
        _LOGREG_CACHE[tag] = None
        return None
    y_bin, _, _ = got
    X = pd.read_csv(fpath)
    X["record_id"] = X["record_id"].astype(str).str.strip()
    X = X.set_index("record_id")
    ids = [i for i in y_bin.index if i in X.index]
    if len(ids) < 20 or y_bin.loc[ids].nunique() < 2:
        _LOGREG_CACHE[tag] = None
        return None
    y = y_bin.loc[ids]
    _LOGREG_CACHE[tag] = M09.logreg_oof(X.loc[ids], y, make_folds(y))
    return _LOGREG_CACHE[tag]


def model_score(tag: str, model: str) -> pd.Series | None:
    """Aligned score of a model on the five-year target (a probability for
    logreg, the aggregated negative prediction otherwise).
    ALWAYS returns a (patients x repetitions) DataFrame, so the caller need not
    distinguish the cases. The five-year logistic has a single column: its
    folds come from make_folds(), a partition of its own, so that
    partition does not depend on the repetitions of 02_train, so its AUC has no
    between-repetition dispersion, unlike that of the regressors.
    """
    if model == "logreg":
        p = logreg_5y(tag)
        return None if p is None else p.to_frame(0)
    got = build_tag(tag)
    if got is None:
        return None
    return got[1].get(model)


# ============================================================
# Inference: patient-level bootstrap plus Nadeau-Bengio
# ============================================================
def boot_indices(n: int, seed: int) -> np.ndarray:
    """Patient draws SHARED across the models and scenarios of one comparison."""
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def _mat(score):
    """(r, n) array from a patients x repetitions DataFrame (or a 1D vector)."""
    a = np.asarray(score, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a.T


def _mauc(label, S, i=None):
    """AUC computed WITHIN each repetition, then averaged (see 09.mean_auc)."""
    lab = label if i is None else label[i]
    v = [M10.fast_auc(lab, r if i is None else r[i]) for r in S]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def auc_ci(label, score, idx):
    S = _mat(score)
    point = _mauc(label, S)
    vals = [_mauc(label, S, i) for i in idx]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size == 0:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def dauc_ci(label, sa, sb, idx):
    """delta AUC = AUC(a) - AUC(b), with a paired bootstrap CI (same draws)."""
    A, B = _mat(sa), _mat(sb)
    point = _mauc(label, A) - _mauc(label, B)
    vals = []
    for i in idx:
        lab = label[i]
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        v = _mauc(label, A, i) - _mauc(label, B, i)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def nb_dauc(label, sa, sb, fold):
    """NB-corrected t on the per-fold delta AUC. Returns (t, df, p, mean_d, K)."""
    A, B = _mat(sa), _mat(sb)
    d = []
    # Differences taken per (repetition, fold): M = r*K, hence df = M-1. The
    # n_test/n_train ratio stays 1/(K-1), since repeating does not change the
    # fold sizes.
    for a_r, b_r in zip(A, B):
        for k in np.unique(fold):
            m = fold == k
            v = M10.fast_auc(label[m], a_r[m]) - M10.fast_auc(label[m], b_r[m])
            if np.isfinite(v):
                d.append(v)
    d = np.asarray(d, dtype=float)
    K = len(d)
    if K < 2:
        return np.nan, np.nan, np.nan, (float(d.mean()) if K else np.nan), K
    mean_d, s2 = float(d.mean()), float(d.var(ddof=1))
    if s2 == 0:
        return (np.inf if mean_d > 0 else -np.inf if mean_d < 0 else 0.0), K - 1, \
               (0.0 if mean_d != 0 else 1.0), mean_d, K
    from scipy.stats import t as student_t
    se = np.sqrt((1.0 / K + NB_TEST_TRAIN_RATIO) * s2)
    tstat = mean_d / se
    return float(tstat), K - 1, float(2.0 * student_t.sf(abs(tstat), df=K - 1)), \
        mean_d, K


# ============================================================
# ARM A - AUC per scenario and model, plus delta AUC (logreg minus regressor)
# ============================================================
def run_auc() -> pd.DataFrame:
    rows = []
    for tag in TAGS:
        got = build_tag(tag)
        if got is None:
            print(f"  ! {tag}: no usable OOF - skipped.")
            continue
        y_bin, _, n_meas = got
        sc = {m: model_score(tag, m) for m in MODEL_ORDER}
        sc = {m: s for m, s in sc.items() if s is not None}
        ids = y_bin.index
        for m in sc:
            ids = ids.intersection(sc[m].dropna().index)
        lab = y_bin.loc[ids].to_numpy(int)
        if len(np.unique(lab)) < 2:
            continue
        idx = boot_indices(len(ids), SEED)          # shared across models
        s_lg = sc["logreg"].loc[ids].to_numpy(float) if "logreg" in sc else None
        for m in MODEL_ORDER:
            if m not in sc:
                continue
            s = sc[m].loc[ids].to_numpy(float)
            auc, lo, hi = auc_ci(lab, s, idx)
            if m == "logreg" or s_lg is None:
                d, dlo, dhi = (0.0, np.nan, np.nan)
            else:
                d, dlo, dhi = dauc_ci(lab, s_lg, s, idx)
            rows.append({
                "tag": tag, "model": m, "n": len(ids), "n_pos": int(lab.sum()),
                "prevalence": round(float(lab.mean()), 4),
                "n_meas_median": float(n_meas.loc[ids].median()),
                "auc": auc, "auc_lo": lo, "auc_hi": hi,
                "dauc_logreg_minus_model": d, "dauc_lo": dlo, "dauc_hi": dhi,
            })
        best = max((r for r in rows if r["tag"] == tag and r["model"] != "logreg"),
                   key=lambda r: r["auc"], default=None)
        lg = next((r for r in rows if r["tag"] == tag and r["model"] == "logreg"),
                  None)
        msg = f"  {tag:34s} n={len(ids)} prev={lab.mean():.3f}"
        if lg:
            msg += f"  logreg={lg['auc']:.3f} [{lg['auc_lo']:.3f};{lg['auc_hi']:.3f}]"
        if best:
            msg += f"  meilleur reg {best['model']}={best['auc']:.3f}"
        print(msg)
    return pd.DataFrame(rows)


# ============================================================
# ARM B - feature-block contrasts on the five-year target
# ============================================================
def run_contrasts() -> pd.DataFrame:
    rows = []
    for ci, (base_tag, treat_tag, name) in enumerate(CONTRASTS):
        gb, gt = build_tag(base_tag), build_tag(treat_tag)
        if gb is None or gt is None:
            print(f"  ! {name}: missing scenario - skipped.")
            continue
        yb, yt = gb[0], gt[0]
        common = yb.index.intersection(yt.index)
        if len(common) == 0:
            continue
        if not (yb.loc[common].values == yt.loc[common].values).all():
            n_bad = int((yb.loc[common].values != yt.loc[common].values).sum())
            print(f"      ! five-year label differs for {n_bad} patients between "
                  f"runs - it should not depend on the feature block.")
        for model in MODEL_ORDER:
            sb, st = model_score(base_tag, model), model_score(treat_tag, model)
            if sb is None or st is None:
                continue
            ids = common.intersection(sb.dropna().index).intersection(
                st.dropna().index)
            lab = yb.loc[ids].to_numpy(int)
            if len(ids) < 20 or len(np.unique(lab)) < 2:
                continue
            a = st.loc[ids].to_numpy(float)
            b = sb.loc[ids].to_numpy(float)
            fold = make_folds(yb.loc[ids]).to_numpy()
            idx = boot_indices(len(ids), SEED + ci)   # shared across models
            d, lo, hi = dauc_ci(lab, a, b, idx)
            tstat, dfree, p, mean_d, K = nb_dauc(lab, a, b, fold)
            rows.append({
                "contrast": name, "baseline": base_tag, "treatment": treat_tag,
                "model": model, "n_paired": len(ids), "n_pos": int(lab.sum()),
                "prevalence": round(float(lab.mean()), 4),
                # Absolute AUCs averaged over the repetitions, like the delta
                # (_mat reshapes to (r, n); at r=1 this is the plain AUC).
                "auc_base": _mauc(lab, _mat(b)), "auc_treat": _mauc(lab, _mat(a)),
                "n_repeats": _mat(a).shape[0],
                "delta_auc": d, "boot_ci_lo": lo, "boot_ci_hi": hi,
                "nb_t": tstat, "nb_df": dfree, "nb_p": p,
                "nb_mean_fold_diff": mean_d, "nb_K": K,
            })
        print(f"  {name:<26} {sum(1 for r in rows if r['contrast'] == name)} models")
    r = pd.DataFrame(rows)
    if not r.empty:
        r["ci_excl0"] = (r["boot_ci_lo"] > 0) | (r["boot_ci_hi"] < 0)
        r["nb_sig"] = r["nb_p"] < ALPHA
    return r


# ============================================================
# Figures
# ============================================================
def auc_figsize(df: pd.DataFrame):
    tags = [t for t in TAGS if t in set(df["tag"])]
    return (max(8.0, 1.6 * len(tags) + 2), 8.2)


def draw_auc(ax, df: pd.DataFrame) -> None:
    """Draw F9 into a supplied axis (standalone and composite figures)."""
    tags = [t for t in TAGS if t in set(df["tag"])]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, len(models) - 1)) for i, m in enumerate(models)}
    width = 0.8 / len(models)
    x = np.arange(len(tags))
    for j, m in enumerate(models):
        s = df[df["model"] == m].set_index("tag").reindex(tags)
        xp = x + (j - (len(models) - 1) / 2) * width
        yerr = np.vstack([(s["auc"] - s["auc_lo"]).to_numpy(),
                          (s["auc_hi"] - s["auc"]).to_numpy()])
        ax.bar(xp, s["auc"].to_numpy(), width=width, color=colors[m],
               edgecolor="black" if m == "logreg" else "none",
               linewidth=1.4 if m == "logreg" else 0.0,
               label=MODEL_LABELS[m], zorder=2)
        ax.errorbar(xp, s["auc"].to_numpy(), yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.8, zorder=3)
    ax.axhline(0.5, color="grey", ls="--", lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TAG_LABELS.get(t, t) for t in tags], rotation=30,
                       ha="right", fontsize=15)
    ax.set_ylabel("AUC: MCID return in [1,5] years", fontsize=12)
    ax.set_ylim(0.40, min(1.0, float(df["auc_hi"].max()) + 0.05))
    ax.legend(ncol=min(len(models), 6), fontsize=12, framealpha=0.9,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def figure_auc(df: pd.DataFrame):
    """F9 - AUC bars per scenario and model with a 95% CI (layout from 09)."""
    fig, ax = plt.subplots(figsize=auc_figsize(df))
    draw_auc(ax, df)
    fig.tight_layout()
    return fig


def contrasts_figsize(r: pd.DataFrame):
    names = [n for _, _, n in CONTRASTS if n in set(r["contrast"])]
    models = [m for m in MODEL_ORDER if m in set(r["model"])]
    return (1.35 * len(names) + 4.2, 0.62 * len(models) + 2.6)


def draw_contrasts(host, ax, r: pd.DataFrame, cbar_pad: float = 0.24) -> None:
    """Draw F10 into a supplied axis; `host` carries the colour bar (a figure or
    a subfigure).

    cbar_pad : ecart colorbar/heatmap, en fraction de la largeur de l'axe. La
    valeur par defaut vise la figure autonome ; un bloc plus etroit (composite)
    en demande une plus faible, sinon la heatmap est ecrasee a gauche.
    """
    names = [n for _, _, n in CONTRASTS if n in set(r["contrast"])]
    models = [m for m in MODEL_ORDER if m in set(r["model"])]
    delta = (r.pivot(index="model", columns="contrast", values="delta_auc")
             .reindex(index=models, columns=names).values.astype(float))
    ci = (r.pivot(index="model", columns="contrast", values="ci_excl0")
          .reindex(index=models, columns=names).values)
    nb = (r.pivot(index="model", columns="contrast", values="nb_sig")
          .reindex(index=models, columns=names).values)

    im = ax.imshow(delta, cmap=CMAP, norm=Normalize(-VMAX, VMAX), aspect="auto")
    for i in range(len(models)):
        for j in range(len(names)):
            if nb[i, j] is not None and bool(nb[i, j]):
                ax.plot(j, i, marker="*", ms=9, color=GLYPH_COLOR, mec="white",
                        mew=0.4, zorder=3)
            elif ci[i, j] is not None and bool(ci[i, j]):
                ax.plot(j, i, marker="o", ms=3.6, color=GLYPH_COLOR, mec="white",
                        mew=0.3, zorder=3)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    for lab in ax.get_yticklabels():
        if lab.get_text() == MODEL_LABELS[PRIMARY]:
            lab.set_fontweight("bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Sustained return to the MCID band over 5 years\n"
                 r"$\Delta$AUC = AUC(with the block) $-$ AUC(without)", fontsize=12)

    handles = [
        Line2D([0], [0], marker="o", color="none", mfc=GLYPH_COLOR, mec="white",
               ms=6, label="bootstrap CI excludes 0"),
        Line2D([0], [0], marker="*", color="none", mfc=GLYPH_COLOR, mec="white",
               ms=11, label=r"+ $p_\mathrm{NB} < 0.05$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=10, frameon=False, title="Significance",
              title_fontsize=10)
    cb = host.colorbar(im, ax=ax, orientation="vertical", fraction=0.05,
                       pad=cbar_pad, extend="both")
    cb.set_label(r"$\Delta$AUC  ($>0$: the block improves)", fontsize=10)
    cb.ax.tick_params(labelsize=9)


def figure_contrasts(r: pd.DataFrame):
    """F10 - single heatmap: rows = models, columns = contrasts.

    Une seule cible (5 ans) => plus de petits-multiples par endpoint : la
    dimension liberee sert aux contrastes.
    """
    fig, ax = plt.subplots(figsize=contrasts_figsize(r))
    draw_contrasts(fig, ax, r)
    fig.tight_layout()
    return fig


def figure_auc_contrasts(auc: pd.DataFrame, con: pd.DataFrame):
    """Figure composite : (a) AUC par scenario (F9), (b) contrastes (F10).

    Blocs EMPILES : les deux panneaux sont larges (9 scenarios x 6 modeles ;
    8 contrastes + colorbar), les mettre cote a cote donnait une figure de plus
    de 30 pouces, illisible une fois ramenee a la largeur d'une page.
    """
    wa, ha = auc_figsize(auc)
    wb, hb = contrasts_figsize(con)
    # NATIVE dimensions preserved: each block stays as dense as its standalone
    # version, with no label overlap.
    fig = plt.figure(figsize=(max(wa, wb), ha + hb), layout="constrained")
    blocks = fig.subfigures(2, 1, height_ratios=[ha, hb])

    draw_auc(blocks[0].subplots(), auc)
    ax_b = blocks[1].subplots()
    draw_contrasts(blocks[1], ax_b, con)

    for sub, tag in zip(blocks, ("(a)", "(b)")):
        sub.text(0.005, 0.995, tag, fontsize=16, fontweight="bold", color="black",
                 ha="left", va="top")
    return fig


# ============================================================
# Table LaTeX
# ============================================================
def build_latex(auc: pd.DataFrame, con: pd.DataFrame) -> str:
    L = [r"% Generated by 11_mcid_5y.py - \usepackage{booktabs}"]
    prev = auc["prevalence"].iloc[0] if not auc.empty else float("nan")
    n = int(auc["n"].iloc[0]) if not auc.empty else 0
    L += [r"\begin{table}[t]", r"  \centering",
          r"  \caption{\emph{Durable} return within the MCID band over five "
          r"years: $y=1$ if " +
          M09.label_desc().replace("Δ", r"$\Delta$") + r" at every real "
          r"measurement of the patient. $n=" + str(n) + r"$, prevalence " +
          f"{prev:.3f}" + r". Out-of-fold AUC with a 95 \% patient-level "
          r"bootstrap CI; the regressor score is "
          r"$-\max_{ep}\hat{\Delta}$ over the same window as the label.}",
          r"  \label{tab:mcid5y_auc}",
          r"  \begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}", r"    \toprule",
          r"    Scenario & " + " & ".join(MODEL_LABELS[m] for m in MODEL_ORDER)
          + r" \\", r"    \midrule"]
    for tag in TAGS:
        s = auc[auc["tag"] == tag].set_index("model")
        if s.empty:
            continue
        cells = []
        for m in MODEL_ORDER:
            if m not in s.index or not np.isfinite(s.loc[m, "auc"]):
                cells.append("--")
                continue
            txt = f"{s.loc[m, 'auc']:.3f}"
            if s.loc[m, "auc"] == s["auc"].max():
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        L.append(rf"    {TAG_LABELS.get(tag, tag)} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \\[2pt] {\footnotesize The highest AUC of each row is in bold.}",
          r"\end{table}", ""]

    if not con.empty:
        L += [r"\begin{table}[t]", r"  \centering",
              r"  \caption{Feature-block contrasts on the same five-year "
              r"target: $\Delta$AUC $=$ AUC(treatment) $-$ AUC(baseline). "
              r"$\bullet$ = bootstrap CI excluding $0$; $\star$ = in addition "
              r"$p_\text{NB}<0.05$ (Nadeau--Bengio). On the regressor rows "
              r"$p_\text{NB}$ is indicative only: the fold partition is rebuilt "
              r"at the patient level and does not coincide with the one under "
              r"which those models were trained.}",
              r"  \label{tab:mcid5y_contrasts}",
              r"  \begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}", r"    \toprule",
              r"    Contrast & " + " & ".join(MODEL_LABELS[m] for m in MODEL_ORDER)
              + r" \\", r"    \midrule"]
        for _, _, name in CONTRASTS:
            s = con[con["contrast"] == name].set_index("model")
            if s.empty:
                continue
            cells = []
            for m in MODEL_ORDER:
                if m not in s.index:
                    cells.append("--")
                    continue
                mark = (r"$^\star$" if s.loc[m, "nb_sig"]
                        else r"$^\bullet$" if s.loc[m, "ci_excl0"] else "")
                cells.append(f"{s.loc[m, 'delta_auc']:+.3f}" + mark)
            L.append(rf"    {name} & " + " & ".join(cells) + r" \\")
        L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 72)
    print("FIVE-YEAR MCID - DURABLE return within the band (ALL rule)")
    print(f"window: {EPS_5Y} d   |   target: {M09.label_desc()} "
          f"at every measurement")
    print(f"inclusion: >= {MIN_MEAS} real measurement(s)   |   B = {N_BOOT}")
    print("=" * 72)

    if REPLOT:
        # Both CSVs are the INPUT here and are not rewritten.
        # mcid5y_contrasts.csv may legitimately be missing (arm B empty at the
        # last computation), in which case F10 is skipped, as in normal mode.
        auc = load_cached_results(OUT_DIR / "mcid5y_auc.csv", "11_mcid_5y.py")
        con_path = OUT_DIR / "mcid5y_contrasts.csv"
        con = (load_cached_results(con_path, "11_mcid_5y.py") if con_path.exists()
               else pd.DataFrame())
    else:
        print("\n--- Arm A: AUC per scenario ---")
        auc = run_auc()
        if auc.empty:
            raise SystemExit("No AUC computable - check models_*/ and prep_*/.")

        print("\n--- Volet B : contrastes de blocs de features ---")
        con = run_contrasts()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not REPLOT:
        auc.to_csv(OUT_DIR / "mcid5y_auc.csv", index=False)
        if not con.empty:
            con.to_csv(OUT_DIR / "mcid5y_contrasts.csv", index=False)
    (OUT_DIR / "mcid5y_table.tex").write_text(build_latex(auc, con))

    fig = figure_auc(auc)
    fig.savefig(FIG_DIR / "F9_mcid5y_auc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "F9_mcid5y_auc.pdf", bbox_inches="tight")
    plt.close(fig)
    if not con.empty:
        fig = figure_contrasts(con)
        fig.savefig(FIG_DIR / "F10_mcid5y_contrasts_heatmap.png", dpi=300,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / "F10_mcid5y_contrasts_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        fig = figure_auc_contrasts(auc, con)
        fig.savefig(FIG_DIR / "F910_mcid5y_auc_contrasts.pdf", bbox_inches="tight")
        plt.close(fig)

    # Summary
    print(f"\nTarget: n = {int(auc['n'].iloc[0])}, prevalence = "
          f"{auc['prevalence'].iloc[0]:.3f}, median of "
          f"{auc['n_meas_median'].iloc[0]:.0f} real measurements per patient.")
    best = auc.loc[auc["auc"].idxmax()]
    print(f"Meilleure AUC : {best['model']} / {best['tag']} = {best['auc']:.3f} "
          f"[{best['auc_lo']:.3f};{best['auc_hi']:.3f}]")
    if "logreg" in set(auc["model"]):
        d = auc[auc["model"] != "logreg"]["dauc_logreg_minus_model"]
        print(f"dAUC(logreg - regressor): median {d.median():+.3f}, "
              f"negative in {int((d < 0).sum())}/{len(d)} cases "
              f"(IC excluant 0 : "
              f"{int(((auc['dauc_lo'] > 0) | (auc['dauc_hi'] < 0)).sum())})")
    if not con.empty:
        print(f"\nContrasts ({len(MODEL_ORDER)} models):")
        for _, _, name in CONTRASTS:
            s = con[con["contrast"] == name]
            if s.empty:
                continue
            print(f"  {name:<26} median dAUC {s['delta_auc'].median():+.3f}   "
                  f"IC≠0 : {int(s['ci_excl0'].sum())}/{len(s)}   "
                  f"p_NB<0.05 : {int(s['nb_sig'].sum())}/{len(s)}")
    print(f"\n[done] Written:")
    print(f"   {FIG_DIR}/F9_mcid5y_auc.png / .pdf")
    print(f"   {FIG_DIR}/F10_mcid5y_contrasts_heatmap.png / .pdf")
    if REPLOT:
        print(f"   {OUT_DIR}/mcid5y_table.tex")
    else:
        print(f"   {OUT_DIR}/mcid5y_auc.csv, mcid5y_contrasts.csv, mcid5y_table.tex")


if __name__ == "__main__":
    main()