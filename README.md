# Segmentation sources, DVH indices and patient-reported outcomes after LDR brachytherapy

Analysis code accompanying the article. Two independent pipelines share the same
cohort and the same dose-volume histogram (DVH) index definitions:

| Directory        | Question |
|------------------|----------|
| `concordance/`   | Does the **source of the organ segmentation** (manual, deterministic automatic, Bayesian Monte-Carlo) change the DVH indices? Paired equivalence analysis (TOST), plus a patient-level quality-threshold analysis. |
| `ipss/`          | Do the DVH indices improve the **prediction of the post-treatment IPSS** beyond the clinical variables? Nested cross-validation, paired contrasts with a corrected Nadeau–Bengio test, and a binary MCID arm. |

The code is published for transparency and reuse of the methods. **No patient
data is included**, and none can be redistributed.

---

## Data

Every script reads individual-level clinical and dosimetric data that is not
part of this repository. The expected inputs are described in the module
docstring of each entry-point script. Nothing is downloaded automatically.

Paths are resolved from environment variables, so no source file needs editing:

| Variable | Default | Contents |
|----------|---------|----------|
| `PROTECTA_DATA_ROOT` | `<repo>/data` | Root of the input tables (patient-level table, longitudinal IPSS, salvage delays). |
| `PROTECTA_DVH_DIR` | `<repo>/data/dvh` | DVH index files, one per segmentation source. |
| `PROTECTA_CROSS_DATASET_DIR` | `<repo>/data/dvh/cross_dataset` | Per-case geometric metrics (`metrics_per_case.csv`) used by the threshold analyses. |
| `PROTECTA_IPSS_DIR` | `<repo>/ipss` | Where the IPSS pipeline writes its outputs. |
| `PROTECTA_CONCORDANCE_DIR` | `<repo>/concordance` | Where the concordance pipeline writes its outputs. |

`.gitignore` blocks `data/`, every tabular format and every imaging format by
default, so patient data cannot be committed by accident.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later is required. The concordance pipeline needs only numpy,
pandas, scipy, statsmodels, scikit-learn and matplotlib; the heavier
dependencies (torch, catboost, xgboost, optuna, shap) are used by the IPSS
pipeline alone.

---

## `concordance/` — segmentation source and DVH indices

Run the scripts in numerical order, from the `concordance/` directory.

```bash
cd concordance
python 00_validate_synthetic.py          # synthetic end-to-end validation, run first
python 01_build_paired_table.py          # paired long table + margin table
python 02_tost_equivalence.py            # paired TOST, Holm on the confirmatory family
python 03_concordance_descriptors.py     # Bland-Altman, CCC, ICC(A,1)  [--figures]
python 04_mc_coverage.py                 # calibration of the Monte-Carlo intervals
python figures_synthese.py               # summary table and figures
```

Extensions, which additionally need the per-case geometric metrics:

```bash
python 07_validate_joint_synthetic.py    # synthetic validation of 07, run first
python 05_dice_threshold.py              # patient-level segmentation-quality threshold
python 06_quality_metric_comparison.py   # which quality metric should carry the threshold
python 07_joint_dice_volume.py           # joint quality x prostate-volume analysis
```

Utilities: `index_availability.py` audits which indices exist in each source;
`diagnostic_effectif_prostate.py` explains where patients are lost between
sources. `utils.py` centralises the panel, the pre-declared equivalence margins,
the harmonised loaders and the statistical primitives, so no definition can
drift between scripts.

Both `00_validate_synthetic.py` and `07_validate_joint_synthetic.py` build a
synthetic cohort whose truth is known in closed form and then run the **real**
functions of the pipeline against it. They are the recommended way to check an
installation before touching real data.

## `ipss/` — prediction of the post-treatment IPSS

Run the scripts in numerical order, from the `ipss/` directory.

```bash
cd ipss
python 00_build_ipss_dataset.py          # build the patient-level table
python 01_prepare_target.py              # features X and endpoint targets y
python 02_train.py                       # 6 algorithms x endpoints, nested CV
python 03_evaluate.py                    # OOF metrics, calibration, SHAP
python 04_contrasts.py                   # paired contrasts, bootstrap CI, Nadeau-Bengio
python 05_headroom.py                    # predictability headroom vs the measurement floor
python 06_figures.py                     # summary figures
```

`config.py` is the single configuration point: endpoints, cross-validation
design, DVH strategy and scenario definitions. Several settings are overridable
by environment variable, so parallel runs need no file edits:

| Variable | Effect |
|----------|--------|
| `PIPE_SCENARIO` | Selects a predefined scenario (DVH source, ablation) — see `config._SCENARIOS`. |
| `PIPE_TAG_SUFFIX` | Suffixes every output directory, so runs do not overwrite each other. |
| `PIPE_ENDPOINTS` | Restricts the run to given endpoints, as `"target:half_window,..."`. |
| `PIPE_N_REPEATS` | Number of repetitions of the outer cross-validation. |
| `PIPE_N_THREADS` | Per-process thread budget, to avoid oversubscription when running scenarios in parallel. |

Downstream analyses read the outputs of `04_contrasts.py` and never retrain:
`04_contrasts_nb_power.py` (power of the Nadeau–Bengio test),
`07_nb_latex_table.py`, `08_all_models_heatmap.py`, and the MCID arm
(`09_mcid_logreg.py`, `10_mcid_contrasts_heatmap.py`, `11_mcid_5y.py`).

The scripts suffixed `b` (`08b`, `10b`) are deliberate **naive twins** of their
counterparts: same material, same effect estimates, but the inference commonly
met in the applied literature (no bootstrap CI, no variance correction, no
multiplicity correction). They are methodological demonstration figures, not
results.

Most figure scripts accept `--replot`, which redraws from the results CSV
without refitting anything.

---

## Methodological notes

- **Sign convention.** In the concordance arm, `diff = val_manual - val_auto`,
  the manual segmentation being the reference. In the prediction arm,
  `delta RMSE = RMSE(baseline) - RMSE(treatment)`, so a positive value means the
  treatment improves the prediction.
- **No imputation of the comparison.** Unmatched patients are dropped and
  counted, never imputed.
- **Nested cross-validation.** Hyperparameters are tuned on an inner CV built
  only on the training part of each outer fold, so the outer test fold never
  enters model selection.
- **Corrected inference.** Model comparisons use a patient-level bootstrap CI as
  the primary inference, with the corrected Nadeau–Bengio test as a secondary,
  deliberately conservative check on the learning randomness.
- **Leakage diagnostic.** Targets estimated by the trajectory model are used for
  training but never for evaluation; the `source` column of the out-of-fold
  files allows the metrics to be re-sliced.

## Reproducibility

Random seeds are fixed in `concordance/utils.py` and `ipss/config.py`. Given the
same inputs and the same library versions, the pipelines are deterministic.
Output directories are suffixed by the active configuration, so a partial or
exploratory run never overwrites a reference run.

## Citation

If you use this code, please cite the accompanying article.
