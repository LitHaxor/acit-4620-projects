# Physics-Aware Multimodal Air Quality Estimation from Street-Level Photos

Predicting continuous **US EPA AQI** (`pm25` in `[1, 530]`) and 6 discrete severity categories (`Good` `0–50`, `Moderate` `51–100`, `USG` `101–150`, `Unhealthy` `151–200`, `Very Unhealthy` `201–300`, `Hazardous` `301–600`) from street-level daytime photographs and coarse geolocation/calendar metadata on the **PM25Vision** benchmark (`8,176` cleaned train images across `2,527` primary stations; `2,920` test images across `729` stations; `0` train/test station overlap).

Built around a **MobileNetV3-Small** visual spine combining **Dimension Consistency Module (DCM) + Light Transformer** representations (inspired by PFRML) with **geospatial + cyclic-month metadata** (`[x, y, z, sin_month, cos_month]`) under spatial-cluster cross-validation (`25 km` separation) to isolate true visual generalization from spatial station memorization.

---

## Headline Final Test Results

All test-set numbers are copied verbatim from [`results/REPORT_NUMBERS.md`](results/REPORT_NUMBERS.md) (Section 6a). Each final neural model was trained once on all `8,176` cleaned training images (`seed=42`) and evaluated once on the held-out test set (`n=2,920`, `729` stations) and its unseen-region subset `test_gt25km` (`>25 km` from every training station; `n=717`, `205` stations). Confidence intervals use `1,000`-resample station-level cluster bootstrap (`seed=42`).

| Name | Model | Split | n | Stations | MAE (95% Station CI) | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F_FUSION_R1`** | `fusion_r1_mobilenetv3_s` | `test` | 2920 | 729 | **35.77** [30.47, 42.72] | 51.07 | 0.7943 | 0.7941 | 0.6070 | 0.4565 | 0.8938 | 0.4150 | 76.92 | 0.2899 |
| **`F_FUSION_R1`** | `fusion_r1_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **39.22** [30.85, 47.74] | 50.68 | 0.6464 | 0.6696 | 0.3743 | 0.3473 | 0.7741 | 0.2352 | 86.00 | 0.0000 |
| `F_E2` | `mobilenetv3_s_pretrained` | `test` | 2920 | 729 | **41.12** [33.97, 50.79] | 61.95 | 0.6749 | 0.7071 | 0.4215 | 0.4240 | 0.8620 | 0.3723 | 96.41 | 0.2252 |
| `F_E2` | `mobilenetv3_s_pretrained` | `test_gt25km` | 717 | 205 | **39.60** [28.06, 54.03] | 54.56 | 0.5689 | 0.5623 | 0.2747 | 0.4003 | 0.8368 | 0.2743 | 116.42 | 0.0000 |
| `E1_metadata_mlp` | `metadata_mlp` | `test` | 2920 | 729 | **40.34** [34.45, 48.31] | 57.70 | 0.7098 | 0.7344 | 0.4982 | 0.4096 | 0.8873 | 0.3397 | 76.82 | 0.1362 |
| `E1_metadata_mlp` | `metadata_mlp` | `test_gt25km` | 717 | 205 | **49.84** [40.18, 58.81] | 65.31 | 0.3797 | 0.4954 | -0.0393 | 0.2929 | 0.7476 | 0.2004 | 88.83 | 0.0000 |
| `F_PFRML_full` | `pfrml_mobilenetv3_s` | `test` | 2920 | 729 | **45.04** [37.51, 54.04] | 68.56 | 0.6039 | 0.6574 | 0.2914 | 0.3986 | 0.8134 | 0.3687 | 96.54 | 0.2752 |
| `F_PFRML_full` | `pfrml_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **46.79** [34.22, 61.06] | 66.14 | 0.4121 | 0.5082 | -0.0660 | 0.3291 | 0.7448 | 0.2221 | 123.82 | 0.0000 |
| `F_FUSION_R1_meta_off` | `fusion_r1_mobilenetv3_s` | `test` | 2920 | 729 | **47.56** [39.57, 58.12] | 66.12 | 0.6936 | 0.6757 | 0.3411 | 0.3428 | 0.7897 | 0.2556 | 119.80 | 0.0805 |
| `F_FUSION_R1_meta_off` | `fusion_r1_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **44.88** [32.15, 58.48] | 59.60 | 0.5091 | 0.5460 | 0.1346 | 0.3333 | 0.7699 | 0.2165 | 121.83 | 0.0000 |

**Key findings from `results/REPORT_NUMBERS.md`:**
- **Fusion (`F_FUSION_R1`) outperforms every single-modality model on the full test set:** lowers MAE by **-5.35** `[-8.62, -2.26]` vs image-only `F_E2` and by **-4.57** `[-8.31, -1.26]` vs metadata-only `E1_metadata_mlp`, reaching `R2 = 0.6070`.
- **Metadata shortcut collapses `>25 km`:** `E1_metadata_mlp` jumps from `40.34` (`R2 = 0.4982`) on the full test set to **`49.84` (`R2 = -0.0393`)** beyond `25 km`, whereas `F_FUSION_R1` (`39.22`, `R2 = 0.3743`) lowers MAE by **-10.63** `[-18.78, -2.59]` vs `E1_metadata_mlp` and preserves `R2 > 0`.
- **Prototypes degrade regression in MobileNetV3-S:** Adding PFRML's prototype classification branch (`R2: E4_mobilenetv3_s_baseline`, `48.14 +- 7.61` station CV MAE) underperforms regression-only `R1` (`E4_mobilenetv3_s_r1_no_prototypes`, `45.50 +- 8.64`, `-2.63 +- 1.10` across `4/5` folds) and `E2_mobilenetv3_s_pretrained` (`45.66 +- 8.41`).

---

## Repository Layout

```text
├── README.md, REPRODUCING.md, requirements.txt
├── train.py, evaluate.py                    # Core training & checkpoint evaluation CLIs
├── configs/                                 # Experiment YAML configurations
│   ├── default.yaml                         # Base settings & shared training defaults
│   ├── pfrml_prototype_multitask.yaml       # R2: core PFRML baseline (toggles off)
│   ├── pfrml_regression_only.yaml           # R1: DCM + Transformer + MLP (no prototypes)
│   ├── pfrml_eq5_regression.yaml            # Eq. 5 literal (reg_input='chi')
│   ├── pfrml_attention_pooling.yaml         # T1: spatial token attention pooling
│   ├── pfrml_physics_gate.yaml              # T2: 4-scalar physics prior gate
│   ├── pfrml_class_weights.yaml             # T3: fold-internal class weights
│   ├── pfrml_dcm_concat.yaml                # T4: DCM concat (collapse variant)
│   ├── pfrml_all_toggles.yaml               # Full PFRML (T1 + T2 + T3)
│   ├── fusion_regression_only_neck.yaml     # Winning F_FUSION_R1 multimodal config
│   └── fusion_pfrml_full.yaml               # Full PFRML + MetadataMLP fusion config
├── src/                                     # Library package (data, features, models, losses, metrics)
├── scripts/
│   ├── final_evaluation.py                  # ONLY script allowed to evaluate held-out test images
│   ├── data/                                # Dataset download, paper PDF->MD, dataset audit
│   ├── experiments/                         # Resumable CV & final experiment queue runners
│   └── analysis/                            # Read-only CPU reporting, bootstrap CIs, diagnostics, figures
├── tests/                                   # 55 CPU unit & regression tests (pytest)
├── docs/                                    # SPEC, EXPERIMENTS registry, development_log, notes
└── results/                                 # Tracked snapshot (~1.8 MB): CSVs, curves, preds, REPORT_NUMBERS.md
```

*Note on git history:* The five original course starter files (`1.eda-image-datasets.ipynb`, `2.feature-extraction.ipynb`, `3.example_train_test.ipynb`, `paper_taxonomy.xlsx`, and the original starter `README.md`) remain removed from the working tree and are recoverable from git history at commit `8fa4cd3`.

---

## Entry Points & Setup

This repository has four main entry points depending on what you want to run:

- **Dataset setup:** `python scripts/data/download_dataset.py` — downloads and unpacks `PM25Vision` into `datasets/` (gitignored).
- **Single-model training & evaluation (`train.py`, `evaluate.py`):** trains baselines or individual CV models (`--model`, `--config`, `--group-mode`, `--fold`) and re-evaluates saved checkpoints.
- **Full-pipeline experiment queues (`scripts/experiments/queue_*.py`):** resumable runners that orchestrate all 5-fold CV runs, diagnostics, and final evaluations on your hardware and write fresh CSVs/curves/checkpoints into `outputs/`.
- **Report & metric compiler (`scripts/analysis/compare_final_models.py`, `scripts/analysis/build_report_numbers.py`):** computes final tables, paired diffs, and 1,000-resample station-bootstrap CIs either from **your local runs** in `outputs/` (default) or **offline** from the tracked snapshot via `--results-dir results`.

### Environment & Dataset Installation

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# Download PM25Vision (~datasets/, gitignored)
python scripts/data/download_dataset.py
```

- **Reference PDFs (copyrighted, gitignored):** Place reference PDFs in `papers/pdfs/` and convert them locally to `papers/markdown/` with `python scripts/data/convert_papers_to_markdown.py`. Neither directory is tracked in git.

---

## Running Everything From Scratch (Computing Your System's Metrics)

`results/` holds our recorded snapshot for reference; to recompute **every metric fresh on your own hardware** into `outputs/` (never touching `results/`), run these steps in order after `download_dataset.py`:

### 1. Verify Environment, Unit Tests, & Dataset Integrity (`~1 min`)

```bash
python -m pytest -q                              # Runs all 55 unit & data-pipeline tests
python scripts/analysis/diagnose_folds.py       # Audits splits -> outputs/diagnostics/folds.md
```

### 2. Optional Pre-Flight Timing & `--smoke` Dry Run (`~1–2 min`)

Each queue script automatically selects `mps -> cuda -> cpu` (`--device auto`), logs to `outputs/logs/`, checkpoints every epoch, and resumes across interruptions. Before committing hours of GPU/MPS time, you can preview runtime estimates or run a tiny 1-epoch/64-image dry run under `outputs/smoke/`:

```bash
# Project total runtime on your hardware without full training
python scripts/experiments/queue_baselines.py --project-only
python scripts/experiments/queue_pfrml_ablations.py --project-only
python scripts/experiments/queue_fusion_cv.py --project-only

# Optional smoke tests (write only to outputs/smoke/, never read test labels)
python scripts/experiments/queue_baselines.py --smoke
python scripts/experiments/queue_pfrml_ablations.py --smoke
python scripts/experiments/queue_fusion_cv.py --smoke
```

*(Note: `queue_fusion_final.py --project-only` / `--smoke` reads `outputs/results/fusion_epoch_selection.md`, which is generated in Stage D below after `queue_fusion_cv.py` writes the 5-fold `E5_fusion_r1_p00` curves.)*

### 3. Fit Tabular & Non-Visual Baselines (`E0_median`, `E0_ridge`, `E1_metadata_mlp`, `~5 min`)

This generates `outputs/results/baselines_cv_results.csv` and `outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz` (required later by the final comparison scripts):

```bash
python train.py --model median --experiment-id E0_median --config configs/default.yaml --group-mode station --fold all --output-csv outputs/results/baselines_cv_results.csv
python train.py --model median --experiment-id E0_median --config configs/default.yaml --group-mode spatial_cluster --fold all --output-csv outputs/results/baselines_cv_results.csv
python train.py --model median --experiment-id E0_median --config configs/default.yaml --fold full --output-csv outputs/results/baselines_cv_results.csv

python train.py --model ridge_haze --experiment-id E0_ridge --config configs/default.yaml --group-mode station --fold all --output-csv outputs/results/baselines_cv_results.csv
python train.py --model ridge_haze --experiment-id E0_ridge --config configs/default.yaml --group-mode spatial_cluster --fold all --output-csv outputs/results/baselines_cv_results.csv
python train.py --model ridge_haze --experiment-id E0_ridge --config configs/default.yaml --fold full --output-csv outputs/results/baselines_cv_results.csv

python train.py --model metadata_mlp --experiment-id E1_metadata_mlp --config configs/default.yaml --group-mode station --fold all --epochs 40 --lr 1e-3 --reg-loss mse --output-csv outputs/results/baselines_cv_results.csv
python train.py --model metadata_mlp --experiment-id E1_metadata_mlp --config configs/default.yaml --group-mode spatial_cluster --fold all --epochs 40 --lr 1e-3 --reg-loss mse --output-csv outputs/results/baselines_cv_results.csv
python train.py --model metadata_mlp --experiment-id E1_metadata_mlp --config configs/default.yaml --fold full --epochs 40 --lr 1e-3 --reg-loss mse --output-csv outputs/results/baselines_cv_results.csv
```

### 4. Run the Full Neural Training & Evaluation Pipeline

Run the four queue stages sequentially; each stage resumes cleanly if interrupted:

```bash
# Stage A: Image baselines & MDNet-T Fold-0 screening + 5-fold station CV -> outputs/results/image_models_cv_results.csv
python scripts/experiments/queue_baselines.py

# Stage B: PFRML ablation ladder (R1, R2, T1-T4, full) + spatial E2 + MDNet-T PFRML -> outputs/results/pfrml_ablation_cv_results.csv
python scripts/experiments/queue_pfrml_ablations.py
python scripts/analysis/diagnose_training_failures.py   # -> outputs/diagnostics/collapse.md
python scripts/analysis/summarize_last_epoch.py         # -> outputs/results/last_epoch_summary.md

# Stage C: Spatial-cluster fusion CV (E5_*) + Eq. 5 probe + one-shot F_E2 & F_PFRML_full -> outputs/results/fusion_cv_results.csv, final_test_results.csv
python scripts/experiments/queue_fusion_cv.py

# Stage D: CMA-3 epoch selection on E5_fusion_r1_p00 curves -> outputs/results/fusion_epoch_selection.md
python scripts/analysis/select_fusion_epochs.py

# Stage E: Epoch-selected 5-fold fusion verification + decision rule + one-shot F_FUSION_R1 final evaluation
python scripts/experiments/queue_fusion_final.py
```

### 5. Compile All Final Tables, Bootstrap CIs, & Master Metrics Report

```bash
# Reads your trained outputs/ (CSVs, curves, predictions) and writes fresh reports:
python scripts/analysis/compare_final_models.py   # -> outputs/results/final_comparison.md
python scripts/analysis/build_report_numbers.py   # -> outputs/results/REPORT_NUMBERS.md
```

Your machine's complete metrics—every 5-fold `mean +- SD`, paired difference, failure diagnostic, `test` / `test_gt25km` score, and `95%` station-bootstrap CI—are now in **`outputs/results/REPORT_NUMBERS.md`** and **`outputs/results/final_comparison.md`**.

### Fast Offline Alternative (Level 1 — No Dataset, No Training)

To verify table/bootstrap computation from the tracked `results/` snapshot without downloading `datasets/`:

```bash
python scripts/analysis/compare_final_models.py --results-dir results
python scripts/analysis/build_report_numbers.py --results-dir results
```

See [`REPRODUCING.md`](REPRODUCING.md) for environment details, hardware notes (`MPS` non-determinism), and per-stage runtimes.

---

## Evaluation Protocol & Naming Conventions

- **Dual Group-Split CV (`5` folds, `StratifiedGroupKFold`):**
  1. `station` mode (`2,527` station groups) measures visual & model generalization without same-station leakage, though geographically nearby stations (`<25 km`) still share folds.
  2. `spatial_cluster` mode (`1,059` connected-component clusters at `25 km` Haversine linkage) isolates unseen-region generalization and exposes coordinate memorization in multimodal models.
- **Last-Epoch Reporting Policy:** All PFRML (`E4_*`), image-model (`E2_*`, `E3_*`, `E3b_*`), and fusion (`E5_*`) CV comparisons in `results/REPORT_NUMBERS.md` quote the **last-epoch** row from `results/curves/*.csv` (`mean +- SD` across 5 folds, paired differences `mean +- SE`), eliminating `min(val_loss)` selection bias.
- **One-Shot Final Test Evaluation:** `scripts/final_evaluation.py` is the only neural evaluation script reading `datasets/test/metadata.csv`. It trains for a fixed epoch count on all `8,176` training images, evaluates the last epoch once per `--name` (refusing repeated names), and computes `1,000`-resample station-cluster bootstrap `95%` CIs on full `test` and `test_gt25km`.
- **Experiment Naming & Registry:** Experiment IDs (`E0_...`–`E5_...`, `F_...`) preserve exact provenance across CSVs, curves, and checkpoints. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the complete experiment table, legacy `Step 1`–`5c` mapping, failed runs (`E4_mdnet_t_baseline_ep60`, `E4_mobilenetv3_s_dcm_concat`), and do-not-quote best-epoch rules.

---

## Known Limitations

1. **Single Random Seed (`42`):** All CV and final runs use one seed; cross-seed variance is not measured.
2. **Fold-0 Hyperparameter Screening:** Learning-rate and epoch screens (`queue_baselines.py`) were performed on Fold 0 before 5-fold promotion; Fold 0 is the easiest station fold (`folds.md`).
3. **Retrospective `N = 7` Early-Stopping Rule:** `F_FUSION_R1`'s `7`-epoch horizon was selected post-hoc via 3-epoch centered moving average on `E5_fusion_r1_p00` spatial CV curves because longer fusion training (`30` ep) progressively overfit station metadata (test `meta_off` MAE degrades from `35.77` to `47.56`).
4. **Small `>25 km` Severe-Pollution Tail:** `test_gt25km` contains `717` images (`205` stations) with relatively few hazardous (`AQI > 300`) samples, yielding `TailF1 = 0.0000` across all models and wider bootstrap CIs on unseen regions.

---

## AI assistance disclosure

TO BE COMPLETED BY THE AUTHOR
