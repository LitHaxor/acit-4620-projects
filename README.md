# Physics-Aware Multimodal Air Quality Estimation from Street-Level Photos

Predicting continuous **US EPA AQI** (`pm25` in `[1, 530]`) and 6 discrete severity categories (`Good` `0–50`, `Moderate` `51–100`, `USG` `101–150`, `Unhealthy` `151–200`, `Very Unhealthy` `201–300`, `Hazardous` `301–600`) from street-level daytime photographs and coarse geolocation/calendar metadata on the **PM25Vision** benchmark (`8,176` cleaned train images across `2,527` primary stations; `2,920` test images across `729` stations; `0` train/test station overlap).

Built around a **MobileNetV3-Small** visual spine combining **Dimension Consistency Module (DCM) + Light Transformer** representations (inspired by PFRML) with **geospatial + cyclic-month metadata** (`[x, y, z, sin_month, cos_month]`) under spatial-cluster cross-validation (`25 km` separation) to isolate true visual generalization from spatial station memorization.

---

## Headline Final Test Results

All test-set numbers are copied verbatim from [`results/REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/results/REPORT_NUMBERS.md) (Section 6a). Each final neural model was trained once on all `8,176` cleaned training images (`seed=42`) and evaluated once on the held-out test set (`n=2,920`, `729` stations) and its unseen-region subset `test_gt25km` (`>25 km` from every training station; `n=717`, `205` stations). Confidence intervals use `1,000`-resample station-level cluster bootstrap (`seed=42`).

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

## Setup

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# Download PM25Vision (~datasets/, gitignored)
python scripts/data/download_dataset.py
```

- **Reference PDFs (copyrighted, gitignored):** Place reference PDFs in `papers/pdfs/` and convert them locally to `papers/markdown/` with `python scripts/data/convert_papers_to_markdown.py`. Neither directory is tracked in git.

---

## Quick Reproduction Commands

See [`REPRODUCING.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/REPRODUCING.md) for **Level 1** (offline regeneration from `results/` with zero datasets/GPUs) and **Level 2** (full training). Summary of exact commands:

```bash
# Regenerate all summary tables & 95% CIs from tracked results/ snapshot (offline)
python scripts/analysis/compare_final_models.py --results-dir results
python scripts/analysis/build_report_numbers.py --results-dir results

# Full Level 2 pipeline (after downloading datasets/):
python -m pytest -q
python scripts/analysis/diagnose_folds.py                       # Data/fold audit -> outputs/diagnostics/folds.md
python scripts/experiments/queue_baselines.py                   # Image baselines & MDNet-T CV
python scripts/experiments/queue_pfrml_ablations.py             # PFRML ablation ladder (R1, R2, T1-T4, all)
python scripts/experiments/queue_fusion_cv.py                   # Spatial-cluster fusion CV + F_E2, F_PFRML_full
python scripts/analysis/select_fusion_epochs.py                 # CMA-3 epoch selection -> N = 7
python scripts/experiments/queue_fusion_final.py                # 7-ep CV verification + one-shot F_FUSION_R1
python scripts/analysis/compare_final_models.py                 # Outputs -> outputs/results/final_comparison.md
python scripts/analysis/build_report_numbers.py                 # Outputs -> outputs/results/REPORT_NUMBERS.md
```

---

## Evaluation Protocol & Naming Conventions

- **Dual Group-Split CV (`5` folds, `StratifiedGroupKFold`):**
  1. `station` mode (`2,527` station groups) measures visual & model generalization without same-station leakage, though geographically nearby stations (`<25 km`) still share folds.
  2. `spatial_cluster` mode (`1,059` connected-component clusters at `25 km` Haversine linkage) isolates unseen-region generalization and exposes coordinate memorization in multimodal models.
- **Last-Epoch Reporting Policy:** All PFRML (`E4_*`), image-model (`E2_*`, `E3_*`, `E3b_*`), and fusion (`E5_*`) CV comparisons in `results/REPORT_NUMBERS.md` quote the **last-epoch** row from `results/curves/*.csv` (`mean +- SD` across 5 folds, paired differences `mean +- SE`), eliminating `min(val_loss)` selection bias.
- **One-Shot Final Test Evaluation:** `scripts/final_evaluation.py` is the only neural evaluation script reading `datasets/test/metadata.csv`. It trains for a fixed epoch count on all `8,176` training images, evaluates the last epoch once per `--name` (refusing repeated names), and computes `1,000`-resample station-cluster bootstrap `95%` CIs on full `test` and `test_gt25km`.
- **Experiment Naming & Registry:** Experiment IDs (`E0_...`–`E5_...`, `F_...`) preserve exact provenance across CSVs, curves, and checkpoints. See [`docs/EXPERIMENTS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/EXPERIMENTS.md) for the complete experiment table, legacy `Step 1`–`5c` mapping, failed runs (`E4_mdnet_t_baseline_ep60`, `E4_mobilenetv3_s_dcm_concat`), and do-not-quote best-epoch rules.

---

## Known Limitations

1. **Single Random Seed (`42`):** All CV and final runs use one seed; cross-seed variance is not measured.
2. **Fold-0 Hyperparameter Screening:** Learning-rate and epoch screens (`queue_baselines.py`) were performed on Fold 0 before 5-fold promotion; Fold 0 is the easiest station fold (`folds.md`).
3. **Retrospective `N = 7` Early-Stopping Rule:** `F_FUSION_R1`'s `7`-epoch horizon was selected post-hoc via 3-epoch centered moving average on `E5_fusion_r1_p00` spatial CV curves because longer fusion training (`30` ep) progressively overfit station metadata (test `meta_off` MAE degrades from `35.77` to `47.56`).
4. **Small `>25 km` Severe-Pollution Tail:** `test_gt25km` contains `717` images (`205` stations) with relatively few hazardous (`AQI > 300`) samples, yielding `TailF1 = 0.0000` across all models and wider bootstrap CIs on unseen regions.

---

## AI assistance disclosure

TO BE COMPLETED BY THE AUTHOR
