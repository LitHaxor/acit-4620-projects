# Tracked Results Snapshot (`results/`)

This directory is a self-contained, git-tracked snapshot (~1.8 MB) of every result CSV, per-epoch validation curve, test-set prediction file, and diagnostic report produced during the project. It allows anyone cloning the repository to regenerate all summary and comparison tables (**Level 1 reproduction**) without GPUs, model checkpoints, or the raw PM25Vision image dataset.

## Contents

- **Aggregated Result CSVs:**
  - `baselines_cv_results.csv` — `E0_median`, `E0_ridge`, `E1_metadata_mlp` (`station` CV, `spatial_cluster` CV, `full_train` test, and `test_gt25km`).
  - `image_models_cv_results.csv` — Image-only CNN & MDNet-T Fold-0 hyperparameter screens and 5-fold `station` CV runs (`E2_*`, `E3_*`, `E3b_*`; note best-epoch CSV columns are superseded — see `REPORT_NUMBERS.md` Section 8).
  - `pfrml_ablation_cv_results.csv` — 5-fold `station` CV PFRML ablation ladder (`R1`, `R2`, `T1–T4`, `R2+all`), `E2_mobilenetv3_s_pretrained_spatial`, and `E4_mdnet_t_baseline_ep60` (note best-epoch CSV columns are superseded — see `REPORT_NUMBERS.md` Section 8).
  - `fusion_cv_results.csv` — `E5_*` multimodal fusion (`spatial_cluster` CV, both normal and `_meta_off`), `E5_mobilenetv3_s_reg_chi` (`station` CV), and `E5_mdnet_t_pfrml_lr2e-4_f0` (`last_epoch` metrics).
  - `final_test_results.csv` — Final one-shot test evaluation rows (`F_E2`, `F_PFRML_full`, `F_FUSION_R1`, `F_FUSION_R1_meta_off`) on full `test` (`n=2,920`) and `test_gt25km` (`n=717`).
- **`curves/` (`102` CSV files):** Per-epoch training/validation curves `{experiment_id}_{group_mode}_fold{fold}.csv`. Authoritative last-epoch CV tables in `REPORT_NUMBERS.md` Sections 3–5 are computed directly from the last row of each 5-fold curve set.
- **`predictions/` (`5` files):** Per-image test predictions (`image_id, station_id, y_true, y_pred`) for `F_E2.csv`, `F_PFRML_full.csv`, `F_FUSION_R1.csv`, and `F_FUSION_R1_meta_off.csv`, plus `E1_metadata_mlp_preds.npz` (copied from `outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz`).
- **`test_gt25km_ids.txt`:** The `717` test `image_id`s located `>25 km` from every training station (used so paired bootstrap differences on `test_gt25km` reproduce identically without requiring `datasets/`).
- **Diagnostics & Epoch Selection:** `folds.md`, `collapse.md`, `fusion_epoch_selection.md`, `fusion_epoch_decision.md`.
- **Generated Reports:** `final_comparison.md` and `REPORT_NUMBERS.md`.

## Regenerating Tables from This Snapshot (No Datasets or GPUs Needed)

```bash
python scripts/analysis/compare_final_models.py --results-dir results
python scripts/analysis/build_report_numbers.py --results-dir results
```

Running both commands overwrites `results/final_comparison.md` and `results/REPORT_NUMBERS.md` with identical numbers (differing only in the provenance commit hash, working-tree status, and generation timestamp).
