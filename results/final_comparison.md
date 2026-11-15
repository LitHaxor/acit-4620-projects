# Final Test Set & >25 km Generalization Comparison

- **Final results source:** `results/final_test_results.csv`
- **Per-image predictions directory:** `results/predictions`
- **Bootstrap protocol:** Station-level cluster bootstrap (`1000` resamples of stations with replacement, `seed=42`).
- **E1 Metadata MLP status:** Loaded `E1_metadata_mlp` from existing `results/baselines_cv_results.csv` and `results/predictions/E1_metadata_mlp_preds.npz` with station-bootstrap 95% CIs on both `test` and `test_gt25km` without retraining.

## 1. All Final Models (Full Test `n=2,920` and `test_gt25km` `n=717`)

| Name | Model | Split | n | Stations | MAE (95% Station CI) | rmse | plcc | srcc | r2 | acc | pm1_acc | macro_f1 | tail_mae | tail_f1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F_E2` | `mobilenetv3_s_pretrained` | `test` | 2920 | 729 | **41.12** [33.97, 50.79] | 61.95 | 0.6749 | 0.7071 | 0.4215 | 0.4240 | 0.8620 | 0.3723 | 96.41 | 0.2252 |
| `F_E2` | `mobilenetv3_s_pretrained` | `test_gt25km` | 717 | 205 | **39.60** [28.06, 54.03] | 54.56 | 0.5689 | 0.5623 | 0.2747 | 0.4003 | 0.8368 | 0.2743 | 116.42 | 0.0000 |
| `F_PFRML_full` | `pfrml_mobilenetv3_s` | `test` | 2920 | 729 | **45.04** [37.51, 54.04] | 68.56 | 0.6039 | 0.6574 | 0.2914 | 0.3986 | 0.8134 | 0.3687 | 96.54 | 0.2752 |
| `F_PFRML_full` | `pfrml_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **46.79** [34.22, 61.06] | 66.14 | 0.4121 | 0.5082 | -0.0660 | 0.3291 | 0.7448 | 0.2221 | 123.82 | 0.0000 |
| `F_FUSION_R1` | `fusion_r1_mobilenetv3_s` | `test` | 2920 | 729 | **35.77** [30.47, 42.72] | 51.07 | 0.7943 | 0.7941 | 0.6070 | 0.4565 | 0.8938 | 0.4150 | 76.92 | 0.2899 |
| `F_FUSION_R1` | `fusion_r1_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **39.22** [30.85, 47.74] | 50.68 | 0.6464 | 0.6696 | 0.3743 | 0.3473 | 0.7741 | 0.2352 | 86.00 | 0.0000 |
| `F_FUSION_R1_meta_off` | `fusion_r1_mobilenetv3_s` | `test` | 2920 | 729 | **47.56** [39.57, 58.12] | 66.12 | 0.6936 | 0.6757 | 0.3411 | 0.3428 | 0.7897 | 0.2556 | 119.80 | 0.0805 |
| `F_FUSION_R1_meta_off` | `fusion_r1_mobilenetv3_s` | `test_gt25km` | 717 | 205 | **44.88** [32.15, 58.48] | 59.60 | 0.5091 | 0.5460 | 0.1346 | 0.3333 | 0.7699 | 0.2165 | 121.83 | 0.0000 |
| `E1_metadata_mlp` | `metadata_mlp` | `test` | 2920 | 729 | **40.34** [34.45, 48.31] | 57.70 | 0.7098 | 0.7344 | 0.4982 | 0.4096 | 0.8873 | 0.3397 | 76.82 | 0.1362 |
| `E1_metadata_mlp` | `metadata_mlp` | `test_gt25km` | 717 | 205 | **49.84** [40.18, 58.81] | 65.31 | 0.3797 | 0.4954 | -0.0393 | 0.2929 | 0.7476 | 0.2004 | 88.83 | 0.0000 |

## 2. Paired Station-Bootstrap MAE Differences — Full Test Set (`n=2,920`, 729 Stations, `Model A - Model B`, 95% CI)

Negative `MAE Diff` means `Model A` achieves lower MAE than `Model B`.

| Model A | Model B | n | Stations | MAE(A) | MAE(B) | MAE Diff (A - B) | 95% Station Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F_E2` | `F_PFRML_full` | 2920 | 729 | 41.12 | 45.04 | **-3.92** | [-6.27, -1.58] |
| `F_E2` | `F_FUSION_R1` | 2920 | 729 | 41.12 | 35.77 | **+5.35** | [+2.26, +8.62] |
| `F_E2` | `F_FUSION_R1_meta_off` | 2920 | 729 | 41.12 | 47.56 | **-6.44** | [-9.39, -3.22] |
| `F_E2` | `E1_metadata_mlp` | 2920 | 729 | 41.12 | 40.34 | **+0.77** | [-3.46, +4.78] |
| `F_PFRML_full` | `F_FUSION_R1` | 2920 | 729 | 45.04 | 35.77 | **+9.26** | [+6.06, +12.52] |
| `F_PFRML_full` | `F_FUSION_R1_meta_off` | 2920 | 729 | 45.04 | 47.56 | **-2.52** | [-6.13, +1.27] |
| `F_PFRML_full` | `E1_metadata_mlp` | 2920 | 729 | 45.04 | 40.34 | **+4.69** | [+0.24, +9.10] |
| `F_FUSION_R1` | `F_FUSION_R1_meta_off` | 2920 | 729 | 35.77 | 47.56 | **-11.79** | [-16.00, -7.74] |
| `F_FUSION_R1` | `E1_metadata_mlp` | 2920 | 729 | 35.77 | 40.34 | **-4.57** | [-8.31, -1.26] |
| `F_FUSION_R1_meta_off` | `E1_metadata_mlp` | 2920 | 729 | 47.56 | 40.34 | **+7.21** | [+1.56, +12.49] |

## 3. Paired Station-Bootstrap MAE Differences — `test_gt25km` Subset (`>25 km`, `n=717`, 205 Stations, `Model A - Model B`, 95% CI)

Negative `MAE Diff` means `Model A` achieves lower MAE than `Model B` on unseen regions (`>25 km` from any training station).

| Model A | Model B | n | Stations | MAE(A) | MAE(B) | MAE Diff (A - B) | 95% Station Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F_E2` | `F_PFRML_full` | 717 | 205 | 39.60 | 46.79 | **-7.18** | [-11.92, -2.53] |
| `F_E2` | `F_FUSION_R1` | 717 | 205 | 39.60 | 39.22 | **+0.38** | [-5.99, +8.09] |
| `F_E2` | `F_FUSION_R1_meta_off` | 717 | 205 | 39.60 | 44.88 | **-5.28** | [-10.67, +0.00] |
| `F_E2` | `E1_metadata_mlp` | 717 | 205 | 39.60 | 49.84 | **-10.24** | [-20.50, +0.86] |
| `F_PFRML_full` | `F_FUSION_R1` | 717 | 205 | 46.79 | 39.22 | **+7.57** | [+1.46, +14.29] |
| `F_PFRML_full` | `F_FUSION_R1_meta_off` | 717 | 205 | 46.79 | 44.88 | **+1.90** | [-2.35, +6.01] |
| `F_PFRML_full` | `E1_metadata_mlp` | 717 | 205 | 46.79 | 49.84 | **-3.06** | [-14.90, +8.29] |
| `F_FUSION_R1` | `F_FUSION_R1_meta_off` | 717 | 205 | 39.22 | 44.88 | **-5.66** | [-11.75, +0.92] |
| `F_FUSION_R1` | `E1_metadata_mlp` | 717 | 205 | 39.22 | 49.84 | **-10.63** | [-18.78, -2.59] |
| `F_FUSION_R1_meta_off` | `E1_metadata_mlp` | 717 | 205 | 44.88 | 49.84 | **-4.96** | [-16.73, +5.42] |
