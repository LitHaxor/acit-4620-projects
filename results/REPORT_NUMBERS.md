# Authoritative Report Numbers (`REPORT_NUMBERS.md`)

> Generated strictly from CSVs (`results/*.csv`), per-epoch curve files (`results/curves/*.csv`), and saved prediction files (`results/predictions/*.csv`, `results/predictions/E1_metadata_mlp_preds.npz`) by `scripts/analysis/build_report_numbers.py`.

## 1. Provenance

- **Git commit hash:** `fd7bea3d525b28c9500ab970d39958a7e795e9d1` (`fd7bea3`)
- **Working tree status:** Uncommitted changes exist (1 modified/untracked paths in working tree)
- **Generated timestamp:** `2026-11-15 21:27:17`
- **Random seed:** `42` (`seed=42` across CV splits, training, and `1000`-resample station bootstrap)
- **Hardware & runtime environment:** Apple M5 Pro, 24 GB unified memory | OS: `macOS-27.2.1-arm64-arm-64bit-Mach-O` | Python `3.14.6` | PyTorch `2.14.1` (`mps` available)

---

## 2. Baselines (`E0_median`, `E0_ridge`, `E1_metadata_mlp`)

Source: `results/baselines_cv_results.csv` and `results/predictions/E1_metadata_mlp_preds.npz`.

### 2a. Summary Table (`station` CV, `spatial_cluster` CV, Full `test`, and `test_gt25km` `>25 km`)

| Experiment | Model | Split / Mode | n | MAE (Mean +- SD or [95% CI]) | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_median` | `median` | `station 5-fold CV` | 8176 | **66.21 +- 4.89** | 88.23 +- 12.81 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | -0.0161 +- 0.0123 | 0.2153 +- 0.0018 | 0.6497 +- 0.0057 | 0.0590 +- 0.0004 | 176.95 +- 33.83 | 0.0000 +- 0.0000 |
| `E0_median` | `median` | `spatial_cluster 5-fold CV` | 8176 | **66.14 +- 5.60** | 88.00 +- 13.71 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | -0.0168 +- 0.0142 | 0.2154 +- 0.0053 | 0.6500 +- 0.0162 | 0.0591 +- 0.0012 | 175.81 +- 29.24 | 0.0000 +- 0.0000 |
| `E0_median` | `median` | `Full test` | 2920 | **63.64** | 81.88 | 0.0000 | 0.0000 | -0.0105 | 0.2158 | 0.6473 | 0.0592 | 155.88 | 0.0000 |
| `E0_median` | `median` | `Test >25 km` | 717 | **56.60** | 64.22 | 0.0000 | 0.0000 | -0.0049 | 0.2022 | 0.6318 | 0.0561 | 102.78 | 0.0000 |
| `E0_ridge` | `ridge_haze` | `station 5-fold CV` | 8176 | **64.11 +- 4.40** | 86.32 +- 14.20 | 0.1998 +- 0.1083 | 0.2354 +- 0.0695 | 0.0301 +- 0.0583 | 0.2450 +- 0.0291 | 0.6925 +- 0.0161 | 0.1330 +- 0.0287 | 158.15 +- 35.36 | 0.0000 +- 0.0000 |
| `E0_ridge` | `ridge_haze` | `spatial_cluster 5-fold CV` | 8176 | **64.56 +- 5.02** | 87.37 +- 14.30 | 0.1526 +- 0.0937 | 0.1982 +- 0.0203 | -0.0018 +- 0.0503 | 0.2415 +- 0.0176 | 0.6915 +- 0.0311 | 0.1325 +- 0.0162 | 160.68 +- 34.63 | 0.0000 +- 0.0000 |
| `E0_ridge` | `ridge_haze` | `Full test` | 2920 | **60.84** | 79.77 | 0.2112 | 0.2258 | 0.0408 | 0.2250 | 0.6815 | 0.1109 | 140.91 | 0.0000 |
| `E0_ridge` | `ridge_haze` | `Test >25 km` | 717 | **53.76** | 65.26 | 0.1435 | 0.0970 | -0.0377 | 0.2092 | 0.6792 | 0.1142 | 107.93 | 0.0000 |
| `E1_metadata_mlp` | `metadata_mlp` | `station 5-fold CV` | 8176 | **43.15 +- 7.47** | 65.43 +- 18.12 | 0.6704 +- 0.1236 | 0.7260 +- 0.0762 | 0.4438 +- 0.1626 | 0.4234 +- 0.0329 | 0.8794 +- 0.0360 | 0.3506 +- 0.0289 | 104.96 +- 40.79 | 0.1632 +- 0.0755 |
| `E1_metadata_mlp` | `metadata_mlp` | `spatial_cluster 5-fold CV` | 8176 | **46.00 +- 7.33** | 70.09 +- 20.07 | 0.6257 +- 0.1404 | 0.7147 +- 0.0694 | 0.3647 +- 0.1768 | 0.4183 +- 0.0305 | 0.8603 +- 0.0441 | 0.3556 +- 0.0386 | 118.49 +- 56.91 | 0.1953 +- 0.0730 |
| `E1_metadata_mlp` | `metadata_mlp` | `Full test` | 2920 | **40.34** [34.45, 48.31] | 57.70 | 0.7098 | 0.7344 | 0.4982 | 0.4096 | 0.8873 | 0.3397 | 76.82 | 0.1362 |
| `E1_metadata_mlp` | `metadata_mlp` | `Test >25 km` | 717 | **49.84** [40.18, 58.81] | 65.31 | 0.3797 | 0.4954 | -0.0393 | 0.2929 | 0.7476 | 0.2004 | 88.83 | 0.0000 |

### 2b. Per-Fold Baseline Validation MAE (`station` and `spatial_cluster`)

| Experiment | Mode | Mean +- SD | F0 | F1 | F2 | F3 | F4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_median` | `station` | **66.21 +- 4.89** | 67.18 | 62.31 | 63.71 | 74.33 | 63.50 |
| `E0_median` | `spatial_cluster` | **66.14 +- 5.60** | 72.22 | 59.91 | 64.20 | 71.86 | 62.53 |
| `E0_ridge` | `station` | **64.11 +- 4.40** | 65.70 | 62.91 | 60.43 | 70.96 | 60.53 |
| `E0_ridge` | `spatial_cluster` | **64.56 +- 5.02** | 69.70 | 59.19 | 61.45 | 70.12 | 62.32 |
| `E1_metadata_mlp` | `station` | **43.15 +- 7.47** | 39.71 | 38.83 | 38.11 | 56.09 | 42.98 |
| `E1_metadata_mlp` | `spatial_cluster` | **46.00 +- 7.33** | 55.86 | 37.61 | 43.39 | 51.05 | 42.10 |

---

## 3. Image-Only Models — 5-Fold `station` CV (LAST-Epoch Reporting)

Source: Last-epoch rows of `results/curves/{experiment_id}_station_fold{0..4}.csv`. Skill is defined relative to `E0_median` (`station` CV `MAE = 66.21`) as $\text{Skill} = 1 - \text{MAE} / \text{MAE}_{\text{median}}$.

### 3a. Per-Fold Last-Epoch MAE & Skill vs `E0_median`

| Experiment | Description | Last-Ep MAE (Mean +- SD) | Overall Skill | Per-Fold Skill (Mean +- SD) | F0 MAE (Skill) | F1 MAE (Skill) | F2 MAE (Skill) | F3 MAE (Skill) | F4 MAE (Skill) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E2_mobilenetv3_s_pretrained` | MobileNetV3-S pretrained (Ref 1) | **45.66 +- 8.41** | **+0.3103** | +0.3120 +- 0.0957 | 36.27 (+0.460) | 46.10 (+0.260) | 42.96 (+0.326) | 59.21 (+0.203) | 43.77 (+0.311) |
| `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` | MobileNetV3-S + MD-Neck (Ref 2, author adaptation) | **44.92 +- 8.54** | **+0.3214** | +0.3229 +- 0.1009 | 34.51 (+0.486) | 45.40 (+0.271) | 43.31 (+0.320) | 58.25 (+0.216) | 43.14 (+0.321) |
| `E2_resnet18_pretrained_lr3e-4_ep40` | ResNet-18 pretrained (lr=3e-4, 40ep) | **44.05 +- 8.70** | **+0.3346** | +0.3369 +- 0.0964 | 34.85 (+0.481) | 43.35 (+0.304) | 42.06 (+0.340) | 58.45 (+0.214) | 41.55 (+0.346) |
| `E3_mdnet_t_reg_lr1e-3_ep60` | MDNet-T regression scratch (lr=1e-3, 60ep) | **53.29 +- 6.38** | **+0.1951** | +0.1961 +- 0.0572 | 48.37 (+0.280) | 50.74 (+0.186) | 53.97 (+0.153) | 64.06 (+0.138) | 49.31 (+0.223) |
| `E2_resnet18_scratch` | ResNet-18 scratch (lr=1e-3, 30ep) | **53.27 +- 7.11** | **+0.1954** | +0.1970 +- 0.0625 | 48.41 (+0.279) | 50.21 (+0.194) | 53.62 (+0.158) | 65.44 (+0.120) | 48.67 (+0.234) |

### 3b. All 10 Last-Epoch Metrics (`mean +- SD`, 5-Fold `station` CV)

| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E2_mobilenetv3_s_pretrained` | 45.66 +- 8.41 | 70.41 +- 21.71 | 0.6152 +- 0.1688 | 0.6695 +- 0.0707 | 0.3450 +- 0.2392 | 0.4044 +- 0.0284 | 0.8488 +- 0.0298 | 0.3776 +- 0.0623 | 119.16 +- 49.30 | 0.3135 +- 0.1350 |
| `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` | 44.92 +- 8.54 | 70.23 +- 21.74 | 0.6173 +- 0.1688 | 0.6780 +- 0.0636 | 0.3461 +- 0.2419 | 0.4137 +- 0.0395 | 0.8482 +- 0.0358 | 0.3789 +- 0.0649 | 118.93 +- 51.53 | 0.2913 +- 0.1183 |
| `E2_resnet18_pretrained_lr3e-4_ep40` | 44.05 +- 8.70 | 68.22 +- 22.66 | 0.6441 +- 0.1653 | 0.7060 +- 0.0578 | 0.3862 +- 0.2454 | 0.4102 +- 0.0447 | 0.8656 +- 0.0300 | 0.3751 +- 0.0804 | 116.26 +- 50.13 | 0.2822 +- 0.1707 |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 53.29 +- 6.38 | 77.16 +- 17.10 | 0.5260 +- 0.1254 | 0.5365 +- 0.0418 | 0.2226 +- 0.1629 | 0.3371 +- 0.0191 | 0.7797 +- 0.0199 | 0.3030 +- 0.0459 | 129.51 +- 41.95 | 0.2175 +- 0.1139 |
| `E2_resnet18_scratch` | 53.27 +- 7.11 | 76.77 +- 18.58 | 0.5177 +- 0.1514 | 0.5367 +- 0.0597 | 0.2314 +- 0.1835 | 0.3276 +- 0.0088 | 0.7943 +- 0.0204 | 0.2975 +- 0.0434 | 126.93 +- 52.35 | 0.2302 +- 0.1391 |

---

## 4. PFRML Ablation Ladder — 5-Fold `station` CV (All `E4` Configs + `E5_mobilenetv3_s_reg_chi`, LAST-Epoch)

Source: Last-epoch rows of `results/curves/{experiment_id}_station_fold{0..4}.csv`. Paired differences report `mean diff +- SE` (`SE = SD / sqrt(5)`, `candidate - reference`, negative = candidate lowers MAE) and `folds won / 5`.

### 4a. Per-Fold Last-Epoch MAE & Paired Differences vs `R1`, `E2`, and `R2`

| Experiment | Rung / Description | Last-Ep MAE (Mean +- SD) | Skill | Diff vs R1 (`E4_..._r1_no_prototypes`) | Won vs R1 | Diff vs E2 (`E2_mobilenetv3_s_pretrained`) | Won vs E2 | Diff vs R2 (`E4_mobilenetv3_s_baseline`) | Won vs R2 | F0 | F1 | F2 | F3 | F4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E4_mobilenetv3_s_r1_no_prototypes` | R1: DCM + Transformer + MLP (no prototypes) | **45.50 +- 8.64** | +0.3127 | +0.00 +- 0.00 | 0/5 | -0.16 +- 0.36 | 4/5 | -2.63 +- 1.10 | 4/5 | 35.71 | 47.22 | 41.94 | 59.05 | 43.59 |
| `E4_mobilenetv3_s_baseline` | R2: PFRML baseline (toggles off, chi_proto) | **48.14 +- 7.61** | +0.2729 | +2.63 +- 1.10 | 1/5 | +2.47 +- 0.95 | 1/5 | +0.00 +- 0.00 | 0/5 | 37.44 | 48.40 | 47.78 | 58.95 | 48.10 |
| `E5_mobilenetv3_s_reg_chi` | R2 (Eq. 5 literal): PFRML baseline with reg_input=chi | **48.04 +- 8.20** | +0.2743 | +2.54 +- 0.42 | 0/5 | +2.38 +- 0.46 | 0/5 | -0.09 +- 0.73 | 2/5 | 37.81 | 49.44 | 45.22 | 60.46 | 47.29 |
| `E4_mobilenetv3_s_attn_pool` | R2+T1: + attention pooling | **48.40 +- 7.74** | +0.2689 | +2.90 +- 1.01 | 1/5 | +2.74 +- 0.97 | 1/5 | +0.27 +- 0.32 | 2/5 | 37.31 | 49.80 | 47.29 | 59.05 | 48.56 |
| `E4_mobilenetv3_s_physics_gate` | R2+T2: + physics prior gate | **47.57 +- 6.73** | +0.2815 | +2.07 +- 1.00 | 1/5 | +1.91 +- 0.78 | 1/5 | -0.57 +- 0.86 | 4/5 | 40.09 | 47.34 | 46.01 | 58.51 | 45.89 |
| `E4_mobilenetv3_s_class_weights` | R2+T3: + class weights | **47.06 +- 8.01** | +0.2892 | +1.55 +- 0.73 | 1/5 | +1.40 +- 0.50 | 0/5 | -1.08 +- 0.49 | 4/5 | 36.97 | 47.20 | 45.97 | 59.38 | 45.75 |
| `E4_mobilenetv3_s_dcm_concat` | R2+T4: + DCM concat | **60.51 +- 5.41** | +0.0860 | +15.01 +- 3.38 | 0/5 | +14.85 +- 3.28 | 0/5 | +12.38 +- 3.49 | 0/5 | 62.64 | 61.11 | 58.86 | 67.36 | 52.60 |
| `E4_mobilenetv3_s_full` | R2+all: PFRML full (T1+T2+T3) | **47.16 +- 7.15** | +0.2878 | +1.65 +- 0.92 | 2/5 | +1.49 +- 0.65 | 1/5 | -0.98 +- 0.90 | 4/5 | 39.45 | 46.77 | 45.74 | 58.91 | 44.91 |
| `E4_mdnet_t_baseline_ep60` | Extra B: PFRML MDNet-T scratch (60ep) | **66.56 +- 4.88** | -0.0053 | +21.05 +- 2.97 | 0/5 | +20.90 +- 2.79 | 0/5 | +18.42 +- 2.95 | 0/5 | 67.64 | 63.23 | 63.93 | 74.66 | 63.33 |

### 4b. All 10 Last-Epoch Metrics for PFRML Ladder (`mean +- SD`, 5-Fold `station` CV)

| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E4_mobilenetv3_s_r1_no_prototypes` | 45.50 +- 8.64 | 71.44 +- 21.21 | 0.6081 +- 0.1659 | 0.6671 +- 0.0651 | 0.3249 +- 0.2375 | 0.4080 +- 0.0402 | 0.8448 +- 0.0310 | 0.3859 +- 0.0712 | 115.37 +- 53.33 | 0.3280 +- 0.1477 |
| `E4_mobilenetv3_s_baseline` | 48.14 +- 7.61 | 75.43 +- 20.35 | 0.5754 +- 0.1616 | 0.6327 +- 0.0701 | 0.2424 +- 0.2467 | 0.4088 +- 0.0297 | 0.8181 +- 0.0388 | 0.3829 +- 0.0535 | 124.19 +- 48.11 | 0.3027 +- 0.1248 |
| `E5_mobilenetv3_s_reg_chi` | 48.04 +- 8.20 | 75.14 +- 20.39 | 0.5778 +- 0.1618 | 0.6406 +- 0.0760 | 0.2538 +- 0.2375 | 0.4025 +- 0.0372 | 0.8182 +- 0.0336 | 0.3705 +- 0.0671 | 123.27 +- 50.24 | 0.2770 +- 0.1558 |
| `E4_mobilenetv3_s_attn_pool` | 48.40 +- 7.74 | 75.42 +- 20.23 | 0.5735 +- 0.1624 | 0.6352 +- 0.0729 | 0.2429 +- 0.2462 | 0.4032 +- 0.0411 | 0.8148 +- 0.0368 | 0.3789 +- 0.0734 | 121.92 +- 47.48 | 0.3131 +- 0.1623 |
| `E4_mobilenetv3_s_physics_gate` | 47.57 +- 6.73 | 74.19 +- 19.93 | 0.5863 +- 0.1508 | 0.6405 +- 0.0525 | 0.2761 +- 0.2172 | 0.4037 +- 0.0213 | 0.8243 +- 0.0247 | 0.3823 +- 0.0450 | 118.49 +- 45.76 | 0.3130 +- 0.1174 |
| `E4_mobilenetv3_s_class_weights` | 47.06 +- 8.01 | 73.67 +- 20.80 | 0.5804 +- 0.1635 | 0.6342 +- 0.0757 | 0.2822 +- 0.2371 | 0.4113 +- 0.0404 | 0.8142 +- 0.0343 | 0.3794 +- 0.0680 | 124.33 +- 47.86 | 0.2886 +- 0.1478 |
| `E4_mobilenetv3_s_dcm_concat` | 60.51 +- 5.41 | 85.22 +- 15.71 | 0.3886 +- 0.0945 | 0.4082 +- 0.0915 | 0.0556 +- 0.1127 | 0.3041 +- 0.0464 | 0.6664 +- 0.0691 | 0.2515 +- 0.0348 | 168.58 +- 34.05 | 0.1918 +- 0.0435 |
| `E4_mobilenetv3_s_full` | 47.16 +- 7.15 | 74.22 +- 18.76 | 0.5828 +- 0.1424 | 0.6398 +- 0.0551 | 0.2774 +- 0.1966 | 0.4150 +- 0.0288 | 0.8136 +- 0.0302 | 0.3861 +- 0.0562 | 121.05 +- 46.12 | 0.3062 +- 0.1314 |
| `E4_mdnet_t_baseline_ep60` | 66.56 +- 4.88 | 87.26 +- 12.45 | 0.1036 +- 0.1133 | 0.0817 +- 0.0760 | 0.0054 +- 0.0151 | 0.1489 +- 0.0401 | 0.4432 +- 0.0538 | 0.1112 +- 0.0366 | 166.28 +- 34.72 | 0.0946 +- 0.0386 |

---

## 5. Multimodal Fusion — 5-Fold `spatial_cluster` CV (`25 km` Cluster Separation, LAST-Epoch)

Source: `results/baselines_cv_results.csv` (`E0_median`, `E1_metadata_mlp`), `results/curves/E2_mobilenetv3_s_pretrained_spatial_spatial_cluster_fold{0..4}.csv`, and `results/fusion_cv_results.csv` (all `E5_fusion_*` runs and their `_meta_off` rows, evaluated at the LAST epoch).

### 5a. Per-Fold Last-Epoch MAE & Paired Differences vs `E1_metadata_mlp` (spatial) and `E2_mobilenetv3_s_pretrained_spatial`

| Experiment | Branch / Eval Mode | Last-Ep MAE (Mean +- SD) | Skill vs `E0_median` | Diff vs `E1_metadata_mlp` (Mean +- SE) | Won vs E1 | Diff vs `E2_..._spatial` (Mean +- SE) | Won vs E2 | F0 | F1 | F2 | F3 | F4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_median` | Median baseline (spatial_cluster) | **66.14 +- 5.60** | +0.0000 | +20.14 +- 1.00 | 0/5 | +15.46 +- 1.04 | 0/5 | 72.22 | 59.91 | 64.20 | 71.86 | 62.53 |
| `E1_metadata_mlp` | Metadata MLP only (spatial_cluster) | **46.00 +- 7.33** | +0.3045 | +0.00 +- 0.00 | 0/5 | -4.68 +- 1.22 | 5/5 | 55.86 | 37.61 | 43.39 | 51.05 | 42.10 |
| `E2_mobilenetv3_s_pretrained_spatial` | Image-only MobileNetV3-S (spatial_cluster) | **50.69 +- 5.60** | +0.2337 | +4.68 +- 1.22 | 0/5 | +0.00 +- 0.00 | 0/5 | 57.38 | 42.62 | 51.64 | 53.57 | 48.21 |
| `E5_fusion_r1_p03` | Vision + Metadata fusion (`normal`) | **49.18 +- 6.63** | +0.2565 | +3.18 +- 1.12 | 1/5 | -1.51 +- 0.72 | 4/5 | 58.44 | 40.93 | 50.40 | 51.04 | 45.07 |
| `E5_fusion_r1_p03_meta_off` | Vision + Metadata zeroed at eval (`metadata_off=True`) | **52.28 +- 5.95** | +0.2096 | +6.28 +- 1.23 | 0/5 | +1.60 +- 0.56 | 0/5 | 60.55 | 44.90 | 53.56 | 53.93 | 48.48 |
| `E5_fusion_full_p03` | Vision + Metadata fusion (`normal`) | **51.68 +- 6.31** | +0.2187 | +5.68 +- 1.11 | 0/5 | +1.00 +- 0.82 | 2/5 | 61.20 | 44.39 | 52.01 | 52.89 | 47.91 |
| `E5_fusion_full_p03_meta_off` | Vision + Metadata zeroed at eval (`metadata_off=True`) | **54.94 +- 5.80** | +0.1694 | +8.94 +- 1.24 | 0/5 | +4.25 +- 0.76 | 0/5 | 63.66 | 48.35 | 55.48 | 55.96 | 51.25 |
| `E5_fusion_r1_p00` | Vision + Metadata fusion (`normal`) | **47.45 +- 6.71** | +0.2827 | +1.44 +- 1.28 | 1/5 | -3.24 +- 0.79 | 5/5 | 56.85 | 38.97 | 49.12 | 48.74 | 43.55 |
| `E5_fusion_r1_p00_meta_off` | Vision + Metadata zeroed at eval (`metadata_off=True`) | **53.54 +- 5.49** | +0.1906 | +7.54 +- 1.58 | 0/5 | +2.85 +- 1.04 | 0/5 | 61.17 | 45.78 | 52.35 | 54.07 | 54.31 |
| `E5_fusion_r1_p00_ep7` | Vision + Metadata fusion (`normal`) | **44.40 +- 7.30** | +0.3288 | -1.60 +- 0.59 | 5/5 | -6.29 +- 1.07 | 5/5 | 55.09 | 35.76 | 43.09 | 47.36 | 40.69 |
| `E5_fusion_r1_p00_ep7_meta_off` | Vision + Metadata zeroed at eval (`metadata_off=True`) | **52.58 +- 5.06** | +0.2051 | +6.58 +- 1.49 | 0/5 | +1.90 +- 1.05 | 1/5 | 58.20 | 45.16 | 50.39 | 55.80 | 53.35 |

### 5b. All 10 Metrics for `spatial_cluster` CV Models (`mean +- SD`, 5 Folds)

| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_median` | 66.14 +- 5.60 | 88.00 +- 13.71 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 | -0.0168 +- 0.0142 | 0.2154 +- 0.0053 | 0.6500 +- 0.0162 | 0.0591 +- 0.0012 | 175.81 +- 29.24 | 0.0000 +- 0.0000 |
| `E1_metadata_mlp` | 46.00 +- 7.33 | 70.09 +- 20.07 | 0.6257 +- 0.1404 | 0.7147 +- 0.0694 | 0.3647 +- 0.1768 | 0.4183 +- 0.0305 | 0.8603 +- 0.0441 | 0.3556 +- 0.0386 | 118.49 +- 56.91 | 0.1953 +- 0.0730 |
| `E2_mobilenetv3_s_pretrained_spatial` | 50.69 +- 5.60 | 77.65 +- 17.70 | 0.5301 +- 0.1282 | 0.6328 +- 0.0528 | 0.2142 +- 0.1477 | 0.3739 +- 0.0284 | 0.8082 +- 0.0348 | 0.2916 +- 0.0406 | 146.79 +- 42.62 | 0.0800 +- 0.0884 |
| `E5_fusion_r1_p03` | 49.18 +- 6.63 | 75.83 +- 19.46 | 0.5741 +- 0.1648 | 0.6722 +- 0.0704 | 0.2484 +- 0.1945 | 0.3769 +- 0.0240 | 0.8245 +- 0.0307 | 0.2908 +- 0.0323 | 145.80 +- 48.31 | 0.0790 +- 0.0862 |
| `E5_fusion_r1_p03_meta_off` | 52.28 +- 5.95 | 78.89 +- 17.93 | 0.5266 +- 0.1540 | 0.6112 +- 0.0628 | 0.1839 +- 0.1736 | 0.3483 +- 0.0221 | 0.7951 +- 0.0286 | 0.2702 +- 0.0307 | 148.42 +- 45.24 | 0.0822 +- 0.0881 |
| `E5_fusion_full_p03` | 51.68 +- 6.31 | 78.71 +- 18.39 | 0.5377 +- 0.1482 | 0.6608 +- 0.0662 | 0.1883 +- 0.1782 | 0.3387 +- 0.0225 | 0.7727 +- 0.0293 | 0.2757 +- 0.0231 | 151.17 +- 45.88 | 0.1308 +- 0.0379 |
| `E5_fusion_full_p03_meta_off` | 54.94 +- 5.80 | 82.15 +- 17.03 | 0.4883 +- 0.1368 | 0.5901 +- 0.0627 | 0.1121 +- 0.1636 | 0.3387 +- 0.0225 | 0.7727 +- 0.0293 | 0.2757 +- 0.0231 | 154.99 +- 43.66 | 0.1308 +- 0.0379 |
| `E5_fusion_r1_p00` | 47.45 +- 6.71 | 73.88 +- 19.60 | 0.5993 +- 0.1562 | 0.6994 +- 0.0717 | 0.2869 +- 0.1956 | 0.3956 +- 0.0345 | 0.8358 +- 0.0217 | 0.3081 +- 0.0453 | 144.53 +- 46.80 | 0.0902 +- 0.0865 |
| `E5_fusion_r1_p00_meta_off` | 53.54 +- 5.49 | 79.81 +- 17.32 | 0.5247 +- 0.1437 | 0.6041 +- 0.0662 | 0.1624 +- 0.1754 | 0.3335 +- 0.0227 | 0.7860 +- 0.0153 | 0.2422 +- 0.0331 | 158.23 +- 36.07 | 0.0519 +- 0.0879 |
| `E5_fusion_r1_p00_ep7` | 44.40 +- 7.30 | 69.68 +- 20.97 | 0.6395 +- 0.1690 | 0.7330 +- 0.0736 | 0.3674 +- 0.2127 | 0.4318 +- 0.0365 | 0.8640 +- 0.0300 | 0.3501 +- 0.0470 | 131.61 +- 53.52 | 0.1359 +- 0.0927 |
| `E5_fusion_r1_p00_ep7_meta_off` | 52.58 +- 5.06 | 78.16 +- 16.75 | 0.5470 +- 0.1553 | 0.6240 +- 0.0671 | 0.1988 +- 0.1595 | 0.3404 +- 0.0321 | 0.7936 +- 0.0345 | 0.2521 +- 0.0494 | 152.01 +- 36.97 | 0.0652 +- 0.0938 |

---

## 6. Final One-Shot Test Results (`test` and `test_gt25km` `>25 km`)

Source: `results/final_test_results.csv`, `results/predictions/*.csv`, `results/predictions/E1_metadata_mlp_preds.npz`, and `results/baselines_cv_results.csv`. Station-level cluster bootstrap uses `1000` resamples with `seed=42`.

### 6a. All Final Models — 10 Metrics & 95% Station-Bootstrap MAE CI

| Name | Model | Split | n | Stations | MAE (95% Station CI) | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
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

### 6b. Paired Station-Bootstrap MAE Differences — Full Test Set (`Model A - Model B`, 95% CI)

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

### 6c. Paired Station-Bootstrap MAE Differences — `test_gt25km` Subset (`>25 km`, `Model A - Model B`, 95% CI)

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

---

## 7. Failed or Unstable Runs

Detailed per-epoch loss trajectories and root-cause analysis are documented in [results/collapse.md](collapse.md).

| Experiment | Mode | Folds | Last-Ep MAE | Last-Ep R2 | Ep 1 `train_loss` (Mean) | Last Ep `train_loss` (Mean) | Per-Fold Last-Ep MAE | Diagnostic Reference & Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E4_mdnet_t_baseline_ep60` | `station` | 5/5 | **66.56 +- 4.88** | 0.0054 +- 0.0151 | 2.1609 | 2.1784 | F0=67.64, F1=63.23, F2=63.93, F3=74.66, F4=63.33 | From-scratch `pfrml_mdnet_t` under hybrid loss (`lr=1e-3`, 60 ep): optimization stalls at near-constant prediction (`last R2 ~ 0` across all folds). See [results/collapse.md](collapse.md). |
| `E5_mdnet_t_pfrml_lr2e-4_f0` | `station` | 1/5 | **58.15** | 0.2910 | 2.2084 | 1.9243 | F0=58.15 | From-scratch `pfrml_mdnet_t` with lower `lr=2e-4` (Fold 0 probe): avoids total collapse but remains substantially worse than regression-only `E3_mdnet_t_reg_lr1e-3_ep60`. See [results/collapse.md](collapse.md). |
| `E4_mobilenetv3_s_dcm_concat` | `station` | 5/5 | **60.51 +- 5.41** | 0.0556 +- 0.1127 | 2.0567 | 1.9196 | F0=62.64, F1=61.11, F2=58.86, F3=67.36, F4=52.60 | DCM concatenation variant: severe optimization instability / underfitting (`train_loss` fails to halve across folds). See [results/collapse.md](collapse.md). |

---

## 8. "Do Not Quote" List (Superseded Best-Epoch Numbers, Missing-Curve IDs & Single-Fold Screens)

1. **Every Best-Epoch Column / Note (`results/image_models_cv_results.csv`, `results/pfrml_ablation_cv_results.csv`, and `best_ep_mae` in `results/fusion_cv_results.csv`):**
   - All image-model and PFRML-ablation CSV rows in `image_models_cv_results.csv` and `pfrml_ablation_cv_results.csv` recorded the minimum-`val_loss` checkpoint (`best_ep`) rather than the final epoch (`last_ep`), and `fusion_cv_results.csv` records `best_ep_mae=...` inside its `notes` column for diagnostic reference only. **Do not quote any best-epoch validation metrics as main results; always quote the last-epoch metrics in Sections 3–5 above.**

| Experiment | Mode | CSV Source | Do-Not-Quote Best-Ep MAE | Authoritative Last-Ep Curve MAE (`Mean +- SD`) |
| :--- | :---: | :--- | :---: | :---: |
| `E2_mobilenetv3_s_pretrained` | `station` | `results/image_models_cv_results.csv` | ~~45.54 +- 10.06~~ | **45.66 +- 8.41** |
| `E2_resnet18_pretrained_lr3e-4_ep40` | `station` | `results/image_models_cv_results.csv` | ~~46.16 +- 13.85~~ | **44.05 +- 8.70** |
| `E2_resnet18_scratch` | `station` | `results/image_models_cv_results.csv` | ~~53.50 +- 7.70~~ | **53.27 +- 7.11** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | `station` | `results/image_models_cv_results.csv` | ~~53.12 +- 6.68~~ | **53.29 +- 6.38** |
| `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` | `station` | `results/image_models_cv_results.csv` | ~~45.85 +- 9.59~~ | **44.92 +- 8.54** |
| `E4_mobilenetv3_s_baseline` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~48.85 +- 5.84~~ | **48.14 +- 7.61** |
| `E4_mobilenetv3_s_r1_no_prototypes` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~45.29 +- 8.66~~ | **45.50 +- 8.64** |
| `E4_mobilenetv3_s_full` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~49.43 +- 6.86~~ | **47.16 +- 7.15** |
| `E4_mobilenetv3_s_attn_pool` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~50.38 +- 5.65~~ | **48.40 +- 7.74** |
| `E4_mobilenetv3_s_physics_gate` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~49.70 +- 5.92~~ | **47.57 +- 6.73** |
| `E4_mobilenetv3_s_class_weights` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~51.03 +- 8.16~~ | **47.06 +- 8.01** |
| `E4_mobilenetv3_s_dcm_concat` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~55.63 +- 3.67~~ | **60.51 +- 5.41** |
| `E2_mobilenetv3_s_pretrained_spatial` | `spatial_cluster` | `results/pfrml_ablation_cv_results.csv` | ~~51.04 +- 5.73~~ | **50.69 +- 5.60** |
| `E4_mdnet_t_baseline_ep60` | `station` | `results/pfrml_ablation_cv_results.csv` | ~~66.46 +- 5.22~~ | **66.56 +- 4.88** |
| `E5_fusion_r1_p03` (`notes: best_ep_mae`) | `spatial_cluster` | `results/fusion_cv_results.csv` | ~~44.55 +- 7.27~~ | **49.18 +- 6.63** |
| `E5_fusion_full_p03` (`notes: best_ep_mae`) | `spatial_cluster` | `results/fusion_cv_results.csv` | ~~51.47 +- 10.41~~ | **51.68 +- 6.31** |
| `E5_fusion_r1_p00` (`notes: best_ep_mae`) | `spatial_cluster` | `results/fusion_cv_results.csv` | ~~42.86 +- 8.86~~ | **47.45 +- 6.71** |
| `E5_mobilenetv3_s_reg_chi` (`notes: best_ep_mae`) | `station` | `results/fusion_cv_results.csv` | ~~51.18 +- 4.97~~ | **48.04 +- 8.20** |
| `E5_fusion_r1_p00_ep7` (`notes: best_ep_mae`) | `spatial_cluster` | `results/fusion_cv_results.csv` | ~~44.11 +- 7.75~~ | **44.40 +- 7.30** |

2. **Superseded `results/pfrml_best_epoch_summary_superseded.md` Best-Epoch Tables:**
   - `results/pfrml_best_epoch_summary_superseded.md` (legacy best-epoch table derived from `pfrml_ablation_cv_results.csv`) reports minimum-`val_loss` (`best_ep`) numbers and Wilcoxon $p$-values at $n=5$. **Do not quote `pfrml_best_epoch_summary_superseded.md`**; use `outputs/results/last_epoch_summary.md` and Section 4 of this document.

3. **Experiment IDs with NO Curve File in `results/curves/` (or Partial `<5` Fold Curve Sets):**
   - `E0_median`: Closed-form / tabular baseline in `results/baselines_cv_results.csv` (no image-epoch curve file in `results/curves/`; quote only from Section 2 / Section 6).
   - `E0_ridge`: Closed-form / tabular baseline in `results/baselines_cv_results.csv` (no image-epoch curve file in `results/curves/`; quote only from Section 2 / Section 6).
   - `E1_metadata_mlp`: Closed-form / tabular baseline in `results/baselines_cv_results.csv` (no image-epoch curve file in `results/curves/`; quote only from Section 2 / Section 6).
   - `E4_mobilenetv3_s_detached_proto`: **0 curve files in `results/curves/` and 0 CSV rows** (never run — do not quote).
   - `E4_mobilenetv3_s_ema099`: **0 curve files in `results/curves/` and 0 CSV rows** (never run — do not quote).
   - `E5_fusion_full_p03_meta_off`: Evaluation-only metadata-zeroed pass evaluated on the parent checkpoint at the end of training (no separate training curve file; quote from `results/fusion_cv_results.csv` / Section 5).
   - `E5_fusion_r1_p00_ep7_meta_off`: Evaluation-only metadata-zeroed pass evaluated on the parent checkpoint at the end of training (no separate training curve file; quote from `results/fusion_cv_results.csv` / Section 5).
   - `E5_fusion_r1_p00_meta_off`: Evaluation-only metadata-zeroed pass evaluated on the parent checkpoint at the end of training (no separate training curve file; quote from `results/fusion_cv_results.csv` / Section 5).
   - `E5_fusion_r1_p03_meta_off`: Evaluation-only metadata-zeroed pass evaluated on the parent checkpoint at the end of training (no separate training curve file; quote from `results/fusion_cv_results.csv` / Section 5).
   - **Single-fold / partial (`<5` folds) curve sets in `results/curves/` (do not quote as 5-fold CV results):**
     - `E2_mobilenetv3_s_pretrained_lr3e-4_ep40` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 40.15`)
     - `E2_resnet18_pretrained` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 40.28`)
     - `E2_resnet18_scratch_lr1e-3_ep60` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 48.48`)
     - `E3_mdnet_t_cls` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 52.50`)
     - `E3_mdnet_t_reg` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 51.58`)
     - `E3b_mobilenetv3_s_md_neck_lr3e-4_ep40` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 35.66`)
     - `E5_mdnet_t_pfrml_lr2e-4_f0` (`station`, folds `[0]`, `1/5` folds, last-epoch `MAE = 58.15`)
