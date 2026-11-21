# Experiment Registry (`docs/EXPERIMENTS.md`)

Authoritative catalogue of every experiment ID recorded in `results/*.csv` and `results/curves/*.csv`, cross-referenced to its config, CV protocol, and section in [results/REPORT_NUMBERS.md](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/results/REPORT_NUMBERS.md).

---

## 1. Master Experiment Table

| Legacy ID | Readable Name | What It Tests | Model | Config File | CV Mode / Split | Result File | `REPORT_NUMBERS.md` Section |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `E0_median` | Constant Median Predictor | Fold training-median AQI baseline (trivial reference) | `median` | `configs/default.yaml` | `station`, `spatial_cluster`, `full_train` | `baselines_cv_results.csv` | Sec. 2, Sec. 5 |
| `E0_ridge` | Haze-Statistics Ridge Regression | 18 interpretable atmospheric haze/contrast features + Ridge | `ridge_haze` | `configs/default.yaml` | `station`, `spatial_cluster`, `full_train` | `baselines_cv_results.csv` | Sec. 2 |
| `E1_metadata_mlp` | GPS + Month Metadata MLP | Non-visual geospatial prior (`[x, y, z, sin(m), cos(m)] -> 64 -> 32 -> 1`, 40 ep) | `metadata_mlp` | `configs/default.yaml` | `station`, `spatial_cluster`, `full_train` | `baselines_cv_results.csv` + `E1_metadata_mlp_preds.npz` | Sec. 2, Sec. 5, Sec. 6 |
| `E2_mobilenetv3_s_pretrained` | MobileNetV3-S Pretrained (Ref 1) | ImageNet-pretrained CNN linear regression baseline (`lr=1e-3`, 30 ep) | `mobilenetv3_s_pretrained` | `configs/default.yaml` | `station` (5-fold) | `image_models_cv_results.csv` / `curves/` | Sec. 3, Sec. 4, Sec. 8 |
| `E2_mobilenetv3_s_pretrained_lr3e-4_ep40` | MobileNetV3-S Pretrained Screen | Fold-0 LR/epoch screen (`lr=3e-4`, 40 ep; not promoted) | `mobilenetv3_s_pretrained` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E2_resnet18_pretrained` | ResNet-18 Pretrained Default | Fold-0 screen at default `lr=1e-3`, 30 ep | `resnet18_pretrained` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E2_resnet18_pretrained_lr3e-4_ep40` | ResNet-18 Pretrained Tuned | Best vision-only backbone (`lr=3e-4`, 40 ep) | `resnet18_pretrained` | `configs/default.yaml` | `station` (5-fold) | `image_models_cv_results.csv` / `curves/` | Sec. 3, Sec. 8 |
| `E2_resnet18_scratch` | ResNet-18 Scratch (30 ep) | Scratch-init ResNet-18 (`lr=1e-3`, 30 ep) | `resnet18_scratch` | `configs/default.yaml` | `station` (5-fold) | `image_models_cv_results.csv` / `curves/` | Sec. 3, Sec. 8 |
| `E2_resnet18_scratch_lr1e-3_ep60` | ResNet-18 Scratch 60-ep Screen | Fold-0 60-ep screen (not promoted) | `resnet18_scratch` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E3_mdnet_t_reg` | MDNet-T Scratch 30-ep Screen | Fold-0 screen for Paper 2 MDNet-T regression (`lr=1e-3`, 30 ep) | `mdnet_t_reg` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E3_mdnet_t_cls` | MDNet-T 6-Class Classification | Fold-0 classification-only screen (`lr=1e-3`, 30 ep) | `mdnet_t_cls` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E3_mdnet_t_reg_lr1e-3_ep60` | MDNet-T Regression Scratch (60 ep) | Full 5-fold Paper 2 MDNet-T regression baseline (`lr=1e-3`, 60 ep) | `mdnet_t_reg` | `configs/default.yaml` | `station` (5-fold) | `image_models_cv_results.csv` / `curves/` | Sec. 3, Sec. 7, Sec. 8 |
| `E3b_mobilenetv3_s_md_neck_lr3e-4_ep40` | MobileNetV3-S + MD-Neck Screen | Fold-0 screen (`lr=3e-4`, 40 ep) | `mobilenetv3_s_md_neck` | `configs/default.yaml` | `station` (Fold 0 only) | `image_models_cv_results.csv` / `curves/` | Sec. 8 *(1/5 folds; do not quote as CV)* |
| `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` | MobileNetV3-S + MD-Neck (Ref 2) | Author adaptation of `MDBlock(128)` neck on MobileNetV3-S (`lr=1e-3`, 40 ep) | `mobilenetv3_s_md_neck` | `configs/default.yaml` | `station` (5-fold) | `image_models_cv_results.csv` / `curves/` | Sec. 3, Sec. 8 |
| `E4_mobilenetv3_s_r1_no_prototypes` | R1: DCM + Transformer (No Prototypes) | PFRML vision spine without prototype classification head (MSE only) | `pfrml_mobilenetv3_s` | `configs/pfrml_regression_only.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E4_mobilenetv3_s_baseline` | R2: PFRML Baseline (Toggles Off) | Core PFRML (`chi_proto` reg + 6 EMA prototypes, toggles off) | `pfrml_mobilenetv3_s` | `configs/pfrml_prototype_multitask.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 7, Sec. 8 |
| `E5_mobilenetv3_s_reg_chi` | R2 (Eq. 5 Literal: `reg_input=chi`) | PFRML with regression MLP reading `chi` only (while prototype `L_cls` stays active) | `pfrml_mobilenetv3_s` | `configs/pfrml_eq5_regression.yaml` | `station` (5-fold) | `fusion_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E4_mobilenetv3_s_attn_pool` | R2 + T1: Attention Pooling | Adds learnable spatial softmax pooling over 196 tokens | `pfrml_mobilenetv3_s` | `configs/pfrml_attention_pooling.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E4_mobilenetv3_s_physics_gate` | R2 + T2: Physics Prior Gate | Adds 4-scalar atmospheric physics sigmoid gate | `pfrml_mobilenetv3_s` | `configs/pfrml_physics_gate.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E4_mobilenetv3_s_class_weights` | R2 + T3: Class Weights | Adds fold-internal inverse-frequency CE class weights (`gamma=0.5`) | `pfrml_mobilenetv3_s` | `configs/pfrml_class_weights.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E4_mobilenetv3_s_dcm_concat` | R2 + T4: DCM Concat **(FAILED)** | Replaces DCM element-wise sum with `concat + LayerNorm2d` (stalls training) | `pfrml_mobilenetv3_s` | `configs/pfrml_dcm_concat.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 7, Sec. 8 |
| `E4_mobilenetv3_s_full` | R2 + All Approved Toggles (T1+T2+T3) | Full PFRML visual model (attention pooling + physics gate + class weights, sum DCM) | `pfrml_mobilenetv3_s` | `configs/pfrml_all_toggles.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 8 |
| `E2_mobilenetv3_s_pretrained_spatial` | Extra A: E2 Spatial-Cluster Reference | Image-only MobileNetV3-S under 25 km spatial cluster CV | `mobilenetv3_s_pretrained` | `configs/default.yaml` | `spatial_cluster` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 5, Sec. 8 |
| `E4_mdnet_t_baseline_ep60` | Extra B: MDNet-T PFRML **(COLLAPSED)** | From-scratch MDNet-T + PFRML hybrid loss (`lr=1e-3`, 60 ep; collapses to constant prediction) | `pfrml_mdnet_t` | `configs/pfrml_prototype_multitask.yaml` | `station` (5-fold) | `pfrml_ablation_cv_results.csv` / `curves/` | Sec. 4, Sec. 7, Sec. 8 |
| `E5_mdnet_t_pfrml_lr2e-4_f0` | MDNet-T PFRML Lower-LR Probe **(UNSTABLE)** | Guarded Fold-0 probe with `lr=2e-4`, 30 ep | `pfrml_mdnet_t` | `configs/pfrml_prototype_multitask.yaml` | `station` (Fold 0 only) | `fusion_cv_results.csv` / `curves/` | Sec. 7, Sec. 8 |
| `E5_fusion_r1_p03` / `_meta_off` | Fusion R1 (`p_drop=0.3`, 30 ep) | R1 vision neck + `MetadataMLP`, modality dropout `0.3`, + zeroed-meta eval | `fusion_r1_mobilenetv3_s` | `configs/fusion_regression_only_neck.yaml` | `spatial_cluster` (5-fold) | `fusion_cv_results.csv` / `curves/` | Sec. 5, Sec. 8 |
| `E5_fusion_full_p03` / `_meta_off` | Fusion PFRML Full (`p_drop=0.3`, 30 ep) | Full PFRML vision neck + `MetadataMLP`, modality dropout `0.3`, + zeroed-meta eval | `fusion_full_mobilenetv3_s` | `configs/fusion_pfrml_full.yaml` | `spatial_cluster` (5-fold) | `fusion_cv_results.csv` / `curves/` | Sec. 5, Sec. 8 |
| `E5_fusion_r1_p00` / `_meta_off` | Fusion R1 (`p_drop=0.0`, 30 ep) | R1 vision neck + `MetadataMLP` without modality dropout + zeroed-meta eval | `fusion_r1_mobilenetv3_s` | `configs/fusion_regression_only_neck.yaml` | `spatial_cluster` (5-fold) | `fusion_cv_results.csv` / `curves/` | Sec. 5, Sec. 8 |
| `E5_fusion_r1_p00_ep7` / `_meta_off` | Fusion R1 (`p_drop=0.0`, CMA-3 `ep=7`) | Selected early-stopped schedule (`N=7` via CMA-3) before metadata overfitting | `fusion_r1_mobilenetv3_s` | `configs/fusion_regression_only_neck.yaml` | `spatial_cluster` (5-fold) | `fusion_cv_results.csv` / `curves/` | Sec. 5, Sec. 8 |
| `F_E2` | Final Image-Only Baseline | Full `8,176`-image train (`30` ep), one-shot test eval | `mobilenetv3_s_pretrained` | `configs/default.yaml` | `full_train` -> `test`, `test_gt25km` | `final_test_results.csv` + `predictions/F_E2.csv` | Sec. 6 |
| `F_PFRML_full` | Final Full PFRML Vision Model | Full `8,176`-image train (`30` ep), one-shot test eval | `pfrml_mobilenetv3_s` | `configs/pfrml_all_toggles.yaml` | `full_train` -> `test`, `test_gt25km` | `final_test_results.csv` + `predictions/F_PFRML_full.csv` | Sec. 6 |
| `F_FUSION_R1` | Final Winning Fusion Model (`ep=7`) | Full `8,176`-image train (`p_drop=0.0`, `7` ep), one-shot test eval | `fusion_r1_mobilenetv3_s` | `configs/fusion_regression_only_neck.yaml` | `full_train` -> `test`, `test_gt25km` | `final_test_results.csv` + `predictions/F_FUSION_R1.csv` | Sec. 6 |
| `F_FUSION_R1_meta_off` | Final Fusion (Metadata Zeroed at Eval) | Vision-only branch evaluation of `F_FUSION_R1` (`metadata_off=True`) | `fusion_r1_mobilenetv3_s` | `configs/fusion_regression_only_neck.yaml` | `full_train` -> `test`, `test_gt25km` | `final_test_results.csv` + `predictions/F_FUSION_R1_meta_off.csv` | Sec. 6 |
| `E4_mobilenetv3_s_detached_proto`, `E4_mobilenetv3_s_ema099` | Unrun Candidate IDs | Mentioned in early notes; **never executed** (`0` curves, `0` CSV rows) | — | — | — | — | Sec. 8 *(Never run — do not quote)* |

---

## 2. Failed Runs & "Do Not Quote" Rules

> [!WARNING]
> Violating either rule below introduces invalidated or collapsed metrics into reports.

1. **Failed / Collapsed Runs (see `results/REPORT_NUMBERS.md` Section 7 & `results/collapse.md`):**
   - `E4_mdnet_t_baseline_ep60` (`66.56 +- 4.88` MAE, `last R2 = 0.0054 +- 0.0151`): From-scratch `pfrml_mdnet_t` under hybrid loss stalls across all 5 folds (`train_loss` stays at `2.16 -> 2.18`), matching the constant median baseline (`66.21`).
   - `E4_mobilenetv3_s_dcm_concat` (`60.51 +- 5.41` MAE, `last R2 = 0.0556 +- 0.1127`): Concatenation + `LayerNorm2d` in DCM fails to reduce `train_loss` below `0.5x` of epoch 1 (`2.06 -> 1.92`).
   - `E5_mdnet_t_pfrml_lr2e-4_f0` (`58.15` MAE on Fold 0): Lower-LR single-fold probe avoids total collapse but still underperforms regression-only `E3_mdnet_t_reg_lr1e-3_ep60` (`48.37` on Fold 0).
2. **Never Quote Best-Epoch CV Numbers (see `results/REPORT_NUMBERS.md` Section 8):**
   - Every row in `image_models_cv_results.csv` (legacy `step3_results.csv`) and `pfrml_ablation_cv_results.csv` (legacy `step4_results.csv`), as well as `best_ep_mae=...` inside `fusion_cv_results.csv` (legacy `step5_results.csv`), recorded the minimum-`val_loss` checkpoint (`best_ep`), which introduces validation leakage and cross-entropy overfitting mismatch.
   - **Always quote last-epoch metrics from `results/REPORT_NUMBERS.md` Sections 3–5** (computed directly from the final row of each fold's curve CSV). Do not quote `pfrml_best_epoch_summary_superseded.md` (legacy `step4_summary.md`).
3. **Never Quote Single-Fold Screens as 5-Fold CV Results:**
   - The `6` single-fold Fold-0 screens (`E2_mobilenetv3_s_pretrained_lr3e-4_ep40`, `E2_resnet18_pretrained`, `E2_resnet18_scratch_lr1e-3_ep60`, `E3_mdnet_t_reg`, `E3_mdnet_t_cls`, `E3b_mobilenetv3_s_md_neck_lr3e-4_ep40`) plus `E5_mdnet_t_pfrml_lr2e-4_f0` have only `1/5` folds in `curves/`.

---

## 3. Legacy Development Step Mapping

Historical entries in [docs/development_log.md](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/development_log.md) use chronological stage labels (`Step 1`–`Step 5c`). This table maps them to the cleaned repository filenames:

| Legacy Name | Current Meaningful Path / Artifact |
| :--- | :--- |
| Step 1 (Data Cleaning & Splits) | `src/data/{cleaning.py,dataset.py,splits.py,transforms.py}`, `tests/test_data_pipeline.py` |
| Step 2 (Baselines & Metrics) | `src/models/baselines.py`, `results/baselines_cv_results.csv` (was `step2_results.csv`), `tests/test_metrics_losses_and_baselines.py` |
| Step 3 (Image Backbones & MDNet-T Queue) | `scripts/experiments/queue_baselines.py` (was `scripts/run_queue.py`), `results/image_models_cv_results.csv` (was `step3_results.csv`), `tests/test_mdnet_and_image_baselines.py`, `tests/test_image_queue_and_md_neck.py` |
| Step 4a / 4b (PFRML Head & Ablation Queue) | `scripts/experiments/queue_pfrml_ablations.py` (was `scripts/run_queue_step4.py`), `results/pfrml_ablation_cv_results.csv` (was `step4_results.csv`), `tests/test_pfrml_head.py` |
| Step 5a (Last-Epoch Reporting, `reg_input='chi'`, Fusion) | `scripts/analysis/summarize_last_epoch.py` (was `summarize_last_epoch_v2.py`), `scripts/analysis/diagnose_training_failures.py` (was `diagnose_collapse.py`), `src/models/fusion.py`, `tests/test_fusion_and_reporting.py` |
| Step 5b (Fusion CV Queue & Final Evaluation) | `scripts/experiments/queue_fusion_cv.py` (was `scripts/run_queue_step5.py`), `scripts/final_evaluation.py` (was `scripts/final_train_eval.py`), `scripts/analysis/compare_final_models.py` (was `scripts/compare_final.py`), `results/fusion_cv_results.csv` (was `step5_results.csv`), `results/final_test_results.csv` (was `final_results.csv`), `tests/test_final_evaluation.py` |
| Step 5c (CMA-3 Epoch Selection, Final Fusion, Master Numbers) | `scripts/analysis/select_fusion_epochs.py` (was `scripts/choose_epochs.py`), `scripts/experiments/queue_fusion_final.py` (was `scripts/run_queue_step5c.py`), `scripts/analysis/build_report_numbers.py`, `results/fusion_epoch_selection.md` (was `step5c_epochs.md`), `results/fusion_epoch_decision.md` (was `step5c_decision.md`), `tests/test_epoch_selection_and_report_numbers.py` |

---

## 4. Recorded Training Environment & Dataset Revision

- **Hardware / OS:** Apple M5 Pro, 24 GB unified memory | `macOS-27.2.1-arm64-arm-64bit-Mach-O`
- **Software:** Python `3.14.6` | PyTorch `2.14.1` (`mps` backend available) | Fixed RNG seed `42`
- **Dataset Source:** Hugging Face Hub `DeadCardassian/PM25Vision` — cached snapshot revision `578bc7c17c99be1aa312ebfe0bda6ce40067ca31`
