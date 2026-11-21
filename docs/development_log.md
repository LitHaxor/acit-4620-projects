# Project Progress Log

## Step 1: Data Pipeline & Fix-Ups — COMPLETED (2026-11-12)

### Files Built & Updated
* [`configs/default.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/default.yaml) & [`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py): Minimal typed YAML configuration loader with override support.
* [`src/data/cleaning.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/cleaning.py): Purges the 18 overlapping train/test `image_id`s from train and deduplicates multi-station `image_id`s via median AQI (`pm25`), updating `aqi_bin` ($0\dots 5$) and `pm25_bin` (`0–50` $\dots$ `301–600`).
* [`src/data/splits.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/splits.py): 5-fold `StratifiedGroupKFold` stratified by `aqi_bin` and grouped by `group_id` (`2,527` groups in `station` mode; `1,059` groups in `spatial_cluster` mode using single-linkage Haversine `AgglomerativeClustering` at `25.0 km`). Includes `compute_val_to_train_nearest_km` for per-fold spatial separation verification.
* [`src/data/transforms.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/transforms.py): Whole-image resize to $224 \times 224$ (no cropping), ImageNet normalization, random horizontal flip ($p=0.5$), and gentle affine transforms ($\pm 5^\circ$ rotation, $\pm 2\%$ translation/scale). Strictly no colour jitter or grayscale.
* [`src/data/dataset.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/dataset.py): Lazy-loading `PM25Dataset` returning `PM25Sample(image, metadata, aqi, aqi_bin, station_id)`.
* [`tests/test_data_pipeline.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_data_pipeline.py): 4 pytest tests covering clean counts, station/spatial-cluster group counts (`2,527` vs. `1,059`), zero cluster overlap + Haversine separation (`test_spatial_cluster_no_overlap_and_haversine_separation`), and dataset tensor shapes/ranges (`4 passed in 3.61s`).
* [`docs/AGENT_RULES.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/AGENT_RULES.md) & [`docs/SPEC.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/SPEC.md): Updated with the `Idea sources` rules (PFRML + MDNet spine, 17-paper toolbox via `docs/related_work_notes.md`, config-toggled ablations with citations).

### Verified Acceptance & Fix-Up Metrics
* **Clean counts:** `11,096` total images across **3,259 unique audited stations** after overlap purge (`2,530` train + `729` test; `2,527` primary train stations + `729` test = `3,256` retained post-dedup; `8,176` train images, `2,920` test images; raw CSVs have `3,261` = `2,532` + `729`).
* **Mode 1 — `station` (`n_groups = 2,527`):**
  * `station_overlaps = [0, 0, 0, 0, 0]` (note: `spatial_cluster_overlaps = [182, 186, 192, 186, 172]`).
  * Per-fold sizes (mean `1635.2`): Fold 0: `1654 (+1.1%)`, Fold 1: `1624 (-0.7%)`, Fold 2: `1627 (-0.5%)`, Fold 3: `1646 (+0.7%)`, Fold 4: `1625 (-0.6%)`.
  * Val-to-nearest-train Haversine distance (km): `min_km = [0.00, 0.00, 0.00, 0.00, 0.01]`, `median_km = [5.62, 8.00, 7.31, 10.67, 8.53]`.
  * All per-fold total counts and per-bin counts stay within `<30%` of their mean (max bin deviation is Bin 5 in Fold 0 at `+28.2%`).
* **Mode 2 — `spatial_cluster` (`n_groups = 1,059`, `AgglomerativeClustering(metric='precomputed', linkage='single', distance_threshold=25.0)`):**
  * `station_overlaps = [0, 0, 0, 0, 0]` AND `spatial_cluster_overlaps = [0, 0, 0, 0, 0]`.
  * Per-fold sizes (mean `1635.2`): Fold 0: `1637 (+0.1%)`, Fold 1: `1595 (-2.5%)`, Fold 2: `1648 (+0.8%)`, Fold 3: `1695 (+3.7%)`, Fold 4: `1601 (-2.1%)` — all total fold sizes within `3.7%` of mean.
  * Val-to-nearest-train Haversine distance (km): `min_km = [24.78, 24.78, 24.51, 24.51, 24.61]`, `median_km = [75.62, 75.23, 65.58, 93.11, 65.79]`.
  * **Per-bin skew in Bin 5 (`AQI > 300`, mean `71.0`/fold):** Bins 0–4 are tightly balanced (`<= 6.1%` deviation), but Bin 5 has `[71, 33, 80, 131, 40]` (`+84.5%` in Fold 3, `-53.5%` in Fold 1, `-43.7%` in Fold 4) because $80\%$ of all training Hazardous images ($282 / 355$) are concentrated in 3 city clusters: `cluster 22` (Beijing, 131 Bin-5 images, Fold 3), `cluster 95` (Delhi/NCR, 80 Bin-5 images, Fold 2), and `cluster 576` (Saltillo/Monterrey, 71 Bin-5 images, Fold 0).
* **Image Cache Decision:** Deferred to Step 3 per user instruction (`240.2 images/sec`, `34.04s/epoch` raw JPEG baseline).

---

## Step 2: Metrics, Losses, Haze Features, CLI & Baselines (E0, E1) — COMPLETED (2026-11-12)

### Docs Updates (Part A)
* [`docs/SPEC.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/SPEC.md): Added approved toggled ablations (physics-prior scalars + sigmoid gate, learnable token attention pooling, long-tail class weighting $w_k \propto N_k^{-\gamma}$), parked ideas (ordinal regulariser, geolocation encoding, ParNet attention), free diagnostics ($\pm 1$-class accuracy, paired Wilcoxon test, prototype-mixture variance), clarified PFRML's prototype/regression head (**one single MLP on a prototype-weighted feature**, citing Sec. II-D Eq. 5, Fig. 5, Sec. III-A3, and noting the ambiguity with Sec. III-C3 Fig. 7), and updated station counts to **3,259 unique**.
* [`docs/related_work_notes.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/related_work_notes.md): Fixed global station count to `3,259 unique`.

### Code Built & Tested (Part B)
* [`src/metrics.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/metrics.py): `MAE`, `RMSE`, `PLCC`, `SRCC`, `R2`, `6-class acc`, `+-1-class acc`, `macro-F1`, tail `MAE`/`F1` (bins 5–6, i.e., 0-indexed `{4, 5}`), `summarize_fold_metrics`, and `paired_wilcoxon_test`.
* [`src/losses.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/losses.py): `LabelSmoothingCrossEntropy` ($\epsilon=0.1$), `RegressionLoss` (`mse` / `huber` on `AQI/100`), `HybridMultitaskLoss` ($L_{\text{cls}} + \beta L_{\text{reg}}$, $\beta=0.5$), and `compute_class_weights` ($w_k \propto N_k^{-\gamma}$).
* [`src/features/haze_features.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/features/haze_features.py): `HazeStatisticsExtractor` (static, differentiable PyTorch ops on pre-norm `[0, 1]` pixels for HSV saturation, Sobel gradient contrast, dark channel across `full`/`top`/`bottom` bands $\to$ 18-D; plus `compute_physics_priors` `[entropy, rms_contrast, transmission, extinction]` for Step 4 reuse).
* [`src/models/baselines.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/baselines.py): `MedianBaseline`, `HazeRidgeBaseline`, and `MetadataMLP` (`5 -> 64 -> 32 -> 1`).
* [`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py) & [`configs/default.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/default.yaml): Extended with `ModelConfig`, `LossConfig`, and `TrainConfig`.
* [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py) & [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py): Generic experiment CLI (`--config`, `--experiment-id`, `--model`, `--seed`, `--group-mode`, `--fold`, `--output-csv`) appending CV, full test, and `test > 25 km` subset metrics to [`outputs/results/step2_results.csv`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step2_results.csv).
* [`tests/test_step2.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step2.py): 5 new unit tests (`9 passed` total across `test_data_pipeline.py` and `test_step2.py`).

### Verified Baseline Results (`outputs/results/step2_results.csv`)
* **E0 (`median`):** Station CV `MAE=66.21+-4.89, R2=-0.0161`; Spatial-cluster CV `MAE=66.14+-5.60, R2=-0.0168`; Test `MAE=63.64, R2=-0.0105`; Test >25km `MAE=56.60, R2=-0.0049`.
* **E0 (`ridge_haze`):** Station CV `MAE=64.11+-4.40, PLCC=0.1998+-0.1083, R2=0.0301+-0.0583`; Spatial-cluster CV `MAE=64.56+-5.02, PLCC=0.1526+-0.0937, R2=-0.0018+-0.0503`; Test `MAE=60.84, PLCC=0.2112, R2=0.0408`; Test >25km `MAE=53.76, PLCC=0.1435, R2=-0.0377`.
* **E1 (`metadata_mlp`):** Station CV `MAE=43.15+-7.47, RMSE=65.43+-18.12, PLCC=0.6704+-0.1236, R2=0.4438+-0.1626, Acc=0.4234+-0.0329`; Spatial-cluster CV `MAE=46.00+-7.33, RMSE=70.09+-20.07, PLCC=0.6257+-0.1404, R2=0.3647+-0.1768, Acc=0.4183+-0.0305`; Full Test `MAE=40.34, R2=0.4982`; Test >25km (`n=717`) collapses to `MAE=49.84, PLCC=0.3797, R2=-0.0393, Acc=0.2929, TailF1=0.0000`.

---

## Step 3: MDNet-T Backbone & Image-Only Baselines (E2, E3) — COMPLETED (2026-11-13)

### Code Built & Tested
* [`src/models/mdnet.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/mdnet.py): Implements `LayerNorm2d`, `MDBlock` (parallel depthwise `k=3 d=2 p=2`, `k=2 d=2 p=1`, `k=1 d=1 p=0`, summed $\to$ `GELU` $\to$ `1x1` conv $\to$ `LayerNorm` $\to$ residual), `MDNet` / `build_mdnet_t` (`[3, 3, 9, 3]`, channels `[96, 192, 384, 768]`, stage shapes `[56, 28, 14, 7]`, `forward_stages` exposed for Step 4, `4,925,473` reg / `4,929,318` cls params), and `compute_mdnet_receptive_field` (**RF = 1136** for MDNet-T, **2288** for base MDNet, matching Paper 2 Table III exactly). Kaiming `fan_out` normal init on `Conv2d` resolved pre-`LayerNorm` gradient amplification identified during Debugging Gate 1.
* [`src/models/baselines.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/baselines.py): Added E2 image-only CNN baselines (`build_resnet18_scratch`, `build_resnet18_pretrained`, `build_mobilenet_v3_small_pretrained`).
* [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py), [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py), [`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py), [`configs/default.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/default.yaml): Extended with `AdamW`, linear warmup (`3` epochs) + cosine schedule, `fp32` default (verified faster than MPS `fp16` autocast: `50.14s` vs `57.81s`), validation-only best-epoch selection, per-epoch curves saved to `outputs/curves/*.csv`, and Step-2-compatible results appended to [`outputs/results/step3_results.csv`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step3_results.csv).
* [`tests/test_step3.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step3.py): 5 new unit tests (`14 passed` total across `test_data_pipeline.py`, `test_step2.py`, and `test_step3.py`).

### Verified Fold-0 Results (`station`, `epochs=30`, `bs=32`, `fp32`, `n=1654`, no test eval)
* **E2c (`mobilenetv3_s_pretrained`):** `MAE=35.72, RMSE=48.11, PLCC=0.8415, SRCC=0.7786, R2=0.7036, Acc=0.4692, +-1Acc=0.9008, MacroF1=0.4865, TailMAE=62.76, TailF1=0.5125` (`14.39 s/ep`, 5-fold est. `36.0 min`).
* **E2b (`resnet18_pretrained`):** `MAE=40.28, RMSE=53.66, PLCC=0.8020, SRCC=0.7493, R2=0.6312, Acc=0.4202, +-1Acc=0.8809, MacroF1=0.4468, TailMAE=77.48, TailF1=0.4901` (`32.48 s/ep`, 5-fold est. `81.2 min`).
* **E2a (`resnet18_scratch`):** `MAE=48.26, RMSE=62.67, PLCC=0.7086, SRCC=0.6220, R2=0.4970, Acc=0.3343, +-1Acc=0.8247, MacroF1=0.3521, TailMAE=83.70, TailF1=0.4136` (`33.06 s/ep`, 5-fold est. `82.7 min`).
* **E3a (`mdnet_t_reg`):** `MAE=51.33, RMSE=68.71, PLCC=0.6495, SRCC=0.5655, R2=0.3952, Acc=0.3168, +-1Acc=0.7987, MacroF1=0.2957, TailMAE=106.81, TailF1=0.2362` (`51.38 s/ep`, 5-fold est. `128.5 min`).
* **E3b (`mdnet_t_cls`):** `MAE=55.29, RMSE=72.08, PLCC=0.5949, SRCC=0.5396, R2=0.3345, Acc=0.3325, +-1Acc=0.6880, MacroF1=0.2747, TailMAE=122.73, TailF1=0.2157` (`51.33 s/ep`, 5-fold est. `128.3 min`).

---

## Training Queue Preparation (E3b + Resumable Runner) — COMPLETED (2026-11-13)

### Code Built & Tested
* [`src/models/mdnet.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/mdnet.py), [`src/models/__init__.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/__init__.py), [`docs/SPEC.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/SPEC.md): Added **E3b** (`MobileNetV3MDNeck` / `build_mobilenet_v3_small_md_neck`, registered as `mobilenetv3_s_md_neck`), combining a pretrained MobileNetV3-Small backbone with a `1x1` conv reduction (`576 -> 128`), `LayerNorm2d(128)`, `2` `MDBlock(128)` layers, global average pooling, `LayerNorm(128)`, and regression head. Marked in docstrings and `SPEC.md` as the author's adaptation of Paper 2's idea, not part of Paper 2.
* [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py): Registered `mobilenetv3_s_md_neck` and upgraded `append_result_row` to use `fcntl.flock` + temp-file `fsync` + `os.replace` for atomic CSV appends.
* [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py): Registered `mobilenetv3_s_md_neck`, added atomic per-epoch checkpointing (`last.pt`) + `--resume` resumption, `--outputs-dir`, and `--max-samples` for smoke tests.
* [`scripts/run_queue.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/run_queue.py): Resumable Stage 1 (fold-0 candidate runs) $\to$ Stage 2 (deterministic MAE selection keeping `(lr=1e-3, 30ep)` default unless improved by $\ge 1.0$) $\to$ Stage 3 (5-fold `station` CV for the 5 regression families; `mdnet_t_cls` remains fold-0 only) runner. Executes each `(experiment, fold)` in its own subprocess, skips completed CSV entries, logs crashes and continues, projects Stage 3 hours with a `12.0h` guard writing `outputs/QUEUE_STOPPED.txt`, logs exclusively to `outputs/logs/queue.log`, and provides `--smoke` (`1` epoch, `64` images, `2` folds under `outputs/smoke/`).
* [`tests/test_queue_and_e3b.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_queue_and_e3b.py): 5 new unit tests (`19 passed` total via `env/bin/python -m pytest -q`).
* **Smoke verification (`--smoke`):** Run 1 completed all 6 Stage-1 fold-0 runs and 5 Stage-3 fold-1 runs under [`outputs/smoke/`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/smoke/logs/queue.log) without modifying [`outputs/results/step3_results.csv`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step3_results.csv) (SHA-1 `a3516b0c66179219741ff8b45bb849f5dc29b2db` unchanged). Run 2 skipped all 6 Stage-1 and all 10 Stage-3 `(experiment, fold)` pairs.

---

## Step 3 queue results (5-fold) (`outputs/results/step3_results.csv`)

### Stage 2 Fold-0 Configuration Selection
1. **`mobilenetv3_s_pretrained` (E2c):** Kept default **`E2_mobilenetv3_s_pretrained`** (`lr=1e-3, 30ep`, Fold-0 `MAE=35.72`) over candidate `E2_mobilenetv3_s_pretrained_lr3e-4_ep40` (`lr=3e-4, 40ep`, Fold-0 `MAE=39.69`, $\Delta = -3.96 < 1.0$).
2. **`mobilenetv3_s_md_neck` (E3b):** Selected **`E3b_mobilenetv3_s_md_neck_lr1e-3_ep40`** (`lr=1e-3, 40ep`, Fold-0 `MAE=34.35`) over `E3b_mobilenetv3_s_md_neck_lr3e-4_ep40` (`lr=3e-4, 40ep`, Fold-0 `MAE=35.44`).
3. **`resnet18_pretrained` (E2b):** Switched to **`E2_resnet18_pretrained_lr3e-4_ep40`** (`lr=3e-4, 40ep`, Fold-0 `MAE=34.66`) over default `E2_resnet18_pretrained` (`lr=1e-3, 30ep`, Fold-0 `MAE=40.28`, $\Delta = +5.62 \ge 1.0$).
4. **`mdnet_t_reg` (E3a):** Switched to **`E3_mdnet_t_reg_lr1e-3_ep60`** (`lr=1e-3, 60ep`, Fold-0 `MAE=46.71`) over default `E3_mdnet_t_reg` (`lr=1e-3, 30ep`, Fold-0 `MAE=51.33`, $\Delta = +4.62 \ge 1.0$).
5. **`resnet18_scratch` (E2a):** Kept default **`E2_resnet18_scratch`** (`lr=1e-3, 30ep`, Fold-0 `MAE=48.26`) over candidate `E2_resnet18_scratch_lr1e-3_ep60` (`lr=1e-3, 60ep`, Fold-0 `MAE=48.96`, $\Delta = -0.70 < 1.0$).
6. **`mdnet_t_cls`:** Fold-0 only (`E3_mdnet_t_cls`, `lr=1e-3, 30ep`, `MAE=55.29, Acc=0.3325, MacroF1=0.2747`).

### Stage 3 5-Fold Station-Mode CV Summary (`mean +- std`, $n=8,176$)
* **`E2_mobilenetv3_s_pretrained` (`lr=1e-3, 30ep`):** `MAE=45.54+-10.06, RMSE=66.88+-20.37, PLCC=0.6421+-0.1608, SRCC=0.6638+-0.0790, R2=0.4111+-0.2102, Acc=0.4055+-0.0496, +-1Acc=0.8454+-0.0377, MacroF1=0.3983+-0.0755, TailMAE=101.51+-40.95, TailF1=0.3907+-0.1148`.
* **`E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` (`lr=1e-3, 40ep`):** `MAE=45.85+-9.59, RMSE=67.51+-20.22, PLCC=0.6279+-0.1682, SRCC=0.6646+-0.0809, R2=0.3979+-0.2120, Acc=0.3845+-0.0630, +-1Acc=0.8557+-0.0315, MacroF1=0.3400+-0.0920, TailMAE=109.67+-48.06, TailF1=0.2609+-0.1356`.
* **`E2_resnet18_pretrained_lr3e-4_ep40` (`lr=3e-4, 40ep`):** `MAE=46.16+-13.85, RMSE=66.72+-21.40, PLCC=0.6461+-0.1691, SRCC=0.6874+-0.0952, R2=0.4133+-0.2236, Acc=0.3995+-0.0673, +-1Acc=0.8519+-0.0640, MacroF1=0.3679+-0.0923, TailMAE=108.20+-41.31, TailF1=0.2757+-0.1709`.
* **`E3_mdnet_t_reg_lr1e-3_ep60` (`lr=1e-3, 60ep`):** `MAE=53.12+-6.68, RMSE=74.00+-15.96, PLCC=0.5314+-0.1351, SRCC=0.5304+-0.0542, R2=0.2840+-0.1476, Acc=0.3213+-0.0434, +-1Acc=0.7989+-0.0245, MacroF1=0.2891+-0.0823, TailMAE=118.28+-38.40, TailF1=0.2728+-0.1547`.
* **`E2_resnet18_scratch` (`lr=1e-3, 30ep`):** `MAE=53.50+-7.70, RMSE=74.34+-17.23, PLCC=0.5310+-0.1581, SRCC=0.5383+-0.0640, R2=0.2796+-0.1591, Acc=0.3176+-0.0169, +-1Acc=0.7958+-0.0252, MacroF1=0.2811+-0.0517, TailMAE=121.40+-45.32, TailF1=0.2289+-0.1218`.

---

## Step 4a: PFRML Head, Backbone Interfaces, Toggled Ideas & Configs — COMPLETED (2026-11-13)

*Implemented purely via new files while a training queue ran in the background; no existing code, config, script, or output files were modified.*

### New Files Built & Tested
* [`src/models/pooling.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/pooling.py): `AttentionPooling` (Toggle #1) — single-layer spatial attention over the `14x14 = 196` tokens (`softmax` mode per Paper 16 Eq. 2, optional `gwap` mode per Paper 8 Eq. 7); `enabled=False` yields exact uniform GAP (`tokens.mean(dim=1)`).
* [`src/models/physics_gate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/physics_gate.py): `PhysicsPriorGate` (Toggle #2) — reuses `HazeStatisticsExtractor` from `src/features/haze_features.py` by import only to compute 4 physics scalars `[entropy, rms_contrast, transmission, extinction]` on pre-normalisation `[0, 1]` pixels, projects $4 \to 224$, and fuses via learned sigmoid gate $\alpha = \sigma(W[F_{\text{deep}}; F_{\text{phys}}] + b)$; `enabled=False` returns `F_deep` unchanged.
* [`src/models/dcm.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/dcm.py):
  * `DimensionConsistencyModule` (`sum` default per Paper 1 Table I, `concat` toggle projecting `4*224 -> 224`) + `SingleFeatureAdapter` for single `(B, C, H, W)` feature maps.
  * `LightTransformerEncoder` (`dim=224`, 2 layers, `nhead=4`, `mlp_ratio=2.0`, `enable_nested_tensor=False`, factored 2D learnable positional encoding + `[CLS]`).
  * `PrototypeClassificationHead` ($K=6$ prototypes stored as checkpoint-saved buffers, orthogonal unit init + `init_prototypes` centroid initializer, eval-safe `no_grad` EMA updates only in `train()` with `targets`, `rsqrt` L2 normalization, `matmul`-based squared Euclidean distance $2 - 2\cos(\theta)$ avoiding `torch.cdist`, learnable positive temperature `log_tau`).
  * `PrototypeGuidedRegressionHead` (single MLP `448 -> 256 -> 1` on `[\chi_i; \tilde{c}_i]`, plus `predict_prototype_mixture_variance` diagnostic).
  * `BackboneTapWrapper`, `PFRMLHead`, and `PFRMLModel` supporting `mobilenetv3_s`, `resnet18`, `mdnet_t`, and `e3b` in both `multistage` (`[C1..C4]`) and `single` (`(B, C, H, W)`) modes.
* [`src/pfrml_losses.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/pfrml_losses.py): `PFRMLMultitaskLoss` (Toggle #3) importing `HybridMultitaskLoss` and `compute_class_weights` from `src/losses.py`; `use_class_weights=False` matches `HybridMultitaskLoss` exactly.
* [`src/models/pfrml_config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/pfrml_config.py) & [`configs/pfrml_*.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/pfrml_baseline.yaml): `PFRMLHeadConfig`, `load_pfrml_config`, `build_pfrml_from_config`, and 6 YAML configs (`pfrml_baseline.yaml`, `pfrml_dcm_concat.yaml`, `pfrml_attn_pool.yaml`, `pfrml_physics_gate.yaml`, `pfrml_class_weights.yaml`, `pfrml_full.yaml`).
* [`tests/test_step4a_pfrml.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step4a_pfrml.py): 15 CPU-only pytest tests (`torch.set_num_threads(2)`, tiny random tensors) verifying all 4 backbone interfaces, gradient flow to all trainable params, buffer/no-grad/eval-safe prototype updates, exact toggle-off baseline equivalence, 16-sample random batch overfitting (`acc=1.0`, `mae < 0.15`), and YAML config loading (**`15 passed in 26.57s`**).

### Tap Channels & Head Parameter Counts (`D=224`, 2-layer Transformer)
* **`mobilenetv3_s`** (`927,008` backbone): multistage `(16, 24, 48, 576)` $\to$ `1,078,402` (`sum`, toggles off) / `1,180,323` (all toggles on) / `1,279,778` (`concat`); single stride-16 `features[8]` (`C=48`) $\to$ `938,402` (off) / `1,040,323` (on).
* **`resnet18`** (`11,689,512` backbone): multistage `(64, 128, 256, 512)` $\to$ `1,144,706` (off) / `1,246,627` (on) / `1,346,082` (`concat`); single stride-16 `layer3` (`C=256`) $\to$ `984,994` (off) / `1,086,915` (on).
* **`mdnet_t`** (`4,925,473` backbone): multistage `(96, 192, 384, 768)` $\to$ `1,252,226` (off) / `1,354,147` (on) / `1,453,602` (`concat`); single stride-16 `stages[2]` (`C=384`) $\to$ `1,013,666` (off) / `1,115,587` (on).
* **`e3b`** (`1,039,393` backbone): multistage `(16, 24, 48, 128)` (`features[1,3,8]` + `md_neck`) $\to$ `978,050` (off) / `1,079,971` (on) / `1,179,426` (`concat`); single post-`md_neck` (`C=128`, stride 32 upsampled `2x`) $\to$ `956,322` (off) / `1,058,243` (on) (or `e3b_single_tap="stride16"` `C=48` bypassing `md_neck`).

### Flagged Ops for Core ML Verification
1. `nn.TransformerEncoder` / `nn.TransformerEncoderLayer` (`nn.MultiheadAttention` / `F.scaled_dot_product_attention`): `enable_nested_tensor=False` is set and token shapes `[B, 197, 224]` are static, but SDPA conversion should be verified with `coremltools` (fallback: explicit `Linear + matmul + softmax`).
2. `F.pad(..., mode="replicate")` inside imported `src/features/haze_features.py`: some `coremltools` versions only support `"constant"`/`"reflect"`.
3. `Tensor.amax` / `Tensor.amin` and `Tensor.std(..., unbiased=False)` inside `src/features/haze_features.py`.
4. `nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)` in DCM / single-tap stride-32 adapter.
5. `torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)` used in place of `F.normalize` / `torch.linalg.norm` (and `matmul` used in place of `torch.cdist`).

### SPEC Ambiguities Noted
1. **DCM norm & avg-pool (`SPEC.md` Sec. 3.2):** Does not specify `BatchNorm2d` vs `LayerNorm2d` or the avg-pool kernel/stride (GAP would collapse `14x14 -> 1x1`); used `LayerNorm2d(224)` + `AvgPool2d(1, 1)` (configurable to `3x3` pad `1`).
2. **DCM `concat` channel width (`SPEC.md` Sec. 3.2):** Does not state whether `4*224=896` projects back to `224` or widens the Transformer; projected `896 -> 224` via `1x1` conv + `LayerNorm2d(224)`.
3. **E3b single-map tap point (`SPEC.md` Sec. 3.1–3.2):** E3b's `MDBlock` neck sits at stride 32 (`[B, 128, 7, 7]`), so strict stride-16 tapping (`C=48`) bypasses the `MDBlock` neck; supported both `multistage` (`(16, 24, 48, 128)`) and `single` (`md_neck` `C=128` upsampled `2x` by default, `stride16` `C=48` optional).
4. **Transformer hyperparameters & `[CLS]` pooling (`SPEC.md` Sec. 3.2):** `nhead`, `mlp_ratio`, `norm_first`, `dropout`, and `196`-token vs `197`-token pooling are unspecified; defaulted to `nhead=4, mlp_ratio=2.0, norm_first=True, dropout=0.1` and pooled over the `196` spatial tokens (`encoded[:, 1:, :]`) so baseline GAP and Toggle #1 (`AttentionPooling`) compare directly.
5. **Prototype EMA, temperature & init (`SPEC.md` Sec. 3.3):** EMA decay, initial `tau`, and initial prototype geometry are unspecified; defaulted to `ema_decay=0.9`, `tau_init=0.2` (`log_tau` param), orthogonal initial unit prototypes + `init_prototypes` centroid initializer.
6. **Regression MLP hidden widths (`SPEC.md` Sec. 3.3):** Cites PFRML Sec. III-A3 (`1024 -> 256 -> 1`) while feeding `[\chi_i; \tilde{c}_i]` (`448`-D); defaulted to `448 -> 256 -> 1` (`reg_hidden_dims=(256,)`).
7. **Toggle #1 pooling mode (`SPEC.md` Sec. 1.2):** Lists both `GWAP` (Paper 8) and `softmax` token pooling (Paper 16); defaulted to `softmax` with `gwap` supported via config.
8. **Toggle #2 placement & scalar norm (`SPEC.md` Sec. 1.1):** Does not specify whether `PhysicsPriorGate` fuses before or after prototype classification or whether the 4 raw scalars are normalized; fused before prototype classification (per Paper 7) with `use_phys_layernorm=False` default.
9. **"Detached vision auxiliary loss" (`SPEC.md` Sec. 3.4, Step 5a):** `SPEC.md` Sec. 3.4 writes `"Modality Dropout (zeroing the metadata embedding with probability p_drop during training) + detached vision auxiliary loss"`. "Detached vision" could be misread as detaching `chi` (`chi.detach()`), which would block gradients from reaching the image backbone and defeat the shortcut-mitigation objective. In Step 5a, we implement an auxiliary vision-only regression head on `chi` alone (loss weight `0.5` added to the total loss) that is **detached from (independent of) the metadata branch**: gradients flow into the image backbone (`chi`) and `aux_reg_head`, and are strictly zero w.r.t. `MetadataMLP`.

---

## Step 4b-1: PFRML Wiring, Smoke Verification & Fold Diagnostics — COMPLETED (2026-11-14)

### Completed Shared-File Edits (Moved from Pending)
1. **[`src/models/__init__.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/__init__.py) — DONE:** Exported `AttentionPooling`, `PhysicsPriorGate`, `BACKBONE_SPECS`, `DimensionConsistencyModule`, `SingleFeatureAdapter`, `LightTransformerEncoder`, `PrototypeClassificationHead`, `PrototypeGuidedRegressionHead`, `PFRMLOutput`, `PFRMLHead`, `BackboneTapWrapper`, `PFRMLModel`, `PFRMLHeadConfig`, `PFRMLFullConfig`, `load_pfrml_config`, and `build_pfrml_from_config`.
2. **[`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py) & [`src/models/pfrml_config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/pfrml_config.py) — DONE:** Added `PFRMLHeadConfig` to `Config.pfrml` and parsed the `pfrml` section inside `load_config` (converting `reg_hidden_dims` list to tuple).
3. **[`configs/default.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/default.yaml) & [`configs/pfrml_*.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/pfrml_full.yaml) — DONE:** Added default `pfrml:` block to `configs/default.yaml` and standardized `model.name: "pfrml_mdnet_t"` across `configs/pfrml_*.yaml`.
4. **[`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py) — DONE:** Registered `pfrml_mobilenetv3_s` (`pretrained=True`), `pfrml_resnet18` (`pretrained=True`), `pfrml_mdnet_t` (`pretrained=False`), and `pfrml_e3b` (`pretrained=True`) with `tap_mode="multistage"` default; wired `build_fold_loss` (`PFRMLMultitaskLoss` with class weights computed strictly from each fold's training bins); passed `targets=targets_bin` only in `model.train()`; wired `init_pfrml_prototypes_from_train_fold` (`model.eval()`, `get_eval_transforms`, `torch.no_grad()`) once at the end of warmup (`epoch == warmup_epochs`, default epoch 3); defaulted Step 4 output CSV to `outputs/results/step4_results.csv` and experiment ID to `E4_<backbone>_<config name>`; logged `reg_src=reg_pred*aqi_scale,cls_src=logits,proto_mix_var=...` in CSV `notes`.
5. **[`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py) — DONE:** Registered `pfrml_*` models, evaluated in `model.eval()` with `targets=None`, extracted `out.reg_pred * aqi_scale` and `out.logits`, and recorded mean validation `predict_prototype_mixture_variance` in `notes`.

### Fold Diagnostics & Tests Built
* [`scripts/diagnose_folds.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/diagnose_folds.py) & [`outputs/diagnostics/folds.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/folds.md): CPU-only diagnostic report covering (a) per-fold station-mode image/bin counts, Bins 5–6 share, and Median-Predictor MAE (`67.18, 62.31, 63.71, 74.33, 63.50`, CV `66.21+-4.89`), (b) Fold 3's 10 busiest $1^\circ \times 1^\circ$ lat/lon cells showing that 4 cells (`(28, 77)` Delhi/NCR `34.50%`, `(25, -101)` Saltillo/Monterrey `20.52%`, `(19, 99)` Chiang Rai `13.54%`, `(18, 73)` Pune `8.30%`) hold `76.86%` (`176 / 229`) of Fold 3's Bins 5–6 images across 7 stations, and (c) per-fold MAE comparison across `E0_median`, `E1_metadata_mlp`, `E2_mobilenetv3_s_pretrained`, and `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40`.
* [`tests/test_step4a_pfrml.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step4a_pfrml.py): Added 6 new unit tests (`test_eval_forward_leaves_prototypes_unchanged_in_train_eval_loop`, `test_class_weights_computed_from_training_bins_only`, and 4 parametrized cases of `test_pfrml_models_build_with_right_pretrained_flag`) — **`40 passed in 36.37s`**.
* **Smoke verification (`outputs/smoke/`):** Verified 1-epoch (`--max-samples 64`) smoke runs for `pfrml_mobilenetv3_s` + `configs/pfrml_full.yaml` (`E4_mobilenetv3_s_full`, `MAE=50.65, Acc=0.0769, proto_mix_var=0.000280`) and `pfrml_mdnet_t` + `configs/pfrml_baseline.yaml` (`E4_mdnet_t_baseline`, `MAE=58.66, Acc=0.3077, proto_mix_var=0.000004`), confirmed `evaluate.py` reproduces `train.py`'s exact validation metrics from the saved checkpoints, and confirmed [`outputs/results/step3_results.csv`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step3_results.csv) SHA-1 (`7f6a85b154be562c74282281d0b1d7c128ff2a7f`) is unchanged.

---

## Step 4b-2: Experiment Ladder Specification, Resumable Queue & Summary Script — COMPLETED (2026-11-14)

### Step 0 Verification of Step 4b-1
* **`pytest`:** `42 passed in 36.33s`.
* **`evaluate.py` on `pfrml_mobilenetv3_s` smoke checkpoint:** `MAE=50.65` (`[E4_mobilenetv3_s_full | station | fold 0] MAE=50.65 RMSE=78.96 PLCC=0.4261 SRCC=0.4353 R2=0.0621 Acc=0.0769 +-1Acc=0.5385 MacroF1=0.0417 TailMAE=246.64 TailF1=0.0000`), matching `train.py`'s recorded validation line (`50.6482`) identically.
* **Why `configs/pfrml_*.yaml` and `src/models/pfrml_config.py` were modified in 4b-1:**
  * `configs/pfrml_*.yaml`: Standardized `model.name: "pfrml_mdnet_t"` across all 6 PFRML YAML configs so `train.py` / `evaluate.py` resolve a registered Step 4 model name when invoked without `--model`.
  * `src/models/pfrml_config.py`: Re-exported `PFRMLHeadConfig` from `src.config` and delegated YAML loading to `src.config.load_config` so `Config.pfrml` and `PFRMLFullConfig` share a single source of truth.
* **Pretrained flags of `pfrml_*` models:** `pfrml_mobilenetv3_s=True`, `pfrml_resnet18=True`, `pfrml_mdnet_t=False`, `pfrml_e3b=True`.

### Code, Configs & Docs Built
* [`docs/SPEC.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/docs/SPEC.md) (**Section 5: Experiment ladder**): Added the full Step 4 PFRML ablation ladder (`R1`, `R2`, `R2+T1`, `R2+T2`, `R2+T3`, `R2+T4`, `R2+all`), existing reference rows (`E2_mobilenetv3_s_pretrained`, `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40`), `Extra run A` (`E2_mobilenetv3_s_pretrained_spatial`, `spatial_cluster`, 5 folds), and `Extra run B` (`E4_mdnet_t_baseline_ep60`, `pfrml_mdnet_t` from scratch, 60 epochs, `lr=1e-3`, 14h guard), with paper citations and the rationale for running all ablations across all 5 folds (`SD ~ 10` AQI points and Fold 0 is the easiest fold per `outputs/diagnostics/folds.md`).
* [`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py), [`src/models/pfrml_config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/pfrml_config.py), [`src/models/dcm.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/dcm.py), [`src/pfrml_losses.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/pfrml_losses.py), [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py), [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py), [`configs/default.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/default.yaml), [`configs/pfrml_r1_no_prototypes.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/pfrml_r1_no_prototypes.yaml): Added `use_prototypes: bool = True` toggle (`false` in `configs/pfrml_r1_no_prototypes.yaml` for `R1`: omits `PrototypeClassificationHead` and prototype buffers, feeds `chi` (`224 -> 256 -> 1`) directly to the single regression MLP, and uses regression-only MSE loss).
* [`scripts/run_queue_step4.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/run_queue_step4.py): Config-major resumable Step 4 queue runner (`R2`, `R1`, `R2+all`, `R2+T1`, `R2+T2`, `R2+T3`, `R2+T4`, `Extra run A`, `Extra run B`) with isolated per-`(experiment, fold)` subprocesses, atomic CSV appends to [`outputs/results/step4_results.csv`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step4_results.csv), `cv_mean`/`cv_std` summary rows, `--pilot` mode (runs only `R2` fold 0 and prints validation MAE), `--smoke` mode under `outputs/smoke/`, and a `14.0 h` total queue runtime guard before `Extra run B` writing `outputs/QUEUE_STOPPED.txt` if exceeded.
* [`scripts/summarize_step4.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/summarize_step4.py): CPU-only, read-only summary generator writing `outputs/results/step4_summary.md` with 5-fold MAE `mean +- SD`, per-fold MAE, per-fold skill $1 - \text{MAE}/\text{MAE}_{\text{median}}$, paired differences (`mean diff +- SE`, `folds won / 5`) vs `R2` and vs `E2_mobilenetv3_s_pretrained`, and two-sided Wilcoxon $p$-values with the note `"n=5: minimum possible two-sided p = 0.0625"`.
* [`tests/test_step4a_pfrml.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step4a_pfrml.py): Added `test_r1_use_prototypes_false_no_buffers_and_regression_only_loss` and `test_step4_queue_specs_and_summarize_empty_csv` (`42 passed`).

---

## Step 4 Final Results (Last-Epoch Reporting from `outputs/curves/*.csv`) — COMPLETED (2026-11-15)

Full 10-metric tables and paired MAE comparisons are saved in [`outputs/results/step4_last_epoch_full.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step4_last_epoch_full.md) (generated by [`scripts/summarize_last_epoch_v2.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/summarize_last_epoch_v2.py)) and [`outputs/results/step4_last_epoch.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step4_last_epoch.md); optimization/collapse diagnostics are saved in [`outputs/diagnostics/collapse.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/collapse.md) (generated by [`scripts/diagnose_collapse.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/diagnose_collapse.py)).

### 1. 5-Fold Last-Epoch Summary (`mean +- SD` across all 10 metrics)

| Experiment | Mode | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E2_mobilenetv3_s_pretrained` (Ref 1) | `station` | **45.66 +- 8.41** | 70.41 +- 21.71 | 0.6152 +- 0.1688 | 0.6695 +- 0.0707 | 0.3450 +- 0.2392 | 0.4044 +- 0.0284 | 0.8488 +- 0.0298 | 0.3776 +- 0.0623 | 119.16 +- 49.30 | 0.3135 +- 0.1350 |
| `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` (Ref 2) | `station` | **44.92 +- 8.54** | 70.23 +- 21.74 | 0.6173 +- 0.1688 | 0.6780 +- 0.0636 | 0.3461 +- 0.2419 | 0.4137 +- 0.0395 | 0.8482 +- 0.0358 | 0.3789 +- 0.0649 | 118.93 +- 51.53 | 0.2913 +- 0.1183 |
| `E2_resnet18_pretrained_lr3e-4_ep40` | `station` | **44.05 +- 8.70** | 68.22 +- 22.66 | 0.6441 +- 0.1653 | 0.7060 +- 0.0578 | 0.3862 +- 0.2454 | 0.4102 +- 0.0447 | 0.8656 +- 0.0300 | 0.3751 +- 0.0804 | 116.26 +- 50.13 | 0.2822 +- 0.1707 |
| `E3_mdnet_t_reg_lr1e-3_ep60` | `station` | **53.29 +- 6.38** | 77.16 +- 17.10 | 0.5260 +- 0.1254 | 0.5365 +- 0.0418 | 0.2226 +- 0.1629 | 0.3371 +- 0.0191 | 0.7797 +- 0.0199 | 0.3030 +- 0.0459 | 129.51 +- 41.95 | 0.2175 +- 0.1139 |
| `E2_resnet18_scratch` | `station` | **53.27 +- 7.11** | 76.77 +- 18.58 | 0.5177 +- 0.1514 | 0.5367 +- 0.0597 | 0.2314 +- 0.1835 | 0.3276 +- 0.0088 | 0.7943 +- 0.0204 | 0.2975 +- 0.0434 | 126.93 +- 52.35 | 0.2302 +- 0.1391 |
| `E4_mobilenetv3_s_r1_no_prototypes` (R1) | `station` | **45.50 +- 8.64** | 71.44 +- 21.21 | 0.6081 +- 0.1659 | 0.6671 +- 0.0651 | 0.3249 +- 0.2375 | 0.4080 +- 0.0402 | 0.8448 +- 0.0310 | 0.3859 +- 0.0712 | 115.37 +- 53.33 | 0.3280 +- 0.1477 |
| `E4_mobilenetv3_s_baseline` (R2) | `station` | **48.14 +- 7.61** | 75.43 +- 20.35 | 0.5754 +- 0.1616 | 0.6327 +- 0.0701 | 0.2424 +- 0.2467 | 0.4088 +- 0.0297 | 0.8181 +- 0.0388 | 0.3829 +- 0.0535 | 124.19 +- 48.11 | 0.3027 +- 0.1248 |
| `E4_mobilenetv3_s_attn_pool` (R2+T1) | `station` | **48.40 +- 7.74** | 75.42 +- 20.23 | 0.5735 +- 0.1624 | 0.6352 +- 0.0729 | 0.2429 +- 0.2462 | 0.4032 +- 0.0411 | 0.8148 +- 0.0368 | 0.3789 +- 0.0734 | 121.92 +- 47.48 | 0.3131 +- 0.1623 |
| `E4_mobilenetv3_s_physics_gate` (R2+T2) | `station` | **47.57 +- 6.73** | 74.19 +- 19.93 | 0.5863 +- 0.1508 | 0.6405 +- 0.0525 | 0.2761 +- 0.2172 | 0.4037 +- 0.0213 | 0.8243 +- 0.0247 | 0.3823 +- 0.0450 | 118.49 +- 45.76 | 0.3130 +- 0.1174 |
| `E4_mobilenetv3_s_class_weights` (R2+T3) | `station` | **47.06 +- 8.01** | 73.67 +- 20.80 | 0.5804 +- 0.1635 | 0.6342 +- 0.0757 | 0.2822 +- 0.2371 | 0.4113 +- 0.0404 | 0.8142 +- 0.0343 | 0.3794 +- 0.0680 | 124.33 +- 47.86 | 0.2886 +- 0.1478 |
| `E4_mobilenetv3_s_dcm_concat` (R2+T4) | `station` | **60.51 +- 5.41** | 85.22 +- 15.71 | 0.3886 +- 0.0945 | 0.4082 +- 0.0915 | 0.0556 +- 0.1127 | 0.3041 +- 0.0464 | 0.6664 +- 0.0691 | 0.2515 +- 0.0348 | 168.58 +- 34.05 | 0.1918 +- 0.0435 |
| `E4_mobilenetv3_s_full` (R2+all) | `station` | **47.16 +- 7.15** | 74.22 +- 18.76 | 0.5828 +- 0.1424 | 0.6398 +- 0.0551 | 0.2774 +- 0.1966 | 0.4150 +- 0.0288 | 0.8136 +- 0.0302 | 0.3861 +- 0.0562 | 121.05 +- 46.12 | 0.3062 +- 0.1314 |
| `E4_mdnet_t_baseline_ep60` (Extra B) | `station` | **66.56 +- 4.88** | 87.26 +- 12.45 | 0.1036 +- 0.1133 | 0.0817 +- 0.0760 | 0.0054 +- 0.0151 | 0.1489 +- 0.0401 | 0.4432 +- 0.0538 | 0.1112 +- 0.0366 | 166.28 +- 34.72 | 0.0946 +- 0.0386 |
| `E2_mobilenetv3_s_pretrained_spatial` (Extra A) | `spatial_cluster` | **50.69 +- 5.60** | 77.65 +- 17.70 | 0.5301 +- 0.1282 | 0.6328 +- 0.0528 | 0.2142 +- 0.1477 | 0.3739 +- 0.0284 | 0.8082 +- 0.0348 | 0.2916 +- 0.0406 | 146.79 +- 42.62 | 0.0800 +- 0.0884 |

### 2. Paired Last-Epoch MAE Differences (`mean diff +- SE`, `folds won / 5`, no p-values)
* **R1 (`E4_mobilenetv3_s_r1_no_prototypes`, `45.50 +- 8.64`):** Beats `E2_mobilenetv3_s_pretrained` on `4/5` folds (`-0.16 +- 0.36`) and beats `R2` (`E4_mobilenetv3_s_baseline`) on `4/5` folds (`-2.63 +- 1.10`).
* **R2 (`E4_mobilenetv3_s_baseline`, `48.14 +- 7.61`):** `+2.47 +- 0.95` vs `E2` (`1/5` won) and `+2.63 +- 1.10` vs `R1` (`1/5` won), indicating that feeding `[\chi_i; \tilde{c}_i]` with noisy 6-class prototype probabilities (`Acc ~ 0.41`) degrades regression relative to `R1`.
* **Toolbox toggles on R2:**
  * `R2+T3` (`class_weights`, `47.06 +- 8.01`): `-1.08 +- 0.49` vs `R2` (`4/5` won).
  * `R2+all` (`full`, `47.16 +- 7.15`): `-0.98 +- 0.90` vs `R2` (`4/5` won).
  * `R2+T2` (`physics_gate`, `47.57 +- 6.73`): `-0.57 +- 0.86` vs `R2` (`4/5` won).
  * `R2+T1` (`attn_pool`, `48.40 +- 7.74`): `+0.27 +- 0.32` vs `R2` (`2/5` won).
  * `R2+T4` (`dcm_concat`, `60.51 +- 5.41`): `+12.38 +- 3.49` vs `R2` (`0/5` won; fails to reduce `train_loss` below `0.5x` of epoch 1).
* **Extra B (`E4_mdnet_t_baseline_ep60`, `66.56 +- 4.88`):** `not learning; last R2 ~ 0` on `5/5` folds (`train_loss` stays at `~2.13–2.20` over 60 epochs), whereas regression-only `E3_mdnet_t_reg_lr1e-3_ep60` learns on `5/5` folds (`MAE = 53.29 +- 6.38`).

---

## Step 5a: Last-Epoch Reporting Policy, `reg_input: chi`, Multimodal Fusion & Diagnostics — COMPLETED (2026-11-15)

### A. Reporting Policy Change (`E5_` Experiments)
* **Last-epoch CSV reporting:** In [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py), any experiment whose `experiment_id` starts with `"E5_"` writes its results-CSV row and `preds.npz` from the **LAST epoch** rather than the minimum-`val_loss` epoch. The best-loss epoch and its MAE are recorded in the CSV `notes` field as `last_ep=<last>/<total>,val_loss=<last_loss>,best_ep=<best>/<total>,best_ep_mae=<best_mae>,...`.
* **Dual checkpoints saved for `E5_` runs:** Each `E5_` fold saves both `model.pt` (`checkpoint_type="last_epoch"`, loaded by `evaluate.py` so standalone evaluation matches the reported CSV row) and `best_model.pt` (`checkpoint_type="best_loss"`).
* **Backwards compatibility:** All earlier experiments (`E0`–`E4`, non-`E5_` IDs) retain their exact previous behaviour (best-loss checkpoint and best-loss CSV row). Existing `step2_results.csv`, `step3_results.csv`, and `step4_results.csv` are completely untouched.

### B. Read-Only CPU Summary & Collapse Diagnostic Scripts
* [`scripts/summarize_last_epoch_v2.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/summarize_last_epoch_v2.py) $\to$ [`outputs/results/step4_last_epoch_full.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step4_last_epoch_full.md): Reports 5-fold `mean +- SD` across all 10 metrics (`mae, rmse, plcc, srcc, r2, acc, pm1_acc, macro_f1, tail_mae, tail_f1`) from the last epoch of `outputs/curves/*.csv`, plus paired MAE difference tables (`mean diff +- SE`, `folds won / 5`, no p-values) for every `E4` config vs `E2_mobilenetv3_s_pretrained`, vs `E4_mobilenetv3_s_r1_no_prototypes`, and vs `E4_mobilenetv3_s_baseline`.
* [`scripts/diagnose_collapse.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/diagnose_collapse.py) $\to$ [`outputs/diagnostics/collapse.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/collapse.md): Per-fold diagnostic table (`train_loss` at epochs `1/5/10/20/last`, `min`/`last` `val_loss`, `min`/`last` `MAE`, `last R2`, and `learning` vs `not learning` + `last R2 ~ 0` verdicts) for `E4_mdnet_t_baseline_ep60`, `E4_mobilenetv3_s_dcm_concat`, `E4_mobilenetv3_s_baseline`, and `E3_mdnet_t_reg_lr1e-3_ep60`.

### C. Regression-Input Flag (`reg_input: "chi"` — PFRML Eq. 5 Literal Variant)
* [`src/config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/config.py), [`src/models/dcm.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/dcm.py), [`src/models/pfrml_config.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/pfrml_config.py), [`configs/pfrml_reg_chi.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/pfrml_reg_chi.yaml): Added `pfrml.reg_input` in `{"chi_proto", "chi"}`. With `reg_input: "chi"`, `PrototypeGuidedRegressionHead` feeds only $\chi_i$ (`224 -> 256 -> 1`, PFRML Sec. II-D Eq. 5 $\hat{y}_i = \hbar(\chi_i)$) while keeping `PrototypeClassificationHead` active (`L_cls` cross-entropy + EMA prototype updates). Cites PFRML Eq. 5 vs Fig. 7 in docstrings.

### D. Multimodal Fusion Models (`fusion_r1_mobilenetv3_s`, `fusion_full_mobilenetv3_s`)
* [`src/models/fusion.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/fusion.py), [`src/pfrml_losses.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/pfrml_losses.py), [`configs/fusion_r1.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/fusion_r1.yaml), [`configs/fusion_full.yaml`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/configs/fusion_full.yaml), [`train.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/train.py), [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py):
  * Registered `fusion_r1_mobilenetv3_s` (R1 image branch + `MetadataMLP` `5 -> 64 -> 32`, regression MLP on `[\chi_i; m]`, `256 -> 256 -> 1`) and `fusion_full_mobilenetv3_s` (`pfrml_full` image branch + `MetadataMLP`, regression MLP on `[\chi_i; \tilde{c}_i; m]`, `480 -> 256 -> 1`; prototype classification branch uses $\chi_i$ only).
  * Per-sample modality dropout (`modality_dropout: 0.3`, CLI `--modality-dropout`) zeroes `m` per sample only during `train()`.
  * Vision-only auxiliary regression head (`aux_reg_head` on $\chi_i$ alone, weight `0.5` added to total loss) with gradients flowing into the image backbone and strictly detached from `MetadataMLP`.
  * Dual validation evaluation at the end of every `E5_` fusion run: normal (`<experiment_id>`) and metadata-zeroed (`<experiment_id>_meta_off`, CLI `--metadata-off`).
* [`tests/test_step5a.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step5a.py): 6 new unit tests (`48 passed` total across the test suite).

---

## Step 5b: Step 5 Queue, Final One-Shot Test Evaluation & Regression Checks — COMPLETED (2026-11-15)

### Part 0: Summary-Script & Regression Checks (CPU, Read-Only)
* **Part 0a ([`scripts/summarize_last_epoch_v2.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/summarize_last_epoch_v2.py) $\to$ [`outputs/results/step4_last_epoch_full.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step4_last_epoch_full.md)):** Enforced exact curve filename parsing `{experiment_id}_{group_mode}_fold{fold}.csv` with `group_mode in {'station', 'spatial_cluster'}` (no fallback regex or prefix glob) and added Section 3 reporting partial (`<5` folds) curve sets:
  * `E3_mdnet_t_reg_lr1e-3_ep60` (`station`, 5/5 folds): `53.29 +- 6.38` before and after.
  * `E2_mobilenetv3_s_pretrained_spatial` (`spatial_cluster`, 5/5 folds): `50.69 +- 5.60` before and after.
  * `E4_mobilenetv3_s_dcm_concat` (`station`, 5/5 folds): `60.51 +- 5.41` before and after.
  * `E2_resnet18_pretrained` (`station`): `1/5` folds (`MAE = 40.28`, `E2_resnet18_pretrained_station_fold0.csv`), cleanly separated from 5-fold `E2_resnet18_pretrained_lr3e-4_ep40` (`44.05 +- 8.70`).
* **Part 0b (Curve mtime & log audit):**
  * `E4_mobilenetv3_s_detached_proto` and `E4_mobilenetv3_s_ema099`: No curve CSVs exist in `outputs/curves/` and no matching lines exist in `outputs/logs/*.log`.
  * `E2_resnet18_pretrained` folds 1–4: No curve CSVs exist for `E2_resnet18_pretrained_station_fold{1..4}.csv` (only `E2_resnet18_pretrained_station_fold0.csv` exists, `13 Nov 00:22`; folds 1–4 belong to `E2_resnet18_pretrained_lr3e-4_ep40_station_fold{1..4}.csv`, `13 Nov 18:59–20:06`).
* **Part 0c (Fold-0 `station` checkpoint evaluation regression check via [`evaluate.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/evaluate.py)):**
  * `E4_mobilenetv3_s_baseline`: Expected `40.11`, Reproduced `40.11`.
  * `E4_mobilenetv3_s_r1_no_prototypes`: Expected `35.72`, Reproduced `35.72`.
  * `E2_mobilenetv3_s_pretrained`: Expected `35.72`, Reproduced `35.72`.
* **Part 0d (Default diff vs `git HEAD`):** All Step 4 defaults are unchanged; only new Step 5 fields (`pfrml.reg_input="chi_proto"`, `pfrml.modality_dropout=0.3`, `pfrml.aux_loss_weight=0.5`, and `PFRMLMultitaskLoss.aux_loss_weight=0.0`) were added.

### Part 1 & Part 2: Scripts Built & Tested
* [`scripts/run_queue_step5.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/run_queue_step5.py): Resumable Step 5 queue runner executing one subprocess per `(experiment, fold)` or final-eval item, skipping completed rows in `outputs/results/step5_results.csv` (and `outputs/results/final_results.csv`), logging crashes and continuing, writing atomic CSV rows and `cv_mean`/`cv_std` summaries (including `<experiment_id>_meta_off` for fusion models), logging to `outputs/logs/queue_step5.log`, supporting `--smoke`, `--pilot` (runs item 1 fold 0 only and prints both normal and `_meta_off` MAE), and `--project-only`, and enforcing an `8.0 h` projected runtime guard before item 7 (`outputs/QUEUE_STOPPED_STEP5.txt`).
* [`scripts/final_train_eval.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/final_train_eval.py): The **ONLY** script that reads `datasets/test/metadata.csv` for neural models. Trains on all `8,176` cleaned training images for fixed `30` epochs, evaluates the **LAST** epoch once per `--name` (refuses with exit code `1` if `--name` already exists in `outputs/results/final_results.csv`), reports all 10 metrics plus a 1,000-resample station-level bootstrap `95% CI` for MAE on both `test` (`n=2,920`) and `test_gt25km` (`n=717`, reusing `compute_test_distance_mask`), writes `<name>_meta_off` rows for fusion models, saves per-image predictions to `outputs/predictions/<name>.csv` (`image_id, station_id, y_true, y_pred`) and checkpoints to `outputs/checkpoints/<name>/full_train/model.pt`, and supports `--smoke-eval-on-val` so smoke tests never touch test labels.
* [`scripts/compare_final.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/compare_final.py): Generates `outputs/results/final_comparison.md` with full (`n=2,920`) and `>25 km` (`n=717`) 10-metric + `95% CI` tables and paired 1,000-resample station-bootstrap MAE differences (`95% CI`) between every pair of final models. Loads `E1_metadata_mlp` from its existing Step 2 `outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz` without retraining or touching test data.
* [`tests/test_step5b.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step5b.py): 3 new unit tests covering Step 5 queue specs, exact curve filename matching, and station-bootstrap CI / paired differences (**`51 passed in 35.94s`** across the full test suite).

---

## Step 5c: Epoch Selection, Fusion R1 Queue, `>25 km` Bootstrap Comparison & Report Numbers — COMPLETED (2026-11-15)

### Part 1: Epoch Selection from `E5_fusion_r1_p00` Curves ([`scripts/choose_epochs.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/choose_epochs.py) $\to$ [`outputs/results/step5c_epochs.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step5c_epochs.md))
* Computed per-epoch 5-fold mean validation MAE and 3-epoch centered moving average (`CMA-3`) across `outputs/curves/E5_fusion_r1_p00_spatial_cluster_fold{0..4}.csv`:
  * Epoch 4: `48.23 +- 6.99` (`CMA-3 = 50.32`)
  * Epoch 6: `47.05 +- 7.79` (`CMA-3 = 47.03`)
  * **Epoch 7 (Chosen `N = 7`):** **`45.74 +- 6.78`** (**`CMA-3 = 46.26`**, lowest across all epochs $\ge 6$)
  * Epoch 8: `45.99 +- 8.45` (`CMA-3 = 46.95`)
  * Epoch 10: `47.74 +- 8.95` (`CMA-3 = 47.97`)
  * Epoch 12: `45.91 +- 7.85` (`CMA-3 = 46.54`)
  * Epoch 15: `47.68 +- 5.85` (`CMA-3 = 46.93`)
  * Epoch 20: `48.11 +- 6.88` (`CMA-3 = 47.85`)
  * Epoch 30: `47.45 +- 6.71` (`CMA-3 = 47.46`)

### Part 2: Resumable Step 5c Queue ([`scripts/run_queue_step5c.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/run_queue_step5c.py))
* **Item 1 (`E5_fusion_r1_p00_ep7`):** `fusion_r1_mobilenetv3_s`, `configs/fusion_r1.yaml`, `spatial_cluster`, 5 folds, `--modality-dropout 0.0`, `--lr 1e-3`, `--epochs 7` (`N=7` read from `outputs/results/step5c_epochs.md`), appending to `outputs/results/step5_results.csv`.
* **Item 2 (Deterministic Decision Rule):** Computes 5-fold mean last-epoch MAE of `E5_fusion_r1_p00_ep7`. If `< 46.95`, sets `E_final = 7`; otherwise sets `E_final = 30`. Writes both numbers and the decision to `outputs/results/step5c_decision.md` and `outputs/logs/queue_step5c.log`.
* **Item 3 (`F_FUSION_R1`):** Runs `scripts/final_train_eval.py --name F_FUSION_R1 --model fusion_r1_mobilenetv3_s --config configs/fusion_r1.yaml --modality-dropout 0.0 --epochs E_final --seed 42` (skips with a clear message if `F_FUSION_R1` already exists in `outputs/results/final_results.csv`).
* **Projected Runtime (`--project-only`):** **`0.43 h`** total (`0.22 h` for Item 1 + `0.21 h` worst-case `E_final=30` for Item 3; well under the `1.00 h` guard).
* **Smoke Verification (`--smoke`):** Verified Run 1 completes Items 1–3 under `outputs/smoke/` and Run 2 skips all items cleanly.

### Part 3: Extended Final Comparison ([`scripts/compare_final.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/compare_final.py) $\to$ [`outputs/results/final_comparison.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/final_comparison.md))
* Added station-bootstrap `95% CI` for `E1_metadata_mlp` on both full `test` (`40.34 [34.45, 48.31]`) and `test_gt25km` (`49.84 [40.18, 58.81]`) from `outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz`.
* Added paired station-bootstrap MAE differences (`1,000` resamples, `seed=42`, `95% CI`) on the `test_gt25km` subset (`n=717`, `205` stations):
  * `F_E2` (`39.60`) vs `F_PFRML_full` (`46.79`): `Diff = -7.18`, `95% CI = [-11.92, -2.53]`
  * `F_E2` (`39.60`) vs `E1_metadata_mlp` (`49.84`): `Diff = -10.24`, `95% CI = [-20.50, +0.86]`
  * `F_PFRML_full` (`46.79`) vs `E1_metadata_mlp` (`49.84`): `Diff = -3.06`, `95% CI = [-14.90, +8.29]`
* Includes `F_FUSION_R1` and `F_FUSION_R1_meta_off` in all tables when present.

### Part 4: Master Report Numbers ([`scripts/build_report_numbers.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/build_report_numbers.py) $\to$ [`outputs/results/REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/REPORT_NUMBERS.md))
* Automatically compiles all 8 required sections strictly from CSVs, per-epoch curves, and prediction files (`(1)` Provenance, `(2)` Baselines, `(3)` Image models 5-fold station CV last-epoch MAE + per-fold + skill, `(4)` PFRML ladder + paired diffs vs R1 & E2, `(5)` Fusion spatial-cluster CV + per-fold + `_meta_off`, `(6)` Final test & `>25 km` results + paired bootstrap tables, `(7)` Failed/unstable runs with pointers to `outputs/diagnostics/collapse.md`, `(8)` Do-not-quote list).
* [`tests/test_step5c.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/tests/test_step5c.py): 4 new unit tests covering Step 5c scripts.

### Command to Run Full Step 5c Queue (Not Started)

```bash
nohup caffeinate -dimsu env/bin/python scripts/run_queue_step5c.py > outputs/logs/step5c_nohup.out 2>&1 &
```

After the queue finishes, regenerate [`outputs/results/final_comparison.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/final_comparison.md) and [`outputs/results/REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/REPORT_NUMBERS.md):

```bash
env/bin/python scripts/compare_final.py
env/bin/python scripts/build_report_numbers.py
```

### Proposed Experiments (Not Created or Run)
* **Decoupled Backbone/Metadata Learning Rates for Fusion (`lr_image=1e-3, lr_meta=3e-4`):** Because `E5_fusion_r1_p00` reaches its lowest `spatial_cluster` validation MAE around epoch `N=7` (`45.74 +- 6.78`) before overfitting station metadata by epoch 30 (`47.45 +- 6.71`), a lower learning rate or stronger weight decay on `MetadataMLP` could prevent late-epoch metadata overfitting without requiring early stopping.
* **Post-Hoc Equal-Weight Ensemble of `F_E2` and `E1_metadata_mlp` on `test` vs `test_gt25km`:** Since per-image predictions are already saved for both `F_E2` and `E1_metadata_mlp`, a distance-aware or equal-weight post-hoc blend can be evaluated purely on saved predictions as an additional diagnostic baseline if requested.
---

## Repository Cleanup — COMPLETED (2026-11-15)

* **Target layout & renames:** Organized scripts into [`scripts/data/`](scripts/data/), [`scripts/experiments/`](scripts/experiments/), [`scripts/analysis/`](scripts/analysis/), and [`scripts/final_evaluation.py`](scripts/final_evaluation.py); renamed configs (`pfrml_prototype_multitask.yaml`, `pfrml_regression_only.yaml`, `pfrml_eq5_regression.yaml`, `pfrml_attention_pooling.yaml`, `pfrml_all_toggles.yaml`, `fusion_regression_only_neck.yaml`, `fusion_pfrml_full.yaml`), tests, and `outputs/results/` & `outputs/logs/` files to descriptive non-chronological names while preserving byte-identical contents for all trained result CSVs and curves.
* **Removed superseded files:** Removed `AGENT_HANDOFF.md`, `PROJECT_EXPLAINER.md`, `ieee papers.md`, `docs/PLAN.md`, `scripts/summarize_last_epoch.py`, and `scripts/summarize_step4.py` via `git rm` (recoverable from tag `final-results`); moved untracked `.DS_Store` to `archive_local/`.
* **Documentation & self-contained `results/` snapshot:** Added [`README.md`](README.md), [`REPRODUCING.md`](REPRODUCING.md), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (including legacy `Step 1–5c` mapping), and a tracked [`results/`](results/) snapshot (`~1.8 MB`) enabling offline regeneration of [`results/final_comparison.md`](results/final_comparison.md) and [`results/REPORT_NUMBERS.md`](results/REPORT_NUMBERS.md) via `--results-dir results`.
* **Verification:** Full test suite passed (`55 passed`), Fold-0 validation checkpoint regressions matched (`40.11` / `35.72`), SHA-1 checksums verified for all renamed result files, and `REPORT_NUMBERS.md` / `final_comparison.md` regenerated with zero numeric differences.
