# Beyond Spatial Shortcuts: Auditing Leakage, Prototype Regularization, and Multimodal Fusion for Global Image-Based Air Quality Estimation

**Technical Research Report — PM25Vision Empirical Benchmark Audit**

---

## Abstract

Estimating Air Quality Index (AQI) values from street-level photographs offers a scalable complement to sparse regulatory monitoring networks. However, auditing recent vision-based AQI literature reveals that 14 of 17 applied studies exhibit spatial, temporal, or co-pollutant leakage—rewarding camera memorization or auxiliary sensor shortcuts over atmospheric scattering cues. We present a leakage-free empirical study on the global **PM25Vision** benchmark (11,096 deduplicated images across 3,256 non-overlapping stations after purging 18 cross-split leaks) using dual 5-fold `StratifiedGroupKFold` validation (`station` and 25 km `spatial_cluster`), an unseen-region test partition (`test_gt25km`, $n=717$, 205 stations), 1,000-resample station-cluster bootstrapping, and strict last-epoch reporting. Four findings emerge. First, coordinate-and-season metadata (`E1_metadata_mlp`) creates a spatial shortcut illusion: competitive on full test ($\text{MAE} = 40.34$, $R^2 = 0.4982$), it collapses beyond 25 km ($49.84$, $R^2 = -0.0393$; $59.96$ $>100\text{ km}$), whereas pure vision (`F_E2`, pretrained `MobileNetV3-Small`) remains invariant ($41.12 \to 39.60$; $31.71$ $>100\text{ km}$). Second, ImageNet pretraining improves `station` CV MAE by 7.6–9.2 points over from-scratch `ResNet-18` or `MDNet-T` ($44.05\text{–}45.66$ vs. $53.27\text{–}53.29$). Third, prototype-guided multi-task classification (`PFRML`) degrades regression on pretrained backbones ($48.14$ vs. prototype-free `R1` at $45.50$) and collapses on scratch `MDNet-T` ($66.56$, $R^2 = 0.0054$) and concatenated fusion (`DCM concat`, $60.51$). Fourth, fusing `R1` with metadata under a detached vision auxiliary loss and `CMA-3` early stopping at epoch $N=7$ (`F_FUSION_R1`, $2.02\text{M}$ params) achieves top generalization across full test ($\text{MAE} = 35.77\;[30.47, 42.72]$, $R^2 = 0.6070$) and $>25\text{ km}$ ($\text{MAE} = 39.22\;[30.85, 47.74]$, $R^2 = 0.3743$).

---

## 1. Introduction

Ambient fine particulate matter ($\text{PM}_{2.5}$) is a leading environmental driver of cardiopulmonary mortality worldwide. While regulatory gravimetric stations deliver calibrated measurements, deployment costs restrict monitor density, leaving large populations kilometers from the nearest sensor. Inferring AQI from crowdsourced street photographs promises ubiquitous spatial coverage via physical Mie and Rayleigh scattering cues: shadows brighten under additive airlight (Dark Channel intuition), distant contours lose contrast, and clear skies desaturate into milky haze.

Yet published vision-AQI benchmarks frequently claim $90\%\text{–}99\%$ accuracy or $R^2 > 0.96$. Our companion review of 17 applied papers (`docs/related_work_notes.md`) alongside two spine methods (**MDNet** and **PFRML**) shows that 14 of 17 studies suffer from train/test leakage. Splitting stationary rooftop webcam frames randomly lets networks memorize static skylines and temporal autocorrelation, failing when transferred to unseen locations.

To separate genuine haze perception from geographic shortcuts and optimization artifacts, we conduct a controlled study on **PM25Vision** (Mapillary street imagery paired globally with WAQI monitors), addressing four research questions:
1. **RQ1 (Spatial Shortcut Illusion):** Under strict station-disjoint and $>25\text{ km}$ protocols, how much predictive skill stems from local geographic priors versus visual cues across distance?
2. **RQ2 (Transfer Learning vs. Domain CNNs):** On ~8,200 heterogeneous scenes, do custom from-scratch dilated atmospheric CNNs (`MDNet-T`) beat compact ImageNet-pretrained backbones (`MobileNetV3-Small`, `ResNet-18`)?
3. **RQ3 (Prototype Regularization & Toggles):** Do Dimension Consistency Modules (`DCM`), spatial Transformers, EMA class prototypes (`PFRML`), attention pooling, physics gating, and class weights improve continuous AQI regression?
4. **RQ4 (Shortcut-Resistant Fusion):** How can geospatial priors be fused with visual features to boost accuracy in monitored corridors without collapsing beyond $25\text{ km}$?

---

## 2. Related Work and Methodological Critique

Rather than repeating the full 19-paper survey (`docs/related_work_notes.md`), we summarize three leakage patterns motivating our protocol alongside the architectural modules tested below.

### 2.1 Structural Leakage in Prior Benchmarks

Across 17 surveyed applied studies, **14 exhibit evaluation leakage** in three forms:
1. **Image-level splitting of fixed cameras:** Random 70/15/15 or 80/20 splits across stationary webcam or narrow multi-city sets (e.g., Beijing, Delhi, Kolkata, Kaggle India/Nepal) leak static backgrounds and autocorrelation. Proper holdout tests reveal sharp drops: Kamble et al. (2025) observed Pearson $r$ fall from $+0.91$ (shuffled 5-min frames) to $-0.22$ (chronological); Nathvani et al. (2023) saw 11-class accuracy on 1.6M Accra images drop from $30\%\text{–}55\%$ (same-site) to $7\%\text{–}20\%$ (unseen sites); and Byler et al. (2025) reported `ResNet-101` accuracy drop from $80.3\%$ (seen US National Parks) to $30.0\%\text{–}65.6\%$ (unseen parks).
2. **Pre-split oversampling:** Applying `SMOTER` or noise synthesis prior to splitting leaks near-duplicate clones across folds, inflating `KNN` $R^2$ above $0.96$.
3. **Co-pollutant sensor shortcuts:** Passing concurrent station gases or $\text{PM}_{10}$ ($r \approx 0.83$ with $\text{PM}_{2.5}$) into multimodal MLPs starves the visual encoder—illustrated by `SkyFuseNet` (Abdelsalam & Sayed, 2026), where zeroing tabular inputs inflated RMSE from $5.70$ to $158.95\;\mu\text{g/m}^3$.

Additionally, several studies apply ImageNet brightness/contrast jitter and blur, which Byler et al. (2025) showed harm AQI generalization by corrupting physical scattering cues.

### 2.2 Target Spines and Domain Toggles

Two hierarchical frameworks anchor our ablations: **MDNet** (Wang et al., IEEE TII 2024; `MDNet-T`, `depths=[3,3,9,3]`, `dims=[96,192,384,768]`) using parallel dilated depthwise `MDBlock` branches (`k=3,d=2`, `k=2,d=2`, `k=1,d=1`), `GELU`, and `LayerNorm`; and **PFRML** (Wang et al., IEEE TGRS 2026), adding a 4-stage Dimension Consistency Module (`DCM`, $14\times 14$ token grid), Light Transformer encoder, and EMA-updated Prototype-Guided Multitask Head ($L = L_{\text{cls}} + \beta L_{\text{reg}}$). Onto this spine, we evaluate softmax token attention pooling (Lee & Manova, 2025; Abdelsalam & Sayed, 2026), differentiable Dark-Channel/contrast/entropy physics gating (Akdim et al., 2026; Yadav et al., 2025), and dampened inverse-frequency class weights ($w_k \propto N_k^{-0.5}$).

---

## 3. Dataset, Cleaning, and Leakage-Free Validation Protocol

### 3.1 PM25Vision Audit and Deterministic Cleaning

**PM25Vision** pairs crowdsourced Mapillary RGB imagery with WAQI station readings globally ($\text{lat} \in [-46.14^\circ, 69.65^\circ]$, $\text{lon} \in [-158.08^\circ, 175.80^\circ]$). Although column `pm25` carries a mass-style name, its $[1.0, 530.0]$ values follow the **US EPA AQI** scale across six categories: **Good** ($0\text{–}50$), **Moderate** ($51\text{–}100$), **USG** ($101\text{–}150$), **Unhealthy** ($151\text{–}200$), **Very Unhealthy** ($201\text{–}300$), and **Hazardous** ($301\text{–}600$).

Raw CSVs contain **8,298 training rows** (8,194 `image_id`s, 2,532 stations) and **2,921 test rows** (2,920 `image_id`s, 729 stations), containing **18 cross-split overlapping `image_id`s** plus duplicate rows matched to multiple monitors. [`src/data/cleaning.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/cleaning.py) removes all 18 leaks from train and collapses multi-station duplicates by deterministic `(image_id, station_id)` sorting and taking **median AQI**, yielding:
- **Cleaned Train:** $n = 8,176$ images across **2,527 stations** (mean $121.80 \pm 88.35$, median $111.0$, IQR $110.0$).
- **Cleaned Test:** $n = 2,920$ images across **729 stations** (mean $119.36 \pm 81.47$, median $114.0$, IQR $111.0$, range $[2.0, 416.0]$).
- **Zero Cross-Split Overlap:** 0 shared `image_id`s or `station_id`s across 3,256 clean stations.

**Figure 1(a)** shows Bins 1–4 (`0–200`) are balanced at $\approx 21.5\%\text{–}21.8\%$ each, while **Bins 5–6 ($\text{AQI} > 200$)** form **13.28%** of train ($1,086$: $731$ Bin 5, $355$ Bin 6) and **13.73%** of test ($401$: $275$ Bin 5, $126$ Bin 6). Neural targets scale by `aqi_scale = 100.0` during optimization and unscale to native AQI for evaluation.

![Figure 1: Dataset US EPA AQI distribution and spatial separation CDFs](outputs/figures/fig1_dataset_spatial_splits.png)

### 3.2 Preprocessing, Dual Spatial CV, and Fold-3 Heterogeneity

[`src/data/transforms.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/transforms.py) resizes full images directly to $224 \times 224$ (`bilinear`, `antialias=True`) **without random cropping** (retaining sky and horizon cues) and excludes all color jitter, brightness/contrast shifts, and grayscale conversion, applying only `RandomHorizontalFlip(p=0.5)` and mild `RandomAffine(5°, translate=0.02, scale=[0.98, 1.02])`.

Because distinct monitors in the same city share background climatologies, [`src/data/splits.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/data/splits.py) builds two 5-fold `StratifiedGroupKFold` protocols (**Figure 1(b)**):
1. **`station` CV (2,527 groups):** Groups by `station_id`. Urban neighbors leave nearest-train fold medians at $5.62\text{–}10.67\text{ km}$ (overall $7.9\text{ km}$, min $0.00\text{–}0.01\text{ km}$).
2. **`spatial_cluster` CV (1,059 clusters):** Groups stations via single-linkage Haversine clustering (`25.0 km` threshold), enforcing $\ge 24.51\text{–}24.78\text{ km}$ separation (fold medians $65.58\text{–}93.11\text{ km}$, overall $70.9\text{ km}$).

Fold diagnostics (`outputs/diagnostics/folds.md`) identify **Fold 3 as an intrinsic tail-variance outlier** across every model (val AQI SD $= 108.20$ vs. $79.34\text{–}88.38$ in Folds 0/1/2/4): just four $1^\circ \times 1^\circ$ cells (7 stations—Delhi `(28,77)`, Saltillo/Monterrey `(25,-101)`, Chiang Rai `(19,99)`, Pune `(18,73)`) concentrate **76.86% (176/229)** of Fold 3's Bins 5–6 images. We therefore report **paired fold differences (`mean diff ± SE`, `folds won / 5`)** alongside `mean ± SD`.

### 3.3 Test Partitions, Metrics, and Last-Epoch Policy

To test regional portability, **`test_gt25km`** isolates **$n=717$ test images (205 stations)** strictly $>25\text{ km}$ from any training image (`mean = 106.54 ± 64.11`, median $106.0$, `9.07%` tail; full test median distance $6.38\text{ km}$, max $1,462.01\text{ km}$). Test 95% CIs and pairwise differences use a **1,000-resample station-cluster bootstrap (`seed=42`)**. All runs report 10 metrics ([`src/metrics.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/metrics.py)): regression (`MAE`, `RMSE`, `PLCC`, `SRCC`, $R^2$, $\text{Skill} = 1 - \text{MAE}/\text{MAE}_{\text{median}}$), 6-class severity (`Acc`, `±1Acc`, `MacroF1`), and tail (`TailMAE`, `TailF1`, $\text{AQI} > 200$).

Crucially, because early Step 3/4 CSV logs saved minimum-`val_loss` checkpoints (`best_ep`)—which overfit composite cross-entropy at epochs 2–6 while MAE still drops—**all results follow a strict last-epoch policy** verified from `outputs/curves/*.csv` and `final_results.csv` (`REPORT_NUMBERS.md` Sec. 8).

---

## 4. Methodology and Architectural Ladder

Models follow a unified progression (**Figure 2**). Unless noted, neural networks train in `fp32` on Apple M5 Pro (`mps`) with AdamW ($\text{wd}=10^{-4}$), grad clipping (`1.0`), batch size `32`, 3-epoch warmup (`lr=1e-3`), cosine decay to $10^{-6}$, and `seed=42`.

![Figure 2: Unified PFRML visual spine, architectural toggles, and F_FUSION_R1 multimodal architecture](outputs/figures/fig2_architecture_fusion.png)

### 4.1 Non-Visual & Vision-Only Baselines (`E0`–`E3b`)

- **`E0_median` / `E0_ridge`:** Constant fold median; and `RidgeCV` ($\alpha \in [10^{-2}, 10^3]$) over **18 static descriptors** (`full/top/bottom` bands $\times$ HSV saturation, Sobel gradient, $15\times 15$ Dark Channel $\times$ `mean/std`).
- **`E1_metadata_mlp` (2,497 params):** Maps unit-sphere $(x,y,z)$ and cyclic month $(\sin,\cos)$ through `5 -> 64 -> 32 -> 1` ReLU MLP.
- **`E2` / `E3` / `E3b`:** MSE-trained image baselines: `ResNet-18` scratch & pretrained (`11.18M`, `512 -> 1`), pretrained `MobileNetV3-Small` (`1.52M`, `1024 -> 1`, ~15–22 s/ep), from-scratch `MDNet-T` (`4.93M`, `depths=[3,3,9,3]`, `dims=[96,192,384,768]`, ~60 s/ep), and author hybrid `E3b_mobilenetv3_s_md_neck` (`1.04M`, `576 -> 128` + 2 `MDBlock` layers).

### 4.2 Unified PFRML Spine, Toggles, and Multimodal Fusion (`E4`–`E5`)

Implemented in [`src/models/dcm.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/dcm.py), PFRML aligns 4 backbone stages (strides `[4,8,16,32]`) onto $14 \times 14$ grids at $D=224$ via $1\times 1$ convs, static resizing, `LayerNorm2d`, and element-wise summation (`DCM`). Flattened 196 tokens + `[CLS]` + factored 2D positional embeddings traverse a 2-layer Transformer (`4` heads, MLP ratio `2.0`, `dropout=0.1`), pooling spatially into $\chi_i \in \mathbb{R}^{224}$. Six unit-hypersphere prototypes $\{\mathbf{p}_k\}_{k=0}^5$ update via **gradient-free EMA buffers** ($\text{decay}=0.9$), scoring logits $-\|\hat{\chi}_i - \mathbf{p}_k\|_2^2/\tau$ ($\tau_{\text{init}}=0.2$) under:
$$\mathcal{L}_{\text{PFRML}} = \mathcal{L}_{\text{cls}}(\text{logits}, y_{\text{bin}}; \epsilon=0.1) + \beta\,\mathcal{L}_{\text{reg}}(\hat{y}, y_{\text{scaled}}), \quad \beta = 0.5.$$
We evaluate **`R1`** (`E4_..._r1_no_prototypes`, `1.95M`, no prototypes/CE, pure MSE from $\chi_i$), **`R2`** (`E4_..._baseline`, `2.01M`, regresses on $[\chi_i; \sum_k p_k \mathbf{p}_k] \in \mathbb{R}^{448}$), and **`R2` Eq. 5 Literal** (`E5_..._reg_chi`, prototype CE active, regresses on $\chi_i$ alone). Onto `R2`, we toggle: **`T1`** (`AttentionPooling`), **`T2`** (`PhysicsPriorGate`, 4-scalar `[Entropy, RMS Contrast, t_DCP, -ln t_DCP]` projection $\mathbf{F}_{\text{phys}}$ blended via learned sigmoid gate $\boldsymbol{\alpha}$), **`T3`** (`class_weights`, $w_k \propto N_k^{-0.5}$), **`T4`** (`dcm_concat`, `896 -> 224`), and **`R2+all`** (`T1+T2+T3`, `2.11M`).

[`PFRMLFusionModel`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/src/models/fusion.py#L1) pairs `R1` (`2.02M`) or `full` (`2.18M`) with `MetadataMLP` (`32`-D $\mathbf{m}$), adding per-sample modality dropout ($p_{\text{drop}} \in \{0.0, 0.3\}$, `meta_off` test mode) and a **detached vision-only auxiliary head** (`224 -> 256 -> 1`, weight $0.5$) supervising $\chi_i$ independently of $\mathbf{m}$.

---

## 5. Experimental Results

All CV figures below represent **5-fold last-epoch metrics** verified against [`REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/REPORT_NUMBERS.md).

### 5.1 Baselines and the Spatial Generalization Gap

**Table 1.** Baseline performance (`E0_median`, `E0_ridge`, `E1_metadata_mlp`) across `station` CV, 25 km `spatial_cluster` CV, full `test`, and `test_gt25km`.

| Experiment | Split / Mode | $n$ | MAE (Mean $\pm$ SD / 95% CI) | RMSE | PLCC | SRCC | $R^2$ | Acc | $\pm 1\text{Acc}$ | TailMAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_median` | `station` / `spatial` CV | 8176 | **66.21 ± 4.89** / **66.14 ± 5.60** | 88.23 / 88.00 | 0.0000 | 0.0000 | -0.0161 | 0.2153 | 0.6497 | 176.95 |
| `E0_ridge` | `station` CV | 8176 | **64.11 ± 4.40** | 86.32 | 0.1998 | 0.2354 | 0.0301 | 0.2450 | 0.6925 | 158.15 |
| `E0_ridge` | `spatial_cluster` CV | 8176 | **64.56 ± 5.02** | 87.37 | 0.1526 | 0.1982 | -0.0018 | 0.2415 | 0.6915 | 160.68 |
| `E1_metadata_mlp` | `station` CV | 8176 | **43.15 ± 7.47** | 65.43 | 0.6704 | 0.7260 | 0.4438 | 0.4234 | 0.8794 | 104.96 |
| `E1_metadata_mlp` | `spatial_cluster` CV | 8176 | **46.00 ± 7.33** | 70.09 | 0.6257 | 0.7147 | 0.3647 | 0.4183 | 0.8603 | 118.49 |
| `E1_metadata_mlp` | Full `test` | 2920 | **40.34** [34.45, 48.31] | 57.70 | 0.7098 | 0.7344 | 0.4982 | 0.4096 | 0.8873 | 76.82 |
| `E1_metadata_mlp` | `test_gt25km` | 717 | **49.84** [40.18, 58.81] | 65.31 | 0.3797 | 0.4954 | -0.0393 | 0.2929 | 0.7476 | 88.83 |

**Table 1** reveals how proximity-heavy splits flatter spatial interpolation. Handcrafted haze features (`E0_ridge`) improve minimally over median guessing ($-2.10$ MAE, $R^2 = 0.0301$). By contrast, `E1_metadata_mlp` attains strong `station` CV ($43.15$, $R^2 = 0.4438$) and full test ($40.34$, $R^2 = 0.4982$) by interpolating neighboring urban monitors ($6.4\text{–}7.9\text{ km}$ median gap). Enforcing 25 km clustering drops `E1` to $46.00$ ($R^2 = 0.3647$; Fold 0 jumps $39.71 \to 55.86$), and on held-out `test_gt25km` it collapses to **$49.84$ MAE ($R^2 = -0.0393$, $\text{TailF1} = 0.0000$)**—underperforming the global mean in explained variance.

### 5.2 Image-Only Backbones: Transfer Learning vs. Scratch Training

**Table 2.** Image-only 5-fold `station` CV last-epoch comparison across scratch and ImageNet-pretrained backbones.

| Experiment | Params | Last-Ep MAE | Skill | RMSE | PLCC | SRCC | $R^2$ | Acc | $\pm 1\text{Acc}$ | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E2_resnet18_scratch` (30ep) | 11.18M | **53.27 ± 7.11** | +0.1954 | 76.77 | 0.5177 | 0.5367 | 0.2314 | 0.3276 | 0.7943 | 0.2975 | 126.93 | 0.2302 |
| `E3_mdnet_t_reg` (60ep) | 4.93M | **53.29 ± 6.38** | +0.1951 | 77.16 | 0.5260 | 0.5365 | 0.2226 | 0.3371 | 0.7797 | 0.3030 | 129.51 | 0.2175 |
| `E2_mobilenetv3_s_pretrained` | 1.52M | **45.66 ± 8.41** | +0.3103 | 70.41 | 0.6152 | 0.6695 | 0.3450 | 0.4044 | 0.8488 | 0.3776 | 119.16 | **0.3135** |
| `E3b_mobilenetv3_s_md_neck` | 1.04M | **44.92 ± 8.54** | +0.3214 | 70.23 | 0.6173 | 0.6780 | 0.3461 | **0.4137** | 0.8482 | **0.3789** | 118.93 | 0.2913 |
| `E2_resnet18_pretrained` (40ep) | 11.18M | **44.05 ± 8.70** | **+0.3346** | **68.22** | **0.6441** | **0.7060** | **0.3862** | 0.4102 | **0.8656** | 0.3751 | **116.26** | 0.2822 |

![Figure 3: Image-only backbone comparison and per-fold Fold 3 sensitivity](outputs/figures/fig3_baselines_and_backbones.png)

**ImageNet pretraining provides a decisive 7.6–9.2 MAE gain** (**Table 2**, **Figure 3**). Despite multi-scale dilated blocks and 60 training epochs, scratch `MDNet-T` ($53.29$) ties scratch `ResNet-18` ($53.27$). Pretrained backbones lift $\text{SRCC}$ from $0.54$ to $0.67\text{–}0.71$ and $\pm 1\text{Acc}$ to $0.85\text{–}0.87$: `ResNet-18` (`40ep`, `lr=3e-4`) leads at **$44.05$**, `E3b` (`MobileNetV3-S` + 128-D `MDBlock` neck) hits **$44.92$** at $1.04\text{M}$ params, and vanilla `MobileNetV3-Small` reaches **$45.66$** with top `TailF1` ($0.3135$). **Figure 3(b)** confirms Fold 3 spikes uniformly by ~15–20 MAE across all models (`E0_median`: $74.33$; `E2_resnet18_pre`: $58.45$).

### 5.3 PFRML Spine Ablations: Why Prototypes Degrade Regression

**Table 3.** PFRML 5-fold `station` CV ablation ladder (`E4` & `E5_reg_chi`) with paired fold differences vs. `R1` and `R2`.

| Rung / Experiment | Last-Ep MAE | Diff vs. `R1` (SE) | Won | Diff vs. `R2` (SE) | Won | $R^2$ | Acc | $\pm 1\text{Acc}$ | MacroF1 | TailMAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R1` (`E4_..._r1_no_prototypes`)** | **45.50 ± 8.64** | +0.00 | — | **-2.63 ± 1.10** | **4/5** | **0.3249** | 0.4080 | **0.8448** | 0.3859 | **115.37** |
| `R2` (`E4_..._baseline`, `chi_proto`) | **48.14 ± 7.61** | +2.63 ± 1.10 | 1/5 | +0.00 | — | 0.2424 | 0.4088 | 0.8181 | 0.3829 | 124.19 |
| `R2` Eq. 5 (`E5_..._reg_chi`) | **48.04 ± 8.20** | +2.54 ± 0.42 | 0/5 | -0.09 ± 0.73 | 2/5 | 0.2538 | 0.4025 | 0.8182 | 0.3705 | 123.27 |
| `R2+T1` (`..._attn_pool`) | **48.40 ± 7.74** | +2.90 ± 1.01 | 1/5 | +0.27 ± 0.32 | 2/5 | 0.2429 | 0.4032 | 0.8148 | 0.3789 | 121.92 |
| `R2+T2` (`..._physics_gate`) | **47.57 ± 6.73** | +2.07 ± 1.00 | 1/5 | -0.57 ± 0.86 | 4/5 | 0.2761 | 0.4037 | 0.8243 | 0.3823 | 118.49 |
| `R2+T3` (`..._class_weights`) | **47.06 ± 8.01** | +1.55 ± 0.73 | 1/5 | **-1.08 ± 0.49** | 4/5 | 0.2822 | 0.4113 | 0.8142 | 0.3794 | 124.33 |
| `R2+T4` (`..._dcm_concat`) | **60.51 ± 5.41** | +15.01 ± 3.38 | 0/5 | +12.38 ± 3.49 | 0/5 | 0.0556 | 0.3041 | 0.6664 | 0.2515 | 168.58 |
| `R2+all` (`E4_..._full`, T1+T2+T3) | **47.16 ± 7.15** | +1.65 ± 0.92 | 2/5 | -0.98 ± 0.90 | 4/5 | 0.2774 | **0.4150** | 0.8136 | **0.3861** | 121.05 |
| Extra B (`E4_mdnet_t_baseline_ep60`) | **66.56 ± 4.88** | +21.05 ± 2.97 | 0/5 | +18.42 ± 2.95 | 0/5 | 0.0054 | 0.1489 | 0.4432 | 0.1112 | 166.28 |

![Figure 4: PFRML ablation ladder and training/validation collapse diagnostics](outputs/figures/fig4_pfrml_ablation_and_collapse.png)

**Table 3** and **Figure 4(a)** isolate three mechanistic takeaways:
1. **`R1` (`DCM` + `LightTransformer` without prototypes) improves over `E2`:** `R1` reaches **$45.50 \pm 8.64$**, beating `E2_mobilenetv3_s_pretrained` on 4/5 folds (`-0.16 ± 0.36`) while improving `TailMAE` ($119.16 \to 115.37$) and `TailF1` ($0.3135 \to 0.3280$).
2. **Prototype classification loss $\mathcal{L}_{\text{cls}}$—not just concatenation—degrades regression:** Enabling EMA prototypes (`R2`) raises MAE by $+2.63 \pm 1.10$ ($48.14$, losing 4/5 folds to `R1`, $R^2: 0.3249 \to 0.2424$). Crucially, **`E5_mobilenetv3_s_reg_chi`** (literal Eq. 5, dropping $\tilde{\mathbf{c}}_i$ from the regressor while keeping prototype CE active) remains equally degraded at **$48.04 \pm 8.20$** (losing **5/5 folds** vs. `R1` and `E2`). When 6-class separability saturates near $\approx 41\%$, pulling $\chi_i$ toward discrete centroids disrupts smooth ordinal haze geometry.
3. **Domain toggles aid `R2` modestly:** Class weights (`T3`, `-1.08 ± 0.49`) and `PhysicsPriorGate` (`T2`, `-0.57 ± 0.86`, tightest SD $6.73$) improve `R2` on 4/5 folds, and `R2+all` (`full`) peaks in CV classification (`Acc = 0.4150`, `MacroF1 = 0.3861`), yet all `R2` variants lag prototype-free `R1`.

### 5.4 Optimization Collapse and Multi-Task Overconfidence Diagnostics

Loss trajectories (`outputs/diagnostics/collapse.md`, **Figure 4(b)**) reveal two optimization failures:
- **Scratch `MDNet-T` + `PFRML` stall (`E4_mdnet_t_baseline_ep60`):** Whereas `E3_mdnet_t_reg` cuts `train_loss` >95% ($0.72\text{–}0.92 \to 0.014\text{–}0.044$), `pfrml_mdnet_t` at `lr=1e-3` stalls across **5/5 folds** (`train_loss`: $2.1609 \to 2.1784$; `MAE = 66.56 ± 4.88`; $R^2 = 0.0054$). A Fold-0 probe at `lr=2e-4` (`E5_mdnet_t_pfrml_lr2e-4_f0`, 1 fold) avoids collapse (`MAE = 58.15`, $R^2 = 0.2910$) but still trails Fold-0 `E3_mdnet_t_reg` ($48.37$).
- **`DCM` concatenation bottleneck (`R2+T4`):** Switching `DCM` from summation to `896 -> 224` concatenation stalls optimization across **5/5 folds** (`train_loss`: $2.0567 \to 1.9196$, ratio $0.85\text{–}0.98$ vs. $2.00 \to 0.46$ in `R2`), degrading MAE by $+12.38 \pm 3.49$ ($60.51 \pm 5.41$, $R^2 = 0.0556$).
- **Cross-entropy U-curve:** In working `R2`, `val_loss` minimizes early at **epochs 2–6** ($1.53\text{–}2.13$) before rising to $2.21\text{–}2.81$ via CE overconfidence even as validation MAE keeps improving to epoch 30 ($48.14$).

### 5.5 Multimodal Fusion and `CMA-3` Early Stopping (`spatial_cluster` CV)

**Table 4.** Multimodal fusion 5-fold `spatial_cluster` CV (25 km separation) under normal and `meta_off` evaluation.

| Experiment | Eval Mode | Last-Ep MAE | Diff vs. `E1_sp` (SE) | Won | Diff vs. `E2_sp` (SE) | Won | $R^2$ | SRCC | Acc | $\pm 1\text{Acc}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E1_metadata_mlp` | Meta Only | **46.00 ± 7.33** | 0.00 | — | -4.68 ± 1.22 | 5/5 | 0.3647 | 0.7147 | 0.4183 | 0.8603 |
| `E2_..._spatial` | Vision Only | **50.69 ± 5.60** | +4.68 ± 1.22 | 0/5 | 0.00 | — | 0.2142 | 0.6328 | 0.3739 | 0.8082 |
| `E5_fusion_full_p03` | Normal / `meta_off` | **51.68** / **54.94** | +5.68 / +8.94 | 0/5 | +1.00 / +4.25 | 2/5 / 0/5 | 0.1883 / 0.1121 | 0.6608 / 0.5901 | 0.3387 | 0.7727 |
| `E5_fusion_r1_p03` | Normal | **49.18 ± 6.63** | +3.18 ± 1.12 | 1/5 | -1.51 ± 0.72 | 4/5 | 0.2484 | 0.6722 | 0.3769 | 0.8245 |
| `E5_fusion_r1_p03` | `meta_off` | **52.28 ± 5.95** | +6.28 ± 1.23 | 0/5 | +1.60 ± 0.56 | 0/5 | 0.1839 | 0.6112 | 0.3483 | 0.7951 |
| `E5_fusion_r1_p00` | Normal | **47.45 ± 6.71** | +1.44 ± 1.28 | 1/5 | -3.24 ± 0.79 | 5/5 | 0.2869 | 0.6994 | 0.3956 | 0.8358 |
| `E5_fusion_r1_p00` | `meta_off` | **53.54 ± 5.49** | +7.54 ± 1.58 | 0/5 | +2.85 ± 1.04 | 0/5 | 0.1624 | 0.6041 | 0.3335 | 0.7860 |
| **`E5_fusion_r1_p00_ep7`** | **Normal** | **44.40 ± 7.30** | **-1.60 ± 0.59** | **5/5** | **-6.29 ± 1.07** | **5/5** | **0.3674** | **0.7330** | **0.4318** | **0.8640** |
| `E5_fusion_r1_p00_ep7` | `meta_off` | **52.58 ± 5.06** | +6.58 ± 1.49 | 0/5 | +1.90 ± 1.05 | 1/5 | 0.1988 | 0.6240 | 0.3404 | 0.7936 |

![Figure 5: Multimodal fusion spatial_cluster CV, CMA-3 epoch selection, and meta_off ablation](outputs/figures/fig5_fusion_early_stopping.png)

On 25 km `spatial_cluster` CV (**Table 4**, **Figure 5**), `R1` again outperforms `PFRML full` (`49.18` vs. `51.68` at `p_drop=0.3`), while `p_drop=0.0` achieves lower 30-epoch multimodal MAE (`47.45`) with `aux_reg_head` retaining `meta_off` rank correlation ($\text{SRCC} = 0.604\text{–}0.624$). Crucially, `E5_fusion_r1_p00`'s trajectory (`outputs/results/step5c_epochs.md`, **Figure 5(a)**) bottoms out at **$45.74 \pm 6.78$ MAE at epoch 7** before drifting upward to $48.11$ (ep 20) and $47.45$ (ep 30) as `MetadataMLP` overfits training-cluster coordinates. Minimizing `CMA-3` for `epoch >= 6` selects **$N = 7$** (`CMA-3 = 46.26 < 46.95` threshold); retraining a fresh 7-epoch cosine schedule (`E5_fusion_r1_p00_ep7`) reaches **$44.40 \pm 7.30$** (`F0=55.09, F1=35.76, F2=43.09, F3=47.36, F4=40.69`), **beating both `E1_metadata_mlp` (`-1.60 ± 0.59`) and `E2_spatial` (`-6.29 ± 1.07`) on all 5/5 folds**.

### 5.6 One-Shot Test & Unseen-Region (`>25 km`) Generalization

**Table 5.** Final one-shot test evaluation across Full Test ($n=2,920$, 729 stations) and `test_gt25km` ($n=717$, 205 stations) with 1,000-resample station-cluster 95% CIs.

| Model | Split | $n$ (St.) | MAE [95% Station CI] | RMSE | PLCC | SRCC | $R^2$ | Acc | $\pm 1\text{Acc}$ | MacroF1 | TailMAE | TailF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F_FUSION_R1`** (7ep) | `test` | 2920 (729) | **35.77** [30.47, 42.72] | **51.07** | **0.7943** | **0.7941** | **0.6070** | **0.4565** | **0.8938** | **0.4150** | 76.92 | **0.2899** |
| `E1_metadata_mlp` | `test` | 2920 (729) | **40.34** [34.45, 48.31] | 57.70 | 0.7098 | 0.7344 | 0.4982 | 0.4096 | 0.8873 | 0.3397 | **76.82** | 0.1362 |
| `F_E2` (`MobileNetV3-S`) | `test` | 2920 (729) | **41.12** [33.97, 50.79] | 61.95 | 0.6749 | 0.7071 | 0.4215 | 0.4240 | 0.8620 | 0.3723 | 96.41 | 0.2252 |
| `F_PFRML_full` | `test` | 2920 (729) | **45.04** [37.51, 54.04] | 68.56 | 0.6039 | 0.6574 | 0.2914 | 0.3986 | 0.8134 | 0.3687 | 96.54 | 0.2752 |
| `F_FUSION_R1_meta_off` | `test` | 2920 (729) | **47.56** [39.57, 58.12] | 66.12 | 0.6936 | 0.6757 | 0.3411 | 0.3428 | 0.7897 | 0.2556 | 119.80 | 0.0805 |
| **`F_FUSION_R1`** (7ep) | `>25 km` | 717 (205) | **39.22** [30.85, 47.74] | **50.68** | **0.6464** | **0.6696** | **0.3743** | 0.3473 | 0.7741 | 0.2352 | **86.00** | 0.0000 |
| `F_E2` (`MobileNetV3-S`) | `>25 km` | 717 (205) | **39.60** [28.06, 54.03] | 54.56 | 0.5689 | 0.5623 | 0.2747 | **0.4003** | **0.8368** | **0.2743** | 116.42 | 0.0000 |
| `F_FUSION_R1_meta_off` | `>25 km` | 717 (205) | **44.88** [32.15, 58.48] | 59.60 | 0.5091 | 0.5460 | 0.1346 | 0.3333 | 0.7699 | 0.2165 | 121.83 | 0.0000 |
| `F_PFRML_full` | `>25 km` | 717 (205) | **46.79** [34.22, 61.06] | 66.14 | 0.4121 | 0.5082 | -0.0660 | 0.3291 | 0.7448 | 0.2221 | 123.82 | 0.0000 |
| `E1_metadata_mlp` | `>25 km` | 717 (205) | **49.84** [40.18, 58.81] | 65.31 | 0.3797 | 0.4954 | -0.0393 | 0.2929 | 0.7476 | 0.2004 | 88.83 | 0.0000 |

![Figure 6: Full test vs. unseen-region >25 km generalization and paired bootstrap differences](outputs/figures/fig6_final_test_generalization.png)

Held-out one-shot evaluation (**Table 5**, **Figure 6**) provides decisive paired station-bootstrap confirmation:
- **Vision-metadata reversal beyond 25 km:** `F_E2` and `E1` tie on full test (`+0.77`, `[-3.46, +4.78]`), but beyond 25 km `E1` loses $9.50$ MAE ($R^2 = -0.0393$) while `F_E2` improves ($39.60$, $R^2 = 0.2747$; `-10.24`, `[-20.50, +0.86]`).
- **`F_E2` beats `F_PFRML_full` significantly:** By **$-3.92$ MAE** (`[-6.27, -1.58]`) on full test and **$-7.18$ MAE** (`[-11.92, -2.53]`) on `test_gt25km` (where `F_PFRML_full` drops to $R^2 = -0.0660$).
- **`F_FUSION_R1` wins both splits:** On full test, `F_FUSION_R1` ($35.77$, $R^2 = 0.6070$, $\pm 1\text{Acc} = 89.38\%$) beats `F_E2` by **$-5.35$** (`[-8.62, -2.26]`), `E1` by **$-4.57$** (`[-8.31, -1.26]`), and `F_PFRML_full` by **$-9.26$** (`[-12.52, -6.06]`). Beyond 25 km, it leads in MAE (**$39.22$**), RMSE ($50.68$), $R^2$ ($0.3743$), `SRCC` ($0.6696$), and `TailMAE` ($86.00$), beating `E1` by **$-10.63$ (`[-18.78, -2.59]`)** and `F_PFRML_full` by **$-7.57$ (`[-14.29, -1.46]`)**.

### 5.7 Distance-Stratified Decay and Severity-Bin Errors

**Table 6.** Test MAE stratified by nearest-train Haversine distance band and US EPA AQI severity bin.

| Stratification | Subset ($n$ / St. / $\mu_{\text{AQI}}$) | `F_FUSION_R1` | `F_E2` | `E1_metadata` | `F_PFRML_full` | `meta_off` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0–5 km` / `5–25 km` | $n=1276$ (299 st.) / $n=927$ (246 st.) | **37.21** / **31.12** | 44.12 / 38.16 | 40.64 / 32.59 | 46.54 / 41.60 | 54.34 / 40.30 |
| `25–100 km` / `>100 km` | $n=498$ (151 st.) / $n=219$ (55 st.) | **40.14** / 37.12 | 43.08 / **31.71** | 45.40 / **59.96** | 51.07 / 37.05 | 48.81 / 35.96 |
| **Bins 1–3 (`0–150`)** | Good / Mod / USG ($629/630/630$) | **22.95 / 26.33** / 29.17 | 29.13 / 28.92 / **28.93** | 32.68 / 31.47 / 32.20 | 28.16 / 30.49 / 38.88 | 26.68 / **22.33** / 33.86 |
| **Bin 4–6 (`151+`)** | Unh ($630$) / V.Unh ($275$) / Haz ($126$) | **38.41** / 54.14 / **126.62** | 42.27 / 69.38 / 155.43 | 41.81 / **44.76** / 146.81 | 49.80 / 71.29 / 151.66 | 61.35 / 92.24 / 179.95 |

![Figure 7: Distance-stratified MAE decay and EPA severity-bin error breakdown](outputs/figures/fig7_distance_and_tail_errors.png)

Decomposing residuals (**Table 6**, **Figure 7**) pinpoints where modalities diverge. Inside `0–25 km`, `F_FUSION_R1` leads both sub-bands (`37.21`, `31.12`); beyond `100 km` (`55` remote stations), `E1` explodes to **`59.96` MAE** whereas pure vision `F_E2` reaches **`31.71`** and `F_FUSION_R1` holds **`37.12`**. Across EPA bins, `F_FUSION_R1` leads Good (`22.95`), Moderate (`26.33`), Unhealthy (`38.41`), and **Hazardous (`>300`, `126.62`)**. While all models underpredict extreme events (`y_pred - y_true` averages $-11.64$ overall and $-74.37$ on Bins 5–6 for `F_FUSION_R1` vs. $-94.91$ for `F_E2`), `F_FUSION_R1` retains the steepest calibration slopes on full test ($0.584$ vs. $0.535$ `E1`, $0.468$ `F_E2`) and `test_gt25km` ($0.446$ vs. $0.376$ `F_E2`, $0.290$ `E1`).

---

## 6. Discussion, Limitations, and Future Work

### 6.1 Visual Physics vs. Geospatial Shortcuts

Geospatial coordinates and month capture legitimate climatology near stations—allowing a 2,497-parameter MLP to rival vision within 25 km—yet fail catastrophically beyond 100 km (`59.96` MAE). Our experiments demonstrate that multimodal fusion need not trade away geographic invariance: combining a prototype-free `R1` visual spine with `MetadataMLP`, a detached vision auxiliary loss ($\lambda_{\text{aux}}=0.5$), and `spatial_cluster` `CMA-3` early stopping (`N=7`) improves local accuracy (`-5.35` MAE vs. `F_E2`) while beating metadata by `-10.63` MAE beyond 25 km. Without early stopping (`ep 7 -> 30`), fast-converging tabular weights gradually crowd out visual refinement.

### 6.2 Why Prototype Multitask Regularization Underperformed

Three factors explain why `PFRML` trailed simpler architectures on global PM25Vision despite strong single-region results: (1) heterogeneous camera optics and skies cap 6-class accuracy near $\approx 41\%$, causing cross-entropy to pull $\chi_i$ toward noisy centroids and warp continuous regression (proven by literal Eq. 5 `E5_mobilenetv3_s_reg_chi` matching `R2` at $48.04$ vs. `R1` at $45.50$); (2) uninitialized `MDNet-T` stages starve under hybrid prototype gradients ($R^2 = 0.0054$); and (3) cross-entropy overconfidence drives premature `val_loss` minima at epochs 2–6 while continuous MAE continues improving.

### 6.3 Limitations and Future Work

Three concrete gaps motivate follow-up research: (1) closing the standalone `meta_off` fallback gap (`47.56` full / `44.88` $>25\text{ km}$ vs. `F_E2` `41.12` / `39.60`) via asymmetric learning rates (`lr_meta < lr_image`) and annealed modality dropout; (2) mitigating Bins 5–6 tail compression (`slope 0.45–0.58`) with $\log(1+\text{AQI})$ or relevance-weighted Huber losses plus spherical-harmonic geolocation (`SatCLIP`); and (3) integrating global surface reanalysis humidity and sky segmentation to separate clean humid fog from hazardous haze without station co-pollutant leakage.

---

## 7. Conclusion

Through deterministic leakage purging, dual `station` and 25 km `spatial_cluster` cross-validation, station bootstrapping, and last-epoch reporting on **PM25Vision**, this study disentangles authentic atmospheric vision from spatial and optimization artifacts. Metadata-only priors collapse beyond 25 km ($R^2 = -0.0393$; `59.96` MAE $>100\text{ km}$), whereas pretrained `MobileNetV3-Small` remains invariant (`39.60` $>25\text{ km}$; `31.71` $>100\text{ km}$). ImageNet pretraining ($-7.6$ to $-9.2$ MAE) and prototype-free `DCM` summation (`R1`, $45.50$) outperform scratch CNNs and prototype multi-task heads (`48.14`–`66.56`). Finally, `F_FUSION_R1` (`2.02M` params, 7 epochs via `CMA-3`) achieves $\text{MAE} = 35.77$ ($R^2 = 0.6070$) on full test and $\text{MAE} = 39.22$ ($R^2 = 0.3743$) beyond 25 km.

---

## Reproducibility & Artifact Provenance

- **Authoritative Metrics:** [`outputs/results/REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/REPORT_NUMBERS.md) (generated by [`scripts/build_report_numbers.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/build_report_numbers.py)).
- **Diagnostics:** [`outputs/diagnostics/folds.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/folds.md), [`outputs/diagnostics/collapse.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/collapse.md), [`outputs/results/step5c_epochs.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/step5c_epochs.md), [`outputs/results/final_comparison.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/results/final_comparison.md).
- **Figure Pipeline:** [`scripts/generate_report_figures.py`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/scripts/generate_report_figures.py) $\to$ `outputs/figures/fig{1..7}_*.png` (300 DPI).
- **Environment:** Apple M5 Pro (24 GB, `macOS-27.2.1-arm64`), Python `3.14.6`, PyTorch `2.14.1` (`mps`, `fp32`), seed `42`.
