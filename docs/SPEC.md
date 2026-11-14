# Technical Specification (`SPEC.md`)

## 1. Idea sources
- The two core papers (**PFRML**, Wang et al., IEEE TGRS 2026; and **MDNet**, Wang et al., IEEE TII 2024) are the spine: the final model must visibly implement their core ideas.
- The other 17 papers in `papers/markdown/` are a toolbox. You may propose and use ideas from them. Read `docs/related_work_notes.md` first; open a full paper only when needed, one at a time, and tell the user which.
- Every borrowed idea gets a config toggle and its own ablation, is cited (paper + section) in the code docstring and in `docs/SPEC.md`, and never replaces the spine.
- Small toggled ideas: you may add them yourself. Anything that changes the architecture or costs more than ~half a day: propose it to the user first.

### Approved Toggled Ablations (from `docs/related_work_notes.md`)
1. **Physics-prior scalars + sigmoid-gated fusion:** 4-scalar atmospheric physics descriptor `[entropy, RMS contrast, dark-channel transmission t, extinction -ln(t)]` computed inside the model as static PyTorch ops on pre-normalisation pixels (in `[0, 1]`), projected and fused via a learned sigmoid gate $\alpha = \sigma(W[F_{\text{deep}}; F_{\text{phys}}] + b)$.
   - *Sources:* **Paper 7** (Akdim et al., IEEE ICCSC 2026, Sec. 3.3 & 3.5 — `[Entropy, Contrast, Transmission]` + sigmoid gate), **Paper 15** (Yadav et al., IEEE ICCSAI 2025, Sec. IV / Table II — extinction coefficient $\propto -\ln t$), and **Paper 2** (Marzagui et al., IEEE WINCOM 2024, Sec. II–III).
2. **Learnable attention pooling over $14 \times 14$ tokens:** Single-layer spatial attention gate (`GWAP` / softmax token pooling) over the $14 \times 14 = 196$ spatial tokens from the DCM/Transformer, replacing uniform global average pooling to emphasize sky/haze regions and suppress foreground clutter.
   - *Sources:* **Paper 8** (Lee & Manova, IEEE Sensors J. 2025, `GWAP` Eq. 7) and **Paper 16** (Abdelsalam & Sayed, IEEE IMSA 2026, `SkyFuseNet` Sec. III / Eq. 2 & Fig. 7).
3. **Long-tail class weighting:** Dampened inverse-class-frequency weights $w_k \propto N_k^{-\gamma}$ (computed strictly within each training fold) applied in the classification/hybrid loss $L_{\text{cls}} + \beta L_{\text{reg}}$.
   - *Sources:* **Paper 7** (Akdim et al., IEEE ICCSC 2026, Eq. 11), **Paper 16** (Abdelsalam & Sayed, IEEE IMSA 2026, Sec. III-D), **Paper 14** (Byler et al., IEEE/CVF WACVW 2025, Sec. 4.1), and **Paper 6** (Mohan et al., IEEE RAIT 2025, `WERCS`).

### Parked Ideas (Do Not Implement)
- **Ordinal regulariser** on the 6 prototype logits (**Paper 4**, Nathvani et al., Sci. Total Environ. 2023, Sec. 3.5.1).
- **Geolocation encoding** (`SatCLIP` / multi-scale spherical-harmonic coordinate encodings in `MetadataMLP`; **Paper 14**, Byler et al. 2025, Sec. 4.2).
- **`ParNet` attention** parallel multi-scale channel-spatial attention block (**Paper 3**, Zhang et al., IEEE ICTech 2024, Sec. II-D).

### Free Evaluation & Uncertainty Diagnostics
- **$\pm 1$-class accuracy:** Adjacent-bin accuracy ($\lvert \hat{b}_i - b_i \rvert \le 1$) alongside exact 6-class accuracy and macro-F1 (**Paper 4**, Nathvani et al. 2023, Sec. 3.6).
- **Paired Wilcoxon signed-rank test across folds:** Non-parametric significance test comparing model variants across the 5 group-split folds (**Paper 17**, Han et al., IEEE Access 2025, Sec. IV; **Paper 11**, Kumaar & Deepak, IEEE ICAAID 2025, Sec. IV).
- **Prototype-mixture variance:** Closed-form predictive variance $\sigma^2(y) = \sum_{k=1}^6 p(y=k \mid \chi_i)\,(\hat{y}_k - \bar{y})^2$ obtained by evaluating the single regression MLP on each prototype-conditioned feature (adapted from **Paper 17**, Han et al. 2025, Sec. III-D uncertainty diagnostic).

---

## 2. Dataset & Step 1 Data Pipeline Specification (`src/data/`)

* **Source Dataset:** PM25Vision (`datasets/train/metadata.csv` with 8,298 rows / 8,194 images / 2,532 stations; `datasets/test/metadata.csv` with 2,921 rows / 2,920 images / 729 stations; 3,261 raw unique stations).
* **Target Variable:** Continuous US EPA AQI (`pm25` $\in [1, 530]$), scaled by `aqi_scale = 100.0` (`aqi = pm25 / 100.0` $\in [0.01, 5.30]$) for loss balance, paired with 6 discrete US EPA AQI bins (`aqi_bin` $\in \{0, 1, 2, 3, 4, 5\}$ corresponding to `0–50`, `51–100`, `101–150`, `151–200`, `201–300`, `301–600`, Paper 2 Table II).
* **Deterministic Cleaning (`src/data/cleaning.py`):**
  1. Purge the 18 overlapping `image_id`s present in both `train` and `test` from `train` (leaving **3,259 unique audited stations**: `2,530` train + `729` test).
  2. Deduplicate multi-station `image_id`s within each split by taking the median `pm25` AQI across rows sharing an `image_id` and recomputing `aqi_bin` / `pm25_bin` (retaining `2,527` primary `station_id`s in train and `729` in test, `3,256` post-dedup).
  3. Resulting clean counts: **11,096 unique images** (**8,176 train** across 2,527 primary stations / 2,530 pre-dedup; **2,920 test** across 729 stations; **3,259 unique audited stations** total; 0 train/test `image_id` or `station_id` overlap).
* **Cross-Validation Splits (`src/data/splits.py`):**
  * 5-fold `StratifiedGroupKFold` stratified by `aqi_bin` ($0\dots 5$) with two grouping modes (`group_mode`):
    1. `"station"`: grouped by `station_id` (`2,527` groups).
    2. `"spatial_cluster"`: grouped by connected-component spatial clusters (`1,059` groups) computed via `AgglomerativeClustering(metric="precomputed", linkage="single", distance_threshold=25.0)` on pairwise station Haversine distances in km.
* **Transforms (`src/data/transforms.py`):**
  * Whole-image bilinear resize to $224 \times 224$ (no cropping, keeping both sky and ground regions), horizontal flip ($p = 0.5$), gentle affine perturbation ($\pm 5^\circ$ rotation, $\pm 2\%$ translation, $0.98\text{--}1.02\times$ scale), and ImageNet normalization (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`). Strictly no colour jitter or grayscale (Paper 2 Sec. III-C).
* **Lazy Dataset (`src/data/dataset.py`):**
  * `PM25Dataset` returns `PM25Sample(image, metadata, aqi, aqi_bin, station_id)` where `metadata` is a 5D `float32` tensor `[x, y, z, sin_month, cos_month]` (3D unit-sphere GPS coordinates + cyclic month).

---

## 3. Spine Architecture Specification (Steps 2–6)

1. **Backbone — Multiscale Dilated CNN (`MDNet-T`, Paper 2 Sec. III-A, III-B, Fig. 1, Table I):**
   * 4 stages with channel widths $[96, 192, 384, 768]$, depths $[3, 3, 9, 3]$ (`4.9M` params, $\text{RF} = 1136$), spatial sizes $[56\times 56, 28\times 28, 14\times 14, 7\times 7]$.
   * Each `MDBlock` sums 3 parallel depthwise (`groups=C`) dilated branches: $(k=3, d=2, p=2)$, $(k=2, d=2, p=1)$, and $(k=1, d=1, p=0)$, followed by `GELU`, pointwise $1 \times 1$ conv ($C \to C$), `LayerNorm`, and residual shortcut.
   * **Author adaptation `E3b` (`MobileNetV3MDNeck` / `mobilenetv3_s_md_neck`, NOT part of Paper 2):** Author's own adaptation of Paper 2's `MDBlock` idea onto an ImageNet-pretrained MobileNetV3-Small backbone (`features` $\to [B, 576, 7, 7]$), followed by a $1\times 1$ conv reduction ($576 \to 128$) + `LayerNorm2d(128)`, $2$ sequential `MDBlock(128)` blocks (summing the 3 dilated depthwise branches), global average pool, `LayerNorm(128)`, and linear regression head (`128 -> 1`). Clearly marked as a user/author adaptation of Paper 2's idea rather than part of the original MDNet paper.
2. **Dimension Consistency Module (DCM) & Lightweight Transformer (Paper 1 Sec. II-B, Fig. 3, Table I):**
   * Projects all 4 stage outputs $C_1 \dots C_4$ to $D = 224$ channels via $1 \times 1$ conv, aligns spatial dimensions to $14 \times 14$ using fixed-stride `MaxPool2d(4,4)`, `MaxPool2d(2,2)`, `Identity`, and `Upsample(scale_factor=2, mode='bilinear')`, normalizes + average-pools, and fuses via element-wise summation (Paper 1 Table I `DCM Stage 5`, with config toggle for concatenation).
   * Flattens to 196 tokens + `[CLS]` token ($197 \times 224$), adds 2D learnable positional encoding, passes through a lightweight Transformer encoder ($\text{Dim}=224$, default 2 layers), and pools to $\chi_i \in \mathbb{R}^{224}$ (uniform GAP by default, or toggled learnable attention pooling over the $14 \times 14$ tokens).
3. **Prototype Multitask Heads & Hybrid Loss (Paper 1 Sec. II-C, II-D, Eq. 1–6, Fig. 5, Sec. III-A3, Sec. III-C3 Fig. 7; Paper 2 Eq. 2):**
   * **Prototype Classification Branch:** $K = 6$ pollution level prototypes $C' = \{c'_1, \dots, c'_6\} \in \mathbb{R}^{6 \times 224}$ with squared Euclidean distance $d(\chi_i, c'_k) = \|\chi_i - c'_k\|_2^2$ and negative-distance Softmax $p(y=k \mid \chi_i) = \text{Softmax}(-d(\chi_i, c'_k)/\tau)$.
   * **Prototype-Guided Regression Branch (Variant: *One MLP on a prototype-weighted feature*):** Computes the prototype-weighted representation $\tilde{c}_i = \sum_{k=1}^6 p(y=k \mid \chi_i) c'_k$, fuses it with $\chi_i$, and predicts scaled AQI via **one single regression MLP** $\hbar([\chi_i, \tilde{c}_i])$ (not 6 separate regression branches). When `use_prototypes: false` (`R1`), prototypes are omitted (`proto_cls = None`) and $\hbar(\chi_i)$ predicts scaled AQI directly from $\chi_i$ (`224 -> 256 -> 1`) under regression-only loss (`cls_loss = 0`).
     * *Paper citation & ambiguity note:* PFRML **Sec. II-D (Eq. 5, Fig. 5)** and **Sec. III-A3** define a single 3-layer regression MLP $\hbar(\cdot)$ (`1024 -> 256 -> 1` in Sec. III-A3) rather than 6 per-class regression branches. However, PFRML is slightly ambiguous on the MLP's input: **Sec. II-D Eq. (5) & Fig. 5** write $\hbar(\chi_i)$ directly on the backbone feature $\chi_i$ in parallel with the prototype branch, whereas **Sec. III-C3 (Fig. 7)** explicitly labels the prototype-constrained representation `"w/ prototype (prototype-weighted feature)"` without stating a fusion formula. We follow the `"prototype-weighted feature"` interpretation (`[\chi_i; \tilde{c}_i] -> \hbar`) so the prototype probabilities directly guide regression while preserving PFRML's single-MLP head.
   * **Hybrid Loss:** $L = L_{\text{cls}} + \beta L_{\text{reg}}$ ($\beta = 0.5$, Paper 1 Sec. III-C5), supporting Label Smoothing Cross-Entropy (Paper 2 Eq. 2, $\epsilon=0.1$) and target-scaled MSE (Paper 1 Eq. 5) / Huber loss, plus optional fold-internal class weights $w_k \propto N_k^{-\gamma}$.
4. **Multimodal Fusion & Shortcut Mitigation:**
   * Optional `MetadataMLP` branch ($5 \to 64 \to 32$) fused with vision features using **Modality Dropout** (zeroing the metadata embedding with probability $p_{\text{drop}}$ during training) + detached vision auxiliary loss.

---

## 5. Experiment ladder

All main PFRML ablation rungs use:
* **Backbone:** `mobilenetv3_s` pretrained (`pfrml_mobilenetv3_s`), multistage taps (`[C1..C4]` at strides `[4, 8, 16, 32]`).
* **Evaluation protocol:** `station` 5-fold cross-validation (`StratifiedGroupKFold`, `2,527` stations), `seed=42`, `lr=1e-3`, `30` epochs (`warmup_epochs=3`), `batch_size=32`, `fp32` (`use_amp=False`) — identical to the `E2_mobilenetv3_s_pretrained` run so comparisons are strictly fair.
* **Why all ablations run on all 5 folds (not only Fold 0):** As documented in [`outputs/diagnostics/folds.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/outputs/diagnostics/folds.md), per-fold validation MAE standard deviation across the 5 station folds is $\approx 10$ AQI points (`10.06` for `E2_mobilenetv3_s_pretrained`, `9.59` for `E3b`), and Fold 0 is the easiest fold (`MAE = 35.72` vs 5-fold mean `45.54`, while Fold 3 reaches `62.56` due to concentrated severe/hazardous episodes). Single-fold ablations on Fold 0 are therefore noisy and optimistic; every rung on the ladder is evaluated across all 5 folds.

### Ladder Rungs, Reference Rows & Extra Runs

| Rung / ID | Experiment ID | Model | Config | Toggles / Settings | Paper Citations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ref 1** *(existing, do not rerun)* | `E2_mobilenetv3_s_pretrained` | `mobilenetv3_s_pretrained` | `configs/default.yaml` | Pretrained MobileNetV3-S + linear regression head (`station` 5-fold CV: `MAE = 45.54 +- 10.06`) | Baseline CNN |
| **Ref 2** *(existing, do not rerun)* | `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40` | `mobilenetv3_s_md_neck` | `configs/default.yaml` | Author adaptation: MobileNetV3-S + 2 `MDBlock(128)` neck, `40` ep (`station` 5-fold CV: `MAE = 45.85 +- 9.59`) | Author adaptation of **Paper 2** (Wang et al., IEEE TII 2024, Sec. III-B) |
| **R1** | `E4_mobilenetv3_s_r1_no_prototypes` | `pfrml_mobilenetv3_s` | `configs/pfrml_r1_no_prototypes.yaml` | `use_prototypes: false` — DCM (`sum`) + Light Transformer (`GAP`) + single regression MLP $\hbar(\chi_i)$ (`224 -> 256 -> 1`) on pooled $\chi_i$; no prototype buffers, regression-only MSE loss (`cls_loss = 0`) | **Paper 1** (Wang et al., IEEE TGRS 2026, Sec. II-B, Sec. III-C3 & Fig. 7 `"w/o prototype"`) |
| **R2** *(PFRML baseline)* | `E4_mobilenetv3_s_baseline` | `pfrml_mobilenetv3_s` | `configs/pfrml_baseline.yaml` | `use_prototypes: true`, all 4 toolbox toggles **off** (`dcm_fusion="sum"`, uniform GAP, no physics gate, no class weights), hybrid loss $L_{\text{cls}} + 0.5 L_{\text{reg}}$ | **Paper 1** (Wang et al., IEEE TGRS 2026, Sec. II-B–II-D, Eq. 1–6, Table I, Fig. 3 & 5); **Paper 2** (Eq. 2 label smoothing $\epsilon=0.1$) |
| **R2+T1** | `E4_mobilenetv3_s_attn_pool` | `pfrml_mobilenetv3_s` | `configs/pfrml_attn_pool.yaml` | R2 + `use_attention_pooling: true` (`softmax` spatial token attention over $14 \times 14 = 196$ tokens) | **Paper 16** (Abdelsalam & Sayed, IEEE IMSA 2026, Sec. III Eq. 2 & Fig. 7); **Paper 8** (Lee & Manova, IEEE Sensors J. 2025, `GWAP` Eq. 7) |
| **R2+T2** | `E4_mobilenetv3_s_physics_gate` | `pfrml_mobilenetv3_s` | `configs/pfrml_physics_gate.yaml` | R2 + `use_physics_gate: true` (4-scalar physics descriptor `[entropy, contrast, t, -ln t]` + sigmoid gate) | **Paper 7** (Akdim et al., IEEE ICCSC 2026, Sec. 3.3 & 3.5); **Paper 15** (Yadav et al., IEEE ICCSAI 2025, Sec. IV / Table II); **Paper 2** (Marzagui et al., WINCOM 2024) |
| **R2+T3** | `E4_mobilenetv3_s_class_weights` | `pfrml_mobilenetv3_s` | `configs/pfrml_class_weights.yaml` | R2 + `use_class_weights: true` (fold-internal inverse-frequency weights $w_k \propto N_k^{-0.5}$) | **Paper 7** (Eq. 11); **Paper 16** (Sec. III-D); **Paper 14** (Byler et al., WACVW 2025, Sec. 4.1); **Paper 6** (Mohan et al., RAIT 2025) |
| **R2+T4** | `E4_mobilenetv3_s_dcm_concat` | `pfrml_mobilenetv3_s` | `configs/pfrml_dcm_concat.yaml` | R2 + `dcm_fusion: "concat"` ($4 \times 224 \to 224$ `1x1` conv + `LayerNorm2d` instead of element-wise sum) | **Paper 1** (Wang et al., IEEE TGRS 2026, Sec. II-B, Table I DCM fusion variant) |
| **R2+all** | `E4_mobilenetv3_s_full` | `pfrml_mobilenetv3_s` | `configs/pfrml_full.yaml` | R2 + T1 (`use_attention_pooling: true`) + T2 (`use_physics_gate: true`) + T3 (`use_class_weights: true`, `dcm_fusion: "sum"`) | **Paper 1** + **Papers 7, 8, 14, 15, 16** |
| **Extra run A** | `E2_mobilenetv3_s_pretrained_spatial` | `mobilenetv3_s_pretrained` | `configs/default.yaml` | Image-only `mobilenetv3_s_pretrained`, `group_mode="spatial_cluster"`, 5 folds, `lr=1e-3`, `30` ep (to compare directly with `E1_metadata_mlp` spatial-cluster CV `MAE = 46.00 +- 7.33`) | Step 1/2 spatial-cluster generalization check |
| **Extra run B** *(last in queue, 14h guard)* | `E4_mdnet_t_baseline_ep60` | `pfrml_mdnet_t` | `configs/pfrml_baseline.yaml` | `pfrml_mdnet_t` from scratch (`pretrained=False`), `pfrml_baseline.yaml`, `60` epochs, `lr=1e-3`, `station` 5-fold CV (the `60`-epoch, `lr=1e-3` schedule selected by Stage 2 for `MDNet-T`) | **Paper 1** (PFRML head) + **Paper 2** (MDNet-T backbone, Sec. III-A–III-B, Table I) |

### Queue Execution Order (`scripts/run_queue_step4.py`)
Config-major order (all 5 folds of a configuration complete before starting the next configuration):
1. `R2` (`E4_mobilenetv3_s_baseline`, `station`, folds `0..4`)
2. `R1` (`E4_mobilenetv3_s_r1_no_prototypes`, `station`, folds `0..4`)
3. `R2+all` (`E4_mobilenetv3_s_full`, `station`, folds `0..4`)
4. `R2+T1` (`E4_mobilenetv3_s_attn_pool`, `station`, folds `0..4`)
5. `R2+T2` (`E4_mobilenetv3_s_physics_gate`, `station`, folds `0..4`)
6. `R2+T3` (`E4_mobilenetv3_s_class_weights`, `station`, folds `0..4`)
7. `R2+T4` (`E4_mobilenetv3_s_dcm_concat`, `station`, folds `0..4`)
8. `Extra run A` (`E2_mobilenetv3_s_pretrained_spatial`, `spatial_cluster`, folds `0..4`)
9. `Extra run B` (`E4_mdnet_t_baseline_ep60`, `station`, folds `0..4` — guarded by a `14.0 h` total queue runtime projection check before starting).
