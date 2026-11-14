# Project State & Handoff Guide for Future AI Agents
> **Purpose:** This document is the primary onboarding briefing for any AI agent or human contributor picking up this project on a fresh branch. It details the project context, exact codebase structure, findings, technical debts/mess, and concrete architectural recommendations to build a clean, successful solution from scratch.

---

## 1. Project Context & Objectives

* **Course / Project:** ACIT 4620 (OsloMet) — *Image-Based Air Quality Estimation*
* **Core Task:** Predict continuous ambient air quality (**PM2.5 AQI on the US EPA scale, 0–500+**) from street-level outdoor photographs.
* **Dataset:** [PM25Vision](https://huggingface.co/datasets/DeadCardassian/PM25Vision) (Mapillary crowdsourced street imagery paired with World Air Quality Index ground stations).
* **Target Variable:** The column in the metadata is named `pm25`, but **critically, it represents US EPA AQI values (1–530)**, not concentrations in $\mu\text{g}/\text{m}^3$.

---

## 2. Directory Layout & Existing Artifacts

```
.
├── datasets/                              # Raw downloaded dataset (~11,114 images)
│   ├── train/                             # 8,298 metadata rows, 8,194 images
│   │   ├── images/                        # <image_id>.jpg
│   │   └── metadata.csv
│   └── test/                              # 2,921 metadata rows, 2,920 images
│       ├── images/                        # <image_id>.jpg
│       └── metadata.csv
├── features/                              # Generated features from Notebooks 1 & 2
│   ├── clean_metadata.csv                 # 11,096 deduplicated images with GPS & region
│   ├── handcrafted_filter_features.csv    # 94 handcrafted haze features (Notebook 1)
│   ├── band_filter_features.csv           # 54 band-sliced features (Notebook 2)
│   ├── minimal_features.csv               # 5 top forward-selected features
│   ├── minimal_feature_spec.json          # Spec defining top 5 features
│   ├── efficientnet_b0_embeddings.npz     # 1,280-d ImageNet embeddings (float16)
│   ├── selected_filters.json              # 12 selected filters from Notebook 1
│   ├── feature_set_comparison.csv         # Benchmark Ridge regression metrics
│   └── mlp_results.csv                    # MLP regression benchmark results
├── papers/                                # 19 research paper PDFs (IEEE 2024–2026)
├── paper_taxonomy.xlsx                    # Taxonomy spreadsheet reviewing 16 IEEE papers
├── download_dataset.py                    # Script that fetches parquet from HF and unpacks datasets/
├── best_model.pt                          # PyTorch weights for baseline CNN (Cell 33 in NB3)
├── requirements.txt                       # Python dependencies (torch, torchvision, pyarrow, etc.)
├── 1.eda-image-datasets.ipynb             # Notebook 1: EDA, filter bank, EfficientNet embeddings
├── 2.feature-extraction.ipynb             # Notebook 2: Band slicing, forward selection, MLP
├── 3.example_train_test.ipynb             # Notebook 3: PyTorch CNNs & ResSE Cross-Validation
└── PROJECT_EXPLAINER.md                   # Beginner guide on ML concepts and physics
```

---

## 3. What Was Done So Far & Key Results

### Phase 1: Data Cleaning & EDA (`1.eda-image-datasets.ipynb`)
* **Deduplication:** 80 images associated with multiple conflicting stations were reduced to median AQI. 18 images present in both train and test were purged from train to eliminate data leakage.
* **Clean Dataset:** Exactly **11,096 images** (8,176 train, 2,920 test) across 3,261 monitoring stations.
* **Physics Filters:** 9 haze filters were designed: Dark Channel Prior (DCP), Transmission map, Saturation, Brightness, Color Attenuation, Blue Ratio, Sobel edges, Laplacian, and Local Contrast.
* **Key Finding (Sky vs Ground):** Haze filters correlate far more strongly in the **sky region (top third)** ($|\rho| \approx 0.28$) than in the ground region ($|\rho| \approx 0.12$).
* **Linear Ridge Baselines (evaluated on unseen test stations):**
  * Train Median (no image): $\text{MAE} = 63.64$, $R^2 = -0.01$
  * 12 Selected Handcrafted Filters: $\text{MAE} = 58.35$, $R^2 = 0.04$, $\rho = 0.41$
  * All 94 Handcrafted Filters: $\text{MAE} = 53.18$, $R^2 = 0.18$, $\rho = 0.52$
  * EfficientNet-B0 Embeddings: $\text{MAE} = 49.02$, $R^2 = 0.30$, $\rho = 0.63$
  * **EfficientNet-B0 + 12 Selected Filters:** $\text{MAE} = \mathbf{48.62}$, $R^2 = \mathbf{0.31}$, $\rho = \mathbf{0.64}$

### Phase 2: Feature Selection & Tabular Models (`2.feature-extraction.ipynb`)
* Sliced images into 3 bands (top $0–64$, middle $64–128$, bottom $128–192$ px) using 6 core filters $\to 54$ features.
* Used greedy forward selection with 5-fold `GroupKFold` by `station_id` to pick 5 top features:
  1. `saturation_top_p90`
  2. `edges_top_p10`
  3. `dark_channel_middle_p10`
  4. `brightness_middle_p10`
  5. `local_contrast_top_p10`
* Tested Multi-Layer Perceptron (MLP) on tabular metadata:
  * **Metadata only (GPS spherical $x, y, z$ + cyclic $\sin/\cos$ month):** Test $\text{MAE} = \mathbf{36.49}$, $R^2 = \mathbf{0.598}$, $\rho = \mathbf{0.782}$.
  * *Crucial Takeaway:* Location and season provide an exceptionally strong prior for air pollution baseline.

### Phase 3: PyTorch Deep Learning (`3.example_train_test.ipynb`)
* Built an on-GPU `HazeFilters` layer that stacks 5 haze maps onto the 3 RGB channels (8 channels total).
* Model 1 (Baseline CNN, 4 stages, max pooling): 5-fold CV $\text{MAE} = \mathbf{57.0 \pm 5.0}$.
* Model 2 (ResSE — Residual Blocks + Squeeze-and-Excitation channel attention, Hasan & Shakir 2026): 5-fold CV $\text{MAE} = \mathbf{48.3 \pm 6.1}$.
* Apple Silicon Acceleration: Runs on `mps` (Apple M5 Pro), completing 1 epoch of ~8,200 images in ~37 seconds.

---

## 4. Why the Current Codebase Needs a Clean Rewrite

The user requested a restart from scratch because the current code grew organically and has significant architectural issues:

1. **Monolithic, Fragile Jupyter Notebooks:**
   * Business logic, data pipelines, model definitions, and plotting are intertwined in 40+ notebook cells.
   * State leaks across cells (e.g., globals like `model`, `learn_images`, `data`).
   * No modular Python package (`src/` or `aqi/`).
2. **In-Memory Image Loading Bottleneck:**
   * Notebook 3 attempts to load all ~8,200 images into a single in-memory PyTorch tensor `train_images` (`learn_images`). This consumes huge RAM and forces fixed image downsampling to $192 \times 144$, which blurs fine distant haze cues.
   * Standard PyTorch `Dataset` with lazy disk loading (`PIL.Image.open` per item) was not used.
3. **Disconnected Modalities (Tabular vs Vision):**
   * Notebook 2 showed that metadata (GPS + Month) achieves $\text{MAE} = 36.5$.
   * Notebook 3 trains a CNN purely on images, achieving $\text{MAE} = 48.3$.
   * **They were never combined into an end-to-end multimodal architecture!**
4. **Model Saving Bug in Notebook 3:**
   * `best_model.pt` on disk contains the *inferior* Baseline CNN (Cell 33, MAE 57.0), because the winning ResSE model (Cell 41, MAE 48.3) was only evaluated in cross-validation and never saved.
5. **No Reproducible CLI / Config:**
   * Running experiments requires clicking through cells or using `nbconvert` rather than passing clean arguments (e.g. `python train.py --model resse --batch-size 64`).

---

## 5. Critical Research Insights & Traps (`papers/` & `paper_taxonomy.xlsx`)

Before writing any new model, the new agent **must** understand the following findings from the 19 reviewed IEEE papers:

### ⚠️ Trap 1: The Stationary Camera Leakage
Many published papers claim $90\%–97\%$ accuracy because they randomly split images from fixed rooftop cameras. The network simply memorized building silhouettes and sun angles.
* **Rule:** Always split or cross-validate grouped by `station_id` (`GroupKFold` or `StratifiedGroupKFold`). Never do a random split.

### ⚠️ Trap 2: Aggressive Color Jitter Data Augmentation
In standard classification (ImageNet), color jittering helps. In AQI estimation, **color (blue ratio) and brightness ARE the physical signal**. Random hue/saturation shifts actively corrupt the label correlation. Use horizontal flips and subtle random crops only.

### 💡 Technique 1: Two-Branch Multimodal Fusion
* Combine an Image Feature Extractor (Backbone) with a Metadata MLP (cyclic month + spherical GPS coords).
* The metadata gives the global prior (e.g., Delhi in December has high baseline pollution); the image provides the local deviation (e.g., it rained today, so the air is clear).

### 💡 Technique 2: Multi-Task Learning (MTL)
* Simultaneously train on:
  1. Continuous AQI regression (L1 loss or Smooth L1/Huber loss).
  2. 6-class AQI classification (Cross-Entropy loss for EPA bins).
* Papers demonstrate that classification loss regularizes the regression feature space and prevents extreme drift.

### 💡 Technique 3: Sky Region Isolation
* The sky region holds $2\times$ the correlation of the foreground. Models that use spatial attention or separate sky-crop encoders consistently outperform whole-image models that get confused by cars and roads.

---

## 6. Recommended Architecture for the Fresh Branch

To build a professional, publication-quality codebase, structure the new repository as follows:

```
├── configs/
│   └── default.yaml               # Hyperparameters, batch size, learning rates
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset: loads image on-the-fly + metadata
│   ├── transforms.py              # Physics-preserving augmentations & normalization
│   ├── filters.py                 # Fast vectorized Dark Channel, Transmission, Saturation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_backbone.py     # EfficientNet-B0 / ResSE / Swin-T
│   │   ├── metadata_mlp.py        # Dense network for GPS + Month
│   │   └── multimodal_net.py      # Dual-branch fusion + Multi-task heads
│   ├── losses.py                  # Smooth L1 + Cross-Entropy multi-task loss
│   └── utils.py                   # Station-grouped CV splits, evaluation metrics
├── train.py                       # Modular CLI training script with checkpoints & logging
├── evaluate.py                    # Evaluation script on held-out test stations
└── notebooks/                     # Kept only for clean presentation figures & inspection
```

### Multimodal Network Design (Recommended Spec)
```
[Image: 224x224] ──► [Pretrained Backbone (e.g. EfficientNet-B0 / ResSE)] ──► [Vision Vector: 128-d] ──┐
                                                                                                        ├──► [Concat: 160-d] ──► [FC Fusion]
[Metadata: GPS + Month] ──► [Dense Layers (64 -> 32)] ──────────────────────► [Meta Vector: 32-d]   ──┘                          │
                                                                                                                                  ├──► Head 1: Continuous AQI (Regression)
                                                                                                                                  └──► Head 2: 6 AQI EPA Bands (Classification)
```

---

## 7. Quick-Start Guide for the New Agent

1. **Verify Environment:**
   * Python environment is at `./env` (`./env/bin/python`).
   * PyTorch has MPS support (`torch.backends.mps.is_available() == True`).
2. **Verify Dataset:**
   * `datasets/train/` (8,194 images) and `datasets/test/` (2,920 images) are already downloaded and verified on disk.
   * `features/clean_metadata.csv` contains pre-cleaned deduplicated labels and GPS coordinates.
3. **If Re-downloading is Ever Needed:**
   * Run `python download_dataset.py`.
4. **First Priority:**
   * Implement `src/dataset.py` with standard PyTorch `Dataset` that yields `(image, metadata_tensor, aqi_target, aqi_bin)`.
   * Implement `train.py` with 5-fold `GroupKFold` grouped by `station_id`.
