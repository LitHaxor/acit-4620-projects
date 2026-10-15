# Image-based Air Quality Estimation (ACIT 4620)

Estimating PM2.5 air quality from ordinary street-level photos. This repository contains the exploratory data analysis (EDA) of the **PM25Vision** dataset, a study of which image filters respond to haze, and the feature extraction (hand-crafted haze filters and pretrained CNN embeddings) that later models will build on.

Course project for ACIT 4620 at OsloMet.

## Contents

| Path | What it is |
|---|---|
| [`1.eda-image-datasets.ipynb`](1.eda-image-datasets.ipynb) | EDA, data cleaning, filter analysis and feature extraction (code cells only, outputs included) |
| [`paper_taxonomy.xlsx`](paper_taxonomy.xlsx) | Literature taxonomy of 16 IEEE papers (2024–2026) on image-based air-quality estimation: datasets, models, accuracy, vision filters, red flags |
| [`requirements.txt`](requirements.txt) | Python dependencies for the notebook |
| `datasets/` | PM25Vision images and metadata (not committed, see below) |
| `features/` | Generated feature files (not committed, rebuilt by the notebook) |
| `papers/` | Local copies of the reviewed papers (not committed, copyrighted) |

## Dataset

[PM25Vision](https://www.kaggle.com/datasets/DeadCardassian/pm25vision) ([paper](https://arxiv.org/abs/2509.16519), [Hugging Face](https://huggingface.co/datasets/DeadCardassian/PM25Vision)) pairs crowdsourced Mapillary street-level photos with PM2.5 readings from World Air Quality Index stations. It is licensed CC BY 4.0.

Download it and place it so the folder looks like this:

```
datasets/
├── train/
│   ├── images/            # <image_id>.jpg
│   ├── samples_by_bin/    # 30 example images per AQI bin (not used by the notebook)
│   └── metadata.csv
└── test/
    ├── images/
    ├── samples_by_bin/
    └── metadata.csv
```

> **Note:** the `pm25` column is the **US EPA AQI value** (1–530), not a concentration in µg/m³. The `pm25_bin` categories (0–50, 51–100, …, 301–600) are the AQI bands.

## Setup

Python 3.12 is what the notebook was run with.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ipykernel          # needed to run the notebook in VS Code or Jupyter
```

Open `1.eda-image-datasets.ipynb`, select the `.venv` kernel and run all cells. The notebook picks the fastest available device automatically (Apple MPS, CUDA, or CPU). On an Apple Silicon Mac the full run takes about 3 minutes: about 85 s for the filter features and about 45 s for the CNN embeddings. The first run downloads the EfficientNet-B0 ImageNet weights (about 20 MB).

## What the notebook does

1. **Loads and checks the metadata.** It drops the columns that are empty or constant (`camera_angle`, `quality_score`, `quality`) and confirms every row has its image file.
2. **Cleans the data.**
   - 80 images were linked to two or three nearby stations, usually with different labels. Each image now keeps one row with the median AQI.
   - 18 images appeared in both train and test. They are removed from train to prevent leakage.
   - Result: 11,096 images (8,176 train, 2,920 test) with no overlap.
3. **Explores the data:** AQI distribution, capture year and season by hemisphere, a world map of image locations, AQI by approximate region, images per station, image sizes and aspect ratios, and sample images for each AQI category.
4. **Builds a bank of 9 haze-related image filters:** dark channel, dark-channel-prior transmission, brightness, saturation, brightness minus saturation (colour attenuation), local contrast, Laplacian, Sobel and blue ratio. It shows each one on a clean, a moderate and a heavily polluted image.
5. **Extracts 94 hand-crafted features per image.** Each filter is summarised by its mean, 10th and 90th percentile over the full image, the sky region (top third) and the ground region. Contrast, entropy, colourfulness, atmospheric light and JPEG size per megapixel are added on top.
6. **Selects filters on the training split only.** Features are ranked by Spearman correlation with AQI, kept only if their direction is the same in at least 80% of geographic regions, and pruned so no two selected features are near-duplicates. This gives 12 features.
7. **Extracts deep features:** 1,280-dimensional EfficientNet-B0 (ImageNet) embeddings for every image.
8. **Checks how much signal each feature set carries** with ridge regression. Cross-validation keeps each station's images in a single fold, and the final numbers come from the test stations, which the model never saw.

## Key findings

- **The split is well designed.** Train and test have identical AQI category shares and share no stations, so test results measure performance at new locations.
- **Images share labels.** 7,596 images share a station and a capture day, and therefore the same daily AQI label, with another image. Any validation should group by station; a random split would leak.
- **Pollution is seasonal and regional.** AQI peaks in each hemisphere's winter. The median ranges from 28 in Oceania to 142 in Asia.
- **The most reliable filters are the physics-based haze filters**: dark-channel transmission, saturation, blue ratio and colour attenuation. Their relationship with AQI points the same way in every region with enough data.
- **The sky region matters.** Haze filters correlate much more strongly with AQI on the sky than on the ground; for transmission, |ρ| is 0.28 versus 0.12.
- **Edge and texture filters are unstable.** Sobel and Laplacian score highest overall but reverse direction in the Middle East. Brightness, airlight and entropy carry almost no signal.
- Many photos are dashcam frames, with the car bonnet and timestamp overlays in the lower part of the image.

Ridge regression on log(1 + AQI), evaluated on unseen test stations:

| Feature set | Test MAE (AQI) | Test R² | Test Spearman ρ |
|---|---|---|---|
| Train median (no image) | 63.6 | −0.01 | — |
| All 94 hand-crafted filters | 53.2 | 0.18 | 0.53 |
| 12 selected filters | 58.3 | 0.04 | 0.41 |
| EfficientNet-B0 embeddings | 49.0 | 0.30 | 0.63 |
| EfficientNet-B0 + 12 selected filters | 48.6 | 0.31 | 0.64 |

These are linear baselines meant to show how much signal each feature set holds, not final models.

## Generated files (`features/`)

| File | Contents |
|---|---|
| `clean_metadata.csv` | One row per cleaned image, with region, hemisphere, year, month and image properties |
| `handcrafted_filter_features.csv` | The 94 filter features per image, plus IDs and labels |
| `efficientnet_b0_embeddings.npz` | `image_id`, `split` and 1,280-dimensional `embeddings` (float16) |
| `selected_filters.json` | The 12 selected filters, with correlation and regional consistency |
| `feature_set_comparison.csv` | The ridge regression comparison table |

## Literature

`paper_taxonomy.xlsx` summarises 16 IEEE papers from 2024–2026 on estimating air quality from photos. It has an overview dashboard, a detailed sheet with datasets, models, accuracy, vision filters and methodological red flags for each paper, and a catalogue of the datasets they use. The paper PDFs are not included in the repository.
