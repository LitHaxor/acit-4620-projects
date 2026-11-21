# Offline PM2.5 estimation on mobile devices from street-level images

ACIT 4620 group project at OsloMet (Mughilvannan Manivannan, Ahamed Hasan, and Mrinmoy Das).

## The problem

Regulatory PM2.5 monitoring stations are expensive and sparse, leaving large parts of Africa, South Asia, and Latin America without local air-quality readings. Airborne particles scatter and absorb sunlight, reducing contrast on distant objects and washing out colours in outdoor photographs. This project tests whether a small neural network running completely offline on a mainstream smartphone can estimate the US EPA Air Quality Index (AQI) from a single street-level photo.

## Research overview

All experiments run on a cleaned version of the global PM25Vision dataset: 8,176 training images from 2,527 stations and 2,920 test images from 729 stations, with zero station overlap and 18 duplicate images removed. To measure performance in areas without nearby monitoring stations, we also evaluate on the 717 test photos taken more than 25 km from any training photo (`test_gt25km`). Model selection uses five-fold `StratifiedGroupKFold` grouped by station or 25 km spatial clusters, each final model is evaluated once on the test set at its fixed last epoch, and confidence intervals come from a 1,000-resample station-level cluster bootstrap.

We compare nine model families under this protocol: a constant median baseline, ridge regression on handcrafted haze filters (18 and 54 features), a location-and-month MLP, two from-scratch CNNs with built-in haze filters (HazeCNN, ResSE), pretrained MobileNetV3-Small, PFRML, and Fusion R1 (a 2.0M-parameter MobileNetV3-Small with a DCM neck, light Transformer, and a metadata branch encoding spherical GPS coordinates and cyclic month). We also run zero-shot evaluations of Claude Opus 5.5 and Gemini 4 Argon through their command-line agents on a paired 200-photo sample, and measure model footprints against mainstream CPU-only smartphone hardware.

Summary of findings:

| Model | Input | Full test MAE | Unseen (>25 km) MAE | fp16 size |
| :--- | :--- | :---: | :---: | :---: |
| Fusion R1 | photo, GPS, month | 35.6 | 39.3 | 3.85 MB |
| Metadata MLP | GPS, month | 36.8 | 48.6 | 0.005 MB |
| MobileNetV3-Small | photo | 41.8 | 39.2 | 2.90 MB |
| PFRML full | photo | 44.1 | 46.5 | 4.02 MB |
| ResSE | photo | 45.1 | 43.5 | 2.28 MB |
| HazeCNN | photo | 51.4 | 45.7 | 1.14 MB |

Fusion R1 achieves the lowest offline error (35.6 AQI MAE, $R^2 = 0.60$, 90% within one EPA category) and stays within 0.3 MAE across training seeds. Location metadata alone nearly matches it near seen stations (36.8) but breaks down beyond 25 km (48.6, $R^2 = -0.05$), where Fusion R1 and image-only MobileNetV3-Small hold steady. On the paired 200-photo sample, Claude Opus 5.5 (28.9 MAE) and Gemini 4 Argon (29.7 MAE) reach comparable accuracy to Fusion R1 (31.9 MAE, differences not statistically significant), so running locally is motivated by privacy, offline availability, low latency, and zero per-query cost. In fp16, Fusion R1 takes 3.85 MB on disk and under 30 MB of working memory, fitting CPU-only phones with at least 3 GB of RAM. Every tested model still underestimates hazardous air quality (AQI > 300).

The full paper is in `report/final_report.tex`.

## Running the project

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/data/download_dataset.py   # downloads PM25Vision into datasets/
python -m pytest -q                       # runs the 55 unit tests
```

Training uses PyTorch on Apple MPS, CUDA, or CPU automatically.

### Notebooks

Run the notebooks in order to reproduce the data audit, feature analysis, model training, frontier comparison, deployment measurements, and all report figures:

1. `1.eda-image-datasets.ipynb` for dataset audit, cleaning, and exploratory analysis.
2. `2.feature-extraction.ipynb` for haze filters, band statistics, and ridge baselines.
3. `3.example_train_test.ipynb` for HazeCNN and ResSE grouped cross-validation.
4. `4.models-eval-and-result.ipynb` for training and scoring all nine models, seed checks, and plots (`outputs/notebook4/`).
5. `5.frontier-and-mobile-deployment.ipynb` for Claude/Gemini agent queries, Fusion R1 head analysis, occlusion maps, model footprints, fp16 checks, and final report figures.

Notebook 5 reuses the 800 saved responses in `results/frontier/predictions.jsonl` and only invokes the `claude` or `agy` CLIs if entries are missing.

### Building the report PDF

PDFs are gitignored. Compile the report locally with Tectonic:

```bash
cd report && tectonic final_report.tex
```

### Command-line scripts

To verify tables and bootstrap intervals from the tracked `results/` snapshot without downloading the dataset:

```bash
python scripts/analysis/compare_final_models.py --results-dir results
python scripts/analysis/build_report_numbers.py --results-dir results
```

For full CLI training queues (`train.py`, `scripts/experiments/queue_*.py`, and `scripts/final_evaluation.py`), see `REPRODUCING.md`.

## Repository layout

```text
1..5 *.ipynb          End-to-end analysis, training, frontier, and figure notebooks
src/                  Data cleaning, grouped splits, haze features, model architectures
configs/              YAML configs for CV and final runs
scripts/              Dataset download, experiment queues, one-shot test eval, analysis
results/              Tracked CSVs, learning curves, frontier logs, footprint tables
report/               LaTeX source (final_report.tex), bibliography, and figures
tests/                Unit and regression tests
```
