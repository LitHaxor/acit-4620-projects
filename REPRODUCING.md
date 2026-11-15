# Reproducing the Project (`REPRODUCING.md`)

This repository supports two levels of reproduction: **Level 1** (fast, CPU-only verification and table regeneration from the tracked `results/` snapshot) and **Level 2** (full training and one-shot test evaluation from the PM25Vision dataset).

---

## Recorded Environment & Data Revision

- **Hardware:** Apple M5 Pro, 24 GB unified memory (`mps` backend)
- **OS:** `macOS-27.2.1-arm64-arm-64bit-Mach-O`
- **Python / PyTorch:** Python `3.14.6` | PyTorch `2.14.1`
- **Random Seed:** `42` (used across `StratifiedGroupKFold`, model initialization, batch shuffling, and `1,000`-resample station bootstrap)
- **Dataset Revision:** Hugging Face dataset `DeadCardassian/PM25Vision` at commit hash `578bc7c17c99be1aa312ebfe0bda6ce40067ca31`

Exact package versions are pinned in [`requirements.txt`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/requirements.txt).

---

## Level 1: Exact Table & Bootstrap Regeneration (No GPUs, No Dataset Download)

Every summary table, paired fold difference, failure diagnosis, and station-bootstrap confidence interval in [`results/REPORT_NUMBERS.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/results/REPORT_NUMBERS.md) and [`results/final_comparison.md`](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/results/final_comparison.md) is computed directly from the tracked CSVs, per-epoch curves, and prediction files inside `results/` (1.8 MB total).

```bash
# 1. Install dependencies (Python >= 3.11)
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# 2. Regenerate final model comparison & station-bootstrap CIs from results/
python scripts/analysis/compare_final_models.py --results-dir results

# 3. Regenerate the complete 8-section REPORT_NUMBERS.md from results/
python scripts/analysis/build_report_numbers.py --results-dir results

# 4. Run offline-capable unit tests (tests not requiring raw dataset images/CSVs pass; full suite runs after Level 2 Step 1)
python -m pytest -q
```

**Expected output:** All metrics, paired differences, and bootstrap CIs match the committed `results/REPORT_NUMBERS.md` and `results/final_comparison.md` identically (only the provenance git commit hash, working-tree line, and generation timestamp update). Note that `results/test_gt25km_ids.txt` provides the `717` unseen-region (`>25 km`) test `image_id`s so paired `test_gt25km` bootstrap tables reproduce without downloading `datasets/`.

---

## Level 2: Full Training & Evaluation From Scratch

Level 2 downloads `PM25Vision`, audits the train/test splits, runs all CV queues into `outputs/`, selects the early-stopping fusion epoch via `CMA-3`, and performs one-shot final evaluation.

> [!CAUTION]
> **Caveats & Hardware Notes**
> 1. **MPS Non-Determinism:** Apple Metal (`mps`) floating-point reductions are not bit-for-bit deterministic across PyTorch/OS updates or different GPU hardware (`CUDA` vs `MPS` vs `CPU`). Retrained MAEs may vary slightly from the recorded snapshot even at `seed=42`.
> 2. **Single Seed:** All experiments use a single random seed (`42`) due to the ~20+ hour compute budget; multi-seed variance is not quantified.
> 3. **Runtime Estimates:** Projected times below were measured on Apple M5 Pro (24 GB). Queue scripts save checkpoints per epoch and skip completed folds automatically on resume. All queues support `--smoke` (tiny subset under `outputs/smoke/`) and `--project-only` (ETA projection without training).

### Stage 0 — Download Dataset & Run Full Test Suite

```bash
python scripts/data/download_dataset.py
python -m pytest -q   # Full 55-test suite
```

### Stage 1 — Data Audit & Non-Visual / Tabular Baselines (`E0_median`, `E0_ridge`, `E1_metadata_mlp`)

```bash
# E0_median + E0_ridge + E1_metadata_mlp across station, spatial_cluster, and full_train (~5 min)
python evaluate.py --model median --config configs/default.yaml --group-mode station --output-csv outputs/results/baselines_cv_results.csv
python evaluate.py --model median --config configs/default.yaml --group-mode spatial_cluster --output-csv outputs/results/baselines_cv_results.csv
python evaluate.py --model ridge_haze --config configs/default.yaml --group-mode station --output-csv outputs/results/baselines_cv_results.csv
python evaluate.py --model ridge_haze --config configs/default.yaml --group-mode spatial_cluster --output-csv outputs/results/baselines_cv_results.csv
python train.py --model metadata_mlp --config configs/default.yaml --group-mode station --all-folds --output-csv outputs/results/baselines_cv_results.csv
python train.py --model metadata_mlp --config configs/default.yaml --group-mode spatial_cluster --all-folds --output-csv outputs/results/baselines_cv_results.csv
```

### Stage 2 — Image-Only Backbones & MDNet-T Queue (~6–8 h)

Runs Fold-0 LR/epoch screens followed by selected 5-fold `station` CV runs (`E2_*`, `E3_*`, `E3b_*`) into `outputs/results/image_models_cv_results.csv`:

```bash
python scripts/experiments/queue_baselines.py --project-only
python scripts/experiments/queue_baselines.py
python scripts/analysis/diagnose_folds.py
```

### Stage 3 — PFRML Ablation Ladder (`R1`, `R2`, `T1–T4`, `all`, Extra A/B) (~8–10 h)

Runs the 5-fold `station` CV PFRML ablation ladder, `E2_mobilenetv3_s_pretrained_spatial` (`spatial_cluster`), and `E4_mdnet_t_baseline_ep60` (guarded at 14 h) into `outputs/results/pfrml_ablation_cv_results.csv`:

```bash
python scripts/experiments/queue_pfrml_ablations.py --project-only
python scripts/experiments/queue_pfrml_ablations.py
python scripts/analysis/diagnose_training_failures.py
python scripts/analysis/summarize_last_epoch.py
```

### Stage 4 — Multimodal Fusion `spatial_cluster` CV & Initial Final Models (~3–4 h)

Runs `E5_fusion_r1_p03`, `E5_fusion_full_p03`, `E5_fusion_r1_p00` (`spatial_cluster`, normal + `_meta_off`), `E5_mobilenetv3_s_reg_chi` (`station`), one-shot test evaluation of `F_E2` and `F_PFRML_full`, and the guarded `E5_mdnet_t_pfrml_lr2e-4_f0` probe:

```bash
python scripts/experiments/queue_fusion_cv.py --project-only
python scripts/experiments/queue_fusion_cv.py
```

### Stage 5 — CMA-3 Epoch Selection, Early-Stopped Fusion (`F_FUSION_R1`), & Report Numbers (~0.5 h)

```bash
# Compute N = 7 via 3-epoch centered moving average on E5_fusion_r1_p00 curves
python scripts/analysis/select_fusion_epochs.py

# Run E5_fusion_r1_p00_ep7 (5 folds), apply decision rule, and evaluate F_FUSION_R1 once on test
python scripts/experiments/queue_fusion_final.py --project-only
python scripts/experiments/queue_fusion_final.py

# Regenerate outputs/results/final_comparison.md and outputs/results/REPORT_NUMBERS.md
python scripts/analysis/compare_final_models.py
python scripts/analysis/build_report_numbers.py
```
