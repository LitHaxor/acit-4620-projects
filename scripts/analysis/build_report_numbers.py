#!/usr/bin/env python3
"""Build authoritative report numbers markdown (`scripts/analysis/build_report_numbers.py`).

CPU-only, read-only script that generates `outputs/results/REPORT_NUMBERS.md`
(or `<results_dir>/REPORT_NUMBERS.md` with `--results-dir`) STRICTLY from CSVs,
per-epoch curve files (`curves/*.csv`), and saved prediction files (`predictions/*.csv`
and E1 `preds.npz`), with zero hand-typed metric numbers.

Sections:
1. Provenance: git commit hash (and whether uncommitted changes exist), date, seed 42, hardware string.
2. Baselines: E0 median, E0 ridge, E1 metadata MLP (station CV, spatial CV, test, >25 km).
3. Image models, 5-fold station CV, LAST-epoch MAE mean +- SD, with per-fold values and skill = 1 - MAE / median MAE.
4. PFRML ladder: all E4 configs plus E5_mobilenetv3_s_reg_chi, with paired differences vs R1 and vs E2 (mean +- SE, folds won).
5. Fusion, spatial CV: E1 spatial, E2 spatial, all E5 fusion runs and their _meta_off rows, per fold.
6. Final test results (full and >25 km) and the paired bootstrap tables.
7. Failed or unstable runs: E4_mdnet_t_baseline_ep60, E5_mdnet_t_pfrml_lr2e-4_f0, E4_mobilenetv3_s_dcm_concat, each with its last-epoch MAE and a pointer to collapse.md.
8. A "Do not quote" list: every best-epoch column, superseded best-epoch summary tables, and any experiment id that has no curve file (or <5 folds).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.compare_final_models import (  # noqa: E402
    METRIC_SPECS,
    _append_paired_bootstrap_table,
    _display_path,
    _get_test_gt25km_mask_and_ids,
    _load_csv_rows,
    _load_predictions_df,
    _order_model_names,
    _parse_float,
    _try_load_e1_metadata_mlp,
    resolve_results_layout,
)
from scripts.analysis.summarize_last_epoch import (  # noqa: E402
    compute_paired_mae_diff,
    discover_curves,
    read_last_epoch_row,
)
from src.config import load_config  # noqa: E402


IMAGE_STATION_MODELS: Tuple[Tuple[str, str], ...] = (
    ("E2_mobilenetv3_s_pretrained", "MobileNetV3-S pretrained (Ref 1)"),
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "MobileNetV3-S + MD-Neck (Ref 2, author adaptation)"),
    ("E2_resnet18_pretrained_lr3e-4_ep40", "ResNet-18 pretrained (lr=3e-4, 40ep)"),
    ("E3_mdnet_t_reg_lr1e-3_ep60", "MDNet-T regression scratch (lr=1e-3, 60ep)"),
    ("E2_resnet18_scratch", "ResNet-18 scratch (lr=1e-3, 30ep)"),
)

PFRML_LADDER_MODELS: Tuple[Tuple[str, str], ...] = (
    ("E4_mobilenetv3_s_r1_no_prototypes", "R1: DCM + Transformer + MLP (no prototypes)"),
    ("E4_mobilenetv3_s_baseline", "R2: PFRML baseline (toggles off, chi_proto)"),
    ("E5_mobilenetv3_s_reg_chi", "R2 (Eq. 5 literal): PFRML baseline with reg_input=chi"),
    ("E4_mobilenetv3_s_attn_pool", "R2+T1: + attention pooling"),
    ("E4_mobilenetv3_s_physics_gate", "R2+T2: + physics prior gate"),
    ("E4_mobilenetv3_s_class_weights", "R2+T3: + class weights"),
    ("E4_mobilenetv3_s_dcm_concat", "R2+T4: + DCM concat"),
    ("E4_mobilenetv3_s_full", "R2+all: PFRML full (T1+T2+T3)"),
    ("E4_mdnet_t_baseline_ep60", "Extra B: PFRML MDNet-T scratch (60ep)"),
)

FAILED_OR_UNSTABLE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    (
        "E4_mdnet_t_baseline_ep60",
        "station",
        "From-scratch `pfrml_mdnet_t` under hybrid loss (`lr=1e-3`, 60 ep): optimization stalls at near-constant prediction (`last R2 ~ 0` across all folds).",
    ),
    (
        "E5_mdnet_t_pfrml_lr2e-4_f0",
        "station",
        "From-scratch `pfrml_mdnet_t` with lower `lr=2e-4` (Fold 0 probe): avoids total collapse but remains substantially worse than regression-only `E3_mdnet_t_reg_lr1e-3_ep60`.",
    ),
    (
        "E4_mobilenetv3_s_dcm_concat",
        "station",
        "DCM concatenation variant: severe optimization instability / underfitting (`train_loss` fails to halve across folds).",
    ),
)

CANDIDATE_UNRUN_IDS: Tuple[str, ...] = (
    "E4_mobilenetv3_s_detached_proto",
    "E4_mobilenetv3_s_ema099",
)


def _mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _git_info() -> Tuple[str, str, bool, List[str]]:
    try:
        full_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True
        ).splitlines()
        dirty = len(status_out) > 0
        return full_hash, short_hash, dirty, status_out
    except Exception:
        return "unknown", "unknown", False, []


def _hardware_string() -> str:
    cpu_brand = platform.processor() or "Apple Silicon"
    mem_gb = ""
    if sys.platform == "darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if brand:
                cpu_brand = brand
        except Exception:
            pass
        try:
            mem_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            mem_gb = f", {mem_bytes / (1024 ** 3):.0f} GB unified memory"
        except Exception:
            pass
    mps_str = "available" if torch.backends.mps.is_available() else "unavailable"
    return (
        f"{cpu_brand}{mem_gb} | OS: `{platform.platform()}` | "
        f"Python `{platform.python_version()}` | PyTorch `{torch.__version__}` (`mps` {mps_str})"
    )


def _read_first_and_last_curve_rows(curve_path: Path) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    if not curve_path.exists() or curve_path.stat().st_size == 0:
        return None, None
    first_row: Optional[Dict[str, str]] = None
    last_row: Optional[Dict[str, str]] = None
    with curve_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if first_row is None:
                first_row = dict(row)
            last_row = dict(row)
    return first_row, last_row


def _format_10_metrics_from_fold_rows(f_rows: Dict[int, Dict[str, str]]) -> List[str]:
    cells: List[str] = []
    for m_key, prec in METRIC_SPECS:
        vals = [_parse_float(f_rows[k].get(m_key)) for k in sorted(f_rows.keys())]
        m_v, sd_v = _mean_sd(vals)
        if math.isnan(m_v):
            cells.append("—")
        elif len(vals) == 1:
            cells.append(f"{m_v:.{prec}f}")
        else:
            cells.append(f"{m_v:.{prec}f} +- {sd_v:.{prec}f}")
    return cells


def _index_csv_by_exp_mode_fold(
    rows: Sequence[Dict[str, str]]
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in rows:
        exp_id = str(r.get("experiment_id", ""))
        g_mode = str(r.get("group_mode", ""))
        fold = str(r.get("fold", ""))
        if exp_id and g_mode and fold:
            out[(exp_id, g_mode, fold)] = r
    return out


def build_report_numbers(
    *,
    outputs_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    output_md: Optional[Path] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> str:
    base_dir = Path(results_dir) if results_dir is not None else (
        Path(outputs_dir) if outputs_dir is not None else PROJECT_ROOT / "outputs"
    )
    layout = resolve_results_layout(base_dir)
    is_flat: bool = bool(layout["is_flat"])
    csv_dir: Path = layout["csv_dir"]
    curves_dir: Path = layout["curves_dir"]
    predictions_dir: Path = layout["predictions_dir"]
    e1_npz_path: Path = layout["e1_npz"]
    collapse_md: Path = layout["collapse_md"]
    if output_md is None:
        output_md = layout["report_numbers_md"]

    baselines_csv: Path = layout["baselines_csv"]
    image_csv: Path = layout["image_csv"]
    pfrml_csv: Path = layout["pfrml_csv"]
    fusion_csv: Path = layout["fusion_csv"]
    final_csv: Path = layout["final_csv"]

    csv_disp = _display_path(csv_dir)
    curves_disp = _display_path(curves_dir)
    preds_disp = _display_path(predictions_dir)
    e1_npz_disp = _display_path(e1_npz_path)
    collapse_disp = _display_path(collapse_md)
    collapse_link = (
        f"[{collapse_disp}](collapse.md)"
        if is_flat
        else f"[{collapse_disp}](file://{collapse_md.resolve()})"
    )

    s2_rows = _load_csv_rows(baselines_csv)
    s3_rows = _load_csv_rows(image_csv)
    s4_rows = _load_csv_rows(pfrml_csv)
    s5_rows = _load_csv_rows(fusion_csv)
    fin_rows = _load_csv_rows(final_csv)

    s2_idx = _index_csv_by_exp_mode_fold(s2_rows)
    s3_idx = _index_csv_by_exp_mode_fold(s3_rows)
    s4_idx = _index_csv_by_exp_mode_fold(s4_rows)
    s5_idx = _index_csv_by_exp_mode_fold(s5_rows)

    discovered_curves = discover_curves(curves_dir)
    last_epoch_by_key: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for key, fold_paths in discovered_curves.items():
        f_map: Dict[int, Dict[str, str]] = {}
        for k, p in sorted(fold_paths.items()):
            lr = read_last_epoch_row(p)
            if lr is not None:
                f_map[k] = lr
        if f_map:
            last_epoch_by_key[key] = f_map

    # Load default config seed
    cfg = load_config(PROJECT_ROOT / "configs/default.yaml")
    cfg_seed = int(cfg.seed)

    full_hash, short_hash, is_dirty, dirty_lines = _git_info()
    dirty_str = (
        f"Uncommitted changes exist ({len(dirty_lines)} modified/untracked paths in working tree)"
        if is_dirty
        else "Clean working tree (no uncommitted changes)"
    )
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hw_str = _hardware_string()

    # Prepare E1 bootstrap CIs and predictions
    pred_dfs: Dict[str, pd.DataFrame] = {}
    if predictions_dir.exists():
        for p in sorted(predictions_dir.glob("*.csv")):
            df = _load_predictions_df(p)
            if df is not None:
                pred_dfs[p.stem] = df
    ref_pred_df: Optional[pd.DataFrame] = None
    for df in pred_dfs.values():
        if ref_pred_df is None or len(df) > len(ref_pred_df):
            ref_pred_df = df

    gt25_mask, gt25_ids, clean_test_df = _get_test_gt25km_mask_and_ids(results_dir=base_dir)
    e1_conv_rows, e1_pred_df, _ = _try_load_e1_metadata_mlp(
        outputs_dir=base_dir,
        baselines_csv=baselines_csv,
        reference_pred_df=ref_pred_df,
        gt25_mask=gt25_mask,
        gt25_ids=gt25_ids,
        clean_test_df=clean_test_df,
        n_boot=n_boot,
        seed=seed,
    )
    if e1_pred_df is not None:
        pred_dfs["E1_metadata_mlp"] = e1_pred_df
    e1_ci_by_split: Dict[str, Tuple[float, float, int]] = {}
    for r in e1_conv_rows:
        sp = str(r.get("eval_split", ""))
        ci_l = _parse_float(r.get("mae_ci95_low"))
        ci_h = _parse_float(r.get("mae_ci95_high"))
        n_st = int(r.get("n_stations") or 0)
        if not math.isnan(ci_l) and not math.isnan(ci_h):
            e1_ci_by_split[sp] = (ci_l, ci_h, n_st)

    lines: List[str] = [
        "# Authoritative Report Numbers (`REPORT_NUMBERS.md`)",
        "",
        f"> Generated strictly from CSVs (`{csv_disp}/*.csv`), per-epoch curve files (`{curves_disp}/*.csv`), and saved prediction files (`{preds_disp}/*.csv`, `{e1_npz_disp}`) by `scripts/analysis/build_report_numbers.py`.",
        "",
        "## 1. Provenance",
        "",
        f"- **Git commit hash:** `{full_hash}` (`{short_hash}`)",
        f"- **Working tree status:** {dirty_str}",
        f"- **Generated timestamp:** `{now_str}`",
        f"- **Random seed:** `{cfg_seed}` (`seed={seed}` across CV splits, training, and `{n_boot}`-resample station bootstrap)",
        f"- **Hardware & runtime environment:** {hw_str}",
        "",
        "---",
        "",
        "## 2. Baselines (`E0_median`, `E0_ridge`, `E1_metadata_mlp`)",
        "",
        f"Source: `{csv_disp}/baselines_cv_results.csv` and `{e1_npz_disp}`.",
        "",
        "### 2a. Summary Table (`station` CV, `spatial_cluster` CV, Full `test`, and `test_gt25km` `>25 km`)",
        "",
        "| Experiment | Model | Split / Mode | n | MAE (Mean +- SD or [95% CI]) | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    baseline_specs: Tuple[Tuple[str, str, str, str], ...] = (
        ("E0_median", "station", "cv_mean", "station 5-fold CV"),
        ("E0_median", "spatial_cluster", "cv_mean", "spatial_cluster 5-fold CV"),
        ("E0_median", "full_train", "test", "Full test"),
        ("E0_median", "full_train", "test_gt25km", "Test >25 km"),
        ("E0_ridge", "station", "cv_mean", "station 5-fold CV"),
        ("E0_ridge", "spatial_cluster", "cv_mean", "spatial_cluster 5-fold CV"),
        ("E0_ridge", "full_train", "test", "Full test"),
        ("E0_ridge", "full_train", "test_gt25km", "Test >25 km"),
        ("E1_metadata_mlp", "station", "cv_mean", "station 5-fold CV"),
        ("E1_metadata_mlp", "spatial_cluster", "cv_mean", "spatial_cluster 5-fold CV"),
        ("E1_metadata_mlp", "full_train", "test", "Full test"),
        ("E1_metadata_mlp", "full_train", "test_gt25km", "Test >25 km"),
    )

    for exp_id, g_mode, fold_key, split_label in baseline_specs:
        r_mean = s2_idx.get((exp_id, g_mode, fold_key))
        if r_mean is None:
            continue
        r_std = s2_idx.get((exp_id, g_mode, "cv_std")) if fold_key == "cv_mean" else None
        model_str = str(r_mean.get("model", ""))
        n_samp = str(r_mean.get("n_samples", ""))
        mae_m = _parse_float(r_mean.get("mae"))
        if r_std is not None:
            mae_s = _parse_float(r_std.get("mae"))
            mae_cell = f"**{mae_m:.2f} +- {mae_s:.2f}**"
        elif exp_id == "E1_metadata_mlp" and fold_key in e1_ci_by_split:
            ci_l, ci_h, _ = e1_ci_by_split[fold_key]
            mae_cell = f"**{mae_m:.2f}** [{ci_l:.2f}, {ci_h:.2f}]"
        else:
            mae_cell = f"**{mae_m:.2f}**"

        other_cells: List[str] = []
        for m_key, prec in METRIC_SPECS[1:]:
            mv = _parse_float(r_mean.get(m_key))
            if r_std is not None:
                sv = _parse_float(r_std.get(m_key))
                other_cells.append(f"{mv:.{prec}f} +- {sv:.{prec}f}")
            else:
                other_cells.append(f"{mv:.{prec}f}")
        lines.append(
            f"| `{exp_id}` | `{model_str}` | `{split_label}` | {n_samp} | {mae_cell} | "
            + " | ".join(other_cells)
            + " |"
        )

    # Per-fold baseline MAE table
    lines.extend(
        [
            "",
            "### 2b. Per-Fold Baseline Validation MAE (`station` and `spatial_cluster`)",
            "",
            "| Experiment | Mode | Mean +- SD | F0 | F1 | F2 | F3 | F4 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for exp_id in ("E0_median", "E0_ridge", "E1_metadata_mlp"):
        for g_mode in ("station", "spatial_cluster"):
            r_m = s2_idx.get((exp_id, g_mode, "cv_mean"))
            r_s = s2_idx.get((exp_id, g_mode, "cv_std"))
            if r_m is None or r_s is None:
                continue
            f_vals = [
                _parse_float(s2_idx.get((exp_id, g_mode, str(k)), {}).get("mae"))
                for k in range(5)
            ]
            f_strs = [f"{v:.2f}" if not math.isnan(v) else "—" for v in f_vals]
            lines.append(
                f"| `{exp_id}` | `{g_mode}` | **{_parse_float(r_m.get('mae')):.2f} +- {_parse_float(r_s.get('mae')):.2f}** | "
                + " | ".join(f_strs)
                + " |"
            )

    # Section 3: Image models, 5-fold station CV, LAST-epoch MAE
    e0_station_fold_mae = {
        k: _parse_float(s2_idx.get(("E0_median", "station", str(k)), {}).get("mae"))
        for k in range(5)
    }
    e0_station_mean_mae = _parse_float(
        s2_idx.get(("E0_median", "station", "cv_mean"), {}).get("mae")
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Image-Only Models — 5-Fold `station` CV (LAST-Epoch Reporting)",
            "",
            f"Source: Last-epoch rows of `{curves_disp}/{{experiment_id}}_station_fold{{0..4}}.csv`. Skill is defined relative to `E0_median` (`station` CV `MAE = {e0_station_mean_mae:.2f}`) as $\\text{{Skill}} = 1 - \\text{{MAE}} / \\text{{MAE}}_{{\\text{{median}}}}$.",
            "",
            "### 3a. Per-Fold Last-Epoch MAE & Skill vs `E0_median`",
            "",
            "| Experiment | Description | Last-Ep MAE (Mean +- SD) | Overall Skill | Per-Fold Skill (Mean +- SD) | F0 MAE (Skill) | F1 MAE (Skill) | F2 MAE (Skill) | F3 MAE (Skill) | F4 MAE (Skill) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for exp_id, desc in IMAGE_STATION_MODELS:
        f_rows = last_epoch_by_key.get((exp_id, "station"), {})
        if len(f_rows) < 5:
            continue
        maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
        m_mae, sd_mae = _mean_sd(maes)
        overall_skill = 1.0 - (m_mae / e0_station_mean_mae)
        fold_skills = [1.0 - (maes[k] / e0_station_fold_mae[k]) for k in range(5)]
        m_sk, sd_sk = _mean_sd(fold_skills)
        f_cells = [f"{maes[k]:.2f} ({fold_skills[k]:+.3f})" for k in range(5)]
        lines.append(
            f"| `{exp_id}` | {desc} | **{m_mae:.2f} +- {sd_mae:.2f}** | **{overall_skill:+.4f}** | "
            f"{m_sk:+.4f} +- {sd_sk:.4f} | " + " | ".join(f_cells) + " |"
        )

    lines.extend(
        [
            "",
            "### 3b. All 10 Last-Epoch Metrics (`mean +- SD`, 5-Fold `station` CV)",
            "",
            "| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for exp_id, _ in IMAGE_STATION_MODELS:
        f_rows = last_epoch_by_key.get((exp_id, "station"), {})
        if len(f_rows) < 5:
            continue
        cells = _format_10_metrics_from_fold_rows(f_rows)
        lines.append(f"| `{exp_id}` | " + " | ".join(cells) + " |")

    # Section 4: PFRML ladder
    ref_e2_rows = last_epoch_by_key.get(("E2_mobilenetv3_s_pretrained", "station"))
    ref_r1_rows = last_epoch_by_key.get(("E4_mobilenetv3_s_r1_no_prototypes", "station"))
    ref_r2_rows = last_epoch_by_key.get(("E4_mobilenetv3_s_baseline", "station"))

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. PFRML Ablation Ladder — 5-Fold `station` CV (All `E4` Configs + `E5_mobilenetv3_s_reg_chi`, LAST-Epoch)",
            "",
            f"Source: Last-epoch rows of `{curves_disp}/{{experiment_id}}_station_fold{{0..4}}.csv`. Paired differences report `mean diff +- SE` (`SE = SD / sqrt(5)`, `candidate - reference`, negative = candidate lowers MAE) and `folds won / 5`.",
            "",
            "### 4a. Per-Fold Last-Epoch MAE & Paired Differences vs `R1`, `E2`, and `R2`",
            "",
            "| Experiment | Rung / Description | Last-Ep MAE (Mean +- SD) | Skill | Diff vs R1 (`E4_..._r1_no_prototypes`) | Won vs R1 | Diff vs E2 (`E2_mobilenetv3_s_pretrained`) | Won vs E2 | Diff vs R2 (`E4_mobilenetv3_s_baseline`) | Won vs R2 | F0 | F1 | F2 | F3 | F4 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for exp_id, desc in PFRML_LADDER_MODELS:
        f_rows = last_epoch_by_key.get((exp_id, "station"), {})
        if len(f_rows) < 5:
            continue
        maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
        m_mae, sd_mae = _mean_sd(maes)
        skill = 1.0 - (m_mae / e0_station_mean_mae)
        d_r1, w_r1, _ = compute_paired_mae_diff(f_rows, ref_r1_rows)
        d_e2, w_e2, _ = compute_paired_mae_diff(f_rows, ref_e2_rows)
        d_r2, w_r2, _ = compute_paired_mae_diff(f_rows, ref_r2_rows)
        f_strs = [f"{maes[k]:.2f}" for k in range(5)]
        lines.append(
            f"| `{exp_id}` | {desc} | **{m_mae:.2f} +- {sd_mae:.2f}** | {skill:+.4f} | "
            f"{d_r1} | {w_r1} | {d_e2} | {w_e2} | {d_r2} | {w_r2} | "
            + " | ".join(f_strs)
            + " |"
        )

    lines.extend(
        [
            "",
            "### 4b. All 10 Last-Epoch Metrics for PFRML Ladder (`mean +- SD`, 5-Fold `station` CV)",
            "",
            "| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for exp_id, _ in PFRML_LADDER_MODELS:
        f_rows = last_epoch_by_key.get((exp_id, "station"), {})
        if len(f_rows) < 5:
            continue
        cells = _format_10_metrics_from_fold_rows(f_rows)
        lines.append(f"| `{exp_id}` | " + " | ".join(cells) + " |")

    # Section 5: Fusion, spatial_cluster CV
    e0_spatial_mean_mae = _parse_float(
        s2_idx.get(("E0_median", "spatial_cluster", "cv_mean"), {}).get("mae")
    )
    e1_spatial_fold_rows = {
        k: s2_idx.get(("E1_metadata_mlp", "spatial_cluster", str(k)), {})
        for k in range(5)
    }
    e2_spatial_fold_rows = last_epoch_by_key.get(
        ("E2_mobilenetv3_s_pretrained_spatial", "spatial_cluster"), {}
    )

    # Discover all spatial_cluster experiments in fusion_cv_results.csv in order of appearance
    fusion_exp_ids: List[str] = []
    for r in s5_rows:
        if r.get("group_mode") == "spatial_cluster":
            eid = str(r.get("experiment_id", ""))
            if eid and eid not in fusion_exp_ids:
                fusion_exp_ids.append(eid)

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Multimodal Fusion — 5-Fold `spatial_cluster` CV (`25 km` Cluster Separation, LAST-Epoch)",
            "",
            f"Source: `{csv_disp}/baselines_cv_results.csv` (`E0_median`, `E1_metadata_mlp`), `{curves_disp}/E2_mobilenetv3_s_pretrained_spatial_spatial_cluster_fold{{0..4}}.csv`, and `{csv_disp}/fusion_cv_results.csv` (all `E5_fusion_*` runs and their `_meta_off` rows, evaluated at the LAST epoch).",
            "",
            "### 5a. Per-Fold Last-Epoch MAE & Paired Differences vs `E1_metadata_mlp` (spatial) and `E2_mobilenetv3_s_pretrained_spatial`",
            "",
            "| Experiment | Branch / Eval Mode | Last-Ep MAE (Mean +- SD) | Skill vs `E0_median` | Diff vs `E1_metadata_mlp` (Mean +- SE) | Won vs E1 | Diff vs `E2_..._spatial` (Mean +- SE) | Won vs E2 | F0 | F1 | F2 | F3 | F4 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    spatial_entries: List[Tuple[str, str, Dict[int, Dict[str, str]]]] = [
        ("E0_median", "Median baseline (spatial_cluster)", {k: s2_idx[("E0_median", "spatial_cluster", str(k))] for k in range(5) if ("E0_median", "spatial_cluster", str(k)) in s2_idx}),
        ("E1_metadata_mlp", "Metadata MLP only (spatial_cluster)", e1_spatial_fold_rows),
        ("E2_mobilenetv3_s_pretrained_spatial", "Image-only MobileNetV3-S (spatial_cluster)", e2_spatial_fold_rows),
    ]
    for eid in fusion_exp_ids:
        f_map = {
            k: s5_idx[(eid, "spatial_cluster", str(k))]
            for k in range(5)
            if (eid, "spatial_cluster", str(k)) in s5_idx
        }
        if len(f_map) == 5:
            desc = "Vision + Metadata zeroed at eval (`metadata_off=True`)" if eid.endswith("_meta_off") else "Vision + Metadata fusion (`normal`)"
            spatial_entries.append((eid, desc, f_map))

    for eid, desc, f_rows in spatial_entries:
        if len(f_rows) < 5:
            continue
        maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
        m_mae, sd_mae = _mean_sd(maes)
        skill = 1.0 - (m_mae / e0_spatial_mean_mae)
        d_e1, w_e1, _ = compute_paired_mae_diff(f_rows, e1_spatial_fold_rows)
        d_e2, w_e2, _ = compute_paired_mae_diff(f_rows, e2_spatial_fold_rows)
        f_strs = [f"{maes[k]:.2f}" for k in range(5)]
        lines.append(
            f"| `{eid}` | {desc} | **{m_mae:.2f} +- {sd_mae:.2f}** | {skill:+.4f} | "
            f"{d_e1} | {w_e1} | {d_e2} | {w_e2} | " + " | ".join(f_strs) + " |"
        )

    lines.extend(
        [
            "",
            "### 5b. All 10 Metrics for `spatial_cluster` CV Models (`mean +- SD`, 5 Folds)",
            "",
            "| Experiment | MAE | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for eid, _, f_rows in spatial_entries:
        if len(f_rows) < 5:
            continue
        cells = _format_10_metrics_from_fold_rows(f_rows)
        lines.append(f"| `{eid}` | " + " | ".join(cells) + " |")

    # Section 6: Final test results and paired bootstrap tables
    all_final_rows: List[Dict[str, Any]] = list(fin_rows)
    existing_final_names = {str(r.get("name", "")) for r in all_final_rows}
    if "E1_metadata_mlp" not in existing_final_names and e1_conv_rows:
        all_final_rows.extend(e1_conv_rows)

    row_names_ordered = _order_model_names([str(r.get("name", "")) for r in all_final_rows])
    ordered_final_rows: List[Dict[str, Any]] = []
    for m_name in row_names_ordered:
        for r in all_final_rows:
            if str(r.get("name", "")) == m_name:
                ordered_final_rows.append(r)

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Final One-Shot Test Results (`test` and `test_gt25km` `>25 km`)",
            "",
            f"Source: `{csv_disp}/final_test_results.csv`, `{preds_disp}/*.csv`, `{e1_npz_disp}`, and `{csv_disp}/baselines_cv_results.csv`. Station-level cluster bootstrap uses `{n_boot}` resamples with `seed={seed}`.",
            "",
            "### 6a. All Final Models — 10 Metrics & 95% Station-Bootstrap MAE CI",
            "",
            "| Name | Model | Split | n | Stations | MAE (95% Station CI) | RMSE | PLCC | SRCC | R2 | Acc | +-1Acc | MacroF1 | TailMAE | TailF1 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for r in ordered_final_rows:
        name = str(r.get("name", ""))
        model = str(r.get("model", ""))
        split = str(r.get("eval_split", ""))
        n_samp = str(r.get("n_samples", ""))
        n_st_str = str(r.get("n_stations", "") or "—")
        mae_v = _parse_float(r.get("mae"))
        ci_l = _parse_float(r.get("mae_ci95_low"))
        ci_h = _parse_float(r.get("mae_ci95_high"))
        if not math.isnan(mae_v) and not math.isnan(ci_l) and not math.isnan(ci_h):
            mae_cell = f"**{mae_v:.2f}** [{ci_l:.2f}, {ci_h:.2f}]"
        elif not math.isnan(mae_v):
            mae_cell = f"**{mae_v:.2f}**"
        else:
            mae_cell = "—"
        other_cells = []
        for m_key, prec in METRIC_SPECS[1:]:
            v = _parse_float(r.get(m_key))
            other_cells.append("—" if math.isnan(v) else f"{v:.{prec}f}")
        lines.append(
            f"| `{name}` | `{model}` | `{split}` | {n_samp} | {n_st_str} | {mae_cell} | "
            + " | ".join(other_cells)
            + " |"
        )

    model_names_boot = _order_model_names(list(pred_dfs.keys()))

    lines.extend(
        [
            "",
            "### 6b. Paired Station-Bootstrap MAE Differences — Full Test Set (`Model A - Model B`, 95% CI)",
            "",
            "| Model A | Model B | n | Stations | MAE(A) | MAE(B) | MAE Diff (A - B) | 95% Station Bootstrap CI |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    _append_paired_bootstrap_table(
        lines=lines,
        model_names=model_names_boot,
        pred_dfs=pred_dfs,
        subset_ids=None,
        n_boot=n_boot,
        seed=seed,
    )

    lines.extend(
        [
            "",
            "### 6c. Paired Station-Bootstrap MAE Differences — `test_gt25km` Subset (`>25 km`, `Model A - Model B`, 95% CI)",
            "",
            "| Model A | Model B | n | Stations | MAE(A) | MAE(B) | MAE Diff (A - B) | 95% Station Bootstrap CI |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    _append_paired_bootstrap_table(
        lines=lines,
        model_names=model_names_boot,
        pred_dfs=pred_dfs,
        subset_ids=gt25_ids,
        n_boot=n_boot,
        seed=seed,
    )

    # Section 7: Failed or unstable runs
    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Failed or Unstable Runs",
            "",
            f"Detailed per-epoch loss trajectories and root-cause analysis are documented in {collapse_link}.",
            "",
            "| Experiment | Mode | Folds | Last-Ep MAE | Last-Ep R2 | Ep 1 `train_loss` (Mean) | Last Ep `train_loss` (Mean) | Per-Fold Last-Ep MAE | Diagnostic Reference & Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for exp_id, g_mode, diagnosis_text in FAILED_OR_UNSTABLE_SPECS:
        f_paths = discovered_curves.get((exp_id, g_mode), {})
        if not f_paths:
            continue
        f_maes: List[float] = []
        f_r2s: List[float] = []
        tr_ep1s: List[float] = []
        tr_lasts: List[float] = []
        per_fold_strs: List[str] = []
        for k in sorted(f_paths.keys()):
            first_r, last_r = _read_first_and_last_curve_rows(f_paths[k])
            if first_r is not None and last_r is not None:
                mae_v = _parse_float(last_r.get("mae"))
                r2_v = _parse_float(last_r.get("r2"))
                tr1 = _parse_float(first_r.get("train_loss"))
                trl = _parse_float(last_r.get("train_loss"))
                f_maes.append(mae_v)
                f_r2s.append(r2_v)
                tr_ep1s.append(tr1)
                tr_lasts.append(trl)
                per_fold_strs.append(f"F{k}={mae_v:.2f}")
        m_mae, sd_mae = _mean_sd(f_maes)
        m_r2, sd_r2 = _mean_sd(f_r2s)
        m_tr1, _ = _mean_sd(tr_ep1s)
        m_trl, _ = _mean_sd(tr_lasts)
        n_f = len(f_maes)
        mae_cell = f"**{m_mae:.2f}**" if n_f == 1 else f"**{m_mae:.2f} +- {sd_mae:.2f}**"
        r2_cell = f"{m_r2:.4f}" if n_f == 1 else f"{m_r2:.4f} +- {sd_r2:.4f}"
        lines.append(
            f"| `{exp_id}` | `{g_mode}` | {n_f}/5 | {mae_cell} | {r2_cell} | "
            f"{m_tr1:.4f} | {m_trl:.4f} | {', '.join(per_fold_strs)} | "
            f"{diagnosis_text} See {collapse_link}. |"
        )

    # Section 8: "Do not quote" list
    lines.extend(
        [
            "",
            "---",
            "",
            "## 8. \"Do Not Quote\" List (Superseded Best-Epoch Numbers, Missing-Curve IDs & Single-Fold Screens)",
            "",
            f"1. **Every Best-Epoch Column / Note (`{csv_disp}/image_models_cv_results.csv`, `{csv_disp}/pfrml_ablation_cv_results.csv`, and `best_ep_mae` in `{csv_disp}/fusion_cv_results.csv`):**",
            "   - All image-model and PFRML-ablation CSV rows in `image_models_cv_results.csv` and `pfrml_ablation_cv_results.csv` recorded the minimum-`val_loss` checkpoint (`best_ep`) rather than the final epoch (`last_ep`), and `fusion_cv_results.csv` records `best_ep_mae=...` inside its `notes` column for diagnostic reference only. **Do not quote any best-epoch validation metrics as main results; always quote the last-epoch metrics in Sections 3–5 above.**",
            "",
            "| Experiment | Mode | CSV Source | Do-Not-Quote Best-Ep MAE | Authoritative Last-Ep Curve MAE (`Mean +- SD`) |",
            "| :--- | :---: | :--- | :---: | :---: |",
        ]
    )
    for csv_name, idx_map in (("image_models_cv_results.csv", s3_idx), ("pfrml_ablation_cv_results.csv", s4_idx)):
        seen_keys: Set[Tuple[str, str]] = set()
        for (eid, g_mode, fold_str), r in idx_map.items():
            if fold_str != "cv_mean" or (eid, g_mode) in seen_keys:
                continue
            seen_keys.add((eid, g_mode))
            best_m = _parse_float(r.get("mae"))
            best_s = _parse_float(idx_map.get((eid, g_mode, "cv_std"), {}).get("mae"))
            f_rows = last_epoch_by_key.get((eid, g_mode), {})
            if len(f_rows) == 5:
                l_maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
                lm, lsd = _mean_sd(l_maes)
                last_str = f"**{lm:.2f} +- {lsd:.2f}**"
            else:
                last_str = "—"
            lines.append(
                f"| `{eid}` | `{g_mode}` | `{csv_disp}/{csv_name}` | ~~{best_m:.2f} +- {best_s:.2f}~~ | {last_str} |"
            )

    # Also parse `best_ep_mae=` from `fusion_cv_results.csv` notes
    s5_best_ep_by_exp: Dict[Tuple[str, str], List[float]] = {}
    for r in s5_rows:
        eid = str(r.get("experiment_id", ""))
        g_mode = str(r.get("group_mode", ""))
        fold_s = str(r.get("fold", ""))
        if not eid or eid.endswith("_meta_off") or fold_s in {"cv_mean", "cv_std"}:
            continue
        m = re.search(r"best_ep_mae=([0-9]+(?:\.[0-9]+)?)", str(r.get("notes", "")))
        if m:
            s5_best_ep_by_exp.setdefault((eid, g_mode), []).append(float(m.group(1)))

    for (eid, g_mode), b_vals in s5_best_ep_by_exp.items():
        if len(b_vals) == 5:
            bm, bsd = _mean_sd(b_vals)
            r_m = s5_idx.get((eid, g_mode, "cv_mean"), {})
            r_s = s5_idx.get((eid, g_mode, "cv_std"), {})
            lm = _parse_float(r_m.get("mae"))
            lsd = _parse_float(r_s.get("mae"))
            lines.append(
                f"| `{eid}` (`notes: best_ep_mae`) | `{g_mode}` | `{csv_disp}/fusion_cv_results.csv` | ~~{bm:.2f} +- {bsd:.2f}~~ | **{lm:.2f} +- {lsd:.2f}** |"
            )

    lines.extend(
        [
            "",
            f"2. **Superseded `{csv_disp}/pfrml_best_epoch_summary_superseded.md` Best-Epoch Tables:**",
            f"   - `{csv_disp}/pfrml_best_epoch_summary_superseded.md` (legacy best-epoch table derived from `pfrml_ablation_cv_results.csv`) reports minimum-`val_loss` (`best_ep`) numbers and Wilcoxon $p$-values at $n=5$. **Do not quote `pfrml_best_epoch_summary_superseded.md`**; use `outputs/results/last_epoch_summary.md` and Section 4 of this document.",
            "",
            f"3. **Experiment IDs with NO Curve File in `{curves_disp}/` (or Partial `<5` Fold Curve Sets):**",
        ]
    )

    curve_exp_ids = {eid for (eid, _) in discovered_curves.keys()}
    all_csv_exp_ids: Set[str] = set()
    for rows in (s2_rows, s3_rows, s4_rows, s5_rows):
        for r in rows:
            eid = str(r.get("experiment_id", ""))
            if eid:
                all_csv_exp_ids.add(eid)

    no_curve_ids = sorted((all_csv_exp_ids | set(CANDIDATE_UNRUN_IDS)) - curve_exp_ids)
    for eid in no_curve_ids:
        if eid in {"E0_median", "E0_ridge", "E1_metadata_mlp"}:
            lines.append(
                f"   - `{eid}`: Closed-form / tabular baseline in `{csv_disp}/baselines_cv_results.csv` (no image-epoch curve file in `{curves_disp}/`; quote only from Section 2 / Section 6)."
            )
        elif eid.endswith("_meta_off"):
            lines.append(
                f"   - `{eid}`: Evaluation-only metadata-zeroed pass evaluated on the parent checkpoint at the end of training (no separate training curve file; quote from `{csv_disp}/fusion_cv_results.csv` / Section 5)."
            )
        else:
            lines.append(
                f"   - `{eid}`: **0 curve files in `{curves_disp}/` and 0 CSV rows** (never run — do not quote)."
            )

    partial_keys = sorted(
        [(eid, g_mode, sorted(fmap.keys())) for (eid, g_mode), fmap in last_epoch_by_key.items() if len(fmap) < 5]
    )
    if partial_keys:
        lines.append(f"   - **Single-fold / partial (`<5` folds) curve sets in `{curves_disp}/` (do not quote as 5-fold CV results):**")
        for eid, g_mode, folds_list in partial_keys:
            fmap = last_epoch_by_key[(eid, g_mode)]
            maes = [_parse_float(fmap[k].get("mae")) for k in folds_list]
            m_v, _ = _mean_sd(maes)
            lines.append(
                f"     - `{eid}` (`{g_mode}`, folds `{folds_list}`, `{len(folds_list)}/5` folds, last-epoch `MAE = {m_v:.2f}`)"
            )

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    return content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build authoritative REPORT_NUMBERS.md strictly from CSVs, curves, and prediction files."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results base directory ('outputs' layout or flat 'results/' snapshot).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.results_dir or args.outputs_dir or (PROJECT_ROOT / "outputs"))
    layout = resolve_results_layout(base_dir)
    output_md = Path(args.output_md) if args.output_md else layout["report_numbers_md"]
    build_report_numbers(
        results_dir=base_dir,
        output_md=output_md,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
