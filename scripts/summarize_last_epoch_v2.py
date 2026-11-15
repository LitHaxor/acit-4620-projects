#!/usr/bin/env python3
"""CPU-only, read-only full last-epoch summary generator (Step 5a, Task B1 / Step 5b Part 0a).

From `outputs/curves/{experiment_id}_{group_mode}_fold{fold}.csv` extracts the
LAST-epoch row per fold by EXACT experiment_id (`group_mode in {'station', 'spatial_cluster'}`),
never by prefix glob, and writes `outputs/results/step4_last_epoch_full.md`:
1. For every experiment with 5 folds, reports `mean +- SD` (sample SD, ddof=1)
   across all 10 evaluation metrics:
   `mae, rmse, plcc, srcc, r2, acc, pm1_acc, macro_f1, tail_mae, tail_f1`.
2. Paired comparison tables of last-epoch MAE (`mean difference +- SE` where
   `SE = SD / sqrt(n)`, and `folds won / 5`, with NO p-values) of every E4
   configuration vs:
   - `E2_mobilenetv3_s_pretrained`
   - `E4_mobilenetv3_s_r1_no_prototypes`
   - `E4_mobilenetv3_s_baseline`
3. Also summarizes any partial (< 5 folds) curve sets discovered in `outputs/curves/`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_GROUP_MODES: Tuple[str, ...] = ("station", "spatial_cluster")
CURVE_FILENAME_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<group_mode>station|spatial_cluster)_fold(?P<fold>[0-4])\.csv$"
)

METRIC_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("mae", "MAE", 2),
    ("rmse", "RMSE", 2),
    ("plcc", "PLCC", 4),
    ("srcc", "SRCC", 4),
    ("r2", "R2", 4),
    ("acc", "Acc", 4),
    ("pm1_acc", "+-1Acc", 4),
    ("macro_f1", "MacroF1", 4),
    ("tail_mae", "TailMAE", 2),
    ("tail_f1", "TailF1", 4),
)

PREFERRED_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("E2_mobilenetv3_s_pretrained", "station", "Ref 1: E2 MobileNetV3-S pretrained"),
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station", "Ref 2: E3b MobileNetV3-S + MD-Neck"),
    ("E2_resnet18_pretrained_lr3e-4_ep40", "station", "E2b: ResNet-18 pretrained (lr=3e-4, 40ep)"),
    ("E3_mdnet_t_reg_lr1e-3_ep60", "station", "E3a: MDNet-T reg scratch (lr=1e-3, 60ep)"),
    ("E2_resnet18_scratch", "station", "E2a: ResNet-18 scratch (30ep)"),
    ("E4_mobilenetv3_s_r1_no_prototypes", "station", "R1: DCM + Transformer + MLP (no proto)"),
    ("E4_mobilenetv3_s_baseline", "station", "R2: PFRML baseline (toggles off)"),
    ("E4_mobilenetv3_s_attn_pool", "station", "R2+T1: + attention pooling"),
    ("E4_mobilenetv3_s_physics_gate", "station", "R2+T2: + physics gate"),
    ("E4_mobilenetv3_s_class_weights", "station", "R2+T3: + class weights"),
    ("E4_mobilenetv3_s_dcm_concat", "station", "R2+T4: + DCM concat"),
    ("E4_mobilenetv3_s_full", "station", "R2+all: PFRML full"),
    ("E4_mdnet_t_baseline_ep60", "station", "Extra B: PFRML MDNet-T scratch (60ep)"),
    ("E2_mobilenetv3_s_pretrained_spatial", "spatial_cluster", "Extra A: E2 MobileNetV3-S (spatial)"),
)

LABELS_BY_KEY: Dict[Tuple[str, str], str] = {
    (exp_id, g_mode): label for exp_id, g_mode, label in PREFERRED_ORDER
}


def _parse_float(val: Optional[str]) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def parse_curve_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """Parse `{experiment_id}_{group_mode}_fold{fold}.csv` with exact `group_mode in {'station', 'spatial_cluster'}`."""
    m = CURVE_FILENAME_RE.match(filename)
    if not m:
        return None
    exp_id = m.group("experiment_id")
    group_mode = m.group("group_mode")
    fold = int(m.group("fold"))
    if group_mode not in VALID_GROUP_MODES or not (0 <= fold <= 4):
        return None
    return exp_id, group_mode, fold


def get_exact_curve_path(
    curves_dir: Path,
    experiment_id: str,
    group_mode: str,
    fold: int,
) -> Path:
    """Return exact path `{curves_dir}/{experiment_id}_{group_mode}_fold{fold}.csv` (never prefix glob)."""
    if group_mode not in VALID_GROUP_MODES:
        raise ValueError(f"Invalid group_mode={group_mode!r}; expected one of {VALID_GROUP_MODES}")
    return curves_dir / f"{experiment_id}_{group_mode}_fold{int(fold)}.csv"


def discover_curves(curves_dir: Path) -> Dict[Tuple[str, str], Dict[int, Path]]:
    """Discover curve files grouped by exact `(experiment_id, group_mode)`."""
    discovered: Dict[Tuple[str, str], Dict[int, Path]] = {}
    if not curves_dir.exists():
        return discovered
    for entry in sorted(curves_dir.iterdir()):
        if not entry.is_file() or entry.suffix != ".csv":
            continue
        parsed = parse_curve_filename(entry.name)
        if parsed is None:
            continue
        exp_id, g_mode, fold = parsed
        exact_path = get_exact_curve_path(curves_dir, exp_id, g_mode, fold)
        if entry == exact_path:
            discovered.setdefault((exp_id, g_mode), {})[fold] = exact_path
    return discovered


def read_last_epoch_row(curve_path: Path) -> Optional[Dict[str, str]]:
    if not curve_path.exists() or curve_path.stat().st_size == 0:
        return None
    last_valid: Optional[Dict[str, str]] = None
    with curve_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = _parse_int(row.get("epoch"))
            mae = _parse_float(row.get("mae"))
            if ep is not None and not math.isnan(mae):
                last_valid = dict(row)
    return last_valid


def compute_paired_mae_diff(
    cand_rows: Dict[int, Dict[str, str]],
    ref_rows: Optional[Dict[int, Dict[str, str]]],
) -> Tuple[str, str, List[str]]:
    """Return `(mean_diff +- SE, folds_won_str, per_fold_diff_strs)` for `cand - ref`."""
    if not ref_rows:
        return "—", "—", ["—"] * 5
    shared = [k for k in range(5) if k in cand_rows and k in ref_rows]
    if not shared:
        return "—", "—", ["—"] * 5

    cand_mae = np.asarray([_parse_float(cand_rows[k].get("mae")) for k in shared], dtype=np.float64)
    ref_mae = np.asarray([_parse_float(ref_rows[k].get("mae")) for k in shared], dtype=np.float64)
    valid = ~(np.isnan(cand_mae) | np.isnan(ref_mae))
    cand_mae_v = cand_mae[valid]
    ref_mae_v = ref_mae[valid]
    n = len(cand_mae_v)
    if n == 0:
        return "—", "—", ["—"] * 5

    diffs = cand_mae_v - ref_mae_v
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0
    wins = int(np.sum(cand_mae_v < ref_mae_v))

    fold_cells: List[str] = []
    for k in range(5):
        if k in cand_rows and k in ref_rows:
            cm = _parse_float(cand_rows[k].get("mae"))
            rm = _parse_float(ref_rows[k].get("mae"))
            if not (math.isnan(cm) or math.isnan(rm)):
                fold_cells.append(f"{cm - rm:+.2f}")
            else:
                fold_cells.append("—")
        else:
            fold_cells.append("—")

    return f"{mean_d:+.2f} +- {se_d:.2f}", f"{wins}/{n}", fold_cells


def generate_full_last_epoch_summary(
    *,
    curves_dir: Path,
    output_md: Path,
) -> str:
    discovered = discover_curves(curves_dir)

    # Separate 5-fold complete experiments vs partial (<5 folds) experiments
    five_fold_rows: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    partial_rows: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for key, fold_paths in discovered.items():
        parsed_folds: Dict[int, Dict[str, str]] = {}
        for k, p in sorted(fold_paths.items()):
            last_row = read_last_epoch_row(p)
            if last_row is not None:
                parsed_folds[k] = last_row
        if len(parsed_folds) == 5 and all(k in parsed_folds for k in range(5)):
            five_fold_rows[key] = parsed_folds
        elif parsed_folds:
            partial_rows[key] = parsed_folds

    preferred_keys = [(exp_id, g_mode) for exp_id, g_mode, _ in PREFERRED_ORDER]
    ordered_keys: List[Tuple[str, str]] = [k for k in preferred_keys if k in five_fold_rows]
    for k in sorted(five_fold_rows.keys()):
        if k not in ordered_keys:
            ordered_keys.append(k)

    lines: List[str] = [
        "# Step 4 Full Last-Epoch Metrics & Paired MAE Comparisons",
        "",
        f"- **Source:** Last-epoch row per fold from `{curves_dir.relative_to(PROJECT_ROOT) if curves_dir.is_relative_to(PROJECT_ROOT) else curves_dir}/{{experiment_id}}_{{group_mode}}_fold{{fold}}.csv` (`n=5` folds per experiment, exact `{{experiment_id}}` match, `group_mode in {{'station', 'spatial_cluster'}}`).",
        "- **Summary statistics:** `mean +- SD` (sample standard deviation, `ddof=1`); paired differences report `mean diff +- SE` (`SE = SD / sqrt(5)`, `candidate - reference`, negative = candidate lowers MAE) and `folds won / 5` (no p-values).",
        "",
        "## 1. 5-Fold Last-Epoch Metrics (`mean +- SD` across all 10 metrics)",
        "",
        "| Experiment | Mode | mae | rmse | plcc | srcc | r2 | acc | pm1_acc | macro_f1 | tail_mae | tail_f1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key in ordered_keys:
        exp_id, g_mode = key
        f_rows = five_fold_rows[key]
        m_cells: List[str] = []
        for m_key, _, prec in METRIC_SPECS:
            vals = [_parse_float(f_rows[k].get(m_key)) for k in range(5)]
            mean_v, sd_v = _mean_sd(vals)
            if math.isnan(mean_v):
                m_cells.append("—")
            else:
                m_cells.append(f"{mean_v:.{prec}f} +- {sd_v:.{prec}f}")
        lines.append(f"| `{exp_id}` | `{g_mode}` | " + " | ".join(m_cells) + " |")

    # Collect E4 configs with 5 folds
    e4_keys: List[Tuple[str, str]] = [
        k for k in ordered_keys if k[0].startswith("E4_")
    ]

    ref_e2 = five_fold_rows.get(("E2_mobilenetv3_s_pretrained", "station"))
    ref_r1 = five_fold_rows.get(("E4_mobilenetv3_s_r1_no_prototypes", "station"))
    ref_r2 = five_fold_rows.get(("E4_mobilenetv3_s_baseline", "station"))

    # Combined overview table of paired differences across all three references
    lines.extend(
        [
            "",
            "## 2. Paired Last-Epoch MAE Overview (All E4 Configs vs E2, R1, and R2)",
            "",
            "| E4 Config | Last-Ep MAE (Mean +- SD) | Diff vs `E2_mobilenetv3_s_pretrained` (Mean +- SE) | Folds Won vs E2 | Diff vs `E4_mobilenetv3_s_r1_no_prototypes` (Mean +- SE) | Folds Won vs R1 | Diff vs `E4_mobilenetv3_s_baseline` (Mean +- SE) | Folds Won vs R2 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for key in e4_keys:
        exp_id, _ = key
        f_rows = five_fold_rows[key]
        maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
        m_mae, sd_mae = _mean_sd(maes)
        mae_str = f"**{m_mae:.2f} +- {sd_mae:.2f}**"
        d_e2, w_e2, _ = compute_paired_mae_diff(f_rows, ref_e2)
        d_r1, w_r1, _ = compute_paired_mae_diff(f_rows, ref_r1)
        d_r2, w_r2, _ = compute_paired_mae_diff(f_rows, ref_r2)
        lines.append(
            f"| `{exp_id}` | {mae_str} | {d_e2} | {w_e2} | {d_r1} | {w_r1} | {d_r2} | {w_r2} |"
        )

    # Individual detailed paired tables (per-fold differences + mean diff +- SE + folds won, NO p-values)
    paired_targets: Tuple[Tuple[str, str, Optional[Dict[int, Dict[str, str]]]], ...] = (
        ("2a", "E2_mobilenetv3_s_pretrained", ref_e2),
        ("2b", "E4_mobilenetv3_s_r1_no_prototypes", ref_r1),
        ("2c", "E4_mobilenetv3_s_baseline", ref_r2),
    )

    for sec_id, ref_name, ref_rows in paired_targets:
        lines.extend(
            [
                "",
                f"### Table {sec_id}. Paired Last-Epoch MAE: E4 Configs vs `{ref_name}`",
                "",
                f"| E4 Config | Last-Ep MAE (Mean +- SD) | Diff vs `{ref_name}` (Mean +- SE) | Folds Won | F0 Diff | F1 Diff | F2 Diff | F3 Diff | F4 Diff |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for key in e4_keys:
            exp_id, _ = key
            f_rows = five_fold_rows[key]
            maes = [_parse_float(f_rows[k].get("mae")) for k in range(5)]
            m_mae, sd_mae = _mean_sd(maes)
            mae_str = f"{m_mae:.2f} +- {sd_mae:.2f}"
            d_str, w_str, f_diffs = compute_paired_mae_diff(f_rows, ref_rows)
            lines.append(
                f"| `{exp_id}` | {mae_str} | **{d_str}** | {w_str} | "
                f"{f_diffs[0]} | {f_diffs[1]} | {f_diffs[2]} | {f_diffs[3]} | {f_diffs[4]} |"
            )

    if partial_rows:
        lines.extend(
            [
                "",
                "## 3. Partial (<5 Folds) Curve Sets in `outputs/curves/` (Exact `experiment_id` Match)",
                "",
                "| Experiment | Mode | Folds Found | Last-Ep MAE | F0 | F1 | F2 | F3 | F4 |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for exp_id, g_mode in sorted(partial_rows.keys()):
            f_rows = partial_rows[(exp_id, g_mode)]
            n_f = len(f_rows)
            maes = [_parse_float(f_rows[k].get("mae")) for k in sorted(f_rows.keys())]
            m_mae, sd_mae = _mean_sd(maes)
            mae_cell = (
                f"{m_mae:.2f} (n={n_f}/5)"
                if n_f == 1
                else f"{m_mae:.2f} +- {sd_mae:.2f} (n={n_f}/5)"
            )
            f_cells = [
                f"{_parse_float(f_rows[k].get('mae')):.2f}" if k in f_rows else "—"
                for k in range(5)
            ]
            lines.append(
                f"| `{exp_id}` | `{g_mode}` | {n_f}/5 | {mae_cell} | "
                + " | ".join(f_cells)
                + " |"
            )

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    return content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full 10-metric last-epoch summary and paired MAE tables (CPU-only, read-only)."
    )
    parser.add_argument(
        "--curves-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/curves"),
        help="Directory containing per-epoch curve CSVs.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step4_last_epoch_full.md"),
        help="Output Markdown file path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_md = Path(args.output_md)
    generate_full_last_epoch_summary(
        curves_dir=Path(args.curves_dir),
        output_md=output_md,
    )
    print(f"Wrote full last-epoch summary to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
