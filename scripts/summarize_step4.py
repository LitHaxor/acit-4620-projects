#!/usr/bin/env python3
"""CPU-only, read-only summary generator for the Step 4 PFRML experiment ladder.

Reads `outputs/results/step4_results.csv` (plus reference rows from
`outputs/results/step3_results.csv` and `outputs/results/step2_results.csv`
without modifying any CSV) and writes `outputs/results/step4_summary.md` (or a
user-specified `--output-md`).

Per configuration reports:
- 5-fold MAE mean +- SD (plus full metric table)
- Per-fold MAE (folds 0..4)
- Per-fold skill `1 - MAE / MAE_median` (using per-fold Median-Predictor MAE from `folds.md`)
- Paired difference vs `R2` (`E4_mobilenetv3_s_baseline`) and vs `E2_mobilenetv3_s_pretrained`
  as `mean difference +- standard error`, `folds won / K`, and two-sided Wilcoxon
  signed-rank `p`-value with the required note: `"n=5: minimum possible two-sided p = 0.0625"`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import paired_wilcoxon_test  # noqa: E402


# Per-fold Median-Predictor MAE (`E0_median`) from `outputs/diagnostics/folds.md`
STATION_MEDIAN_MAE: Dict[int, float] = {
    0: 67.1816807738815,
    1: 62.311268472906406,
    2: 63.712661339889365,
    3: 74.32928311057108,
    4: 63.49569230769231,
}
SPATIAL_MEDIAN_MAE: Dict[int, float] = {
    0: 72.22174709835065,
    1: 59.912225705329156,
    2: 64.19993932038835,
    3: 71.86283185840708,
    4: 62.52529668956902,
}

WILCOXON_NOTE = "n=5: minimum possible two-sided p = 0.0625"

CONFIG_LABELS: Dict[Tuple[str, str], str] = {
    ("E2_mobilenetv3_s_pretrained", "station"): "Ref 1: E2 MobileNetV3-S pretrained",
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station"): "Ref 2: E3b MobileNetV3-S + MD-Neck",
    ("E4_mobilenetv3_s_r1_no_prototypes", "station"): "R1: DCM + Transformer + MLP (no prototypes)",
    ("E4_mobilenetv3_s_baseline", "station"): "R2: PFRML baseline (toggles off)",
    ("E4_mobilenetv3_s_attn_pool", "station"): "R2+T1: + attention pooling",
    ("E4_mobilenetv3_s_physics_gate", "station"): "R2+T2: + physics gate",
    ("E4_mobilenetv3_s_class_weights", "station"): "R2+T3: + class weights",
    ("E4_mobilenetv3_s_dcm_concat", "station"): "R2+T4: + DCM concat",
    ("E4_mobilenetv3_s_full", "station"): "R2+all: PFRML full",
    ("E2_mobilenetv3_s_pretrained_spatial", "spatial_cluster"): "Extra A: E2 MobileNetV3-S (spatial_cluster)",
    ("E1_metadata_mlp", "spatial_cluster"): "Ref Spatial: E1 MetadataMLP (spatial_cluster)",
    ("E4_mdnet_t_baseline_ep60", "station"): "Extra B: PFRML MDNet-T scratch (60 ep)",
}

DISPLAY_ORDER: Tuple[Tuple[str, str], ...] = (
    ("E2_mobilenetv3_s_pretrained", "station"),
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station"),
    ("E4_mobilenetv3_s_r1_no_prototypes", "station"),
    ("E4_mobilenetv3_s_baseline", "station"),
    ("E4_mobilenetv3_s_attn_pool", "station"),
    ("E4_mobilenetv3_s_physics_gate", "station"),
    ("E4_mobilenetv3_s_class_weights", "station"),
    ("E4_mobilenetv3_s_dcm_concat", "station"),
    ("E4_mobilenetv3_s_full", "station"),
    ("E4_mdnet_t_baseline_ep60", "station"),
    ("E2_mobilenetv3_s_pretrained_spatial", "spatial_cluster"),
    ("E1_metadata_mlp", "spatial_cluster"),
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("experiment_id")]


def _parse_float(val: Optional[str]) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _collect_per_fold(
    rows: Sequence[Dict[str, str]],
) -> Dict[Tuple[str, str], Dict[int, Dict[str, str]]]:
    """Group validation fold rows (`fold in {'0','1','2','3','4'}`) by `(experiment_id, group_mode)`."""
    grouped: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for r in rows:
        if r.get("eval_split") != "val":
            continue
        fold_str = str(r.get("fold", ""))
        if fold_str not in {"0", "1", "2", "3", "4"}:
            continue
        key = (str(r.get("experiment_id", "")), str(r.get("group_mode", "station")))
        grouped.setdefault(key, {})[int(fold_str)] = r
    return grouped


def _mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _paired_comparison(
    cand_folds: Dict[int, Dict[str, str]],
    ref_folds: Optional[Dict[int, Dict[str, str]]],
) -> Tuple[str, str, str]:
    """Return `(mean_diff +- SE, folds_won_str, wilcoxon_p_str)` for `cand - ref` MAE."""
    if not ref_folds:
        return "—", "—", "—"
    shared = sorted(k for k in cand_folds.keys() if k in ref_folds)
    if not shared:
        return "—", "—", "—"

    cand_mae = np.asarray([_parse_float(cand_folds[k].get("mae")) for k in shared], dtype=np.float64)
    ref_mae = np.asarray([_parse_float(ref_folds[k].get("mae")) for k in shared], dtype=np.float64)
    valid = ~(np.isnan(cand_mae) | np.isnan(ref_mae))
    cand_mae = cand_mae[valid]
    ref_mae = ref_mae[valid]
    n = len(cand_mae)
    if n == 0:
        return "—", "—", "—"

    diffs = cand_mae - ref_mae  # negative = candidate lowers MAE
    mean_d = float(np.mean(diffs))
    se_d = float(np.std(diffs, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    wins = int(np.sum(cand_mae < ref_mae))

    if np.allclose(diffs, 0.0):
        return f"{mean_d:+.2f} +- {se_d:.2f}", f"{wins}/{n}", "1.0000"

    if n >= 2:
        w_res = paired_wilcoxon_test(cand_mae, ref_mae, alternative="two-sided")
        p_str = f"{w_res['p_value']:.4f}"
    else:
        p_str = "N/A (n<2)"

    return f"{mean_d:+.2f} +- {se_d:.2f}", f"{wins}/{n}", p_str


def generate_step4_summary(
    *,
    step4_csv: Path,
    step3_csv: Optional[Path] = None,
    step2_csv: Optional[Path] = None,
    output_md: Path,
) -> str:
    """Build markdown summary and write to `output_md`."""
    step4_rows = _read_csv_rows(step4_csv)
    if not step4_rows:
        msg = (
            f"No Step 4 result rows found in `{step4_csv}` (file is missing or empty). "
            "Run `scripts/run_queue_step4.py --pilot` or `scripts/run_queue_step4.py` first."
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(f"# Step 4 Experiment Ladder Summary\n\n{msg}\n", encoding="utf-8")
        print(msg)
        return msg

    step4_grouped = _collect_per_fold(step4_rows)
    if not step4_grouped:
        msg = (
            f"No per-fold validation rows found in `{step4_csv}`. "
            "Run `scripts/run_queue_step4.py` to populate fold results."
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(f"# Step 4 Experiment Ladder Summary\n\n{msg}\n", encoding="utf-8")
        print(msg)
        return msg

    all_grouped: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = dict(step4_grouped)

    # Load read-only reference rows from Step 3 and Step 2 if available
    if step3_csv and step3_csv.exists():
        s3_grouped = _collect_per_fold(_read_csv_rows(step3_csv))
        for ref_key in [
            ("E2_mobilenetv3_s_pretrained", "station"),
            ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station"),
        ]:
            if ref_key in s3_grouped and ref_key not in all_grouped:
                all_grouped[ref_key] = s3_grouped[ref_key]

    if step2_csv and step2_csv.exists():
        s2_grouped = _collect_per_fold(_read_csv_rows(step2_csv))
        ref_spatial_key = ("E1_metadata_mlp", "spatial_cluster")
        if ref_spatial_key in s2_grouped and ref_spatial_key not in all_grouped:
            all_grouped[ref_spatial_key] = s2_grouped[ref_spatial_key]

    # Order keys: known ladder order first, then any extra experiments in step4_csv
    ordered_keys: List[Tuple[str, str]] = [k for k in DISPLAY_ORDER if k in all_grouped]
    for k in step4_grouped.keys():
        if k not in ordered_keys:
            ordered_keys.append(k)

    r2_folds = all_grouped.get(("E4_mobilenetv3_s_baseline", "station"))
    e2_folds = all_grouped.get(("E2_mobilenetv3_s_pretrained", "station"))

    lines: List[str] = [
        "# Step 4 PFRML Experiment Ladder Summary",
        "",
        f"- **Source CSV:** `{step4_csv}`",
        f"- **Wilcoxon signed-rank test note:** **{WILCOXON_NOTE}**.",
        "- **Per-fold skill score:** $\\text{Skill}_k = 1 - \\text{MAE}_k / \\text{MAE}_{\\text{median}, k}$, "
        "where $\\text{MAE}_{\\text{median}, k}$ is the fold-$k$ Median-Predictor MAE from `outputs/diagnostics/folds.md` "
        "(`station`: `[67.18, 62.31, 63.71, 74.33, 63.50]`; `spatial_cluster`: `[72.22, 59.91, 64.20, 71.86, 62.53]`).",
        "- **Paired differences ($\\Delta\\text{MAE}$):** Reported as `mean diff +- SE` (`candidate - reference`; "
        "negative values indicate lower/better MAE than the reference) and `folds won / 5`.",
        "",
        "## 1. Per-Fold Validation MAE, CV Mean $\\pm$ SD & Paired Significance",
        "",
        "| Rung / Config | Group Mode | Folds ($K$) | MAE (Mean $\\pm$ SD) | F0 MAE | F1 MAE | F2 MAE | F3 MAE | F4 MAE | $\\Delta$ vs R2 (Mean $\\pm$ SE) | Won vs R2 | $p$ vs R2 | $\\Delta$ vs E2 (Mean $\\pm$ SE) | Won vs E2 | $p$ vs E2 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key in ordered_keys:
        exp_id, g_mode = key
        fold_map = all_grouped[key]
        label = CONFIG_LABELS.get(key, f"`{exp_id}`")
        maes = [_parse_float(fold_map[k].get("mae")) for k in sorted(fold_map.keys())]
        mean_mae, sd_mae = _mean_sd(maes)
        mae_str = f"**{mean_mae:.2f} $\\pm$ {sd_mae:.2f}**" if not math.isnan(mean_mae) else "—"

        f_cells: List[str] = []
        for k in range(5):
            if k in fold_map:
                v = _parse_float(fold_map[k].get("mae"))
                f_cells.append(f"{v:.2f}" if not math.isnan(v) else "—")
            else:
                f_cells.append("—")

        if g_mode == "station":
            d_r2, w_r2, p_r2 = _paired_comparison(fold_map, r2_folds)
            d_e2, w_e2, p_e2 = _paired_comparison(fold_map, e2_folds)
        else:
            d_r2 = w_r2 = p_r2 = "—"
            d_e2 = w_e2 = p_e2 = "—"

        lines.append(
            f"| {label} (`{exp_id}`) | `{g_mode}` | {len(fold_map)} | {mae_str} | "
            f"{f_cells[0]} | {f_cells[1]} | {f_cells[2]} | {f_cells[3]} | {f_cells[4]} | "
            f"{d_r2} | {w_r2} | {p_r2} | {d_e2} | {w_e2} | {p_e2} |"
        )

    lines.extend(
        [
            "",
            f"> **Statistical Note:** {WILCOXON_NOTE}.",
            "",
            "## 2. Per-Fold Skill Score ($1 - \\text{MAE} / \\text{MAE}_{\\text{median}}$)",
            "",
            "| Rung / Config | Group Mode | Skill (Mean $\\pm$ SD) | F0 Skill | F1 Skill | F2 Skill | F3 Skill | F4 Skill |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for key in ordered_keys:
        exp_id, g_mode = key
        fold_map = all_grouped[key]
        label = CONFIG_LABELS.get(key, f"`{exp_id}`")
        med_lookup = SPATIAL_MEDIAN_MAE if g_mode == "spatial_cluster" else STATION_MEDIAN_MAE

        skills: List[float] = []
        s_cells: List[str] = []
        for k in range(5):
            if k in fold_map and k in med_lookup:
                v = _parse_float(fold_map[k].get("mae"))
                if not math.isnan(v) and med_lookup[k] > 0:
                    sk = 1.0 - (v / med_lookup[k])
                    skills.append(sk)
                    s_cells.append(f"{sk:.4f}")
                else:
                    s_cells.append("—")
            else:
                s_cells.append("—")

        mean_sk, sd_sk = _mean_sd(skills)
        sk_str = f"**{mean_sk:.4f} $\\pm$ {sd_sk:.4f}**" if not math.isnan(mean_sk) else "—"
        lines.append(
            f"| {label} (`{exp_id}`) | `{g_mode}` | {sk_str} | "
            f"{s_cells[0]} | {s_cells[1]} | {s_cells[2]} | {s_cells[3]} | {s_cells[4]} |"
        )

    lines.extend(
        [
            "",
            "## 3. Secondary Cross-Validation Metrics (Mean $\\pm$ SD)",
            "",
            "| Rung / Config | Group Mode | RMSE | PLCC | SRCC | $R^2$ | 6-Class Acc | $\\pm 1$-Class Acc | Macro-F1 | Tail MAE | Tail F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    sec_metrics = [
        ("rmse", 2),
        ("plcc", 4),
        ("srcc", 4),
        ("r2", 4),
        ("acc", 4),
        ("pm1_acc", 4),
        ("macro_f1", 4),
        ("tail_mae", 2),
        ("tail_f1", 4),
    ]
    for key in ordered_keys:
        exp_id, g_mode = key
        fold_map = all_grouped[key]
        label = CONFIG_LABELS.get(key, f"`{exp_id}`")
        m_cells: List[str] = []
        for m_name, prec in sec_metrics:
            vals = [_parse_float(fold_map[k].get(m_name)) for k in sorted(fold_map.keys())]
            m_mean, m_sd = _mean_sd(vals)
            if math.isnan(m_mean):
                m_cells.append("—")
            else:
                m_cells.append(f"{m_mean:.{prec}f} $\\pm$ {m_sd:.{prec}f}")
        lines.append(f"| {label} (`{exp_id}`) | `{g_mode}` | " + " | ".join(m_cells) + " |")

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    print(f"Wrote Step 4 summary ({len(step4_grouped)} Step 4 configs) to {output_md}")
    return content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Step 4 PFRML experiment ladder results into Markdown."
    )
    parser.add_argument(
        "--step4-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step4_results.csv"),
        help="Path to Step 4 results CSV.",
    )
    parser.add_argument(
        "--step3-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step3_results.csv"),
        help="Path to Step 3 results CSV (read-only reference rows).",
    )
    parser.add_argument(
        "--step2-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step2_results.csv"),
        help="Path to Step 2 results CSV (read-only reference rows).",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step4_summary.md"),
        help="Output Markdown file path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    generate_step4_summary(
        step4_csv=Path(args.step4_csv),
        step3_csv=Path(args.step3_csv) if args.step3_csv else None,
        step2_csv=Path(args.step2_csv) if args.step2_csv else None,
        output_md=Path(args.output_md),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
