#!/usr/bin/env python3
"""CPU-only, read-only last-epoch summary generator for Step 4 and reference runs.

Discovers per-epoch training curves under `outputs/curves/` matching:
    `outputs/curves/{experiment_id}_{group_mode}_fold{fold}.csv`
(with columns `epoch,lr,train_loss,val_loss,mae,...,epoch_time_s`), extracts the
LAST-epoch row (not best-epoch) for each finished `(experiment_id, group_mode, fold)`,
and writes `outputs/results/step4_last_epoch.md` with:
- Per-experiment, per-fold last-epoch MAE and Mean +- SD
- Best-epoch MAE (Mean +- SD) from the results CSVs (`step4_results.csv` /
  `step3_results.csv`) for direct comparison
- Per-fold skill score `1 - MAE / MAE_median` using per-fold Median-Predictor MAE
  from `outputs/diagnostics/folds.md` (and `E0_median` in `step2_results.csv`)
- Explicit handling of missing reference curves (`E2`, `E3b`: named and left blank)
- Explicit handling of partial queues (unfinished folds marked `"pending"`)
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

CURVE_FILENAME_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<group_mode>station|spatial_cluster|full_train)_fold(?P<fold>[0-4])\.csv$"
)
CURVE_FILENAME_FALLBACK_RE = re.compile(
    r"^(?P<experiment_id>.+)_(?P<group_mode>[a-zA-Z0-9]+)_fold(?P<fold>\d+)\.csv$"
)
BEST_EP_RE = re.compile(r"best_ep=([0-9]+/[0-9]+)")

# Per-fold Median-Predictor MAE (`E0_median`) matching `outputs/diagnostics/folds.md`
DEFAULT_MEDIAN_MAE: Dict[str, Dict[int, float]] = {
    "station": {
        0: 67.1816807738815,
        1: 62.311268472906406,
        2: 63.712661339889365,
        3: 74.32928311057108,
        4: 63.49569230769231,
    },
    "spatial_cluster": {
        0: 72.22174709835065,
        1: 59.912225705329156,
        2: 64.19993932038835,
        3: 71.86283185840708,
        4: 62.52529668956902,
    },
}

REFERENCE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("E2_mobilenetv3_s_pretrained", "station"),
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station"),
)

LADDER_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("E2_mobilenetv3_s_pretrained", "station", "Ref 1: E2 MobileNetV3-S pretrained"),
    ("E3b_mobilenetv3_s_md_neck_lr1e-3_ep40", "station", "Ref 2: E3b MobileNetV3-S + MD-Neck"),
    ("E4_mobilenetv3_s_r1_no_prototypes", "station", "R1: DCM + Transformer + MLP (no proto)"),
    ("E4_mobilenetv3_s_baseline", "station", "R2: PFRML baseline (toggles off)"),
    ("E4_mobilenetv3_s_attn_pool", "station", "R2+T1: + attention pooling"),
    ("E4_mobilenetv3_s_physics_gate", "station", "R2+T2: + physics gate"),
    ("E4_mobilenetv3_s_class_weights", "station", "R2+T3: + class weights"),
    ("E4_mobilenetv3_s_dcm_concat", "station", "R2+T4: + DCM concat"),
    ("E4_mobilenetv3_s_full", "station", "R2+all: PFRML full"),
    ("E4_mdnet_t_baseline_ep60", "station", "Extra B: PFRML MDNet-T scratch (60 ep)"),
    ("E2_mobilenetv3_s_pretrained_spatial", "spatial_cluster", "Extra A: E2 MobileNetV3-S (spatial)"),
)

LADDER_LABELS: Dict[Tuple[str, str], str] = {
    (exp_id, g_mode): label for exp_id, g_mode, label in LADDER_SPECS
}


def _parse_float(val: Optional[str]) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        out = float(val)
    except ValueError:
        return float("nan")
    return out


def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def parse_curve_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """Parse `{experiment_id}_{group_mode}_fold{fold}.csv` into `(experiment_id, group_mode, fold)`."""
    m = CURVE_FILENAME_RE.match(filename) or CURVE_FILENAME_FALLBACK_RE.match(filename)
    if not m:
        return None
    return m.group("experiment_id"), m.group("group_mode"), int(m.group("fold"))


def discover_curves(curves_dir: Path) -> Dict[Tuple[str, str], Dict[int, Path]]:
    """Return `{ (experiment_id, group_mode): { fold: curve_csv_path } }`."""
    discovered: Dict[Tuple[str, str], Dict[int, Path]] = {}
    if not curves_dir.exists():
        return discovered
    for path in sorted(curves_dir.glob("*.csv")):
        parsed = parse_curve_filename(path.name)
        if parsed is None:
            continue
        exp_id, g_mode, fold = parsed
        discovered.setdefault((exp_id, g_mode), {})[fold] = path
    return discovered


def read_last_epoch_row(curve_path: Path) -> Optional[Dict[str, str]]:
    """Return the last valid epoch row from a per-epoch curve CSV."""
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


def index_results_val_rows(
    csv_paths: Sequence[Path],
) -> Dict[Tuple[str, str], Dict[int, Dict[str, str]]]:
    """Index `eval_split='val'` fold rows (`0..4`) across results CSVs (earlier paths win)."""
    grouped: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for path in csv_paths:
        for r in _read_csv_rows(path):
            if r.get("eval_split") != "val":
                continue
            fold_int = _parse_int(r.get("fold"))
            if fold_int is None or fold_int not in {0, 1, 2, 3, 4}:
                continue
            exp_id = str(r.get("experiment_id", "")).strip()
            g_mode = str(r.get("group_mode", "station")).strip()
            if not exp_id:
                continue
            key = (exp_id, g_mode)
            grouped.setdefault(key, {}).setdefault(fold_int, r)
    return grouped


def load_median_mae_lookup(
    folds_md: Optional[Path],
    step2_csv: Optional[Path],
) -> Dict[str, Dict[int, float]]:
    """Load per-fold Median-Predictor MAE from `step2_results.csv` / `folds.md` with exact fallback."""
    lookup: Dict[str, Dict[int, float]] = {
        mode: dict(vals) for mode, vals in DEFAULT_MEDIAN_MAE.items()
    }
    if folds_md and folds_md.exists():
        text = folds_md.read_text(encoding="utf-8")
        # Verify station median MAE rows from folds.md table (a) when present
        row_re = re.compile(
            r"^\|\s*\*\*([0-4])\*\*\s*\|.*\|\s*\*\*([0-9]+(?:\.[0-9]+)?)\*\*\s*\|$",
            re.MULTILINE,
        )
        parsed_station = {int(m.group(1)): float(m.group(2)) for m in row_re.finditer(text)}
        if len(parsed_station) == 5:
            # Keep full-precision E0_median if step2_results.csv is available; otherwise use folds.md
            lookup["station"] = parsed_station

    if step2_csv and step2_csv.exists():
        for r in _read_csv_rows(step2_csv):
            if r.get("experiment_id") == "E0_median" and r.get("eval_split") == "val":
                fold_int = _parse_int(r.get("fold"))
                g_mode = str(r.get("group_mode", "")).strip()
                mae = _parse_float(r.get("mae"))
                if fold_int is not None and g_mode in lookup and not math.isnan(mae):
                    lookup[g_mode][fold_int] = mae
    return lookup


def _extract_best_ep(row: Optional[Dict[str, str]]) -> str:
    if not row:
        return "—"
    m = BEST_EP_RE.search(row.get("notes", "") or "")
    return m.group(1) if m else "—"


def _format_mean_sd(vals: Sequence[float], total_folds: int, has_pending: bool, prec: int = 2) -> str:
    valid = [v for v in vals if not math.isnan(v)]
    if not valid:
        return "pending" if has_pending else ""
    mean_v, sd_v = _mean_sd(valid)
    if len(valid) == total_folds and not has_pending:
        return f"**{mean_v:.{prec}f} +- {sd_v:.{prec}f}**"
    if len(valid) == 1:
        tag = f"n=1/{total_folds}, pending" if has_pending else f"n=1/{total_folds}"
        return f"{mean_v:.{prec}f} ({tag})"
    tag = f"n={len(valid)}/{total_folds}, pending" if has_pending else f"n={len(valid)}/{total_folds}"
    return f"{mean_v:.{prec}f} +- {sd_v:.{prec}f} ({tag})"


def generate_last_epoch_summary(
    *,
    curves_dir: Path,
    step4_csv: Path,
    step3_csv: Path,
    step2_csv: Optional[Path],
    folds_md: Optional[Path],
    output_md: Path,
) -> str:
    """Build `step4_last_epoch.md` and return its content."""
    curves = discover_curves(curves_dir)
    results_by_key = index_results_val_rows([step4_csv, step3_csv])
    median_lookup = load_median_mae_lookup(folds_md, step2_csv)

    # Check missing curves for reference runs (E2, E3b)
    missing_ref_curves: List[str] = []
    for ref_key in REFERENCE_KEYS:
        ref_res = results_by_key.get(ref_key, {})
        ref_curves = curves.get(ref_key, {})
        for k in range(5):
            if k in ref_res and k not in ref_curves:
                missing_ref_curves.append(f"{ref_key[0]} ({ref_key[1]}, fold {k})")

    # Build ordered keys for the Step 4 ladder + reference runs
    ladder_keys: List[Tuple[str, str]] = [
        (exp_id, g_mode) for exp_id, g_mode, _ in LADDER_SPECS
    ]
    step4_only_keys = sorted(
        k
        for k in (set(results_by_key.keys()) | set(curves.keys()))
        if k not in set(ladder_keys) and k[0].startswith("E4_")
    )
    main_keys = ladder_keys + step4_only_keys

    # Any other discovered curves (e.g., Step 3 ResNet18 / MDNet-T / Stage-1 fold-0 runs)
    other_keys = sorted(k for k in curves.keys() if k not in set(main_keys))

    def build_key_stats(key: Tuple[str, str]) -> Dict[str, object]:
        exp_id, g_mode = key
        res_folds = results_by_key.get(key, {})
        curve_folds = curves.get(key, {})
        med_map = median_lookup.get(g_mode, median_lookup["station"])
        is_ref = key in REFERENCE_KEYS

        fold_records: Dict[int, Dict[str, object]] = {}
        last_maes: List[float] = []
        best_maes: List[float] = []
        skills: List[float] = []
        done_count = 0
        pending_count = 0

        for k in range(5):
            in_results = k in res_folds
            curve_path = curve_folds.get(k)
            last_row = read_last_epoch_row(curve_path) if curve_path else None

            if in_results:
                done_count += 1
                best_mae = _parse_float(res_folds[k].get("mae"))
                best_ep = _extract_best_ep(res_folds[k])
                if not math.isnan(best_mae):
                    best_maes.append(best_mae)

                if last_row is not None:
                    last_ep = _parse_int(last_row.get("epoch"))
                    last_mae = _parse_float(last_row.get("mae"))
                    last_maes.append(last_mae)
                    sk = 1.0 - (last_mae / med_map[k]) if (k in med_map and med_map[k] > 0) else float("nan")
                    if not math.isnan(sk):
                        skills.append(sk)
                    fold_records[k] = {
                        "status": "done",
                        "last_ep": str(last_ep) if last_ep is not None else "—",
                        "last_mae": last_mae,
                        "last_cell": f"{last_mae:.2f}",
                        "skill_cell": f"{sk:.4f}" if not math.isnan(sk) else "—",
                        "best_ep": best_ep,
                        "best_mae": best_mae,
                    }
                else:
                    # Missing curve for a finished run (leave blank for E2/E3b reference runs)
                    fold_records[k] = {
                        "status": "missing_curve",
                        "last_ep": "",
                        "last_mae": float("nan"),
                        "last_cell": "" if is_ref else "missing",
                        "skill_cell": "" if is_ref else "missing",
                        "best_ep": best_ep,
                        "best_mae": best_mae,
                    }
            else:
                pending_count += 1
                partial_ep = _parse_int(last_row.get("epoch")) if last_row else None
                status_str = f"pending (ep {partial_ep})" if partial_ep is not None else "pending"
                fold_records[k] = {
                    "status": status_str,
                    "last_ep": f"{partial_ep} (running)" if partial_ep is not None else "pending",
                    "last_mae": float("nan"),
                    "last_cell": "pending",
                    "skill_cell": "pending",
                    "best_ep": "pending",
                    "best_mae": float("nan"),
                }

        return {
            "key": key,
            "label": LADDER_LABELS.get(key, f"`{exp_id}`"),
            "exp_id": exp_id,
            "g_mode": g_mode,
            "done_count": done_count,
            "pending_count": pending_count,
            "last_maes": last_maes,
            "best_maes": best_maes,
            "skills": skills,
            "folds": fold_records,
        }

    main_stats = [build_key_stats(k) for k in main_keys]

    if missing_ref_curves:
        ref_status_line = "- **Missing reference curves (left blank):** " + "; ".join(
            f"`{item}`" for item in missing_ref_curves
        )
    else:
        ref_status_line = "- **Reference curves (E2, E3b):** All 5 folds present for `E2_mobilenetv3_s_pretrained` and `E3b_mobilenetv3_s_md_neck_lr1e-3_ep40`."

    lines: List[str] = [
        "# Step 4 Last-Epoch vs Best-Epoch Summary",
        "",
        f"- **Curve path pattern:** `{curves_dir.relative_to(PROJECT_ROOT) if curves_dir.is_relative_to(PROJECT_ROOT) else curves_dir}/{{experiment_id}}_{{group_mode}}_fold{{fold}}.csv`",
        ref_status_line,
        "- **Median-Predictor MAE (`outputs/diagnostics/folds.md`):** `station` `[67.18, 62.31, 63.71, 74.33, 63.50]`; `spatial_cluster` `[72.22, 59.91, 64.20, 71.86, 62.53]`.",
        "",
        "## 1. Last-Epoch MAE vs Best-Epoch MAE (Step 4 Ladder & Reference Runs)",
        "",
        "| Rung / Experiment | Mode | Done | Last-Ep MAE (Mean +- SD) | Best-Ep MAE (Mean +- SD) | F0 | F1 | F2 | F3 | F4 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for st in main_stats:
        has_pending = int(st["pending_count"]) > 0
        last_summary = _format_mean_sd(st["last_maes"], 5, has_pending, prec=2)  # type: ignore[arg-type]
        best_summary = _format_mean_sd(st["best_maes"], 5, has_pending, prec=2)  # type: ignore[arg-type]
        folds_map: Dict[int, Dict[str, object]] = st["folds"]  # type: ignore[assignment]
        f_cells = [str(folds_map[k]["last_cell"]) for k in range(5)]
        lines.append(
            f"| {st['label']} (`{st['exp_id']}`) | `{st['g_mode']}` | {st['done_count']}/5 | "
            f"{last_summary} | {best_summary} | "
            f"{f_cells[0]} | {f_cells[1]} | {f_cells[2]} | {f_cells[3]} | {f_cells[4]} |"
        )

    lines.extend(
        [
            "",
            "## 2. Per-Fold Skill Score (`1 - MAE_last / MAE_median`)",
            "",
            "| Rung / Experiment | Mode | Skill (Mean +- SD) | F0 Skill | F1 Skill | F2 Skill | F3 Skill | F4 Skill |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for st in main_stats:
        has_pending = int(st["pending_count"]) > 0
        sk_summary = _format_mean_sd(st["skills"], 5, has_pending, prec=4)  # type: ignore[arg-type]
        folds_map = st["folds"]  # type: ignore[assignment]
        s_cells = [str(folds_map[k]["skill_cell"]) for k in range(5)]
        lines.append(
            f"| {st['label']} (`{st['exp_id']}`) | `{st['g_mode']}` | {sk_summary} | "
            f"{s_cells[0]} | {s_cells[1]} | {s_cells[2]} | {s_cells[3]} | {s_cells[4]} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Fold Comparison: Last Epoch vs Best Epoch",
            "",
            "| Rung / Experiment | Mode | Fold | Status | Last Ep | Last-Ep MAE | Best Ep | Best-Ep MAE | Diff (Last - Best) | Skill |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for st in main_stats:
        folds_map = st["folds"]  # type: ignore[assignment]
        for k in range(5):
            rec = folds_map[k]
            last_mae = rec["last_mae"]  # type: ignore[assignment]
            best_mae = rec["best_mae"]  # type: ignore[assignment]
            if isinstance(last_mae, float) and not math.isnan(last_mae) and isinstance(best_mae, float) and not math.isnan(best_mae):
                diff_str = f"{last_mae - best_mae:+.2f}"
                last_mae_str = f"{last_mae:.2f}"
                best_mae_str = f"{best_mae:.2f}"
            elif rec["status"] == "missing_curve":
                diff_str = ""
                last_mae_str = ""
                best_mae_str = f"{best_mae:.2f}" if isinstance(best_mae, float) and not math.isnan(best_mae) else ""
            else:
                diff_str = "pending"
                last_mae_str = "pending"
                best_mae_str = "pending"
            lines.append(
                f"| {st['label']} (`{st['exp_id']}`) | `{st['g_mode']}` | {k} | "
                f"{rec['status']} | {rec['last_ep']} | {last_mae_str} | "
                f"{rec['best_ep']} | {best_mae_str} | {diff_str} | {rec['skill_cell']} |"
            )

    if other_keys:
        lines.extend(
            [
                "",
                "## 4. Other Discovered Curve CSVs in `outputs/curves/` (Step 3 Runs & Stage-1 Candidates)",
                "",
                "| Experiment | Mode | Folds Found | Last-Ep MAE (Mean +- SD) | Best-Ep MAE (Mean +- SD) | F0 | F1 | F2 | F3 | F4 |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for key in other_keys:
            exp_id, g_mode = key
            curve_folds = curves[key]
            res_folds = results_by_key.get(key, {})
            med_map = median_lookup.get(g_mode, median_lookup["station"])
            last_maes: List[float] = []
            best_maes: List[float] = []
            f_cells: List[str] = []
            for k in range(5):
                if k in curve_folds:
                    row = read_last_epoch_row(curve_folds[k])
                    if row is not None:
                        lm = _parse_float(row.get("mae"))
                        last_maes.append(lm)
                        f_cells.append(f"{lm:.2f}")
                    else:
                        f_cells.append("—")
                else:
                    f_cells.append("—")
                if k in res_folds:
                    bm = _parse_float(res_folds[k].get("mae"))
                    if not math.isnan(bm):
                        best_maes.append(bm)
            n_found = len(curve_folds)
            last_str = _format_mean_sd(last_maes, 5, has_pending=False, prec=2)
            best_str = _format_mean_sd(best_maes, 5, has_pending=False, prec=2)
            lines.append(
                f"| `{exp_id}` | `{g_mode}` | {n_found}/5 | {last_str} | {best_str} | "
                f"{f_cells[0]} | {f_cells[1]} | {f_cells[2]} | {f_cells[3]} | {f_cells[4]} |"
            )

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    return content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize last-epoch metrics from per-epoch curve CSVs into Markdown (CPU-only, read-only)."
    )
    parser.add_argument(
        "--curves-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/curves"),
        help="Directory containing per-epoch curve CSVs.",
    )
    parser.add_argument(
        "--step4-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step4_results.csv"),
        help="Path to Step 4 results CSV (read-only).",
    )
    parser.add_argument(
        "--step3-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step3_results.csv"),
        help="Path to Step 3 results CSV (read-only).",
    )
    parser.add_argument(
        "--step2-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step2_results.csv"),
        help="Path to Step 2 results CSV for E0_median lookup (read-only).",
    )
    parser.add_argument(
        "--folds-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/diagnostics/folds.md"),
        help="Path to folds diagnostic Markdown (read-only).",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step4_last_epoch.md"),
        help="Output Markdown file path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    curves_dir = Path(args.curves_dir)
    output_md = Path(args.output_md)
    generate_last_epoch_summary(
        curves_dir=curves_dir,
        step4_csv=Path(args.step4_csv),
        step3_csv=Path(args.step3_csv),
        step2_csv=Path(args.step2_csv) if args.step2_csv else None,
        folds_md=Path(args.folds_md) if args.folds_md else None,
        output_md=output_md,
    )
    print(f"Wrote last-epoch summary to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
