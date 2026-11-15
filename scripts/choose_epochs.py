#!/usr/bin/env python3
"""Choose optimal epoch count N for E5_fusion_r1_p00 (Step 5c, Part 1).

CPU-only, read-only script:
- Reads `outputs/curves/E5_fusion_r1_p00_spatial_cluster_fold{0..4}.csv`.
- Per epoch computes the mean validation MAE across the 5 folds, then a 3-epoch
  centered moving average.
- N = epoch with the lowest 3-epoch centered moving average, restricted to N >= 6.
- Prints mean MAE at epochs 4, 6, 8, 10, 12, 15, 20, 30 and the chosen N, and
  writes `outputs/results/step5c_epochs.md`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_EPOCHS: Tuple[int, ...] = (4, 6, 8, 10, 12, 15, 20, 30)
MIN_ALLOWED_EPOCH: int = 6


def load_fold_epoch_maes(
    curves_dir: Path,
    experiment_id: str = "E5_fusion_r1_p00",
    group_mode: str = "spatial_cluster",
    folds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[int, Dict[int, float]]:
    """Return `{epoch: {fold: mae}}` across the specified folds."""
    per_epoch: Dict[int, Dict[int, float]] = {}
    for fold in folds:
        curve_path = curves_dir / f"{experiment_id}_{group_mode}_fold{fold}.csv"
        if not curve_path.exists():
            raise FileNotFoundError(f"Missing curve file: {curve_path}")
        with curve_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row["epoch"])
                mae = float(row["mae"])
                per_epoch.setdefault(ep, {})[fold] = mae
    return per_epoch


def compute_epoch_stats(
    per_epoch: Dict[int, Dict[int, float]],
    folds: Sequence[int] = (0, 1, 2, 3, 4),
    min_epoch: int = MIN_ALLOWED_EPOCH,
) -> Tuple[List[Dict[str, float]], int]:
    """Compute per-epoch 5-fold mean MAE, SD, 3-epoch centered moving average, and chosen N."""
    epochs = sorted(ep for ep, fmap in per_epoch.items() if all(k in fmap for k in folds))
    if not epochs:
        raise ValueError("No complete epochs found across all folds.")

    mean_by_ep: Dict[int, float] = {}
    sd_by_ep: Dict[int, float] = {}
    for ep in epochs:
        vals = np.asarray([per_epoch[ep][k] for k in folds], dtype=np.float64)
        mean_by_ep[ep] = float(np.mean(vals))
        sd_by_ep[ep] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    rows: List[Dict[str, float]] = []
    for ep in epochs:
        # 3-epoch centered window [ep - 1, ep, ep + 1]
        if (ep - 1) in mean_by_ep and (ep + 1) in mean_by_ep:
            cma3 = (mean_by_ep[ep - 1] + mean_by_ep[ep] + mean_by_ep[ep + 1]) / 3.0
        else:
            window_vals = [
                mean_by_ep[t] for t in (ep - 1, ep, ep + 1) if t in mean_by_ep
            ]
            cma3 = float(np.mean(window_vals))
        entry: Dict[str, float] = {
            "epoch": float(ep),
            "mean_mae": mean_by_ep[ep],
            "sd_mae": sd_by_ep[ep],
            "cma3_mae": cma3,
        }
        for k in folds:
            entry[f"fold{k}"] = per_epoch[ep][k]
        rows.append(entry)

    eligible = [r for r in rows if int(r["epoch"]) >= min_epoch]
    if not eligible:
        raise ValueError(f"No epochs >= {min_epoch} available.")
    best_row = min(eligible, key=lambda r: (r["cma3_mae"], r["mean_mae"], int(r["epoch"])))
    chosen_n = int(best_row["epoch"])
    return rows, chosen_n


def read_chosen_n_from_markdown(md_path: Path) -> int:
    """Parse `CHOSEN_N: <int>` or `N = <int>` from `outputs/results/step5c_epochs.md`."""
    if not md_path.exists():
        raise FileNotFoundError(f"Step 5c epochs file not found: {md_path}")
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"CHOSEN_N:\s*([0-9]+)", text)
    if not m:
        m = re.search(r"\bN\s*=\s*([0-9]+)", text)
    if not m:
        raise ValueError(f"Could not parse chosen epoch N from {md_path}")
    return int(m.group(1))


def generate_step5c_epochs_report(
    *,
    curves_dir: Path,
    output_md: Path,
    report_epochs: Sequence[int] = REPORT_EPOCHS,
    min_epoch: int = MIN_ALLOWED_EPOCH,
) -> Tuple[List[Dict[str, float]], int, str]:
    per_epoch = load_fold_epoch_maes(curves_dir=curves_dir)
    rows, chosen_n = compute_epoch_stats(per_epoch, min_epoch=min_epoch)
    rows_by_ep = {int(r["epoch"]): r for r in rows}
    best_row = rows_by_ep[chosen_n]

    lines: List[str] = [
        "# Step 5c Epoch Selection (`E5_fusion_r1_p00`, 5-Fold `spatial_cluster` CV)",
        "",
        "- **Source curves:** `outputs/curves/E5_fusion_r1_p00_spatial_cluster_fold{0..4}.csv`",
        f"- **Selection rule:** Minimize 3-epoch centered moving average (`CMA-3`) of 5-fold mean validation MAE subject to `epoch >= {min_epoch}`.",
        f"- **CHOSEN_N:** {chosen_n} (`N = {chosen_n}`, 5-fold mean MAE = `{best_row['mean_mae']:.2f} +- {best_row['sd_mae']:.2f}`, `CMA-3 = {best_row['cma3_mae']:.2f}`).",
        "",
        "## 1. Selected Checkpoint Epochs (`4, 6, 8, 10, 12, 15, 20, 30`)",
        "",
        "| Epoch | 5-Fold Mean MAE | SD | 3-Epoch Centered MA (`CMA-3`) | F0 | F1 | F2 | F3 | F4 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ep in report_epochs:
        if ep not in rows_by_ep:
            continue
        r = rows_by_ep[ep]
        mark = " **(N)**" if ep == chosen_n else ""
        lines.append(
            f"| {ep}{mark} | **{r['mean_mae']:.2f}** | {r['sd_mae']:.2f} | {r['cma3_mae']:.2f} | "
            f"{r['fold0']:.2f} | {r['fold1']:.2f} | {r['fold2']:.2f} | {r['fold3']:.2f} | {r['fold4']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Full Per-Epoch Trajectory (`1..30`)",
            "",
            "| Epoch | 5-Fold Mean MAE | SD | 3-Epoch Centered MA (`CMA-3`) | F0 | F1 | F2 | F3 | F4 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for r in rows:
        ep = int(r["epoch"])
        mark = " **(N)**" if ep == chosen_n else ""
        lines.append(
            f"| {ep}{mark} | {r['mean_mae']:.2f} | {r['sd_mae']:.2f} | {r['cma3_mae']:.2f} | "
            f"{r['fold0']:.2f} | {r['fold1']:.2f} | {r['fold2']:.2f} | {r['fold3']:.2f} | {r['fold4']:.2f} |"
        )

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    return rows, chosen_n, content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 5-fold mean validation MAE and 3-epoch centered moving average for E5_fusion_r1_p00."
    )
    parser.add_argument(
        "--curves-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/curves"),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step5c_epochs.md"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_md = Path(args.output_md)
    rows, chosen_n, _ = generate_step5c_epochs_report(
        curves_dir=Path(args.curves_dir),
        output_md=output_md,
    )
    rows_by_ep = {int(r["epoch"]): r for r in rows}
    print("Epoch | Mean MAE | SD    | CMA-3 MAE | F0    | F1    | F2    | F3    | F4")
    print("------+----------+-------+-----------+-------+-------+-------+-------+------")
    for ep in REPORT_EPOCHS:
        if ep in rows_by_ep:
            r = rows_by_ep[ep]
            print(
                f"{ep:5d} | {r['mean_mae']:8.2f} | {r['sd_mae']:5.2f} | {r['cma3_mae']:9.2f} | "
                f"{r['fold0']:5.2f} | {r['fold1']:5.2f} | {r['fold2']:5.2f} | {r['fold3']:5.2f} | {r['fold4']:5.2f}"
            )
    best_r = rows_by_ep[chosen_n]
    print(
        f"Chosen N = {chosen_n} (mean MAE = {best_r['mean_mae']:.2f} +- {best_r['sd_mae']:.2f}, "
        f"3-epoch centered MA = {best_r['cma3_mae']:.2f})"
    )
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
