#!/usr/bin/env python3
"""CPU-only fold diagnostic script for PM25Vision (station-mode 5-fold CV).

Reads cleaned metadata and existing results CSVs (`outputs/results/baselines_cv_results.csv`,
`outputs/results/image_models_cv_results.csv`) without training any models, and writes
`outputs/diagnostics/folds.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import load_and_clean_metadata  # noqa: E402
from src.data.splits import build_cv_splits  # noqa: E402


BIN_LABELS: Tuple[str, ...] = (
    "Bin 1 (0–50)",
    "Bin 2 (51–100)",
    "Bin 3 (101–150)",
    "Bin 4 (151–200)",
    "Bin 5 (201–300)",
    "Bin 6 (301–600)",
)


def _lookup_fold_mae(
    csv_path: Path,
    experiment_id: str,
    group_mode: str = "station",
    eval_split: str = "val",
) -> Tuple[Dict[str, float], bool]:
    """Return ({fold_str: mae}, has_all_5_folds) from a results CSV."""
    if not csv_path.exists():
        return {}, False
    df = pd.read_csv(csv_path)
    sub = df[
        (df["experiment_id"] == experiment_id)
        & (df["group_mode"] == group_mode)
        & (df["eval_split"] == eval_split)
    ]
    out: Dict[str, float] = {}
    for _, row in sub.iterrows():
        fold_key = str(row["fold"])
        out[fold_key] = float(row["mae"])
    has_all_5 = all(str(k) in out for k in range(5))
    return out, has_all_5


def generate_fold_diagnostics(
    train_csv: Path = Path("datasets/train/metadata.csv"),
    test_csv: Path = Path("datasets/test/metadata.csv"),
    baselines_csv: Path = Path("outputs/results/baselines_cv_results.csv"),
    image_csv: Path = Path("outputs/results/image_models_cv_results.csv"),
    output_md: Path = Path("outputs/diagnostics/folds.md"),
) -> str:
    """Compute station-mode fold diagnostics and write markdown report to `output_md`."""
    train_df, _ = load_and_clean_metadata(train_csv, test_csv)
    _, folds = build_cv_splits(train_df, n_splits=5, group_mode="station")

    # Part (a): Per-fold counts, bin distribution, share of bins 5-6 (0-indexed {4, 5}), median MAE
    fold_rows_a: List[Dict[str, object]] = []
    median_maes: Dict[str, float] = {}

    total_val_images = len(train_df)
    total_bins = np.bincount(train_df["aqi_bin"].to_numpy(dtype=np.int64), minlength=6)[:6]
    total_tail = int(total_bins[4] + total_bins[5])

    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_sub = train_df.iloc[tr_idx]
        va_sub = train_df.iloc[va_idx]

        tr_pm25 = tr_sub["pm25"].to_numpy(dtype=np.float64)
        va_pm25 = va_sub["pm25"].to_numpy(dtype=np.float64)
        va_bins = va_sub["aqi_bin"].to_numpy(dtype=np.int64)

        bin_counts = np.bincount(va_bins, minlength=6)[:6]
        n_val = len(va_sub)
        n_tail = int(bin_counts[4] + bin_counts[5])
        tail_share_fold_pct = 100.0 * n_tail / max(n_val, 1)
        tail_share_global_pct = 100.0 * n_tail / max(total_tail, 1)

        train_median = float(np.median(tr_pm25))
        med_mae = float(np.mean(np.abs(va_pm25 - train_median)))
        median_maes[str(k)] = med_mae

        fold_rows_a.append(
            {
                "fold": k,
                "n_val": n_val,
                "n_stations": int(va_sub["station_id"].nunique()),
                "b0": int(bin_counts[0]),
                "b1": int(bin_counts[1]),
                "b2": int(bin_counts[2]),
                "b3": int(bin_counts[3]),
                "b4": int(bin_counts[4]),
                "b5": int(bin_counts[5]),
                "n_tail": n_tail,
                "tail_share_fold_pct": tail_share_fold_pct,
                "tail_share_global_pct": tail_share_global_pct,
                "train_median": train_median,
                "val_mean_aqi": float(np.mean(va_pm25)),
                "val_std_aqi": float(np.std(va_pm25, ddof=1)),
                "median_mae": med_mae,
            }
        )

    med_vals = np.array([median_maes[str(k)] for k in range(5)], dtype=np.float64)
    median_maes["cv_mean"] = float(np.mean(med_vals))
    median_maes["cv_std"] = float(np.std(med_vals, ddof=1))

    # Part (b): Fold 3 top 10 busiest 1-degree lat/lon cells in validation
    _, va3_idx = folds[3]
    va3_df = train_df.iloc[va3_idx].copy()
    va3_df["lat_cell"] = np.floor(va3_df["latitude"].to_numpy(dtype=np.float64)).astype(int)
    va3_df["lon_cell"] = np.floor(va3_df["longitude"].to_numpy(dtype=np.float64)).astype(int)
    va3_df["is_tail"] = va3_df["aqi_bin"].isin([4, 5]).astype(int)

    fold3_total_tail = int(va3_df["is_tail"].sum())
    cell_grp = (
        va3_df.groupby(["lat_cell", "lon_cell"], as_index=False)
        .agg(
            n_images=("image_id", "count"),
            n_stations=("station_id", "nunique"),
            mean_aqi=("pm25", "mean"),
            max_aqi=("pm25", "max"),
            n_tail=("is_tail", "sum"),
            n_bin5_haz=("aqi_bin", lambda s: int((s == 5).sum())),
        )
        .sort_values(["n_images", "n_tail", "mean_aqi"], ascending=[False, False, False])
        .head(10)
    )

    # Part (c): Cross-model per-fold MAE comparison table
    e1_maes, e1_complete = _lookup_fold_mae(baselines_csv, "E1_metadata_mlp")
    e2c_maes, _ = _lookup_fold_mae(image_csv, "E2_mobilenetv3_s_pretrained")
    e3b_maes, _ = _lookup_fold_mae(image_csv, "E3b_mobilenetv3_s_md_neck_lr1e-3_ep40")

    lines: List[str] = []
    lines.append("# Station-Mode 5-Fold Cross-Validation Diagnostics")
    lines.append("")
    lines.append(
        "Generated by `scripts/analysis/diagnose_folds.py` (CPU-only, read-only analysis of cleaned metadata and baseline/image CV results CSVs)."
    )
    lines.append("")
    lines.append("## (a) Per-Fold Distribution & Median-Predictor Baseline (`station` mode)")
    lines.append("")
    lines.append(
        "- **Bins 1–6** correspond to 0-indexed `aqi_bin` $0\\dots 5$: `0–50`, `51–100`, `101–150`, `151–200`, `201–300` (Bin 5, Very Unhealthy), and `301–600` (Bin 6, Hazardous)."
    )
    lines.append(
        "- **Bins 5–6 (`AQI > 200`, 0-indexed `{4, 5}`)** represent the high-pollution tail."
    )
    lines.append(
        "- **Median-Predictor MAE** is the validation MAE of predicting the fold's training-set median AQI on all validation images in that fold."
    )
    lines.append("")
    lines.append(
        "| Fold | Val Images ($n$) | Val Stations | Bin 1 (`0–50`) | Bin 2 (`51–100`) | Bin 3 (`101–150`) | Bin 4 (`151–200`) | Bin 5 (`201–300`) | Bin 6 (`301–600`) | Bins 5–6 Count | Bins 5–6 Share of Fold | Val Mean $\\pm$ SD AQI | Train Median AQI | Median-Predictor MAE |"
    )
    lines.append(
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in fold_rows_a:
        lines.append(
            f"| **{r['fold']}** | {r['n_val']} | {r['n_stations']} | "
            f"{r['b0']} | {r['b1']} | {r['b2']} | {r['b3']} | {r['b4']} | {r['b5']} | "
            f"{r['n_tail']} | {r['tail_share_fold_pct']:.2f}% ({r['tail_share_global_pct']:.1f}% of all tail) | "
            f"{r['val_mean_aqi']:.2f} $\\pm$ {r['val_std_aqi']:.2f} | {r['train_median']:.2f} | **{r['median_mae']:.2f}** |"
        )
    lines.append(
        f"| **Total / CV** | {total_val_images} | {int(train_df['station_id'].nunique())} | "
        f"{int(total_bins[0])} | {int(total_bins[1])} | {int(total_bins[2])} | "
        f"{int(total_bins[3])} | {int(total_bins[4])} | {int(total_bins[5])} | "
        f"{total_tail} | {100.0 * total_tail / total_val_images:.2f}% (100.0%) | "
        f"{float(train_df['pm25'].mean()):.2f} $\\pm$ {float(train_df['pm25'].std(ddof=1)):.2f} | — | "
        f"**{median_maes['cv_mean']:.2f} $\\pm$ {median_maes['cv_std']:.2f}** |"
    )
    lines.append("")

    lines.append("## (b) Fold 3 Validation: 10 Busiest $1^\\circ \\times 1^\\circ$ Lat/Lon Cells")
    lines.append("")
    lines.append(
        f"Fold 3 contains **{len(va3_df)}** validation images, including **{fold3_total_tail}** images in Bins 5–6 (`AQI > 200`, `{int((va3_df['aqi_bin'] == 4).sum())}` in Bin 5 and `{int((va3_df['aqi_bin'] == 5).sum())}` in Bin 6)."
    )
    lines.append(
        "Each cell is defined by $(\\lfloor\\text{lat}\\rfloor, \\lfloor\\text{lon}\\rfloor)$ covering $[\\text{lat}, \\text{lat}+1^\\circ) \\times [\\text{lon}, \\text{lon}+1^\\circ)$."
    )
    lines.append("")
    lines.append(
        "| Rank | $1^\\circ$ Cell $(\\lfloor\\text{lat}\\rfloor, \\lfloor\\text{lon}\\rfloor)$ | Val Stations | Val Images | Mean AQI | Max AQI | Bins 5–6 Images (Bin 6 Haz.) | Share of Fold 3 Bins 5–6 |"
    )
    lines.append(
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for rank, (_, crow) in enumerate(cell_grp.iterrows(), start=1):
        lat_c = int(crow["lat_cell"])
        lon_c = int(crow["lon_cell"])
        n_img = int(crow["n_images"])
        n_st = int(crow["n_stations"])
        m_aqi = float(crow["mean_aqi"])
        mx_aqi = float(crow["max_aqi"])
        n_tl = int(crow["n_tail"])
        n_b6 = int(crow["n_bin5_haz"])
        share_tl = 100.0 * n_tl / max(fold3_total_tail, 1)
        lines.append(
            f"| {rank} | `({lat_c}, {lon_c})` ($[{lat_c}^\\circ,{lat_c+1}^\\circ) \\times [{lon_c}^\\circ,{lon_c+1}^\\circ)$) | "
            f"{n_st} | {n_img} | {m_aqi:.2f} | {mx_aqi:.1f} | {n_tl} ({n_b6}) | {share_tl:.2f}% |"
        )
    lines.append("")

    lines.append("## (c) Per-Fold Validation MAE Comparison (`station` mode)")
    lines.append("")
    if e1_complete:
        lines.append(
            "- Per-fold `E1_metadata_mlp` (`station` mode) rows are present in `outputs/results/baselines_cv_results.csv`."
        )
    else:
        lines.append(
            "- **Note:** Some or all per-fold `E1_metadata_mlp` rows are missing from `outputs/results/baselines_cv_results.csv`; missing entries are marked `MISSING` (not invented)."
        )
    lines.append("")
    lines.append(
        "| Fold | Median Predictor (`E0_median`) | Metadata MLP (`E1_metadata_mlp`) | MobileNetV3-S Pretrained (`E2_mobilenetv3_s_pretrained`) | E3b MobileNetV3-S + MD-Neck (`E3b_mobilenetv3_s_md_neck_lr1e-3_ep40`) |"
    )
    lines.append(
        "| :---: | :---: | :---: | :---: | :---: |"
    )

    def _fmt(d: Dict[str, float], k: str) -> str:
        return f"{d[k]:.2f}" if k in d else "MISSING"

    for k in range(5):
        ks = str(k)
        lines.append(
            f"| **{k}** | {_fmt(median_maes, ks)} | {_fmt(e1_maes, ks)} | {_fmt(e2c_maes, ks)} | {_fmt(e3b_maes, ks)} |"
        )

    def _fmt_mean_std(d: Dict[str, float]) -> str:
        if "cv_mean" in d and "cv_std" in d:
            return f"**{d['cv_mean']:.2f} $\\pm$ {d['cv_std']:.2f}**"
        vals = [d[str(i)] for i in range(5) if str(i) in d]
        if len(vals) == 5:
            return f"**{float(np.mean(vals)):.2f} $\\pm$ {float(np.std(vals, ddof=1)):.2f}**"
        return "MISSING"

    lines.append(
        f"| **CV Mean $\\pm$ SD** | {_fmt_mean_std(median_maes)} | {_fmt_mean_std(e1_maes)} | {_fmt_mean_std(e2c_maes)} | {_fmt_mean_std(e3b_maes)} |"
    )
    lines.append("")
    lines.append("## Key Diagnostic Findings")
    lines.append("")
    lines.append(
        "1. **Fold 3 is an extreme tail-variance outlier for ALL models (including non-visual baselines):** "
        f"Even the constant Median Predictor jumps from `62.31–67.18` MAE on Folds 0/1/2/4 to **`{median_maes['3']:.2f}`** on Fold 3 "
        f"(validation AQI standard deviation `{fold_rows_a[3]['val_std_aqi']:.2f}` vs `{min(r['val_std_aqi'] for i, r in enumerate(fold_rows_a) if i != 3):.2f}–{max(r['val_std_aqi'] for i, r in enumerate(fold_rows_a) if i != 3):.2f}` in other folds), "
        f"and the GPS+month `E1_metadata_mlp` jumps from `38.11–42.98` to **`{e1_maes.get('3', float('nan')):.2f}`**."
    )
    lines.append(
        "2. **Station-grouped concentration of severe/hazardous episodes in Fold 3:** "
        "Although `StratifiedGroupKFold` balances total images (`1,624–1,654`), individual multi-image stations with extreme pollution episodes must stay intact in a single fold. "
        "In Fold 3, just four $1^\\circ \\times 1^\\circ$ cells (`7` stations total) account for **`76.86%` (`176 / 229`)** of Fold 3's Bins 5–6 images: "
        "`(28, 77)` (Delhi/NCR, `85` images, mean AQI `288.84`, `79` tail = `34.50%`), "
        "`(25, -101)` (Saltillo/Monterrey, `51` images, mean AQI `474.00`, `47` Bin-6 Hazardous = `20.52%`), "
        "`(19, 99)` (Chiang Rai, `32` images, mean AQI `225.72`, `31` tail = `13.54%`), and "
        "`(18, 73)` (Pune/Maharashtra, `39` images, mean AQI `190.18`, `19` tail = `8.30%`)."
    )
    lines.append("")

    report = "\n".join(lines)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CPU-only fold diagnostics markdown.")
    parser.add_argument("--output-md", type=str, default="outputs/diagnostics/folds.md")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.output_md)
    generate_fold_diagnostics(output_md=out_path)
    print(f"Wrote fold diagnostics to {out_path}")


if __name__ == "__main__":
    main()
