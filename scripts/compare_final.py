#!/usr/bin/env python3
"""Compare final models on full test and >25 km test sets (`scripts/compare_final.py`).

Generates `outputs/results/final_comparison.md`:
1. One table of all final models (both `test` full and `test_gt25km` >25 km subset)
   with all 10 metrics (`mae, rmse, plcc, srcc, r2, acc, pm1_acc, macro_f1, tail_mae, tail_f1`)
   and 95% station-bootstrap CI for MAE.
2. Paired station-bootstrap MAE differences (`mean diff`, `95% CI [low, high]`, `1,000`
   station resamples with replacement) between every pair of final models on both
   the full test set (`n=2,920`) and the `test_gt25km` subset (`n=717`).
3. For `E1_metadata_mlp`: if saved test predictions (`outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz`
   or `outputs/predictions/E1_metadata_mlp.csv`) and summary rows in `outputs/results/step2_results.csv`
   exist, loads them WITHOUT retraining and without any new test peek (aligning with the
   `image_id, station_id, y_true` columns already saved in `outputs/predictions/<final_name>.csv`);
   otherwise skips `E1_metadata_mlp` in the pairs and notes why.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_train_eval import station_bootstrap_mae_ci  # noqa: E402
from src.metrics import METRIC_ORDER  # noqa: E402


METRIC_SPECS: Tuple[Tuple[str, int], ...] = (
    ("mae", 2),
    ("rmse", 2),
    ("plcc", 4),
    ("srcc", 4),
    ("r2", 4),
    ("acc", 4),
    ("pm1_acc", 4),
    ("macro_f1", 4),
    ("tail_mae", 2),
    ("tail_f1", 4),
)


def _parse_float(val: Any) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def paired_station_bootstrap_mae_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    station_ids: Sequence[Any],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute `(obs_diff, ci_low, ci_high)` for `MAE(A) - MAE(B)` via paired station bootstrap."""
    err_a = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred_a, dtype=np.float64))
    err_b = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred_b, dtype=np.float64))
    diff_i = err_a - err_b
    obs_diff = float(np.mean(diff_i)) if len(diff_i) > 0 else float("nan")

    stations = np.asarray([str(s) for s in station_ids])
    unique_stations, inv = np.unique(stations, return_inverse=True)
    n_stations = len(unique_stations)
    if n_stations <= 1:
        return obs_diff, obs_diff, obs_diff

    station_counts = np.bincount(inv, minlength=n_stations).astype(np.float64)
    station_diff_sums = np.bincount(inv, weights=diff_i, minlength=n_stations).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_stations, size=(n_boot, n_stations))
    boot_sums = station_diff_sums[idx].sum(axis=1)
    boot_counts = station_counts[idx].sum(axis=1)
    boot_diffs = boot_sums / np.maximum(boot_counts, 1.0)
    low, high = np.percentile(boot_diffs, [2.5, 97.5])
    return obs_diff, float(low), float(high)


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_predictions_df(pred_csv: Path) -> Optional[pd.DataFrame]:
    if not pred_csv.exists() or pred_csv.stat().st_size == 0:
        return None
    df = pd.read_csv(pred_csv)
    required = {"image_id", "station_id", "y_true", "y_pred"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def _try_load_e1_metadata_mlp(
    outputs_dir: Path,
    step2_csv: Path,
    reference_pred_df: Optional[pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], Optional[pd.DataFrame], str]:
    """Load existing `E1_metadata_mlp` test rows & saved predictions WITHOUT retraining or reading test CSV."""
    s2_rows = [
        r
        for r in _load_csv_rows(step2_csv)
        if r.get("experiment_id") == "E1_metadata_mlp"
        and r.get("eval_split") in {"test", "test_gt25km"}
    ]
    e1_pred_csv = outputs_dir / "predictions" / "E1_metadata_mlp.csv"
    e1_npz = outputs_dir / "checkpoints" / "E1_metadata_mlp" / "full_train" / "preds.npz"

    e1_pred_df: Optional[pd.DataFrame] = _load_predictions_df(e1_pred_csv)
    if e1_pred_df is None and e1_npz.exists() and reference_pred_df is not None:
        data = np.load(e1_npz)
        if "y_pred" in data and len(data["y_pred"]) == len(reference_pred_df):
            e1_pred_df = reference_pred_df[["image_id", "station_id", "y_true"]].copy()
            e1_pred_df["y_pred"] = np.asarray(data["y_pred"], dtype=np.float64)

    converted_rows: List[Dict[str, Any]] = []
    for r in s2_rows:
        split = str(r.get("eval_split", ""))
        row_dict: Dict[str, Any] = {
            "name": "E1_metadata_mlp",
            "model": r.get("model", "metadata_mlp"),
            "config": "configs/default.yaml",
            "seed": r.get("seed", "42"),
            "epochs": "40",
            "eval_split": split,
            "n_samples": r.get("n_samples", ""),
            "n_stations": "",
            "mae_ci95_low": "",
            "mae_ci95_high": "",
            "notes": r.get("notes", ""),
        }
        for k in METRIC_ORDER:
            row_dict[k] = r.get(k, "")
        if e1_pred_df is not None and split == "test":
            ci_l, ci_h = station_bootstrap_mae_ci(
                e1_pred_df["y_true"].to_numpy(dtype=np.float64),
                e1_pred_df["y_pred"].to_numpy(dtype=np.float64),
                e1_pred_df["station_id"].astype(str).to_numpy(),
            )
            row_dict["mae_ci95_low"] = ci_l
            row_dict["mae_ci95_high"] = ci_h
            row_dict["n_stations"] = int(e1_pred_df["station_id"].nunique())
        converted_rows.append(row_dict)

    if converted_rows and e1_pred_df is not None:
        status = (
            "Loaded `E1_metadata_mlp` from existing `outputs/results/step2_results.csv` and "
            f"`{e1_npz.relative_to(PROJECT_ROOT) if e1_npz.is_relative_to(PROJECT_ROOT) else e1_npz}` "
            "without retraining or reading `datasets/test/`."
        )
    elif converted_rows:
        status = (
            "Loaded `E1_metadata_mlp` summary rows from `outputs/results/step2_results.csv` "
            "(paired per-image bootstrap requires at least one final model prediction CSV in "
            "`outputs/predictions/<name>.csv` to supply `(image_id, station_id, y_true)` without opening test metadata)."
        )
    else:
        status = "Skipped `E1_metadata_mlp`: no saved test predictions or Step 2 rows found."
    return converted_rows, e1_pred_df, status


def generate_final_comparison(
    *,
    final_csv: Path,
    predictions_dir: Path,
    outputs_dir: Path,
    step2_csv: Path,
    output_md: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> str:
    final_rows = _load_csv_rows(final_csv)

    # Load all available per-image prediction CSVs in predictions_dir
    pred_dfs: Dict[str, pd.DataFrame] = {}
    if predictions_dir.exists():
        for p in sorted(predictions_dir.glob("*.csv")):
            df = _load_predictions_df(p)
            if df is not None:
                pred_dfs[p.stem] = df

    ref_pred_df: Optional[pd.DataFrame] = None
    for df in pred_dfs.values():
        if len(df) == 2920 or ref_pred_df is None:
            ref_pred_df = df
            if len(df) == 2920:
                break

    e1_rows, e1_pred_df, e1_status = _try_load_e1_metadata_mlp(
        outputs_dir=outputs_dir,
        step2_csv=step2_csv,
        reference_pred_df=ref_pred_df,
    )
    if e1_pred_df is not None:
        pred_dfs["E1_metadata_mlp"] = e1_pred_df

    all_rows: List[Dict[str, Any]] = list(final_rows)
    existing_names = {str(r.get("name", "")) for r in all_rows}
    if "E1_metadata_mlp" not in existing_names and e1_rows:
        all_rows.extend(e1_rows)

    lines: List[str] = [
        "# Final Test Set & >25 km Generalization Comparison",
        "",
        f"- **Final results source:** `{final_csv.relative_to(PROJECT_ROOT) if final_csv.is_relative_to(PROJECT_ROOT) else final_csv}`",
        f"- **Per-image predictions directory:** `{predictions_dir.relative_to(PROJECT_ROOT) if predictions_dir.is_relative_to(PROJECT_ROOT) else predictions_dir}`",
        f"- **Bootstrap protocol:** Station-level cluster bootstrap (`{n_boot}` resamples of stations with replacement, `seed={seed}`).",
        f"- **E1 Metadata MLP status:** {e1_status}",
        "",
        "## 1. All Final Models (Full Test `n=2,920` and `test_gt25km` `n=717`)",
        "",
        "| Name | Model | Split | n | MAE (95% Station CI) | rmse | plcc | srcc | r2 | acc | pm1_acc | macro_f1 | tail_mae | tail_f1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    if not all_rows:
        lines.append("| — | — | — | 0 | No final runs recorded yet | — | — | — | — | — | — | — | — | — |")
    else:
        for r in all_rows:
            name = str(r.get("name", ""))
            model = str(r.get("model", ""))
            split = str(r.get("eval_split", ""))
            n_samp = str(r.get("n_samples", ""))
            mae_v = _parse_float(r.get("mae"))
            ci_l = _parse_float(r.get("mae_ci95_low"))
            ci_h = _parse_float(r.get("mae_ci95_high"))
            if not math.isnan(mae_v) and not math.isnan(ci_l) and not math.isnan(ci_h):
                mae_cell = f"**{mae_v:.2f}** [{ci_l:.2f}, {ci_h:.2f}]"
            elif not math.isnan(mae_v):
                mae_cell = f"**{mae_v:.2f}**"
            else:
                mae_cell = "—"

            other_cells: List[str] = []
            for m_key, prec in METRIC_SPECS[1:]:
                v = _parse_float(r.get(m_key))
                other_cells.append("—" if math.isnan(v) else f"{v:.{prec}f}")
            lines.append(
                f"| `{name}` | `{model}` | `{split}` | {n_samp} | {mae_cell} | "
                + " | ".join(other_cells)
                + " |"
            )

    lines.extend(
        [
            "",
            "## 2. Paired Station-Bootstrap MAE Differences (`Model A - Model B`, 95% CI)",
            "",
            "Negative `MAE Diff` means `Model A` achieves lower MAE than `Model B`.",
            "",
            "| Model A | Model B | n | Stations | MAE(A) | MAE(B) | MAE Diff (A - B) | 95% Station Bootstrap CI |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    model_names = list(pred_dfs.keys())
    if len(model_names) < 2:
        lines.append("| — | — | 0 | 0 | — | — | Fewer than 2 prediction CSVs available for paired comparison | — |")
    else:
        for name_a, name_b in combinations(model_names, 2):
            df_a = pred_dfs[name_a]
            df_b = pred_dfs[name_b]
            merged = pd.merge(
                df_a[["image_id", "station_id", "y_true", "y_pred"]],
                df_b[["image_id", "y_pred"]],
                on="image_id",
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue
            yt = merged["y_true"].to_numpy(dtype=np.float64)
            yp_a = merged["y_pred_a"].to_numpy(dtype=np.float64)
            yp_b = merged["y_pred_b"].to_numpy(dtype=np.float64)
            sts = merged["station_id"].astype(str).to_numpy()
            mae_a = float(np.mean(np.abs(yt - yp_a)))
            mae_b = float(np.mean(np.abs(yt - yp_b)))
            diff_ab, ci_l, ci_h = paired_station_bootstrap_mae_diff(
                yt, yp_a, yp_b, sts, n_boot=n_boot, seed=seed
            )
            n_st = int(len(np.unique(sts)))
            lines.append(
                f"| `{name_a}` | `{name_b}` | {len(yt)} | {n_st} | "
                f"{mae_a:.2f} | {mae_b:.2f} | **{diff_ab:+.2f}** | [{ci_l:+.2f}, {ci_h:+.2f}] |"
            )

    content = "\n".join(lines) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(content, encoding="utf-8")
    return content


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final models and compute paired station-bootstrap MAE CIs."
    )
    parser.add_argument(
        "--final-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/final_results.csv"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/predictions"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
    )
    parser.add_argument(
        "--step2-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/step2_results.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "outputs/results/final_comparison.md"),
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_md = Path(args.output_md)
    generate_final_comparison(
        final_csv=Path(args.final_csv),
        predictions_dir=Path(args.predictions_dir),
        outputs_dir=Path(args.outputs_dir),
        step2_csv=Path(args.step2_csv),
        output_md=output_md,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    print(f"Wrote final comparison report to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
