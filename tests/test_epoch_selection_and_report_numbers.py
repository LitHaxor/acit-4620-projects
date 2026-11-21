"""Unit tests for fusion epoch selection, final queue decision rule & specs, >25km comparison, and report numbers builder."""

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.analysis.build_report_numbers import build_report_numbers
from scripts.analysis.compare_final_models import generate_final_comparison
from scripts.analysis.select_fusion_epochs import (
    compute_epoch_stats,
    load_fold_epoch_maes,
    read_chosen_n_from_markdown,
)
from scripts.experiments.queue_fusion_final import (
    DECISION_THRESHOLD_MAE,
    FALLBACK_FINAL_EPOCHS,
    build_fusion_epn_cv_spec,
    build_fusion_final_spec,
    compute_item1_last_epoch_mean_mae,
    evaluate_decision_rule,
)


def _local_or_tracked(local: str, tracked: str) -> Path:
    """Prefer a fresh local run in outputs/ (gitignored); fall back to the tracked results/ snapshot."""
    local_path = Path(local)
    return local_path if local_path.exists() else Path(tracked)


def test_choose_epochs_cma3_and_min_epoch_floor() -> None:
    """Verify 3-epoch centered moving average selection and the N >= 6 floor."""
    # Construct 5 synthetic folds where epochs 2-4 have artificially low MAE (10.0),
    # while among epochs >= 6, epoch 7 has the lowest 3-epoch centered moving average.
    base_curve = {
        1: 60.0,
        2: 10.0,
        3: 10.0,
        4: 10.0,
        5: 50.0,
        6: 46.0,
        7: 44.0,
        8: 45.0,
        9: 49.0,
        10: 50.0,
    }
    per_epoch = {
        ep: {k: val + (k - 2) * 0.5 for k in range(5)}
        for ep, val in base_curve.items()
    }

    rows, chosen_n = compute_epoch_stats(per_epoch, min_epoch=6)
    assert chosen_n == 7
    row7 = next(r for r in rows if int(r["epoch"]) == 7)
    assert np.isclose(row7["mean_mae"], 44.0)
    assert np.isclose(row7["cma3_mae"], (46.0 + 44.0 + 45.0) / 3.0)

    # Also verify real E5_fusion_r1_p00 curves give N = 7
    real_per_ep = load_fold_epoch_maes(_local_or_tracked("outputs/curves", "results/curves"))
    _, real_n = compute_epoch_stats(real_per_ep, min_epoch=6)
    assert real_n == 7
    assert read_chosen_n_from_markdown(_local_or_tracked("outputs/results/fusion_epoch_selection.md", "results/fusion_epoch_selection.md")) == 7


def test_fusion_final_decision_rule_and_specs(tmp_path: Path) -> None:
    """Verify final fusion queue Item 1/2/3 specs and the exact decision rule (< 46.95 -> N, else 30)."""
    spec1 = build_fusion_epn_cv_spec(chosen_n=7, smoke=False)
    assert spec1.experiment_id == "E5_fusion_r1_p00_ep7"
    assert spec1.model == "fusion_r1_mobilenetv3_s"
    assert spec1.config == "configs/fusion_regression_only_neck.yaml"
    assert spec1.group_mode == "spatial_cluster"
    assert spec1.folds == (0, 1, 2, 3, 4)
    assert spec1.epochs == 7
    assert abs(spec1.lr - 1e-3) < 1e-12
    assert spec1.modality_dropout == 0.0

    spec3 = build_fusion_final_spec(e_final=7, smoke=False)
    assert spec3.is_final and spec3.final_name == "F_FUSION_R1"
    assert spec3.model == "fusion_r1_mobilenetv3_s"
    assert spec3.config == "configs/fusion_regression_only_neck.yaml"
    assert spec3.modality_dropout == 0.0
    assert spec3.epochs == 7

    fusion_csv = tmp_path / "fusion_cv_results.csv"
    # Case A: 5-fold mean MAE = 46.20 < 46.95 -> E_final = 7
    rows_below = [
        {
            "experiment_id": "E5_fusion_r1_p00_ep7",
            "group_mode": "spatial_cluster",
            "fold": str(k),
            "eval_split": "val",
            "mae": 46.20,
        }
        for k in range(5)
    ]
    pd.DataFrame(rows_below).to_csv(fusion_csv, index=False)
    mean_mae_a, _, _ = compute_item1_last_epoch_mean_mae(spec1, fusion_csv)
    e_final_a, _ = evaluate_decision_rule(
        mean_mae=mean_mae_a,
        chosen_n=7,
        threshold=DECISION_THRESHOLD_MAE,
        fallback_epochs=FALLBACK_FINAL_EPOCHS,
    )
    assert np.isclose(mean_mae_a, 46.20)
    assert e_final_a == 7

    # Case B: 5-fold mean MAE = 47.10 >= 46.95 -> E_final = 30
    rows_above = [
        {
            "experiment_id": "E5_fusion_r1_p00_ep7",
            "group_mode": "spatial_cluster",
            "fold": str(k),
            "eval_split": "val",
            "mae": 47.10,
        }
        for k in range(5)
    ]
    pd.DataFrame(rows_above).to_csv(fusion_csv, index=False)
    mean_mae_b, _, _ = compute_item1_last_epoch_mean_mae(spec1, fusion_csv)
    e_final_b, _ = evaluate_decision_rule(
        mean_mae=mean_mae_b,
        chosen_n=7,
        threshold=DECISION_THRESHOLD_MAE,
        fallback_epochs=FALLBACK_FINAL_EPOCHS,
    )
    assert np.isclose(mean_mae_b, 47.10)
    assert e_final_b == 30


def test_compare_final_includes_fusion_and_meta_off(tmp_path: Path) -> None:
    """Verify compare_final includes F_FUSION_R1 and F_FUSION_R1_meta_off when present, plus >25km paired differences."""
    final_csv = tmp_path / "final_test_results.csv"
    rows = []
    for name, mae in [("F_E2", 38.0), ("F_PFRML_full", 39.5), ("F_FUSION_R1", 37.2), ("F_FUSION_R1_meta_off", 38.4)]:
        for split, n_samp, n_stat in [("test", 4, 2), ("test_gt25km", 2, 1)]:
            rows.append(
                {
                    "name": name,
                    "model": "fusion_r1_mobilenetv3_s" if "FUSION" in name else "mobilenetv3_s_pretrained",
                    "config": "configs/fusion_regression_only_neck.yaml",
                    "seed": 42,
                    "epochs": 7,
                    "eval_split": split,
                    "n_samples": n_samp,
                    "n_stations": n_stat,
                    "mae": mae,
                    "rmse": mae + 10.0,
                    "plcc": 0.8,
                    "srcc": 0.8,
                    "r2": 0.6,
                    "acc": 0.5,
                    "pm1_acc": 0.9,
                    "macro_f1": 0.4,
                    "tail_mae": mae + 20.0,
                    "tail_f1": 0.5,
                    "mae_ci95_low": mae - 2.0,
                    "mae_ci95_high": mae + 2.0,
                    "notes": "unit_test",
                }
            )
    pd.DataFrame(rows).to_csv(final_csv, index=False)

    preds_dir = tmp_path / "predictions"
    preds_dir.mkdir()
    for idx, name in enumerate(["F_E2", "F_PFRML_full", "F_FUSION_R1", "F_FUSION_R1_meta_off"]):
        pd.DataFrame(
            {
                "image_id": ["i0", "i1", "i2", "i3"],
                "station_id": ["s0", "s0", "s1", "s1"],
                "y_true": [50.0, 60.0, 100.0, 110.0],
                "y_pred": [52.0 + idx, 62.0 + idx, 98.0 - idx, 108.0 - idx],
            }
        ).to_csv(preds_dir / f"{name}.csv", index=False)

    out_md = tmp_path / "final_comparison.md"
    md = generate_final_comparison(
        final_csv=final_csv,
        predictions_dir=preds_dir,
        outputs_dir=tmp_path,
        baselines_csv=tmp_path / "no_baselines.csv",
        output_md=out_md,
        n_boot=50,
        seed=42,
    )
    assert "`F_FUSION_R1`" in md
    assert "`F_FUSION_R1_meta_off`" in md

    # Also verify real final_comparison.md has Section 3 (>25 km paired bootstrap differences)
    real_fc = _local_or_tracked("outputs/results/final_comparison.md", "results/final_comparison.md").read_text(encoding="utf-8")
    assert "## 3. Paired Station-Bootstrap MAE Differences — `test_gt25km` Subset" in real_fc


def test_build_report_numbers_has_all_8_sections(tmp_path: Path) -> None:
    """Verify build_report_numbers produces all 8 required sections from CSVs/curves/preds."""
    out_md = tmp_path / "REPORT_NUMBERS.md"
    text = build_report_numbers(
        outputs_dir=Path("outputs"),
        output_md=out_md,
        n_boot=20,
        seed=42,
    )
    assert out_md.exists()
    for sec in [
        "## 1. Provenance",
        "## 2. Baselines",
        "## 3. Image-Only Models — 5-Fold `station` CV (LAST-Epoch Reporting)",
        "## 4. PFRML Ablation Ladder — 5-Fold `station` CV",
        "## 5. Multimodal Fusion — 5-Fold `spatial_cluster` CV",
        "## 6. Final One-Shot Test Results (`test` and `test_gt25km` `>25 km`)",
        "## 7. Failed or Unstable Runs",
        "## 8. \"Do Not Quote\" List",
    ]:
        assert sec in text
