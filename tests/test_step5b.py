"""Unit tests for Step 5b: exact curve parsing, queue step 5 specs, final_train_eval guard/bootstrap, and compare_final."""

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.compare_final import (
    generate_final_comparison,
    paired_station_bootstrap_mae_diff,
)
from scripts.final_train_eval import (
    append_final_row,
    name_exists_in_final_csv,
    station_bootstrap_mae_ci,
)
from scripts.run_queue_step5 import build_step5_specs
from scripts.summarize_last_epoch_v2 import discover_curves, parse_curve_filename


def test_parse_curve_filename_exact_experiment_id_and_group_mode(tmp_path: Path) -> None:
    """Verify exact curve filename parsing separates prefix collisions like E2_resnet18_pretrained vs _lr3e-4_ep40."""
    assert parse_curve_filename("E2_resnet18_pretrained_station_fold0.csv") == (
        "E2_resnet18_pretrained",
        "station",
        0,
    )
    assert parse_curve_filename("E2_resnet18_pretrained_lr3e-4_ep40_station_fold0.csv") == (
        "E2_resnet18_pretrained_lr3e-4_ep40",
        "station",
        0,
    )
    assert parse_curve_filename("E2_mobilenetv3_s_pretrained_spatial_spatial_cluster_fold4.csv") == (
        "E2_mobilenetv3_s_pretrained_spatial",
        "spatial_cluster",
        4,
    )
    assert parse_curve_filename("invalid_group_mode_other_fold0.csv") is None

    (tmp_path / "E2_resnet18_pretrained_station_fold0.csv").write_text("epoch,mae\n30,40.28\n")
    for k in range(5):
        (tmp_path / f"E2_resnet18_pretrained_lr3e-4_ep40_station_fold{k}.csv").write_text("epoch,mae\n40,35.0\n")

    disc = discover_curves(tmp_path)
    assert list(disc[("E2_resnet18_pretrained", "station")].keys()) == [0]
    assert list(disc[("E2_resnet18_pretrained_lr3e-4_ep40", "station")].keys()) == [0, 1, 2, 3, 4]


def test_step5_queue_specs_order_and_parameters() -> None:
    """Verify the 7 Step 5 queue items match the exact prompt specification."""
    specs = build_step5_specs(smoke=False)
    assert len(specs) == 7

    assert specs[0].experiment_id == "E5_fusion_r1_p03"
    assert specs[0].model == "fusion_r1_mobilenetv3_s"
    assert specs[0].config == "configs/fusion_r1.yaml"
    assert specs[0].group_mode == "spatial_cluster"
    assert specs[0].folds == (0, 1, 2, 3, 4)
    assert specs[0].modality_dropout == 0.3

    assert specs[1].experiment_id == "E5_fusion_full_p03"
    assert specs[1].model == "fusion_full_mobilenetv3_s"
    assert specs[1].config == "configs/fusion_full.yaml"
    assert specs[1].group_mode == "spatial_cluster"
    assert specs[1].modality_dropout == 0.3

    assert specs[2].experiment_id == "E5_fusion_r1_p00"
    assert specs[2].model == "fusion_r1_mobilenetv3_s"
    assert specs[2].config == "configs/fusion_r1.yaml"
    assert specs[2].group_mode == "spatial_cluster"
    assert specs[2].modality_dropout == 0.0

    assert specs[3].experiment_id == "E5_mobilenetv3_s_reg_chi"
    assert specs[3].model == "pfrml_mobilenetv3_s"
    assert specs[3].config == "configs/pfrml_reg_chi.yaml"
    assert specs[3].group_mode == "station"
    assert specs[3].folds == (0, 1, 2, 3, 4)

    assert specs[4].is_final and specs[4].final_name == "F_E2"
    assert specs[4].model == "mobilenetv3_s_pretrained"
    assert specs[4].config == "configs/default.yaml"

    assert specs[5].is_final and specs[5].final_name == "F_PFRML_full"
    assert specs[5].model == "pfrml_mobilenetv3_s"
    assert specs[5].config == "configs/pfrml_full.yaml"

    assert specs[6].experiment_id == "E5_mdnet_t_pfrml_lr2e-4_f0"
    assert specs[6].model == "pfrml_mdnet_t"
    assert specs[6].config == "configs/pfrml_baseline.yaml"
    assert specs[6].group_mode == "station"
    assert specs[6].folds == (0,)
    assert abs(specs[6].lr - 2e-4) < 1e-12
    assert specs[6].epochs == 30
    assert specs[6].is_guarded_item7 is True


def test_station_bootstrap_and_compare_final(tmp_path: Path) -> None:
    """Verify station bootstrap CIs, one-shot name check, and compare_final report generation."""
    y_true = np.array([50.0, 60.0, 100.0, 120.0, 200.0, 210.0])
    y_pred_a = np.array([55.0, 65.0, 110.0, 130.0, 190.0, 200.0])
    y_pred_b = np.array([70.0, 80.0, 130.0, 150.0, 240.0, 250.0])
    stations = ["s1", "s1", "s2", "s2", "s3", "s3"]

    ci_l, ci_h = station_bootstrap_mae_ci(y_true, y_pred_a, stations, n_boot=200, seed=42)
    assert 5.0 <= ci_l <= ci_h <= 10.0

    diff_ab, d_low, d_high = paired_station_bootstrap_mae_diff(
        y_true, y_pred_a, y_pred_b, stations, n_boot=200, seed=42
    )
    assert diff_ab < 0.0 and d_high < 0.0 and d_low <= diff_ab <= d_high

    final_csv = tmp_path / "final_results.csv"
    assert not name_exists_in_final_csv(final_csv, "F_E2")
    append_final_row(
        final_csv,
        {
            "name": "F_E2",
            "model": "mobilenetv3_s_pretrained",
            "config": "configs/default.yaml",
            "seed": 42,
            "epochs": 30,
            "eval_split": "test",
            "n_samples": 6,
            "n_stations": 3,
            "mae": 8.33,
            "rmse": 8.66,
            "plcc": 0.99,
            "srcc": 0.99,
            "r2": 0.95,
            "acc": 0.8,
            "pm1_acc": 1.0,
            "macro_f1": 0.8,
            "tail_mae": 10.0,
            "tail_f1": 0.9,
            "mae_ci95_low": ci_l,
            "mae_ci95_high": ci_h,
            "notes": "smoke",
        },
    )
    assert name_exists_in_final_csv(final_csv, "F_E2")

    preds_dir = tmp_path / "predictions"
    preds_dir.mkdir()
    pd.DataFrame(
        {"image_id": [f"img{i}" for i in range(6)], "station_id": stations, "y_true": y_true, "y_pred": y_pred_a}
    ).to_csv(preds_dir / "F_E2.csv", index=False)
    pd.DataFrame(
        {"image_id": [f"img{i}" for i in range(6)], "station_id": stations, "y_true": y_true, "y_pred": y_pred_b}
    ).to_csv(preds_dir / "F_PFRML_full.csv", index=False)

    out_md = tmp_path / "final_comparison.md"
    md_text = generate_final_comparison(
        final_csv=final_csv,
        predictions_dir=preds_dir,
        outputs_dir=tmp_path,
        step2_csv=tmp_path / "missing_step2.csv",
        output_md=out_md,
        n_boot=100,
        seed=42,
    )
    assert out_md.exists()
    assert "`F_E2`" in md_text and "`F_PFRML_full`" in md_text
