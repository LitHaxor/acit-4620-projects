"""Unit tests for E3b (MobileNetV3MDNeck), atomic CSV appends, resume, and queue selection."""

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from evaluate import CSV_FIELDNAMES, append_result_row, build_result_row
from scripts.run_queue import (
    EARLIER_DEFAULTS,
    build_stage1_specs,
    index_val_rows,
    project_stage3_hours,
    run_queue,
    select_stage2_configs,
)
from src.config import load_config
from src.metrics import METRIC_ORDER
from src.models.mdnet import MobileNetV3MDNeck, build_mobilenet_v3_small_md_neck
from train import _train_predict_image_model, build_image_model


def test_e3b_mobilenetv3_md_neck_shapes_grads_and_docstring() -> None:
    doc = (MobileNetV3MDNeck.__doc__ or "") + "\n" + (build_mobilenet_v3_small_md_neck.__doc__ or "")
    assert "Paper 2" in doc
    assert "NOT part of Paper 2" in doc

    model = build_mobilenet_v3_small_md_neck(head_type="regression", pretrained=False)
    assert len(model.md_neck) == 2
    assert model.neck_channels == 128

    x = torch.randn(2, 3, 224, 224)
    feat = model.forward_features(x)
    assert feat.shape == (2, 128)
    out = model(x)
    assert out.shape == (2,)

    out.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    cfg = load_config("configs/default.yaml")
    reg_model, head_type = build_image_model("mobilenetv3_s_md_neck", cfg)
    assert isinstance(reg_model, MobileNetV3MDNeck)
    assert head_type == "regression"


def test_atomic_append_result_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    metrics = {k: float(i) for i, k in enumerate(METRIC_ORDER)}
    row1 = build_result_row(
        experiment_id="exp1",
        model="mobilenetv3_s_md_neck",
        seed=42,
        group_mode="station",
        eval_split="val",
        fold="0",
        n_samples=100,
        metrics=metrics,
        notes="sec_per_ep=1.23",
    )
    row2 = build_result_row(
        experiment_id="exp1",
        model="mobilenetv3_s_md_neck",
        seed=42,
        group_mode="station",
        eval_split="val",
        fold="1",
        n_samples=100,
        metrics=metrics,
        notes="sec_per_ep=1.25",
    )
    append_result_row(csv_path, row1)
    append_result_row(csv_path, row2)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == CSV_FIELDNAMES
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["fold"] == "0"
    assert rows[1]["fold"] == "1"


def test_stage2_selection_rule_threshold_and_e3b() -> None:
    specs = build_stage1_specs(smoke=False)
    rows = [
        # Defaults (lr=1e-3, 30ep)
        {
            "experiment_id": EARLIER_DEFAULTS["mobilenetv3_s_pretrained"].experiment_id,
            "model": "mobilenetv3_s_pretrained",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "35.72",
        },
        {
            "experiment_id": EARLIER_DEFAULTS["resnet18_pretrained"].experiment_id,
            "model": "resnet18_pretrained",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "40.28",
        },
        {
            "experiment_id": EARLIER_DEFAULTS["resnet18_scratch"].experiment_id,
            "model": "resnet18_scratch",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "48.26",
        },
        {
            "experiment_id": EARLIER_DEFAULTS["mdnet_t_reg"].experiment_id,
            "model": "mdnet_t_reg",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "51.33",
        },
        # Stage 1 candidates:
        # mobilenetv3_s_pretrained improves by 0.99 (< 1.0) -> MUST keep default
        {
            "experiment_id": "E2_mobilenetv3_s_pretrained_lr3e-4_ep40",
            "model": "mobilenetv3_s_pretrained",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "34.73",
        },
        # resnet18_pretrained improves by 1.00 (>= 1.0) -> MUST switch
        {
            "experiment_id": "E2_resnet18_pretrained_lr3e-4_ep40",
            "model": "resnet18_pretrained",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "39.28",
        },
        # resnet18_scratch crashed (missing) -> MUST fall back to default
        # mdnet_t_reg improves by 3.33 -> MUST switch
        {
            "experiment_id": "E3_mdnet_t_reg_lr1e-3_ep60",
            "model": "mdnet_t_reg",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "48.00",
        },
        # E3b: no earlier default; picks lower of 3e-4 (36.1) vs 1e-3 (34.9)
        {
            "experiment_id": "E3b_mobilenetv3_s_md_neck_lr3e-4_ep40",
            "model": "mobilenetv3_s_md_neck",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "36.10",
        },
        {
            "experiment_id": "E3b_mobilenetv3_s_md_neck_lr1e-3_ep40",
            "model": "mobilenetv3_s_md_neck",
            "group_mode": "station",
            "eval_split": "val",
            "fold": "0",
            "mae": "34.90",
        },
    ]

    indexed = index_val_rows(rows)
    decisions: Dict[str, str] = {
        d.family: d.selected.experiment_id for d in select_stage2_configs(specs, indexed)
    }
    assert decisions["mobilenetv3_s_pretrained"] == "E2_mobilenetv3_s_pretrained"
    assert decisions["resnet18_pretrained"] == "E2_resnet18_pretrained_lr3e-4_ep40"
    assert decisions["resnet18_scratch"] == "E2_resnet18_scratch"
    assert decisions["mdnet_t_reg"] == "E3_mdnet_t_reg_lr1e-3_ep60"
    assert decisions["mobilenetv3_s_md_neck"] == "E3b_mobilenetv3_s_md_neck_lr1e-3_ep40"


def test_queue_12h_guard_writes_stop_file(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    csv_path = tmp_path / "results.csv"

    # Populate Stage 1 + defaults with high sec_per_ep so Stage 3 exceeds 12h
    specs = build_stage1_specs(smoke=False)
    metrics = {k: 40.0 for k in METRIC_ORDER}
    for spec in [*EARLIER_DEFAULTS.values(), *specs]:
        row = build_result_row(
            experiment_id=spec.experiment_id,
            model=spec.model,
            seed=42,
            group_mode="station",
            eval_split="val",
            fold="0",
            n_samples=1654,
            metrics=metrics,
            notes="sec_per_ep=500.0",
        )
        append_result_row(csv_path, row)

    rc = run_queue(smoke=False, outputs_dir=outputs_dir, results_csv=csv_path, max_hours=12.0)
    assert rc == 0
    stop_file = outputs_dir / "QUEUE_STOPPED.txt"
    assert stop_file.exists()
    content = stop_file.read_text(encoding="utf-8")
    assert "Projected Stage 3 total hours" in content
    log_file = outputs_dir / "logs" / "queue.log"
    assert log_file.exists()
    assert "STOPPED: Projected hours" in log_file.read_text(encoding="utf-8")


def test_per_epoch_checkpoint_and_resume(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    records = []
    for i in range(4):
        fname = f"img_{i}.jpg"
        arr = np.full((32, 32, 3), 50 * (i + 1), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)
        records.append(
            {
                "filename": fname,
                "pm25": float(40 * (i + 1)),
                "aqi_bin": min(i, 5),
                "latitude": 39.9,
                "longitude": 116.4,
                "captured_at": "2024-06-01T12:00:00Z",
                "station_id": i,
            }
        )
    df = pd.DataFrame(records)
    cfg = load_config("configs/default.yaml")
    cfg.outputs_dir = str(tmp_path / "out")
    cfg.data.num_workers = 0
    cfg.train.batch_size = 2
    cfg.train.epochs = 1
    cfg.train.warmup_epochs = 0

    ckpt_dir = tmp_path / "out" / "checkpoints" / "test_resume" / "station" / "fold_0"
    _train_predict_image_model(
        model_name="resnet18_scratch",
        experiment_id="test_resume",
        group_mode="station",
        fold_label="0",
        train_df=df.iloc[:2],
        eval_df=df.iloc[2:],
        train_images_dir=img_dir,
        eval_images_dir=img_dir,
        cfg=cfg,
        seed=42,
        device=torch.device("cpu"),
        ckpt_dir=ckpt_dir,
        resume=False,
    )
    last_pt = ckpt_dir / "last.pt"
    assert last_pt.exists()

    # Simulate interruption at epoch 1 of a 2-epoch run
    ckpt_data = torch.load(last_pt, map_location="cpu", weights_only=False)
    ckpt_data["epochs"] = 2
    torch.save(ckpt_data, last_pt)

    cfg.train.epochs = 2
    _train_predict_image_model(
        model_name="resnet18_scratch",
        experiment_id="test_resume",
        group_mode="station",
        fold_label="0",
        train_df=df.iloc[:2],
        eval_df=df.iloc[2:],
        train_images_dir=img_dir,
        eval_images_dir=img_dir,
        cfg=cfg,
        seed=42,
        device=torch.device("cpu"),
        ckpt_dir=ckpt_dir,
        resume=True,
    )
    curve_path = tmp_path / "out" / "curves" / "test_resume_station_fold0.csv"
    with curve_path.open("r", newline="", encoding="utf-8") as f:
        curve_rows = list(csv.DictReader(f))
    assert [int(r["epoch"]) for r in curve_rows] == [1, 2]
