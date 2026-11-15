#!/usr/bin/env python3
"""Resumable Step 5 queue runner (`scripts/run_queue_step5.py`).

Design mirrors `scripts/run_queue_step4.py`:
- One isolated subprocess per `(experiment, fold)` (or per final full-train run).
- Skips rows already present in `outputs/results/step5_results.csv` (and
  `outputs/results/final_results.csv` for items 5 & 6).
- Crashes are logged and the queue continues.
- Logs exclusively to `outputs/logs/queue_step5.log` (or `outputs/smoke/logs/queue_step5.log`
  in `--smoke` mode).
- Atomic CSV appends via `evaluate.append_result_row` and `cv_mean`/`cv_std` summary rows
  (including `<experiment_id>_meta_off` summary rows for fusion runs).
- Every command passes explicit `--model`, `--config`, `--group-mode`, `--epochs`, `--lr`
  (and `--modality-dropout` where relevant), using absolute paths and `--resume`.
- `--pilot`: runs item 1 (`E5_fusion_r1_p03`) fold 0 only and prints MAE for both the
  normal row (`E5_fusion_r1_p03`) and the metadata-off row (`E5_fusion_r1_p03_meta_off`).
- `--smoke`: runs 1 epoch on 64 samples under `outputs/smoke/`, passing `--smoke-eval-on-val`
  to `scripts/final_train_eval.py` so smoke runs never touch the test set.
- Time guard before item 7 (`E5_mdnet_t_pfrml_lr2e-4_f0`): projects total queue hours from
  measured `sec/epoch` (timing ~20 real training batches per model type when not already
  recorded); if total projected queue time exceeds `8.0 h`, skips item 7 and writes
  `outputs/QUEUE_STOPPED_STEP5.txt` explaining why.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import append_result_row, build_result_row, resolve_device  # noqa: E402
from scripts.final_train_eval import name_exists_in_final_csv  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.cleaning import dedupe_split_by_median_aqi  # noqa: E402
from src.data.dataset import PM25Dataset  # noqa: E402
from src.data.transforms import get_train_transforms  # noqa: E402
from src.metrics import METRIC_ORDER, summarize_fold_metrics  # noqa: E402
from src.models.dcm import PFRMLModel  # noqa: E402
from src.models.fusion import FUSION_MODEL_SPECS, PFRMLFusionModel  # noqa: E402
from train import build_fold_loss, build_image_model, sync_device  # noqa: E402


EVAL_SPLIT = "val"
DEFAULT_MAX_HOURS = 8.0
BATCHES_PER_FOLD_BS32 = 205       # ~6,541 train images / 32
BATCHES_FULL_TRAIN_BS32 = 256     # 8,176 train images / 32


@dataclass(frozen=True)
class Step5RunSpec:
    """Specification for an item in the Step 5 queue."""

    order_idx: int
    rung: str
    experiment_id: str
    model: str
    config: str
    group_mode: str = "station"
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    lr: float = 1e-3
    epochs: int = 30
    modality_dropout: Optional[float] = None
    is_final: bool = False
    final_name: Optional[str] = None
    is_guarded_item7: bool = False

    @property
    def is_fusion(self) -> bool:
        return self.model in FUSION_MODEL_SPECS


def build_step5_specs(smoke: bool = False) -> List[Step5RunSpec]:
    """Return the 7 Step 5 queue items in exact specified order."""

    def ep(n: int) -> int:\
        return 1 if smoke else n

    cv_folds: Tuple[int, ...] = (0, 1) if smoke else (0, 1, 2, 3, 4)

    return [
        Step5RunSpec(
            order_idx=1,
            rung="1_E5_fusion_r1_p03",
            experiment_id="E5_fusion_r1_p03",
            model="fusion_r1_mobilenetv3_s",
            config="configs/fusion_r1.yaml",
            group_mode="spatial_cluster",
            folds=cv_folds,
            lr=1e-3,
            epochs=ep(30),
            modality_dropout=0.3,
        ),
        Step5RunSpec(
            order_idx=2,
            rung="2_E5_fusion_full_p03",
            experiment_id="E5_fusion_full_p03",
            model="fusion_full_mobilenetv3_s",
            config="configs/fusion_full.yaml",
            group_mode="spatial_cluster",
            folds=cv_folds,
            lr=1e-3,
            epochs=ep(30),
            modality_dropout=0.3,
        ),
        Step5RunSpec(
            order_idx=3,
            rung="3_E5_fusion_r1_p00",
            experiment_id="E5_fusion_r1_p00",
            model="fusion_r1_mobilenetv3_s",
            config="configs/fusion_r1.yaml",
            group_mode="spatial_cluster",
            folds=cv_folds,
            lr=1e-3,
            epochs=ep(30),
            modality_dropout=0.0,
        ),
        Step5RunSpec(
            order_idx=4,
            rung="4_E5_mobilenetv3_s_reg_chi",
            experiment_id="E5_mobilenetv3_s_reg_chi",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_reg_chi.yaml",
            group_mode="station",
            folds=cv_folds,
            lr=1e-3,
            epochs=ep(30),
            modality_dropout=None,
        ),
        Step5RunSpec(
            order_idx=5,
            rung="5_FINAL_E2",
            experiment_id="F_E2",
            model="mobilenetv3_s_pretrained",
            config="configs/default.yaml",
            group_mode="full_train",
            folds=(),
            lr=1e-3,
            epochs=ep(30),
            is_final=True,
            final_name="F_E2",
        ),
        Step5RunSpec(
            order_idx=6,
            rung="6_FINAL_PFRML",
            experiment_id="F_PFRML_full",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_full.yaml",
            group_mode="full_train",
            folds=(),
            lr=1e-3,
            epochs=ep(30),
            is_final=True,
            final_name="F_PFRML_full",
        ),
        Step5RunSpec(
            order_idx=7,
            rung="7_E5_mdnet_t_pfrml_lr2e-4_f0",
            experiment_id="E5_mdnet_t_pfrml_lr2e-4_f0",
            model="pfrml_mdnet_t",
            config="configs/pfrml_baseline.yaml",
            group_mode="station",
            folds=(0,),
            lr=2e-4,
            epochs=ep(30),
            is_guarded_item7=True,
        ),
    ]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_f: TextIO, msg: str) -> None:
    log_f.write(f"[{_ts()}] {msg}\n")
    log_f.flush()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(text)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_results_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_val_rows(
    rows: Sequence[Dict[str, str]],
    eval_split: str = EVAL_SPLIT,
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Index CSV rows by `(experiment_id, group_mode, fold)` for `eval_split='val'`."""
    indexed: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in rows:
        if r.get("eval_split") == eval_split:
            exp_id = str(r.get("experiment_id", ""))
            g_mode = str(r.get("group_mode", ""))
            fold = str(r.get("fold", ""))
            if exp_id and g_mode and fold:
                indexed[(exp_id, g_mode, fold)] = r
    return indexed


def _parse_float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(val)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def extract_sec_per_epoch_from_row(row: Dict[str, str]) -> Optional[float]:
    m = re.search(r"sec_per_ep=([0-9]+(?:\.[0-9]+)?)", row.get("notes", "") or "")
    if m:
        return float(m.group(1))
    return None


def extract_epochs_from_row(row: Dict[str, str], default_epochs: int) -> int:
    m = re.search(r"(?:last_ep|best_ep)=[0-9]+/([0-9]+)", row.get("notes", "") or "")
    if m:
        return int(m.group(1))
    return default_epochs


def benchmark_real_batches_sec_per_epoch(
    model_name: str,
    config_path: str,
    n_batches: int = 20,
    batch_size: int = 32,
    is_full_train: bool = False,
) -> float:
    """Time `n_batches` real training batches from `datasets/train/` and extrapolate to an epoch."""
    import pandas as pd

    cfg = load_config(PROJECT_ROOT / config_path, overrides={"model.name": model_name})
    device = resolve_device(cfg.train.device)

    raw_train = pd.read_csv(PROJECT_ROOT / cfg.data.train_csv)
    clean_train = dedupe_split_by_median_aqi(raw_train)
    n_needed = max((n_batches + 2) * batch_size, 128)
    sub_df = clean_train.iloc[: min(len(clean_train), n_needed)].reset_index(drop=True)

    tf = get_train_transforms(
        image_size=cfg.data.image_size,
        horizontal_flip_prob=cfg.data.horizontal_flip_prob,
        rotation_deg=cfg.data.rotation_deg,
        translate_frac=cfg.data.translate_frac,
        scale_range=cfg.data.scale_range,
    )
    ds = PM25Dataset(
        sub_df,
        images_dir=PROJECT_ROOT / cfg.data.train_images_dir,
        transform=tf,
        aqi_scale=float(cfg.data.aqi_scale),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model, head_type = build_image_model(model_name, cfg)
    model = model.to(device)
    model.train()

    use_protos_flag = (
        bool(model.head.use_prototypes)
        if isinstance(model, (PFRMLModel, PFRMLFusionModel))
        else False
    )
    is_fusion = model_name in FUSION_MODEL_SPECS
    aux_w = float(cfg.pfrml.aux_loss_weight) if is_fusion else 0.0
    loss_fn = build_fold_loss(
        head_type,
        cfg,
        train_bins=sub_df["aqi_bin"].to_numpy(dtype=int),
        use_prototypes_override=use_protos_flag,
        aux_loss_weight=aux_w,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    it = iter(loader)
    # 2 warmup steps
    for _ in range(2):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        optimizer.zero_grad(set_to_none=True)
        images = batch.image.to(device)
        y_aqi = batch.aqi.to(device)
        y_bin = batch.aqi_bin.to(device)
        if head_type == "multitask":
            if isinstance(model, PFRMLFusionModel):
                out = model(images, metadata=batch.metadata.to(device), targets=y_bin)
            else:
                out = model(images, targets=y_bin)
            loss, _ = loss_fn(out, y_bin, y_aqi)
        else:
            out = model(images)
            loss = loss_fn(out.float(), y_aqi)
        loss.backward()
        optimizer.step()
    sync_device(device)

    t0 = time.perf_counter()
    measured = 0
    while measured < n_batches:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        optimizer.zero_grad(set_to_none=True)
        images = batch.image.to(device)
        y_aqi = batch.aqi.to(device)
        y_bin = batch.aqi_bin.to(device)
        if head_type == "multitask":
            if isinstance(model, PFRMLFusionModel):
                out = model(images, metadata=batch.metadata.to(device), targets=y_bin)
            else:
                out = model(images, targets=y_bin)
            loss, _ = loss_fn(out, y_bin, y_aqi)
        else:
            out = model(images)
            loss = loss_fn(out.float(), y_aqi)
        loss.backward()
        optimizer.step()
        measured += 1
    sync_device(device)
    elapsed = time.perf_counter() - t0
    sec_per_batch = elapsed / max(measured, 1)

    if is_full_train:
        return sec_per_batch * BATCHES_FULL_TRAIN_BS32
    return sec_per_batch * (BATCHES_PER_FOLD_BS32 + 52 * 0.35)


def estimate_spec_sec_per_epoch(
    spec: Step5RunSpec,
    step5_rows: Sequence[Dict[str, str]],
    final_rows: Sequence[Dict[str, str]],
    step4_rows: Sequence[Dict[str, str]],
    step3_rows: Sequence[Dict[str, str]],
    outputs_dir: Path,
    bench_cache: Dict[Tuple[str, str, bool], float],
    smoke: bool = False,
) -> float:
    """Estimate `sec/epoch` for `spec` from recorded CSV rows or 20 real training batches."""
    if smoke:
        return 1.0

    if spec.is_final:
        for r in final_rows:
            if r.get("name") == spec.final_name:
                t = extract_sec_per_epoch_from_row(r)
                if t is not None:
                    return t

    exact_times = [
        t
        for r in step5_rows
        if r.get("experiment_id") == spec.experiment_id
        and r.get("fold") not in {"cv_mean", "cv_std"}
        and (t := extract_sec_per_epoch_from_row(r)) is not None
    ]
    if exact_times:
        return sum(exact_times) / len(exact_times)

    curve_path = outputs_dir / "curves" / f"{spec.experiment_id}_{spec.group_mode}_fold0.csv"
    if curve_path.exists():
        with curve_path.open("r", newline="", encoding="utf-8") as cf:
            c_times = [
                t
                for cr in csv.DictReader(cf)
                if (t := _parse_float(cr.get("epoch_time_s"))) is not None
            ]
            if c_times:
                return sum(c_times) / len(c_times)

    # Check same model in step5_rows, step4_rows, step3_rows
    for rows in (step5_rows, step4_rows, step3_rows):
        m_times = [
            t
            for r in rows
            if r.get("model") == spec.model
            and r.get("fold") not in {"cv_mean", "cv_std"}
            and (t := extract_sec_per_epoch_from_row(r)) is not None
        ]
        if m_times:
            base_t = sum(m_times) / len(m_times)
            if spec.is_final:
                return base_t * (256.0 / (205.0 + 52.0 * 0.35))
            return base_t

    cache_key = (spec.model, spec.config, spec.is_final)
    if cache_key not in bench_cache:
        bench_cache[cache_key] = benchmark_real_batches_sec_per_epoch(
            model_name=spec.model,
            config_path=spec.config,
            n_batches=20,
            batch_size=32,
            is_full_train=spec.is_final,
        )
    return bench_cache[cache_key]


def project_total_queue_hours_step5(
    all_specs: Sequence[Step5RunSpec],
    step5_rows: Sequence[Dict[str, str]],
    final_rows: Sequence[Dict[str, str]],
    step4_rows: Sequence[Dict[str, str]],
    step3_rows: Sequence[Dict[str, str]],
    indexed_val: Dict[Tuple[str, str, str], Dict[str, str]],
    outputs_dir: Path,
    smoke: bool = False,
    force_benchmark: bool = False,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """Return `(total_projected_hours, item7_hours, per_spec_breakdown)`."""
    total_seconds = 0.0
    item7_seconds = 0.0
    breakdown: List[Dict[str, object]] = []
    bench_cache: Dict[Tuple[str, str, bool], float] = {}

    for spec in all_specs:
        if force_benchmark and not smoke:
            cache_key = (spec.model, spec.config, spec.is_final)
            if cache_key not in bench_cache:
                bench_cache[cache_key] = benchmark_real_batches_sec_per_epoch(
                    model_name=spec.model,
                    config_path=spec.config,
                    n_batches=20,
                    batch_size=32,
                    is_full_train=spec.is_final,
                )
            sec_per_ep = bench_cache[cache_key]
        else:
            sec_per_ep = estimate_spec_sec_per_epoch(
                spec=spec,
                step5_rows=step5_rows,
                final_rows=final_rows,
                step4_rows=step4_rows,
                step3_rows=step3_rows,
                outputs_dir=outputs_dir,
                bench_cache=bench_cache,
                smoke=smoke,
            )

        spec_sec = 0.0
        done_units: List[str] = []
        rem_units: List[str] = []

        if spec.is_final:
            done = any(r.get("name") == spec.final_name for r in final_rows)
            if done:
                done_units.append("full_train")
            else:
                rem_units.append("full_train")
            spec_sec = sec_per_ep * spec.epochs
        else:
            for k in spec.folds:
                key = (spec.experiment_id, spec.group_mode, str(k))
                if key in indexed_val:
                    row = indexed_val[key]
                    row_sec = extract_sec_per_epoch_from_row(row) or sec_per_ep
                    row_ep = extract_epochs_from_row(row, spec.epochs)
                    spec_sec += row_sec * row_ep
                    done_units.append(str(k))
                else:
                    spec_sec += sec_per_ep * spec.epochs
                    rem_units.append(str(k))

        total_seconds += spec_sec
        if spec.is_guarded_item7:
            item7_seconds = spec_sec

        n_runs = 1 if spec.is_final else len(spec.folds)
        min_per_run = (spec_sec / 60.0) / max(n_runs, 1)
        breakdown.append(
            {
                "order_idx": spec.order_idx,
                "rung": spec.rung,
                "experiment_id": spec.experiment_id,
                "model": spec.model,
                "config": spec.config,
                "group_mode": spec.group_mode,
                "n_runs": n_runs,
                "sec_per_ep": sec_per_ep,
                "epochs": spec.epochs,
                "min_per_run": min_per_run,
                "done_units": done_units,
                "remaining_units": rem_units,
                "projected_minutes": spec_sec / 60.0,
                "projected_hours": spec_sec / 3600.0,
            }
        )

    return total_seconds / 3600.0, item7_seconds / 3600.0, breakdown


def _run_train_subprocess(
    *,
    spec: Step5RunSpec,
    fold: int,
    results_csv: Path,
    outputs_dir: Path,
    smoke: bool,
    log_f: TextIO,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(PROJECT_ROOT / spec.config),
        "--experiment-id",
        spec.experiment_id,
        "--model",
        spec.model,
        "--group-mode",
        spec.group_mode,
        "--fold",
        str(fold),
        "--epochs",
        str(spec.epochs),
        "--lr",
        str(spec.lr),
        "--output-csv",
        str(results_csv),
        "--outputs-dir",
        str(outputs_dir),
        "--resume",
    ]
    if spec.modality_dropout is not None:
        cmd.extend(["--modality-dropout", str(spec.modality_dropout)])
    if smoke:
        cmd.extend(["--max-samples", "64", "--num-workers", "0"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    mod_str = f" modality_dropout={spec.modality_dropout}" if spec.modality_dropout is not None else ""
    log_line(
        log_f,
        f"START rung={spec.rung} exp={spec.experiment_id} model={spec.model} "
        f"group_mode={spec.group_mode} fold={fold} lr={spec.lr:g} epochs={spec.epochs}{mod_str}",
    )
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log_f.flush()
    if proc.returncode == 0:
        log_line(
            log_f,
            f"DONE  rung={spec.rung} exp={spec.experiment_id} fold={fold} (rc=0)",
        )
    else:
        log_line(
            log_f,
            f"CRASH rung={spec.rung} exp={spec.experiment_id} fold={fold} "
            f"(rc={proc.returncode}) — continuing queue",
        )
    return int(proc.returncode)


def _run_final_subprocess(
    *,
    spec: Step5RunSpec,
    final_csv: Path,
    outputs_dir: Path,
    smoke: bool,
    log_f: TextIO,
) -> int:
    assert spec.final_name is not None
    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "final_train_eval.py"),
        "--name",
        spec.final_name,
        "--model",
        spec.model,
        "--config",
        str(PROJECT_ROOT / spec.config),
        "--epochs",
        str(spec.epochs),
        "--seed",
        "42",
        "--lr",
        str(spec.lr),
        "--outputs-dir",
        str(outputs_dir),
        "--output-csv",
        str(final_csv),
        "--resume",
    ]
    if spec.modality_dropout is not None:
        cmd.extend(["--modality-dropout", str(spec.modality_dropout)])
    if smoke:
        cmd.extend(["--max-samples", "64", "--num-workers", "0", "--smoke-eval-on-val"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_line(
        log_f,
        f"START FINAL rung={spec.rung} name={spec.final_name} model={spec.model} "
        f"config={spec.config} lr={spec.lr:g} epochs={spec.epochs}",
    )
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log_f.flush()
    if proc.returncode == 0:
        log_line(
            log_f,
            f"DONE  FINAL rung={spec.rung} name={spec.final_name} (rc=0)",
        )
    else:
        log_line(
            log_f,
            f"CRASH FINAL rung={spec.rung} name={spec.final_name} "
            f"(rc={proc.returncode}) — continuing queue",
        )
    return int(proc.returncode)


def _maybe_append_cv_summary_for_exp(
    exp_id: str,
    model_name: str,
    group_mode: str,
    rung: str,
    cv_folds: Sequence[int],
    results_csv: Path,
    log_f: TextIO,
) -> None:
    if len(cv_folds) < 2:
        return
    rows = load_results_rows(results_csv)
    indexed = index_val_rows(rows)
    mean_key = (exp_id, group_mode, "cv_mean")
    std_key = (exp_id, group_mode, "cv_std")
    if mean_key in indexed and std_key in indexed:
        return

    fold_metrics: List[Dict[str, float]] = []
    seed_val = 42
    for k in cv_folds:
        row = indexed.get((exp_id, group_mode, str(k)))
        if row is None:
            return
        try:
            seed_val = int(row.get("seed", 42))
        except ValueError:
            seed_val = 42
        m: Dict[str, float] = {}
        for metric_name in METRIC_ORDER:
            v = _parse_float(row.get(metric_name))
            m[metric_name] = float("nan") if v is None else v
        fold_metrics.append(m)

    summary = summarize_fold_metrics(fold_metrics)
    mean_dict = {k: summary[k][0] for k in METRIC_ORDER}
    std_dict = {k: summary[k][1] for k in METRIC_ORDER}
    total_n = sum(
        int(float(indexed[(exp_id, group_mode, str(k))].get("n_samples", 0) or 0))
        for k in cv_folds
    )
    notes_str = f"folds={','.join(str(k) for k in cv_folds)}"

    if mean_key not in indexed:
        append_result_row(
            results_csv,
            build_result_row(
                experiment_id=exp_id,
                model=model_name,
                seed=seed_val,
                group_mode=group_mode,
                eval_split=EVAL_SPLIT,
                fold="cv_mean",
                n_samples=total_n,
                metrics=mean_dict,
                notes=notes_str,
            ),
        )
    if std_key not in indexed:
        append_result_row(
            results_csv,
            build_result_row(
                experiment_id=exp_id,
                model=model_name,
                seed=seed_val,
                group_mode=group_mode,
                eval_split=EVAL_SPLIT,
                fold="cv_std",
                n_samples=total_n,
                metrics=std_dict,
                notes=notes_str,
            ),
        )
    log_line(
        log_f,
        f"CV_SUMMARY rung={rung} exp={exp_id} group_mode={group_mode} "
        f"folds={list(cv_folds)} MAE={mean_dict['mae']:.2f}+-{std_dict['mae']:.2f} "
        f"R2={mean_dict['r2']:.4f}+-{std_dict['r2']:.4f}",
    )


def _maybe_append_cv_summary(
    spec: Step5RunSpec,
    results_csv: Path,
    log_f: TextIO,
) -> None:
    if spec.is_final or len(spec.folds) < 2:
        return
    _maybe_append_cv_summary_for_exp(
        exp_id=spec.experiment_id,
        model_name=spec.model,
        group_mode=spec.group_mode,
        rung=spec.rung,
        cv_folds=spec.folds,
        results_csv=results_csv,
        log_f=log_f,
    )
    if spec.is_fusion:
        _maybe_append_cv_summary_for_exp(
            exp_id=f"{spec.experiment_id}_meta_off",
            model_name=spec.model,
            group_mode=spec.group_mode,
            rung=f"{spec.rung}_meta_off",
            cv_folds=spec.folds,
            results_csv=results_csv,
            log_f=log_f,
        )


def run_queue_step5(
    *,
    smoke: bool = False,
    pilot: bool = False,
    outputs_dir: Optional[Path] = None,
    results_csv: Optional[Path] = None,
    final_csv: Optional[Path] = None,
    max_hours: float = DEFAULT_MAX_HOURS,
) -> int:
    if outputs_dir is None:
        outputs_dir = PROJECT_ROOT / ("outputs/smoke" if smoke else "outputs")
    if results_csv is None:
        results_csv = (
            outputs_dir / "step5_results.csv"
            if smoke
            else PROJECT_ROOT / "outputs/results/step5_results.csv"
        )
    if final_csv is None:
        final_csv = (
            outputs_dir / "final_results.csv"
            if smoke
            else PROJECT_ROOT / "outputs/results/final_results.csv"
        )
    log_path = outputs_dir / "logs" / "queue_step5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    specs = build_step5_specs(smoke=smoke)

    step4_csv = PROJECT_ROOT / "outputs/results/step4_results.csv"
    step3_csv = PROJECT_ROOT / "outputs/results/step3_results.csv"
    step4_rows = load_results_rows(step4_csv)
    step3_rows = load_results_rows(step3_csv)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_line(
            log_f,
            f"=== STEP 5 QUEUE START (smoke={smoke}, pilot={pilot}, "
            f"results_csv={results_csv}, final_csv={final_csv}, outputs_dir={outputs_dir}) ===",
        )

        if pilot:
            item1 = specs[0]
            indexed = index_val_rows(load_results_rows(results_csv))
            key_norm = (item1.experiment_id, item1.group_mode, "0")
            key_mo = (f"{item1.experiment_id}_meta_off", item1.group_mode, "0")
            if key_norm not in indexed or key_mo not in indexed:
                _run_train_subprocess(
                    spec=item1,
                    fold=0,
                    results_csv=results_csv,
                    outputs_dir=outputs_dir,
                    smoke=smoke,
                    log_f=log_f,
                )
            else:
                log_line(
                    log_f,
                    f"SKIP  pilot exp={item1.experiment_id} fold=0 (already in CSV)",
                )
            indexed = index_val_rows(load_results_rows(results_csv))
            row_norm = indexed.get(key_norm)
            row_mo = indexed.get(key_mo)
            mae_norm = _parse_float(row_norm.get("mae") if row_norm else None)
            mae_mo = _parse_float(row_mo.get("mae") if row_mo else None)
            norm_str = f"{mae_norm:.2f}" if mae_norm is not None else "N/A"
            mo_str = f"{mae_mo:.2f}" if mae_mo is not None else "N/A"
            msg = (
                f"PILOT {item1.experiment_id} ({item1.group_mode} fold 0) validation MAE: "
                f"normal={norm_str}, {item1.experiment_id}_meta_off={mo_str}"
            )
            log_line(log_f, msg)
            print(msg, flush=True)
            log_line(log_f, "=== STEP 5 PILOT COMPLETE ===")
            return 0

        for spec in specs:
            if spec.is_guarded_item7:
                step5_rows = load_results_rows(results_csv)
                final_rows = load_results_rows(final_csv)
                indexed = index_val_rows(step5_rows)
                rem_folds = [
                    k
                    for k in spec.folds
                    if (spec.experiment_id, spec.group_mode, str(k)) not in indexed
                ]
                if rem_folds:
                    total_proj_h, item7_h, breakdown = project_total_queue_hours_step5(
                        all_specs=specs,
                        step5_rows=step5_rows,
                        final_rows=final_rows,
                        step4_rows=step4_rows,
                        step3_rows=step3_rows,
                        indexed_val=indexed,
                        outputs_dir=outputs_dir,
                        smoke=smoke,
                    )
                    log_line(
                        log_f,
                        f"TIME_GUARD before item 7 ({spec.experiment_id}): "
                        f"item7_est={item7_h:.2f}h, total_projected_queue={total_proj_h:.2f}h "
                        f"(limit={max_hours:.2f}h)",
                    )
                    if total_proj_h > float(max_hours):
                        stop_path = outputs_dir / "QUEUE_STOPPED_STEP5.txt"
                        lines = [
                            f"Step 5 queue time guard triggered at {_ts()}",
                            f"Skipped run: {spec.rung} ({spec.experiment_id}, model={spec.model}, "
                            f"config={spec.config}, lr={spec.lr:g}, epochs={spec.epochs}, folds={rem_folds})",
                            f"Reason: Total projected Step 5 queue time ({total_proj_h:.2f} h, including "
                            f"{item7_h:.2f} h for {spec.experiment_id}) exceeds the {max_hours:.2f} h limit.",
                            "",
                            "Per-item runtime breakdown:",
                        ]
                        for item in breakdown:
                            lines.append(
                                f"- {item['rung']} ({item['experiment_id']}): "
                                f"sec_per_ep={item['sec_per_ep']:.2f}s, epochs={item['epochs']}, "
                                f"done={item['done_units']}, remaining={item['remaining_units']}, "
                                f"est_hours={item['projected_hours']:.2f}h"
                            )
                        _atomic_write_text(stop_path, "\n".join(lines) + "\n")
                        log_line(
                            log_f,
                            f"SKIPPED item 7 ({spec.experiment_id}): projected total queue time "
                            f"{total_proj_h:.2f}h > {max_hours:.2f}h. Wrote {stop_path}.",
                        )
                        continue

            if spec.is_final:
                assert spec.final_name is not None
                if name_exists_in_final_csv(final_csv, spec.final_name):
                    log_line(
                        log_f,
                        f"SKIP  FINAL rung={spec.rung} name={spec.final_name} (already in {final_csv.name})",
                    )
                    continue
                try:
                    _run_final_subprocess(
                        spec=spec,
                        final_csv=final_csv,
                        outputs_dir=outputs_dir,
                        smoke=smoke,
                        log_f=log_f,
                    )
                except Exception as exc:
                    log_line(
                        log_f,
                        f"CRASH FINAL rung={spec.rung} name={spec.final_name} exc={exc!r} — continuing",
                    )
                continue

            for fold in spec.folds:
                indexed = index_val_rows(load_results_rows(results_csv))
                if (spec.experiment_id, spec.group_mode, str(fold)) in indexed:
                    log_line(
                        log_f,
                        f"SKIP  rung={spec.rung} exp={spec.experiment_id} "
                        f"group_mode={spec.group_mode} fold={fold} (already in CSV)",
                    )
                    continue
                try:
                    _run_train_subprocess(
                        spec=spec,
                        fold=fold,
                        results_csv=results_csv,
                        outputs_dir=outputs_dir,
                        smoke=smoke,
                        log_f=log_f,
                    )
                except Exception as exc:
                    log_line(
                        log_f,
                        f"CRASH rung={spec.rung} exp={spec.experiment_id} fold={fold} "
                        f"exc={exc!r} — continuing",
                    )

            _maybe_append_cv_summary(spec, results_csv, log_f)

        log_line(log_f, "=== STEP 5 QUEUE COMPLETE ===")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable Step 5 fusion, reg_chi, and final test evaluation queue."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch on 64 images across 2 folds under outputs/smoke/ only (with --smoke-eval-on-val for finals).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run item 1 (E5_fusion_r1_p03) fold 0 only and print MAE for both normal and _meta_off rows.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="Skip item 7 if total projected queue hours exceed this limit (default: 8.0).",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Benchmark 20 real training batches per model type and print the projected runtime breakdown without training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.project_only:
        specs = build_step5_specs(smoke=False)
        outputs_dir = PROJECT_ROOT / "outputs"
        s5_rows = load_results_rows(outputs_dir / "results" / "step5_results.csv")
        fin_rows = load_results_rows(outputs_dir / "results" / "final_results.csv")
        s4_rows = load_results_rows(outputs_dir / "results" / "step4_results.csv")
        s3_rows = load_results_rows(outputs_dir / "results" / "step3_results.csv")
        total_h, item7_h, breakdown = project_total_queue_hours_step5(
            all_specs=specs,
            step5_rows=s5_rows,
            final_rows=fin_rows,
            step4_rows=s4_rows,
            step3_rows=s3_rows,
            indexed_val=index_val_rows(s5_rows),
            outputs_dir=outputs_dir,
            smoke=False,
            force_benchmark=True,
        )
        for item in breakdown:
            print(
                f"{item['order_idx']}. {item['experiment_id']} ({item['model']}, {item['group_mode']}, "
                f"runs={item['n_runs']}, ep={item['epochs']}): "
                f"{item['sec_per_ep']:.2f} s/ep -> {item['min_per_run']:.2f} min/run "
                f"({item['projected_minutes']:.2f} min total, {item['projected_hours']:.2f} h)"
            )
        print(f"TOTAL PROJECTED HOURS: {total_h:.2f} h (item 7 = {item7_h:.2f} h, guard = {args.max_hours:.2f} h)")
        return 0

    return run_queue_step5(
        smoke=bool(args.smoke),
        pilot=bool(args.pilot),
        max_hours=float(args.max_hours),
    )


if __name__ == "__main__":
    raise SystemExit(main())
