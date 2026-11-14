#!/usr/bin/env python3
"""Resumable Step 4 PFRML experiment-ladder queue runner (`scripts/run_queue_step4.py`).

Design mirrors `scripts/run_queue.py`:
- One isolated subprocess (`train.py --resume`) per `(experiment_id, fold)`.
- Skips any `(experiment_id, group_mode, fold)` already present in
  `outputs/results/step4_results.csv` (or `outputs/smoke/step4_results.csv` in `--smoke`).
- Crashes are logged and the queue continues.
- Logs exclusively to `outputs/logs/queue_step4.log` (or `outputs/smoke/logs/queue_step4.log`).
- Atomic CSV appends via `evaluate.append_result_row` and `cv_mean`/`cv_std` summary rows.
- `--pilot`: runs only `R2` (`E4_mobilenetv3_s_baseline`) fold 0 and prints its validation MAE.
- `--smoke`: runs 1 epoch on 64 samples across 2 folds under `outputs/smoke/`.
- Time guard before Extra run B (`E4_mdnet_t_baseline_ep60`): projects its duration from
  measured `sec/epoch` (or benchmarks `pfrml_mdnet_t` batches if no prior `mdnet_t` timing
  is available in the CSVs); if total projected queue time (completed + remaining) exceeds
  `14.0 h`, skips Extra run B and writes `outputs/QUEUE_STOPPED.txt`.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import append_result_row, build_result_row, resolve_device  # noqa: E402
from src.config import load_config  # noqa: E402
from src.metrics import METRIC_ORDER, summarize_fold_metrics  # noqa: E402
from train import build_fold_loss, build_image_model, sync_device  # noqa: E402


EVAL_SPLIT = "val"
DEFAULT_MAX_HOURS = 14.0
FOLDS_PER_SPLIT = 5
BATCHES_PER_FOLD_BS32 = 205  # ~6541 train images / 32


@dataclass(frozen=True)
class Step4RunSpec:
    """Specification for a single rung / run in the Step 4 queue."""

    rung: str
    experiment_id: str
    model: str
    config: str
    group_mode: str = "station"
    lr: float = 1e-3
    epochs: int = 30
    is_extra_b: bool = False


def build_step4_specs(smoke: bool = False) -> List[Step4RunSpec]:
    """Return Step 4 queue configurations in config-major order (`SPEC.md` Sec. 5)."""

    def ep(n: int) -> int:
        return 1 if smoke else n

    return [
        Step4RunSpec(
            rung="R2",
            experiment_id="E4_mobilenetv3_s_baseline",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_baseline.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R1",
            experiment_id="E4_mobilenetv3_s_r1_no_prototypes",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_r1_no_prototypes.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R2+all",
            experiment_id="E4_mobilenetv3_s_full",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_full.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R2+T1",
            experiment_id="E4_mobilenetv3_s_attn_pool",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_attn_pool.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R2+T2",
            experiment_id="E4_mobilenetv3_s_physics_gate",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_physics_gate.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R2+T3",
            experiment_id="E4_mobilenetv3_s_class_weights",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_class_weights.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="R2+T4",
            experiment_id="E4_mobilenetv3_s_dcm_concat",
            model="pfrml_mobilenetv3_s",
            config="configs/pfrml_dcm_concat.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="Extra run A",
            experiment_id="E2_mobilenetv3_s_pretrained_spatial",
            model="mobilenetv3_s_pretrained",
            config="configs/default.yaml",
            group_mode="spatial_cluster",
            lr=1e-3,
            epochs=ep(30),
        ),
        Step4RunSpec(
            rung="Extra run B",
            experiment_id="E4_mdnet_t_baseline_ep60",
            model="pfrml_mdnet_t",
            config="configs/pfrml_baseline.yaml",
            group_mode="station",
            lr=1e-3,
            epochs=ep(60),
            is_extra_b=True,
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
    m = re.search(r"best_ep=[0-9]+/([0-9]+)", row.get("notes", "") or "")
    if m:
        return int(m.group(1))
    return default_epochs


def estimate_spec_sec_per_epoch(
    spec: Step4RunSpec,
    step4_rows: Sequence[Dict[str, str]],
    step3_rows: Sequence[Dict[str, str]],
    outputs_dir: Path,
    smoke: bool = False,
) -> float:
    """Estimate `sec/epoch` for `spec` from recorded CSV rows, curve CSVs, or a short batch timing."""
    # 1. Any completed fold of this exact experiment_id in step4_rows
    exact_times: List[float] = []
    for r in step4_rows:
        if r.get("experiment_id") == spec.experiment_id and r.get("fold") not in {"cv_mean", "cv_std"}:
            t = extract_sec_per_epoch_from_row(r)
            if t is not None:
                exact_times.append(t)
    if exact_times:
        return sum(exact_times) / len(exact_times)

    # 2. Curve CSV for fold 0 of this experiment
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

    # 3. Same model in step4_rows
    same_model_times = [
        t
        for r in step4_rows
        if r.get("model") == spec.model
        and r.get("fold") not in {"cv_mean", "cv_std"}
        and (t := extract_sec_per_epoch_from_row(r)) is not None
    ]
    if same_model_times:
        return sum(same_model_times) / len(same_model_times)

    # 4. For Extra run B (`pfrml_mdnet_t`) or `mobilenetv3_s_pretrained`, check Step 3 CSV rows
    if spec.model == "mobilenetv3_s_pretrained":
        s3_times = [
            t
            for r in step3_rows
            if r.get("model") == "mobilenetv3_s_pretrained"
            and (t := extract_sec_per_epoch_from_row(r)) is not None
        ]
        if s3_times:
            return sum(s3_times) / len(s3_times)

    if spec.model == "pfrml_mdnet_t":
        # Compute ratio from pfrml_mobilenetv3_s vs mobilenetv3_s_pretrained or add PFRML head overhead to mdnet_t_reg
        mdnet_s3_times = [
            t
            for r in step3_rows
            if r.get("model") == "mdnet_t_reg"
            and (t := extract_sec_per_epoch_from_row(r)) is not None
        ]
        mob_s3_times = [
            t
            for r in step3_rows
            if r.get("model") == "mobilenetv3_s_pretrained"
            and (t := extract_sec_per_epoch_from_row(r)) is not None
        ]
        pfrml_mob_times = [
            t
            for r in step4_rows
            if r.get("model") == "pfrml_mobilenetv3_s"
            and r.get("fold") not in {"cv_mean", "cv_std"}
            and (t := extract_sec_per_epoch_from_row(r)) is not None
        ]
        if mdnet_s3_times and not smoke:
            base_mdnet = sum(mdnet_s3_times) / len(mdnet_s3_times)
            head_overhead = 2.0
            if mob_s3_times and pfrml_mob_times:
                head_overhead = max(
                    1.0,
                    (sum(pfrml_mob_times) / len(pfrml_mob_times))
                    - (sum(mob_s3_times) / len(mob_s3_times)),
                )
            return base_mdnet + head_overhead

    # 5. Fallback: time a few synthetic batches on device
    if smoke:
        return 1.0
    return benchmark_batches_sec_per_epoch(spec.model, spec.config, n_batches=10)


def benchmark_batches_sec_per_epoch(
    model_name: str,
    config_path: str,
    n_batches: int = 20,
    batch_size: int = 32,
) -> float:
    """Time `n_batches` training steps at `batch_size=32` and extrapolate to a fold epoch."""
    cfg = load_config(PROJECT_ROOT / config_path, overrides={"model.name": model_name})
    device = resolve_device(cfg.train.device)
    model, head_type = build_image_model(model_name, cfg)
    model = model.to(device)
    model.train()

    dummy_bins = list(range(6)) * 20
    loss_fn = build_fold_loss(head_type, cfg, train_bins=dummy_bins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y_aqi = torch.rand(batch_size, device=device) * 3.0
    y_bin = torch.randint(0, 6, (batch_size,), device=device)

    # Warmup 2 steps
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        if head_type == "multitask":
            out = model(x, targets=y_bin)
            loss, _ = loss_fn(out, y_bin, y_aqi)
        else:
            out = model(x)
            loss = loss_fn(out.float(), y_aqi)
        loss.backward()
        optimizer.step()
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(n_batches):
        optimizer.zero_grad(set_to_none=True)
        if head_type == "multitask":
            out = model(x, targets=y_bin)
            loss, _ = loss_fn(out, y_bin, y_aqi)
        else:
            out = model(x)
            loss = loss_fn(out.float(), y_aqi)
        loss.backward()
        optimizer.step()
    sync_device(device)
    elapsed = time.perf_counter() - t0
    sec_per_batch = elapsed / max(n_batches, 1)
    # 205 train batches + ~52 eval batches (~0.35x cost of train batch)
    return sec_per_batch * (BATCHES_PER_FOLD_BS32 + 52 * 0.35)


def _completed_epochs_from_checkpoint(ckpt_path: Path, expected_epochs: int) -> int:
    if not ckpt_path.exists():
        return 0
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if int(ckpt.get("epochs", -1)) != expected_epochs:
            return 0
        return max(0, int(ckpt.get("epoch", 0)))
    except Exception:
        return 0


def project_total_queue_hours_before_extra_b(
    all_specs: Sequence[Step4RunSpec],
    cv_folds: Sequence[int],
    step4_rows: Sequence[Dict[str, str]],
    step3_rows: Sequence[Dict[str, str]],
    indexed_rows: Dict[Tuple[str, str, str], Dict[str, str]],
    outputs_dir: Path,
    smoke: bool = False,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """Return `(total_projected_hours, extra_b_hours, per_spec_breakdown)`."""
    total_seconds = 0.0
    extra_b_seconds = 0.0
    breakdown: List[Dict[str, object]] = []

    for spec in all_specs:
        sec_per_ep = estimate_spec_sec_per_epoch(
            spec=spec,
            step4_rows=step4_rows,
            step3_rows=step3_rows,
            outputs_dir=outputs_dir,
            smoke=smoke,
        )
        spec_sec = 0.0
        done_folds: List[int] = []
        rem_folds: List[int] = []
        for k in cv_folds:
            key = (spec.experiment_id, spec.group_mode, str(k))
            if key in indexed_rows:
                row = indexed_rows[key]
                row_sec = extract_sec_per_epoch_from_row(row) or sec_per_ep
                row_ep = extract_epochs_from_row(row, spec.epochs)
                spec_sec += row_sec * row_ep
                done_folds.append(k)
            else:
                ckpt_path = (
                    outputs_dir
                    / "checkpoints"
                    / spec.experiment_id
                    / spec.group_mode
                    / f"fold_{k}"
                    / "last.pt"
                )
                done_ep = _completed_epochs_from_checkpoint(ckpt_path, spec.epochs)
                # Count both already-run epochs in partial checkpoint and remaining epochs
                spec_sec += sec_per_ep * spec.epochs
                rem_folds.append(k)

        total_seconds += spec_sec
        if spec.is_extra_b:
            extra_b_seconds = spec_sec

        breakdown.append(
            {
                "rung": spec.rung,
                "experiment_id": spec.experiment_id,
                "model": spec.model,
                "sec_per_ep": sec_per_ep,
                "epochs": spec.epochs,
                "done_folds": done_folds,
                "remaining_folds": rem_folds,
                "projected_hours": spec_sec / 3600.0,
            }
        )

    return total_seconds / 3600.0, extra_b_seconds / 3600.0, breakdown


def _run_train_subprocess(
    *,
    spec: Step4RunSpec,
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
    if smoke:
        cmd.extend(["--max-samples", "64", "--num-workers", "0"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_line(
        log_f,
        f"START rung={spec.rung} exp={spec.experiment_id} model={spec.model} "
        f"group_mode={spec.group_mode} fold={fold} lr={spec.lr:g} epochs={spec.epochs}",
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


def _maybe_append_cv_summary(
    spec: Step4RunSpec,
    cv_folds: Sequence[int],
    results_csv: Path,
    log_f: TextIO,
) -> None:
    rows = load_results_rows(results_csv)
    indexed = index_val_rows(rows)
    mean_key = (spec.experiment_id, spec.group_mode, "cv_mean")
    std_key = (spec.experiment_id, spec.group_mode, "cv_std")
    if mean_key in indexed and std_key in indexed:
        return

    fold_metrics: List[Dict[str, float]] = []
    seed_val = 42
    for k in cv_folds:
        row = indexed.get((spec.experiment_id, spec.group_mode, str(k)))
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
        int(
            float(
                indexed[(spec.experiment_id, spec.group_mode, str(k))].get("n_samples", 0) or 0
            )
        )
        for k in cv_folds
    )
    notes_str = f"folds={','.join(str(k) for k in cv_folds)}"

    if mean_key not in indexed:
        append_result_row(
            results_csv,
            build_result_row(
                experiment_id=spec.experiment_id,
                model=spec.model,
                seed=seed_val,
                group_mode=spec.group_mode,
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
                experiment_id=spec.experiment_id,
                model=spec.model,
                seed=seed_val,
                group_mode=spec.group_mode,
                eval_split=EVAL_SPLIT,
                fold="cv_std",
                n_samples=total_n,
                metrics=std_dict,
                notes=notes_str,
            ),
        )
    log_line(
        log_f,
        f"CV_SUMMARY rung={spec.rung} exp={spec.experiment_id} group_mode={spec.group_mode} "
        f"folds={list(cv_folds)} MAE={mean_dict['mae']:.2f}+-{std_dict['mae']:.2f} "
        f"R2={mean_dict['r2']:.4f}+-{std_dict['r2']:.4f}",
    )


def run_queue_step4(
    *,
    smoke: bool = False,
    pilot: bool = False,
    outputs_dir: Optional[Path] = None,
    results_csv: Optional[Path] = None,
    max_hours: float = DEFAULT_MAX_HOURS,
) -> int:
    if outputs_dir is None:
        outputs_dir = PROJECT_ROOT / ("outputs/smoke" if smoke else "outputs")
    if results_csv is None:
        results_csv = (
            outputs_dir / "step4_results.csv"
            if smoke
            else PROJECT_ROOT / "outputs/results/step4_results.csv"
        )
    log_path = outputs_dir / "logs" / "queue_step4.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cv_folds = [0, 1] if smoke else [0, 1, 2, 3, 4]
    specs = build_step4_specs(smoke=smoke)

    step3_csv = PROJECT_ROOT / "outputs/results/step3_results.csv"
    step3_rows = load_results_rows(step3_csv)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_line(
            log_f,
            f"=== STEP 4 QUEUE START (smoke={smoke}, pilot={pilot}, "
            f"results_csv={results_csv}, outputs_dir={outputs_dir}) ===",
        )

        if pilot:
            r2_spec = specs[0]
            indexed = index_val_rows(load_results_rows(results_csv))
            key = (r2_spec.experiment_id, r2_spec.group_mode, "0")
            if key not in indexed:
                _run_train_subprocess(
                    spec=r2_spec,
                    fold=0,
                    results_csv=results_csv,
                    outputs_dir=outputs_dir,
                    smoke=smoke,
                    log_f=log_f,
                )
            else:
                log_line(
                    log_f,
                    f"SKIP  pilot exp={r2_spec.experiment_id} fold=0 (already in CSV)",
                )
            indexed = index_val_rows(load_results_rows(results_csv))
            row = indexed.get(key)
            mae_val = _parse_float(row.get("mae") if row else None)
            mae_str = f"{mae_val:.2f}" if mae_val is not None else "N/A"
            msg = f"PILOT R2 ({r2_spec.experiment_id}) fold 0 validation MAE: {mae_str}"
            log_line(log_f, msg)
            print(msg, flush=True)
            log_line(log_f, "=== STEP 4 PILOT COMPLETE ===")
            return 0

        for spec in specs:
            if spec.is_extra_b:
                # Time guard before Extra run B
                step4_rows = load_results_rows(results_csv)
                indexed = index_val_rows(step4_rows)
                rem_b_folds = [
                    k
                    for k in cv_folds
                    if (spec.experiment_id, spec.group_mode, str(k)) not in indexed
                ]
                if rem_b_folds:
                    total_proj_h, extra_b_h, breakdown = project_total_queue_hours_before_extra_b(
                        all_specs=specs,
                        cv_folds=cv_folds,
                        step4_rows=step4_rows,
                        step3_rows=step3_rows,
                        indexed_rows=indexed,
                        outputs_dir=outputs_dir,
                        smoke=smoke,
                    )
                    log_line(
                        log_f,
                        f"TIME_GUARD before Extra run B ({spec.experiment_id}): "
                        f"extra_b_est={extra_b_h:.2f}h, total_projected_queue={total_proj_h:.2f}h "
                        f"(limit={max_hours:.2f}h)",
                    )
                    if total_proj_h > float(max_hours):
                        stop_path = outputs_dir / "QUEUE_STOPPED.txt"
                        lines = [
                            f"Step 4 queue time guard triggered at {_ts()}",
                            f"Skipped run: {spec.rung} ({spec.experiment_id}, model={spec.model}, "
                            f"config={spec.config}, epochs={spec.epochs}, folds={rem_b_folds})",
                            f"Reason: Total projected queue time ({total_proj_h:.2f} h, including "
                            f"{extra_b_h:.2f} h for {spec.experiment_id}) exceeds the {max_hours:.2f} h limit.",
                            "",
                            "Per-config runtime breakdown:",
                        ]
                        for item in breakdown:
                            lines.append(
                                f"- {item['rung']} ({item['experiment_id']}): "
                                f"sec_per_ep={item['sec_per_ep']:.2f}s, epochs={item['epochs']}, "
                                f"done_folds={item['done_folds']}, "
                                f"remaining_folds={item['remaining_folds']}, "
                                f"est_hours={item['projected_hours']:.2f}h"
                            )
                        _atomic_write_text(stop_path, "\n".join(lines) + "\n")
                        log_line(
                            log_f,
                            f"SKIPPED Extra run B ({spec.experiment_id}): projected total queue time "
                            f"{total_proj_h:.2f}h > {max_hours:.2f}h. Wrote {stop_path}.",
                        )
                        continue

            for fold in cv_folds:
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

            _maybe_append_cv_summary(spec, cv_folds, results_csv, log_f)

        log_line(log_f, "=== STEP 4 QUEUE COMPLETE ===")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable Step 4 PFRML experiment-ladder training queue."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch on 64 images across 2 folds writing to outputs/smoke/ only.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run only R2 (E4_mobilenetv3_s_baseline) fold 0 and print its validation MAE.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="Skip Extra run B if total projected queue hours exceed this limit (default: 14.0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_queue_step4(
        smoke=bool(args.smoke),
        pilot=bool(args.pilot),
        max_hours=float(args.max_hours),
    )


if __name__ == "__main__":
    raise SystemExit(main())
