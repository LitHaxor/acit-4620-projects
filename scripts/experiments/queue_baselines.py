#!/usr/bin/env python3
"""Resumable Stage 1 -> Stage 2 -> Stage 3 training queue runner.

Executes each training run in an isolated subprocess, checkpoints every epoch
for crash/interrupt resumption, skips any (experiment_id, fold) already present
in the results CSV, enforces the deterministic Stage 2 selection rule, guards
Stage 3 against >12h projected runtime, and logs exclusively to
`outputs/logs/queue_baselines.log` (or `outputs/smoke/logs/queue_baselines.log` in `--smoke` mode).
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
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import append_result_row, build_result_row  # noqa: E402
from src.metrics import METRIC_ORDER, summarize_fold_metrics  # noqa: E402


GROUP_MODE = "station"
EVAL_SPLIT = "val"
DEFAULT_MAX_HOURS = 12.0


@dataclass(frozen=True)
class RunSpec:
    """Specification for a single experiment configuration."""

    family: str
    experiment_id: str
    model: str
    lr: float
    epochs: int
    is_default: bool = False


@dataclass(frozen=True)
class SelectionDecision:
    """Result of Stage 2 config selection for a model family."""

    family: str
    selected: RunSpec
    fold0_mae: float
    reason: str


# Initial defaults (lr=1e-3, 30 epochs) already evaluated on fold 0
EARLIER_DEFAULTS: Dict[str, RunSpec] = {
    "mobilenetv3_s_pretrained": RunSpec(
        family="mobilenetv3_s_pretrained",
        experiment_id="E2_mobilenetv3_s_pretrained",
        model="mobilenetv3_s_pretrained",
        lr=1e-3,
        epochs=30,
        is_default=True,
    ),
    "resnet18_pretrained": RunSpec(
        family="resnet18_pretrained",
        experiment_id="E2_resnet18_pretrained",
        model="resnet18_pretrained",
        lr=1e-3,
        epochs=30,
        is_default=True,
    ),
    "resnet18_scratch": RunSpec(
        family="resnet18_scratch",
        experiment_id="E2_resnet18_scratch",
        model="resnet18_scratch",
        lr=1e-3,
        epochs=30,
        is_default=True,
    ),
    "mdnet_t_reg": RunSpec(
        family="mdnet_t_reg",
        experiment_id="E3_mdnet_t_reg",
        model="mdnet_t_reg",
        lr=1e-3,
        epochs=30,
        is_default=True,
    ),
}

STAGE3_FAMILIES: Tuple[str, ...] = (
    "mobilenetv3_s_pretrained",
    "resnet18_pretrained",
    "resnet18_scratch",
    "mdnet_t_reg",
    "mobilenetv3_s_md_neck",
)


def build_stage1_specs(smoke: bool = False) -> List[RunSpec]:
    """Return ordered Stage 1 fold-0 candidate configurations."""
    def ep(n: int) -> int:
        return 1 if smoke else n

    return [
        RunSpec(
            family="mobilenetv3_s_pretrained",
            experiment_id="E2_mobilenetv3_s_pretrained_lr3e-4_ep40",
            model="mobilenetv3_s_pretrained",
            lr=3e-4,
            epochs=ep(40),
        ),
        RunSpec(
            family="resnet18_pretrained",
            experiment_id="E2_resnet18_pretrained_lr3e-4_ep40",
            model="resnet18_pretrained",
            lr=3e-4,
            epochs=ep(40),
        ),
        RunSpec(
            family="resnet18_scratch",
            experiment_id="E2_resnet18_scratch_lr1e-3_ep60",
            model="resnet18_scratch",
            lr=1e-3,
            epochs=ep(60),
        ),
        RunSpec(
            family="mdnet_t_reg",
            experiment_id="E3_mdnet_t_reg_lr1e-3_ep60",
            model="mdnet_t_reg",
            lr=1e-3,
            epochs=ep(60),
        ),
        RunSpec(
            family="mobilenetv3_s_md_neck",
            experiment_id="E3b_mobilenetv3_s_md_neck_lr3e-4_ep40",
            model="mobilenetv3_s_md_neck",
            lr=3e-4,
            epochs=ep(40),
        ),
        RunSpec(
            family="mobilenetv3_s_md_neck",
            experiment_id="E3b_mobilenetv3_s_md_neck_lr1e-3_ep40",
            model="mobilenetv3_s_md_neck",
            lr=1e-3,
            epochs=ep(40),
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
    group_mode: str = GROUP_MODE,
    eval_split: str = EVAL_SPLIT,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Index CSV rows by (experiment_id, fold) for station validation."""
    indexed: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in rows:
        if r.get("group_mode") == group_mode and r.get("eval_split") == eval_split:
            exp_id = str(r.get("experiment_id", ""))
            fold = str(r.get("fold", ""))
            if exp_id and fold:
                indexed[(exp_id, fold)] = r
    return indexed


def _parse_float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(val)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def select_stage2_configs(
    stage1_specs: Sequence[RunSpec],
    indexed_rows: Dict[Tuple[str, str], Dict[str, str]],
    defaults: Dict[str, RunSpec] = EARLIER_DEFAULTS,
) -> List[SelectionDecision]:
    """Apply Stage 2 selection rule with zero discretion.

    Per model family: pick the config with the lowest fold-0 validation MAE,
    but only switch away from the earlier default (lr 1e-3, 30 epochs) if it
    improves MAE by at least 1.0.
    """
    by_family: Dict[str, List[RunSpec]] = {fam: [] for fam in STAGE3_FAMILIES}
    for spec in stage1_specs:
        if spec.family in by_family:
            by_family[spec.family].append(spec)

    decisions: List[SelectionDecision] = []
    for fam in STAGE3_FAMILIES:
        cands = by_family.get(fam, [])
        valid_cands: List[Tuple[RunSpec, float]] = []
        for c in cands:
            row = indexed_rows.get((c.experiment_id, "0"))
            mae = _parse_float(row.get("mae") if row else None)
            if mae is not None:
                valid_cands.append((c, mae))

        default_spec = defaults.get(fam)
        default_row = indexed_rows.get((default_spec.experiment_id, "0")) if default_spec else None
        default_mae = _parse_float(default_row.get("mae") if default_row else None)

        if default_spec is not None and default_mae is not None:
            if not valid_cands:
                decisions.append(
                    SelectionDecision(
                        family=fam,
                        selected=default_spec,
                        fold0_mae=default_mae,
                        reason="fallback_to_default (no valid Stage 1 candidate)",
                    )
                )
                continue
            best_cand, best_mae = min(valid_cands, key=lambda x: x[1])
            improvement = default_mae - best_mae
            if improvement >= 1.0:
                decisions.append(
                    SelectionDecision(
                        family=fam,
                        selected=best_cand,
                        fold0_mae=best_mae,
                        reason=(
                            f"switched_to_candidate (default_mae={default_mae:.4f}, "
                            f"cand_mae={best_mae:.4f}, improvement={improvement:.4f} >= 1.0)"
                        ),
                    )
                )
            else:
                decisions.append(
                    SelectionDecision(
                        family=fam,
                        selected=default_spec,
                        fold0_mae=default_mae,
                        reason=(
                            f"kept_default (default_mae={default_mae:.4f}, "
                            f"best_cand={best_cand.experiment_id} mae={best_mae:.4f}, "
                            f"improvement={improvement:.4f} < 1.0)"
                        ),
                    )
                )
        else:
            # E3b (or no earlier default present in CSV): pick lowest fold-0 MAE
            if not valid_cands:
                continue
            best_cand, best_mae = min(valid_cands, key=lambda x: x[1])
            decisions.append(
                SelectionDecision(
                    family=fam,
                    selected=best_cand,
                    fold0_mae=best_mae,
                    reason=f"lowest_mae_no_default (cand_mae={best_mae:.4f})",
                )
            )
    return decisions


def extract_sec_per_epoch(
    spec: RunSpec,
    rows: Sequence[Dict[str, str]],
    outputs_dir: Path,
) -> Optional[float]:
    """Measure seconds/epoch from CSV `notes` or saved fold-0 curve CSV."""
    # 1. Exact experiment_id fold-0 row
    for r in reversed(rows):
        if (
            r.get("experiment_id") == spec.experiment_id
            and r.get("group_mode") == GROUP_MODE
            and r.get("eval_split") == EVAL_SPLIT
        ):
            m = re.search(r"sec_per_ep=([0-9]+(?:\.[0-9]+)?)", r.get("notes", "") or "")
            if m:
                return float(m.group(1))

    # 2. Curve CSV for fold 0
    curve_path = outputs_dir / "curves" / f"{spec.experiment_id}_{GROUP_MODE}_fold0.csv"
    if curve_path.exists():
        with curve_path.open("r", newline="", encoding="utf-8") as cf:
            times: List[float] = []
            for cr in csv.DictReader(cf):
                t = _parse_float(cr.get("epoch_time_s"))
                if t is not None:
                    times.append(t)
            if times:
                return sum(times) / len(times)

    # 3. Any row for the same model family
    for r in reversed(rows):
        if r.get("model") == spec.model:
            m = re.search(r"sec_per_ep=([0-9]+(?:\.[0-9]+)?)", r.get("notes", "") or "")
            if m:
                return float(m.group(1))
    return None


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


def project_stage3_hours(
    decisions: Sequence[SelectionDecision],
    cv_folds: Sequence[int],
    rows: Sequence[Dict[str, str]],
    indexed_rows: Dict[Tuple[str, str], Dict[str, str]],
    outputs_dir: Path,
) -> Tuple[float, List[Dict[str, object]]]:
    """Compute projected Stage 3 remaining hours from measured sec/epoch."""
    total_seconds = 0.0
    breakdown: List[Dict[str, object]] = []

    for dec in decisions:
        spec = dec.selected
        sec_per_ep = extract_sec_per_epoch(spec, rows, outputs_dir)
        rem_epochs = 0
        rem_folds: List[int] = []

        for k in cv_folds:
            if (spec.experiment_id, str(k)) in indexed_rows:
                continue
            ckpt_path = (
                outputs_dir / "checkpoints" / spec.experiment_id / GROUP_MODE / f"fold_{k}" / "last.pt"
            )
            done_ep = _completed_epochs_from_checkpoint(ckpt_path, spec.epochs)
            left = max(0, spec.epochs - done_ep)
            rem_epochs += left
            rem_folds.append(k)

        est_sec = (sec_per_ep or 0.0) * rem_epochs
        total_seconds += est_sec
        breakdown.append(
            {
                "family": dec.family,
                "experiment_id": spec.experiment_id,
                "sec_per_ep": sec_per_ep,
                "remaining_folds": rem_folds,
                "remaining_epochs": rem_epochs,
                "projected_hours": est_sec / 3600.0,
            }
        )

    return total_seconds / 3600.0, breakdown


def _run_train_subprocess(
    *,
    spec: RunSpec,
    fold: int,
    results_csv: Path,
    outputs_dir: Path,
    smoke: bool,
    log_f: TextIO,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--experiment-id",
        spec.experiment_id,
        "--model",
        spec.model,
        "--group-mode",
        GROUP_MODE,
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
        f"START exp={spec.experiment_id} model={spec.model} fold={fold} "
        f"lr={spec.lr:g} epochs={spec.epochs}",
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
        log_line(log_f, f"DONE  exp={spec.experiment_id} fold={fold} (rc=0)")
    else:
        log_line(
            log_f,
            f"CRASH exp={spec.experiment_id} fold={fold} (rc={proc.returncode}) — continuing queue",
        )
    return int(proc.returncode)


def _maybe_append_cv_summary(
    spec: RunSpec,
    cv_folds: Sequence[int],
    results_csv: Path,
    log_f: TextIO,
) -> None:
    rows = load_results_rows(results_csv)
    indexed = index_val_rows(rows)
    if (spec.experiment_id, "cv_mean") in indexed and (spec.experiment_id, "cv_std") in indexed:
        return

    fold_metrics: List[Dict[str, float]] = []
    seed_val = 42
    for k in cv_folds:
        row = indexed.get((spec.experiment_id, str(k)))
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
    total_n = sum(int(float(indexed[(spec.experiment_id, str(k))].get("n_samples", 0) or 0)) for k in cv_folds)

    if (spec.experiment_id, "cv_mean") not in indexed:
        append_result_row(
            results_csv,
            build_result_row(
                experiment_id=spec.experiment_id,
                model=spec.model,
                seed=seed_val,
                group_mode=GROUP_MODE,
                eval_split=EVAL_SPLIT,
                fold="cv_mean",
                n_samples=total_n,
                metrics=mean_dict,
                notes=f"folds={','.join(str(k) for k in cv_folds)}",
            ),
        )
    if (spec.experiment_id, "cv_std") not in indexed:
        append_result_row(
            results_csv,
            build_result_row(
                experiment_id=spec.experiment_id,
                model=spec.model,
                seed=seed_val,
                group_mode=GROUP_MODE,
                eval_split=EVAL_SPLIT,
                fold="cv_std",
                n_samples=total_n,
                metrics=std_dict,
                notes=f"folds={','.join(str(k) for k in cv_folds)}",
            ),
        )
    log_line(
        log_f,
        f"CV_SUMMARY exp={spec.experiment_id} folds={list(cv_folds)} "
        f"MAE={mean_dict['mae']:.2f}+-{std_dict['mae']:.2f} "
        f"R2={mean_dict['r2']:.4f}+-{std_dict['r2']:.4f}",
    )


def run_queue(
    *,
    smoke: bool = False,
    outputs_dir: Optional[Path] = None,
    results_csv: Optional[Path] = None,
    max_hours: float = DEFAULT_MAX_HOURS,
) -> int:
    if outputs_dir is None:
        outputs_dir = PROJECT_ROOT / ("outputs/smoke" if smoke else "outputs")
    if results_csv is None:
        results_csv = (
            outputs_dir / "image_models_cv_results.csv"
            if smoke
            else PROJECT_ROOT / "outputs/results/image_models_cv_results.csv"
        )
    log_path = outputs_dir / "logs" / "queue_baselines.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cv_folds = [0, 1] if smoke else [0, 1, 2, 3, 4]
    stage1_specs = build_stage1_specs(smoke=smoke)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_line(
            log_f,
            f"=== QUEUE START (smoke={smoke}, results_csv={results_csv}, outputs_dir={outputs_dir}) ===",
        )

        # ---- STAGE 1: fold-0 candidate runs ----
        log_line(log_f, "--- STAGE 1: Fold-0 Candidate Runs (station, val only) ---")
        for spec in stage1_specs:
            indexed = index_val_rows(load_results_rows(results_csv))
            if (spec.experiment_id, "0") in indexed:
                log_line(log_f, f"SKIP  Stage1 exp={spec.experiment_id} fold=0 (already in CSV)")
                continue
            try:
                _run_train_subprocess(
                    spec=spec,
                    fold=0,
                    results_csv=results_csv,
                    outputs_dir=outputs_dir,
                    smoke=smoke,
                    log_f=log_f,
                )
            except Exception as exc:
                log_line(
                    log_f,
                    f"CRASH Stage1 exp={spec.experiment_id} fold=0 exc={exc!r} — continuing",
                )

        # ---- STAGE 2: deterministic config selection ----
        log_line(log_f, "--- STAGE 2: Deterministic Config Selection ---")
        rows = load_results_rows(results_csv)
        indexed = index_val_rows(rows)
        decisions = select_stage2_configs(stage1_specs, indexed)
        for dec in decisions:
            log_line(
                log_f,
                f"SELECT family={dec.family} -> exp={dec.selected.experiment_id} "
                f"(lr={dec.selected.lr:g}, epochs={dec.selected.epochs}, "
                f"fold0_mae={dec.fold0_mae:.4f}) | {dec.reason}",
            )

        # ---- STAGE 3 TIME PROJECTION & 12H GUARD ----
        log_line(log_f, "--- STAGE 3: Runtime Projection ---")
        projected_hours, breakdown = project_stage3_hours(
            decisions=decisions,
            cv_folds=cv_folds,
            rows=rows,
            indexed_rows=indexed,
            outputs_dir=outputs_dir,
        )
        for item in breakdown:
            sec_ep = item["sec_per_ep"]
            sec_str = f"{sec_ep:.2f}s/ep" if sec_ep is not None else "unknown s/ep"
            log_line(
                log_f,
                f"PROJECT {item['experiment_id']}: {sec_str}, "
                f"remaining_folds={item['remaining_folds']}, "
                f"remaining_epochs={item['remaining_epochs']}, "
                f"est={item['projected_hours']:.2f}h",
            )
        log_line(
            log_f,
            f"PROJECTED TOTAL STAGE 3 HOURS: {projected_hours:.2f}h (limit={max_hours:.2f}h)",
        )

        if projected_hours > float(max_hours):
            stop_path = outputs_dir / "QUEUE_STOPPED.txt"
            lines = [
                f"Queue stopped before Stage 3 at {_ts()}",
                f"Projected Stage 3 total hours: {projected_hours:.2f}h > {max_hours:.2f}h limit",
                "",
                "Per-model breakdown:",
            ]
            for item in breakdown:
                lines.append(
                    f"- {item['experiment_id']}: sec_per_ep={item['sec_per_ep']}, "
                    f"remaining_folds={item['remaining_folds']}, "
                    f"remaining_epochs={item['remaining_epochs']}, "
                    f"est_hours={item['projected_hours']:.2f}"
                )
            _atomic_write_text(stop_path, "\n".join(lines) + "\n")
            log_line(
                log_f,
                f"STOPPED: Projected hours ({projected_hours:.2f}h) exceed {max_hours:.2f}h. "
                f"Wrote {stop_path} and exiting without waiting for input.",
            )
            return 0

        # ---- STAGE 3: 5-fold CV (2-fold in smoke) ----
        log_line(log_f, f"--- STAGE 3: Cross-Validation (folds={cv_folds}, station, val only) ---")
        for dec in decisions:
            spec = dec.selected
            for fold in cv_folds:
                indexed = index_val_rows(load_results_rows(results_csv))
                if (spec.experiment_id, str(fold)) in indexed:
                    log_line(
                        log_f,
                        f"SKIP  Stage3 exp={spec.experiment_id} fold={fold} (already in CSV)",
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
                        f"CRASH Stage3 exp={spec.experiment_id} fold={fold} exc={exc!r} — continuing",
                    )

            _maybe_append_cv_summary(spec, cv_folds, results_csv, log_f)

        log_line(log_f, "=== QUEUE COMPLETE ===")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable training queue for PM25Vision.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch on 64 images across 2 folds writing to outputs/smoke/ only.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="Stop before Stage 3 if projected total hours exceed this threshold (default: 12.0).",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Print the queue plan and projected runtime breakdown without training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.project_only:
        outputs_dir = PROJECT_ROOT / "outputs"
        results_csv = outputs_dir / "results" / "image_models_cv_results.csv"
        rows = load_results_rows(results_csv)
        indexed = index_val_rows(rows)
        stage1_specs = build_stage1_specs(smoke=False)
        decisions = select_stage2_configs(stage1_specs, indexed)
        cv_folds = [0, 1, 2, 3, 4]
        projected_hours, breakdown = project_stage3_hours(
            decisions=decisions,
            cv_folds=cv_folds,
            rows=rows,
            indexed_rows=indexed,
            outputs_dir=outputs_dir,
        )
        print("Stage 1 fold-0 candidate plan:")
        for spec in stage1_specs:
            status = "done" if (spec.experiment_id, "0") in indexed else "pending"
            print(
                f"  - {spec.experiment_id} (model={spec.model}, lr={spec.lr:g}, ep={spec.epochs}) [{status}]"
            )
        print("Stage 3 5-fold CV selected plan:")
        for item in breakdown:
            sec_ep = item["sec_per_ep"]
            sec_str = f"{sec_ep:.2f} s/ep" if sec_ep is not None else "unknown s/ep"
            print(
                f"  - {item['experiment_id']}: {sec_str}, "
                f"remaining_folds={item['remaining_folds']}, "
                f"remaining_epochs={item['remaining_epochs']}, "
                f"est={item['projected_hours']:.2f} h"
            )
        print(f"TOTAL PROJECTED HOURS: {projected_hours:.2f} h (guard = {args.max_hours:.2f} h)")
        return 0
    return run_queue(smoke=bool(args.smoke), max_hours=float(args.max_hours))


if __name__ == "__main__":
    raise SystemExit(main())
