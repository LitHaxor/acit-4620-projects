#!/usr/bin/env python3
"""Resumable Step 5c queue runner (`scripts/run_queue_step5c.py`).

Design mirrors `scripts/run_queue_step5.py`:
- Logs to `outputs/logs/queue_step5c.log` (or `outputs/smoke/logs/queue_step5c.log` in `--smoke`).
- Prints projected total hours before starting (expected under 1.0 h; `--project-only` prints and exits).
- Item 1: `E5_fusion_r1_p00_ep{N}` (`fusion_r1_mobilenetv3_s`, `configs/fusion_r1.yaml`,
  `spatial_cluster`, 5 folds, `--modality-dropout 0.0`, `--lr 1e-3`, `--epochs N`, where `N`
  is read from `outputs/results/step5c_epochs.md`). Results go to `outputs/results/step5_results.csv`.
- Item 2 (deterministic decision rule, no discretion): compute 5-fold mean last-epoch MAE of Item 1.
  If `< 46.95`, `E_final = N`, else `E_final = 30`. Write both numbers and the decision to
  `outputs/results/step5c_decision.md` and to the queue log.
- Item 3: `scripts/final_train_eval.py --name F_FUSION_R1 --model fusion_r1_mobilenetv3_s
  --config configs/fusion_r1.yaml --modality-dropout 0.0 --epochs E_final --seed 42`.
  If `F_FUSION_R1` already exists in `outputs/results/final_results.csv`, skip with a clear message.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.choose_epochs import (  # noqa: E402
    generate_step5c_epochs_report,
    read_chosen_n_from_markdown,
)
from scripts.final_train_eval import name_exists_in_final_csv  # noqa: E402
from scripts.run_queue_step5 import (  # noqa: E402
    Step5RunSpec,
    _atomic_write_text,
    _maybe_append_cv_summary,
    _parse_float,
    _run_final_subprocess,
    _run_train_subprocess,
    index_val_rows,
    load_results_rows,
    log_line,
    project_total_queue_hours_step5,
)


DECISION_THRESHOLD_MAE: float = 46.95
FALLBACK_FINAL_EPOCHS: int = 30
DEFAULT_MAX_HOURS: float = 1.0


def resolve_chosen_n(epochs_md_path: Optional[Path] = None) -> int:
    """Read chosen epoch `N` from `outputs/results/step5c_epochs.md` (generating it if missing)."""
    target_md = epochs_md_path or (PROJECT_ROOT / "outputs/results/step5c_epochs.md")
    if not target_md.exists():
        generate_step5c_epochs_report(
            curves_dir=PROJECT_ROOT / "outputs/curves",
            output_md=target_md,
        )
    return read_chosen_n_from_markdown(target_md)


def build_step5c_item1_spec(chosen_n: int, smoke: bool = False) -> Step5RunSpec:
    """Build the `Step5RunSpec` for Item 1 (`E5_fusion_r1_p00_ep{N}`)."""
    cv_folds: Tuple[int, ...] = (0, 1) if smoke else (0, 1, 2, 3, 4)
    ep = 1 if smoke else int(chosen_n)
    return Step5RunSpec(
        order_idx=1,
        rung=f"1_E5_fusion_r1_p00_ep{chosen_n}",
        experiment_id=f"E5_fusion_r1_p00_ep{chosen_n}",
        model="fusion_r1_mobilenetv3_s",
        config="configs/fusion_r1.yaml",
        group_mode="spatial_cluster",
        folds=cv_folds,
        lr=1e-3,
        epochs=ep,
        modality_dropout=0.0,
        is_final=False,
    )


def build_step5c_item3_spec(e_final: int, smoke: bool = False) -> Step5RunSpec:
    """Build the `Step5RunSpec` for Item 3 (`F_FUSION_R1`)."""
    ep = 1 if smoke else int(e_final)
    return Step5RunSpec(
        order_idx=3,
        rung="3_FINAL_FUSION_R1",
        experiment_id="F_FUSION_R1",
        model="fusion_r1_mobilenetv3_s",
        config="configs/fusion_r1.yaml",
        group_mode="full_train",
        folds=(),
        lr=1e-3,
        epochs=ep,
        modality_dropout=0.0,
        is_final=True,
        final_name="F_FUSION_R1",
    )


def compute_item1_last_epoch_mean_mae(
    spec: Step5RunSpec,
    results_csv: Path,
) -> Tuple[float, float, Dict[int, float]]:
    """Compute `(mean_mae, sd_mae, per_fold_mae)` of Item 1 across `spec.folds`."""
    rows = load_results_rows(results_csv)
    indexed = index_val_rows(rows)
    per_fold: Dict[int, float] = {}
    for k in spec.folds:
        row = indexed.get((spec.experiment_id, spec.group_mode, str(k)))
        if row is None:
            raise RuntimeError(
                f"Missing validation row for ({spec.experiment_id}, {spec.group_mode}, fold={k}) in {results_csv}"
            )
        mae_val = _parse_float(row.get("mae"))
        if mae_val is None or math.isnan(mae_val):
            raise RuntimeError(
                f"Invalid MAE in ({spec.experiment_id}, {spec.group_mode}, fold={k})"
            )
        per_fold[int(k)] = float(mae_val)

    vals = np.asarray([per_fold[int(k)] for k in spec.folds], dtype=np.float64)
    mean_mae = float(np.mean(vals))
    sd_mae = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_mae, sd_mae, per_fold


def evaluate_decision_rule(
    mean_mae: float,
    chosen_n: int,
    threshold: float = DECISION_THRESHOLD_MAE,
    fallback_epochs: int = FALLBACK_FINAL_EPOCHS,
) -> Tuple[int, str]:
    """Deterministic decision rule: if `mean_mae < 46.95`, `E_final = N`, else `E_final = 30`."""
    if mean_mae < threshold:
        e_final = int(chosen_n)
        verdict = (
            f"mean_last_epoch_mae ({mean_mae:.2f}) < threshold ({threshold:.2f}) "
            f"-> E_final = N = {e_final}"
        )
    else:
        e_final = int(fallback_epochs)
        verdict = (
            f"mean_last_epoch_mae ({mean_mae:.2f}) >= threshold ({threshold:.2f}) "
            f"-> E_final = {e_final}"
        )
    return e_final, verdict


def read_e_final_from_decision_md(decision_md: Path) -> Optional[int]:
    if not decision_md.exists() or decision_md.stat().st_size == 0:
        return None
    text = decision_md.read_text(encoding="utf-8")
    m = re.search(r"E_final\s*=\s*([0-9]+)", text)
    if m:
        return int(m.group(1))
    return None


def write_decision_markdown(
    decision_md: Path,
    *,
    exp_id: str,
    chosen_n: int,
    mean_mae: float,
    sd_mae: float,
    per_fold: Dict[int, float],
    threshold: float,
    e_final: int,
    verdict: str,
) -> None:
    fold_Parts = ", ".join(f"F{k}={v:.2f}" for k, v in sorted(per_fold.items()))
    lines = [
        "# Step 5c Epoch Decision (`step5c_decision.md`)",
        "",
        f"- **Item 1 Experiment:** `{exp_id}` (`E5_fusion_r1_p00_epN` with `N = {chosen_n}`)",
        f"- **Candidate Epoch Count (`N`):** `{chosen_n}`",
        f"- **Item 1 5-Fold Last-Epoch Mean MAE:** `{mean_mae:.2f} +- {sd_mae:.2f}` ({fold_Parts})",
        f"- **Decision Threshold:** `{threshold:.2f}` (`E5_fusion_r1_p00` 3-epoch centered MA at epoch 8 / benchmark reference)",
        f"- **Decision Rule:** If 5-fold mean last-epoch MAE `< {threshold:.2f}`, set `E_final = N = {chosen_n}`; otherwise set `E_final = 30`.",
        f"- **Result:** `{verdict}`",
        f"- **Selected Final Training Epochs:** `E_final = {e_final}`",
    ]
    _atomic_write_text(decision_md, "\n".join(lines) + "\n")


def project_step5c_hours(
    *,
    chosen_n: int,
    e_final_for_proj: int,
    outputs_dir: Path,
    results_csv: Path,
    final_csv: Path,
    smoke: bool = False,
    force_benchmark: bool = False,
) -> Tuple[float, List[Dict[str, object]]]:
    """Project total runtime in hours for Step 5c (Item 1 + Item 3)."""
    spec1 = build_step5c_item1_spec(chosen_n=chosen_n, smoke=smoke)
    spec3 = build_step5c_item3_spec(e_final=e_final_for_proj, smoke=smoke)
    all_specs = [spec1, spec3]

    s5_rows = load_results_rows(results_csv)
    if not s5_rows and (PROJECT_ROOT / "outputs/results/step5_results.csv").exists():
        s5_rows = load_results_rows(PROJECT_ROOT / "outputs/results/step5_results.csv")
    fin_rows = load_results_rows(final_csv)
    s4_rows = load_results_rows(PROJECT_ROOT / "outputs/results/step4_results.csv")
    s3_rows = load_results_rows(PROJECT_ROOT / "outputs/results/step3_results.csv")

    total_h, _, breakdown = project_total_queue_hours_step5(
        all_specs=all_specs,
        step5_rows=s5_rows,
        final_rows=fin_rows,
        step4_rows=s4_rows,
        step3_rows=s3_rows,
        indexed_val=index_val_rows(load_results_rows(results_csv)),
        outputs_dir=outputs_dir,
        smoke=smoke,
        force_benchmark=force_benchmark,
    )
    return total_h, breakdown


def run_queue_step5c(
    *,
    smoke: bool = False,
    outputs_dir: Optional[Path] = None,
    results_csv: Optional[Path] = None,
    final_csv: Optional[Path] = None,
    decision_md: Optional[Path] = None,
    epochs_md: Optional[Path] = None,
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
    if decision_md is None:
        decision_md = (
            outputs_dir / "results" / "step5c_decision.md"
            if smoke
            else PROJECT_ROOT / "outputs/results/step5c_decision.md"
        )

    chosen_n = resolve_chosen_n(epochs_md)
    log_path = outputs_dir / "logs" / "queue_step5c.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Worst-case E_final before Item 2 decision is max(chosen_n, 30) = 30
    existing_e_final = read_e_final_from_decision_md(decision_md)
    proj_e_final = existing_e_final if existing_e_final is not None else max(chosen_n, FALLBACK_FINAL_EPOCHS)
    total_proj_h, breakdown = project_step5c_hours(
        chosen_n=chosen_n,
        e_final_for_proj=proj_e_final,
        outputs_dir=outputs_dir,
        results_csv=results_csv,
        final_csv=final_csv,
        smoke=smoke,
        force_benchmark=False,
    )

    with log_path.open("a", encoding="utf-8") as log_f:
        log_line(
            log_f,
            f"=== STEP 5C QUEUE START (smoke={smoke}, N={chosen_n}, "
            f"results_csv={results_csv}, final_csv={final_csv}, decision_md={decision_md}) ===",
        )
        proj_msg = (
            f"TIME_GUARD Step 5c projected total runtime: {total_proj_h:.2f} h "
            f"(N={chosen_n}, E_final_est={proj_e_final}, limit={max_hours:.2f} h)"
        )
        log_line(log_f, proj_msg)
        print(proj_msg, flush=True)

        if not smoke and total_proj_h > float(max_hours):
            stop_path = outputs_dir / "QUEUE_STOPPED_STEP5C.txt"
            _atomic_write_text(
                stop_path,
                f"Step 5c time guard triggered: projected {total_proj_h:.2f}h > {max_hours:.2f}h\n",
            )
            log_line(log_f, f"STOPPED: projected {total_proj_h:.2f}h > {max_hours:.2f}h ({stop_path})")
            return 1

        # Item 1: E5_fusion_r1_p00_ep{N} across folds
        spec1 = build_step5c_item1_spec(chosen_n=chosen_n, smoke=smoke)
        any_item1_ran = False
        for fold in spec1.folds:
            indexed = index_val_rows(load_results_rows(results_csv))
            if (spec1.experiment_id, spec1.group_mode, str(fold)) in indexed:
                log_line(
                    log_f,
                    f"SKIP  rung={spec1.rung} exp={spec1.experiment_id} "
                    f"group_mode={spec1.group_mode} fold={fold} (already in CSV)",
                )
                continue
            any_item1_ran = True
            try:
                _run_train_subprocess(
                    spec=spec1,
                    fold=fold,
                    results_csv=results_csv,
                    outputs_dir=outputs_dir,
                    smoke=smoke,
                    log_f=log_f,
                )
            except Exception as exc:
                log_line(
                    log_f,
                    f"CRASH rung={spec1.rung} exp={spec1.experiment_id} fold={fold} "
                    f"exc={exc!r} — continuing",
                )

        _maybe_append_cv_summary(spec1, results_csv, log_f)

        # Item 2: Deterministic decision rule
        mean_mae, sd_mae, per_fold = compute_item1_last_epoch_mean_mae(spec1, results_csv)
        e_final, verdict = evaluate_decision_rule(
            mean_mae=mean_mae,
            chosen_n=chosen_n,
            threshold=DECISION_THRESHOLD_MAE,
            fallback_epochs=FALLBACK_FINAL_EPOCHS,
        )
        if decision_md.exists() and not any_item1_ran:
            log_line(
                log_f,
                f"SKIP  ITEM 2 decision for {spec1.experiment_id} (already in {decision_md.name}: "
                f"mean_mae={mean_mae:.2f}, threshold={DECISION_THRESHOLD_MAE:.2f}, E_final={e_final})",
            )
        else:
            write_decision_markdown(
                decision_md,
                exp_id=spec1.experiment_id,
                chosen_n=chosen_n,
                mean_mae=mean_mae,
                sd_mae=sd_mae,
                per_fold=per_fold,
                threshold=DECISION_THRESHOLD_MAE,
                e_final=e_final,
                verdict=verdict,
            )
            log_line(
                log_f,
                f"DECISION ITEM 2 exp={spec1.experiment_id}: mean_mae={mean_mae:.2f}+-{sd_mae:.2f} "
                f"vs threshold={DECISION_THRESHOLD_MAE:.2f} -> E_final={e_final} (wrote {decision_md})",
            )

        # Item 3: Final evaluation F_FUSION_R1
        spec3 = build_step5c_item3_spec(e_final=e_final, smoke=smoke)
        assert spec3.final_name is not None
        if name_exists_in_final_csv(final_csv, spec3.final_name):
            skip_msg = (
                f"SKIP  FINAL rung={spec3.rung} name={spec3.final_name} "
                f"(already exists in {final_csv})"
            )
            log_line(log_f, skip_msg)
            print(skip_msg, flush=True)
        else:
            try:
                _run_final_subprocess(
                    spec=spec3,
                    final_csv=final_csv,
                    outputs_dir=outputs_dir,
                    smoke=smoke,
                    log_f=log_f,
                )
            except Exception as exc:
                log_line(
                    log_f,
                    f"CRASH FINAL rung={spec3.rung} name={spec3.final_name} exc={exc!r} — continuing",
                )

        log_line(log_f, "=== STEP 5C QUEUE COMPLETE ===")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable Step 5c queue runner (E5_fusion_r1_p00_epN, decision rule, F_FUSION_R1)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch on 64 images across 2 folds under outputs/smoke/ only (with --smoke-eval-on-val for final).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="Time guard limit in hours (default: 1.0).",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Print projected runtime breakdown and exit without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    chosen_n = resolve_chosen_n()
    if args.project_only:
        outputs_dir = PROJECT_ROOT / "outputs"
        results_csv = outputs_dir / "results" / "step5_results.csv"
        final_csv = outputs_dir / "results" / "final_results.csv"
        decision_md = outputs_dir / "results" / "step5c_decision.md"
        existing_e_final = read_e_final_from_decision_md(decision_md)
        e_final_est = existing_e_final if existing_e_final is not None else max(chosen_n, FALLBACK_FINAL_EPOCHS)
        total_h, breakdown = project_step5c_hours(
            chosen_n=chosen_n,
            e_final_for_proj=e_final_est,
            outputs_dir=outputs_dir,
            results_csv=results_csv,
            final_csv=final_csv,
            smoke=False,
            force_benchmark=False,
        )
        for item in breakdown:
            print(
                f"{item['order_idx']}. {item['experiment_id']} ({item['model']}, {item['group_mode']}, "
                f"runs={item['n_runs']}, ep={item['epochs']}): "
                f"{item['sec_per_ep']:.2f} s/ep -> {item['min_per_run']:.2f} min/run "
                f"({item['projected_minutes']:.2f} min total, {item['projected_hours']:.2f} h)"
            )
        print(
            f"TOTAL PROJECTED HOURS: {total_h:.2f} h "
            f"(N={chosen_n}, E_final_est={e_final_est}, guard={args.max_hours:.2f} h)"
        )
        return 0

    return run_queue_step5c(
        smoke=bool(args.smoke),
        max_hours=float(args.max_hours),
    )


if __name__ == "__main__":
    raise SystemExit(main())
