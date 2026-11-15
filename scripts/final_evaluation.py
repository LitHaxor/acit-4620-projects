#!/usr/bin/env python3
"""Final full-training and one-shot test evaluation CLI (`scripts/final_evaluation.py`).

THIS IS THE ONLY SCRIPT ALLOWED TO EVALUATE THE TEST SET.

Features:
- Trains on ALL 8,176 cleaned training images for a fixed 30 epochs and evaluates
  the LAST epoch (no validation-based selection, no hyperparameter changes).
- Refuses (exit code 1 with clear error message) if `--name` already exists in
  `outputs/results/final_test_results.csv` (or the target `--output-csv`).
- Reports all 10 metrics (`mae, rmse, plcc, srcc, r2, acc, pm1_acc, macro_f1, tail_mae, tail_f1`)
  plus 95% station-bootstrap CI for MAE (`1,000` resamples of stations with replacement)
  on:
  (i) full test set (`n=2,920`), and
  (ii) `test_gt25km` subset (test images > 25 km from any training image, `n=717`,
       reusing `compute_test_distance_mask`).
- For fusion models (`fusion_r1_mobilenetv3_s`, `fusion_full_mobilenetv3_s`), also
  evaluates and writes a separate `<name>_meta_off` row for both `test` and `test_gt25km`.
- Saves per-image predictions to `outputs/predictions/<name>.csv` (and
  `outputs/predictions/<name>_meta_off.csv` for fusion models) with columns:
  `image_id,station_id,y_true,y_pred`.
- Saves the final last-epoch checkpoint to `outputs/checkpoints/final/<name>/model.pt`.
- `--smoke-eval-on-val`: trains on fold 0 training split and evaluates on fold 0
  validation split so smoke runs NEVER touch test labels or `datasets/test/metadata.csv`.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (  # noqa: E402
    compute_test_distance_mask,
    format_metrics_oneline,
    resolve_device,
)
from src.config import load_config  # noqa: E402
from src.data.cleaning import (  # noqa: E402
    dedupe_split_by_median_aqi,
    load_and_clean_metadata,
)
from src.data.dataset import PM25Dataset  # noqa: E402
from src.data.splits import build_cv_splits, haversine_distance_matrix_km  # noqa: E402
from src.data.transforms import get_eval_transforms, get_train_transforms  # noqa: E402
from src.metrics import METRIC_ORDER, compute_all_metrics  # noqa: E402
from src.models.dcm import PFRMLModel  # noqa: E402
from src.models.fusion import FUSION_MODEL_SPECS, PFRMLFusionModel  # noqa: E402
from train import (  # noqa: E402
    ALL_MODEL_NAMES,
    _atomic_torch_save,
    _evaluate_image_model,
    _subsample_indices,
    build_fold_loss,
    build_image_model,
    build_warmup_cosine_scheduler,
    init_pfrml_prototypes_from_train_fold,
    seed_everything,
    sync_device,
)


FINAL_CSV_FIELDNAMES: Tuple[str, ...] = (
    "name",
    "model",
    "config",
    "seed",
    "epochs",
    "eval_split",
    "n_samples",
    "n_stations",
    *METRIC_ORDER,
    "mae_ci95_low",
    "mae_ci95_high",
    "notes",
)


def station_bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    station_ids: Sequence[Any],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% CI `(ci_low, ci_high)` for MAE via station-level cluster bootstrap.

    Resamples `G` stations with replacement `n_boot` times; each resampled station
    contributes all of its images (weighting station mean errors by station image count).
    """
    errors = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    stations = np.asarray([str(s) for s in station_ids])
    unique_stations, inv = np.unique(stations, return_inverse=True)
    n_stations = len(unique_stations)
    if n_stations <= 1:
        m = float(np.mean(errors)) if len(errors) > 0 else float("nan")
        return m, m

    station_counts = np.bincount(inv, minlength=n_stations).astype(np.float64)
    station_error_sums = np.bincount(inv, weights=errors, minlength=n_stations).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_stations, size=(n_boot, n_stations))
    boot_sums = station_error_sums[idx].sum(axis=1)
    boot_counts = station_counts[idx].sum(axis=1)
    boot_maes = boot_sums / np.maximum(boot_counts, 1.0)
    low, high = np.percentile(boot_maes, [2.5, 97.5])
    return float(low), float(high)


def load_final_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def name_exists_in_final_csv(csv_path: Path, name: str) -> bool:
    for row in load_final_rows(csv_path):
        row_name = str(row.get("name", "") or row.get("experiment_id", "")).strip()
        if row_name == name:
            return True
    return False


def append_final_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Atomically append a row to `final_test_results.csv`."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_name(f".{csv_path.name}.lock")
    ordered = {k: row.get(k, "") for k in FINAL_CSV_FIELDNAMES}

    with lock_path.open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            existing_content = ""
            if csv_path.exists() and csv_path.stat().st_size > 0:
                existing_content = csv_path.read_text(encoding="utf-8")
                if existing_content and not existing_content.endswith("\n"):
                    existing_content += "\n"

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(FINAL_CSV_FIELDNAMES))
            if not existing_content:
                writer.writeheader()
            writer.writerow(ordered)

            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{csv_path.name}.",
                suffix=".tmp",
                dir=str(csv_path.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as tmp_f:
                    if existing_content:
                        tmp_f.write(existing_content)
                    tmp_f.write(buf.getvalue())
                    tmp_f.flush()
                    os.fsync(tmp_f.fileno())
                os.replace(tmp_path, csv_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def save_predictions_csv(
    pred_path: Path,
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Atomically write per-image predictions `(image_id, station_id, y_true, y_pred)`."""
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{pred_path.name}.",
        suffix=".tmp",
        dir=str(pred_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "station_id", "y_true", "y_pred"])
            img_ids = eval_df["image_id"].astype(str).to_numpy()
            st_ids = eval_df["station_id"].astype(str).to_numpy()
            for img_id, st_id, yt, yp in zip(img_ids, st_ids, y_true, y_pred):
                writer.writerow([img_id, st_id, f"{float(yt):.6f}", f"{float(yp):.6f}"])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, pred_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_train_only_cleaned(train_csv_path: str | Path) -> pd.DataFrame:
    """Load and deduplicate `train/metadata.csv` for `--smoke-eval-on-val` without opening `test/metadata.csv`."""
    raw_train = pd.read_csv(train_csv_path)
    return dedupe_split_by_median_aqi(raw_train)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final full-train and one-shot test evaluation."
    )
    parser.add_argument("--name", type=str, required=True, help="Unique final run name (e.g., F_E2, F_PFRML_full).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(ALL_MODEL_NAMES),
        help="Model architecture name.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--modality-dropout", type=float, default=None, help="Optional modality dropout override.")
    parser.add_argument("--epochs", type=int, default=30, help="Fixed number of training epochs (default: 30).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--lr", type=float, default=None, help="Optional learning rate override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader workers override.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples cap for smoke testing.")
    parser.add_argument(
        "--smoke-eval-on-val",
        action="store_true",
        help="Evaluate on fold-0 validation split instead of the test set so smoke runs never touch test labels.",
    )
    parser.add_argument("--outputs-dir", type=str, default=None, help="Optional outputs directory override.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional results CSV path override.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training from last.pt if present.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    overrides: Dict[str, Any] = {
        "model.name": args.model,
        "seed": int(args.seed),
        "train.epochs": int(args.epochs),
    }
    if args.outputs_dir is not None:
        overrides["outputs_dir"] = args.outputs_dir
    if args.modality_dropout is not None:
        overrides["pfrml.modality_dropout"] = float(args.modality_dropout)
    if args.lr is not None:
        overrides["train.lr"] = float(args.lr)
    if args.num_workers is not None:
        overrides["data.num_workers"] = int(args.num_workers)

    cfg = load_config(args.config, overrides=overrides)
    outputs_dir = Path(cfg.outputs_dir)
    if args.output_csv is not None:
        results_csv = Path(args.output_csv)
    else:
        results_csv = outputs_dir / "results" / "final_test_results.csv"

    run_name = args.name.strip()
    if not run_name:
        print("ERROR: --name must be non-empty.", file=sys.stderr)
        return 1

    # Refuse if --name already exists in final_test_results.csv (one-shot test protection)
    if name_exists_in_final_csv(results_csv, run_name):
        print(
            f"REFUSED: Final evaluation name {run_name!r} already exists in {results_csv}. "
            "Test set evaluation is strictly one-shot per --name.",
            file=sys.stderr,
        )
        return 1

    seed = int(cfg.seed)
    total_epochs = int(cfg.train.epochs)
    warmup_epochs = int(cfg.train.warmup_epochs)
    aqi_scale = float(cfg.data.aqi_scale)
    model_name = cfg.model.name
    is_fusion = model_name in FUSION_MODEL_SPECS

    seed_everything(seed)
    device = resolve_device(cfg.train.device)

    if args.smoke_eval_on_val:
        # Never touch test labels or test metadata CSV during smoke runs
        clean_train_only = _load_train_only_cleaned(cfg.data.train_csv)
        _, folds = build_cv_splits(
            clean_train_only,
            n_splits=cfg.data.n_splits,
            group_mode="station",
            spatial_cluster_km=cfg.data.spatial_cluster_km,
        )
        tr_idx, va_idx = folds[0]
        if args.max_samples is not None:
            n_tr = max(1, int(round(args.max_samples * 0.8)))
            n_va = max(1, args.max_samples - n_tr)
            tr_idx = _subsample_indices(tr_idx, n_tr)
            va_idx = _subsample_indices(va_idx, n_va)
        train_df = clean_train_only.iloc[tr_idx].reset_index(drop=True)
        eval_df = clean_train_only.iloc[va_idx].reset_index(drop=True)
        train_images_dir = Path(cfg.data.train_images_dir)
        eval_images_dir = Path(cfg.data.train_images_dir)
        full_split_label = "val_smoke"
        gt25_split_label = "val_smoke_gt25km"
        _, gt25_mask = compute_test_distance_mask(train_df, eval_df, min_km=25.0)
        if not np.any(gt25_mask):
            dmat = haversine_distance_matrix_km(
                eval_df[["latitude", "longitude"]].to_numpy(dtype=np.float64),
                train_df[["latitude", "longitude"]].to_numpy(dtype=np.float64),
            )
            nearest_km = dmat.min(axis=1)
            gt25_mask = nearest_km >= float(np.median(nearest_km))
    else:
        train_df, test_df = load_and_clean_metadata(cfg.data.train_csv, cfg.data.test_csv)
        if args.max_samples is not None:
            n_tr = max(1, int(round(args.max_samples * 0.8)))
            n_te = max(1, args.max_samples - n_tr)
            tr_idx = _subsample_indices(np.arange(len(train_df)), n_tr)
            te_idx = _subsample_indices(np.arange(len(test_df)), n_te)
            train_df = train_df.iloc[tr_idx].reset_index(drop=True)
            eval_df = test_df.iloc[te_idx].reset_index(drop=True)
        else:
            eval_df = test_df.reset_index(drop=True)
        train_images_dir = Path(cfg.data.train_images_dir)
        eval_images_dir = Path(cfg.data.test_images_dir)
        full_split_label = "test"
        gt25_split_label = "test_gt25km"
        _, gt25_mask = compute_test_distance_mask(train_df, eval_df, min_km=25.0)
        print(
            f"[{run_name}] Cleaned full train n={len(train_df)}, test n={len(eval_df)}, "
            f"test_gt25km n={int(gt25_mask.sum())}",
            flush=True,
        )

    train_tf = get_train_transforms(
        image_size=cfg.data.image_size,
        horizontal_flip_prob=cfg.data.horizontal_flip_prob,
        rotation_deg=cfg.data.rotation_deg,
        translate_frac=cfg.data.translate_frac,
        scale_range=cfg.data.scale_range,
    )
    eval_tf = get_eval_transforms(image_size=cfg.data.image_size)

    train_ds = PM25Dataset(
        train_df,
        images_dir=train_images_dir,
        transform=train_tf,
        aqi_scale=aqi_scale,
    )
    eval_ds = PM25Dataset(
        eval_df,
        images_dir=eval_images_dir,
        transform=eval_tf,
        aqi_scale=aqi_scale,
    )

    num_workers = 0 if (args.max_samples is not None and args.num_workers is None) else int(cfg.data.num_workers)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=g,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model, head_type = build_image_model(model_name, cfg)
    model = model.to(device)

    train_bins = train_df["aqi_bin"].to_numpy(dtype=np.int64)
    use_protos_flag = (
        bool(model.head.use_prototypes)
        if isinstance(model, (PFRMLModel, PFRMLFusionModel))
        else False
    )
    aux_w = float(cfg.pfrml.aux_loss_weight) if is_fusion else 0.0
    loss_fn = build_fold_loss(
        head_type=head_type,
        cfg=cfg,
        train_bins=train_bins,
        use_prototypes_override=use_protos_flag,
        aux_loss_weight=aux_w,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=total_epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=warmup_epochs,
        base_lr=float(cfg.train.lr),
        min_lr=float(cfg.train.min_lr),
    )

    use_amp = bool(cfg.train.use_amp) and device.type in {"mps", "cuda"}
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    grad_clip = float(cfg.train.grad_clip_norm) if cfg.train.grad_clip_norm else 0.0

    ckpt_dir = outputs_dir / "checkpoints" / "final" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pt"

    start_epoch = 1
    epoch_times: List[float] = []
    if args.resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("model_name") == model_name and int(ckpt.get("epochs", -1)) == total_epochs:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            epoch_times = [float(x) for x in ckpt.get("epoch_times", [])]
            start_epoch = int(ckpt["epoch"]) + 1
            if "rng_python" in ckpt:
                random.setstate(ckpt["rng_python"])
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])
            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])
            print(f"[{run_name}] Resuming full-train from epoch {start_epoch - 1}/{total_epochs}", flush=True)

    # Train strictly on training set for fixed `total_epochs` without peeking at eval_loader
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            images = batch.image.to(device)
            targets_aqi = batch.aqi.to(device)
            targets_bin = batch.aqi_bin.to(device)
            bs = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                if head_type == "multitask":
                    if isinstance(model, PFRMLFusionModel):
                        out = model(
                            images,
                            metadata=batch.metadata.to(device),
                            targets=targets_bin,
                            metadata_off=False,
                        )
                    else:
                        out = model(images, targets=targets_bin)
                    loss, _ = loss_fn(out, targets_bin, targets_aqi)
                else:
                    out = model(images)
                    if head_type == "regression":
                        loss = loss_fn(out.float(), targets_aqi)
                    else:
                        loss = loss_fn(out.float(), targets_bin)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * bs
            n_seen += bs

        if (
            head_type == "multitask"
            and warmup_epochs > 0
            and epoch == warmup_epochs
            and isinstance(model, (PFRMLModel, PFRMLFusionModel))
            and model.head.use_prototypes
        ):
            init_pfrml_prototypes_from_train_fold(
                model=model,
                train_df=train_df,
                train_images_dir=train_images_dir,
                cfg=cfg,
                device=device,
            )

        sync_device(device)
        ep_sec = time.perf_counter() - t0
        epoch_times.append(ep_sec)
        train_loss = running_loss / max(n_seen, 1)

        cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _atomic_torch_save(
            {
                "epoch": epoch,
                "epochs": total_epochs,
                "model_name": model_name,
                "head_type": head_type,
                "model_state": cur_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch_times": epoch_times,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
            },
            last_ckpt_path,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == total_epochs:
            print(
                f"[{run_name}] ep {epoch:02d}/{total_epochs:02d} tr={train_loss:.4f} ({ep_sec:.1f}s)",
                flush=True,
            )

    mean_ep_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_torch_save(
        {
            "state_dict": final_state,
            "checkpoint_type": "last_epoch",
            "name": run_name,
            "model_name": model_name,
            "config": str(args.config),
            "head_type": head_type,
            "seed": seed,
            "epochs": total_epochs,
            "epoch": total_epochs,
            "max_samples": args.max_samples,
            "smoke_eval_on_val": bool(args.smoke_eval_on_val),
        },
        ckpt_dir / "model.pt",
    )
    if last_ckpt_path.exists():
        last_ckpt_path.unlink()

    # Evaluate the LAST-epoch model ONCE on eval_loader
    eval_loss, y_pred, y_pred_bin, proto_var = _evaluate_image_model(
        model=model,
        loader=eval_loader,
        head_type=head_type,
        loss_fn=loss_fn,
        device=device,
        aqi_scale=aqi_scale,
        use_amp=use_amp,
        metadata_off=False,
    )

    y_true = eval_df["pm25"].to_numpy(dtype=np.float64)
    y_bin = eval_df["aqi_bin"].to_numpy(dtype=np.int64)
    st_ids = eval_df["station_id"].astype(str).to_numpy()

    preds_dir = outputs_dir / "predictions"
    save_predictions_csv(preds_dir / f"{run_name}.csv", eval_df, y_true, y_pred)

    base_notes = (
        f"last_ep={total_epochs}/{total_epochs},eval_loss={eval_loss:.4f},"
        f"bs={int(cfg.train.batch_size)},lr={cfg.train.lr:g},sec_per_ep={mean_ep_s:.2f}"
    )
    if proto_var is not None:
        base_notes += f",proto_mix_var={proto_var:.6f}"

    def _record_split(
        rec_name: str,
        split_name: str,
        mask: Optional[np.ndarray],
        preds_arr: np.ndarray,
        bins_arr: Optional[np.ndarray],
        notes_str: str,
    ) -> Dict[str, float]:
        yt = y_true if mask is None else y_true[mask]
        yp = preds_arr if mask is None else preds_arr[mask]
        yb = y_bin if mask is None else y_bin[mask]
        ypb = (
            bins_arr
            if (bins_arr is None or mask is None)
            else bins_arr[mask]
        )
        sts = st_ids if mask is None else st_ids[mask]
        m = compute_all_metrics(y_true=yt, y_pred=yp, y_true_bin=yb, y_pred_bin=ypb)
        ci_low, ci_high = station_bootstrap_mae_ci(yt, yp, sts, n_boot=1000, seed=seed)
        n_st = int(len(np.unique(sts)))
        row: Dict[str, Any] = {
            "name": rec_name,
            "model": model_name,
            "config": str(args.config),
            "seed": seed,
            "epochs": total_epochs,
            "eval_split": split_name,
            "n_samples": int(len(yt)),
            "n_stations": n_st,
            "mae_ci95_low": ci_low,
            "mae_ci95_high": ci_high,
            "notes": notes_str,
        }
        for k in METRIC_ORDER:
            row[k] = float(m[k])
        append_final_row(results_csv, row)
        print(
            f"[{rec_name} | {split_name} (n={len(yt)}, stations={n_st})] "
            f"{format_metrics_oneline(m)} MAE_95CI=[{ci_low:.2f}, {ci_high:.2f}]",
            flush=True,
        )
        return m

    _record_split(run_name, full_split_label, None, y_pred, y_pred_bin, base_notes)
    if np.any(gt25_mask):
        _record_split(
            run_name,
            gt25_split_label,
            gt25_mask,
            y_pred,
            y_pred_bin,
            base_notes + ",nearest_train_km>25.0",
        )

    if is_fusion:
        mo_name = f"{run_name}_meta_off"
        mo_loss, mo_pred, mo_pred_bin, mo_proto_var = _evaluate_image_model(
            model=model,
            loader=eval_loader,
            head_type=head_type,
            loss_fn=loss_fn,
            device=device,
            aqi_scale=aqi_scale,
            use_amp=use_amp,
            metadata_off=True,
        )
        save_predictions_csv(preds_dir / f"{mo_name}.csv", eval_df, y_true, mo_pred)
        mo_notes = (
            f"last_ep={total_epochs}/{total_epochs},eval_loss={mo_loss:.4f},"
            f"bs={int(cfg.train.batch_size)},lr={cfg.train.lr:g},sec_per_ep={mean_ep_s:.2f},"
            f"metadata_off=True"
        )
        if mo_proto_var is not None:
            mo_notes += f",proto_mix_var={mo_proto_var:.6f}"
        _record_split(mo_name, full_split_label, None, mo_pred, mo_pred_bin, mo_notes)
        if np.any(gt25_mask):
            _record_split(
                mo_name,
                gt25_split_label,
                gt25_mask,
                mo_pred,
                mo_pred_bin,
                mo_notes + ",nearest_train_km>25.0",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
