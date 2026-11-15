"""Evaluation CLI and shared metric/reporting utilities for PM25Vision."""

import argparse
import csv
import fcntl
import io
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config, load_config
from src.data.cleaning import load_and_clean_metadata
from src.data.dataset import PM25Dataset, encode_metadata_features
from src.data.splits import build_cv_splits, haversine_distance_matrix_km
from src.data.transforms import get_eval_transforms
from src.metrics import (
    METRIC_ORDER,
    compute_all_metrics,
    format_mean_std,
    summarize_fold_metrics,
)
from src.models.baselines import (
    MetadataMLP,
    build_mobilenet_v3_small_pretrained,
    build_resnet18_pretrained,
    build_resnet18_scratch,
)
from src.models.dcm import PFRMLModel
from src.models.fusion import (
    FUSION_MODEL_NAMES,
    FUSION_MODEL_SPECS,
    PFRMLFusionModel,
)
from src.models.mdnet import build_mdnet_t, build_mobilenet_v3_small_md_neck

EPA_BIN_MIDPOINTS = np.array([25.0, 75.0, 125.0, 175.0, 250.0, 450.0], dtype=np.float64)

PFRML_MODEL_SPECS: Dict[str, Tuple[str, bool]] = {
    "pfrml_mobilenetv3_s": ("mobilenetv3_s", True),
    "pfrml_resnet18": ("resnet18", True),
    "pfrml_mdnet_t": ("mdnet_t", False),
    "pfrml_e3b": ("e3b", True),
}
PFRML_MODEL_NAMES: Tuple[str, ...] = tuple(PFRML_MODEL_SPECS.keys())

CSV_FIELDNAMES: Tuple[str, ...] = (
    "experiment_id",
    "model",
    "seed",
    "group_mode",
    "eval_split",
    "fold",
    "n_samples",
    *METRIC_ORDER,
    "notes",
)


def default_pfrml_experiment_id(
    model_name: str,
    config_path: Union[str, Path],
    reg_input: str = "chi_proto",
) -> str:
    """Format default Step 4/5 PFRML experiment ID.

    Step 4 configs use `E4_<backbone>_<config name>`; Step 5 `pfrml_reg_chi`
    (`reg_input='chi'`) uses `E5_<backbone>_<config name>`.
    """
    backbone = PFRML_MODEL_SPECS[model_name][0]
    stem = Path(config_path).stem
    cfg_tag = stem[len("pfrml_") :] if stem.startswith("pfrml_") else stem
    prefix = "E5" if (cfg_tag == "reg_chi" or reg_input == "chi") else "E4"
    return f"{prefix}_{backbone}_{cfg_tag}"


def default_fusion_experiment_id(model_name: str) -> str:
    """Format default Step 5 fusion experiment ID as `E5_<model_name>`."""
    return f"E5_{model_name}"


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _subsample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Select up to `count` evenly-spaced indices from `indices`."""
    if count <= 0 or len(indices) <= count:
        return indices
    pos = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return indices[pos]


def compute_test_distance_mask(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_km: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nearest_km, mask_gt_min_km) for test images relative to train images."""
    dmat = haversine_distance_matrix_km(
        test_df[["latitude", "longitude"]].to_numpy(dtype=np.float64),
        train_df[["latitude", "longitude"]].to_numpy(dtype=np.float64),
    )
    nearest_km = dmat.min(axis=1)
    return nearest_km, nearest_km > float(min_km)


def append_result_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Atomically append a single metrics row to a CSV file, creating headers if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_name(f".{csv_path.name}.lock")
    ordered = {k: row.get(k, "") for k in CSV_FIELDNAMES}

    with lock_path.open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            existing_content = ""
            if csv_path.exists() and csv_path.stat().st_size > 0:
                existing_content = csv_path.read_text(encoding="utf-8")
                if existing_content and not existing_content.endswith("\n"):
                    existing_content += "\n"

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(CSV_FIELDNAMES))
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


def build_result_row(
    experiment_id: str,
    model: str,
    seed: int,
    group_mode: str,
    eval_split: str,
    fold: str,
    n_samples: int,
    metrics: Dict[str, float],
    notes: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "model": model,
        "seed": int(seed),
        "group_mode": group_mode,
        "eval_split": eval_split,
        "fold": str(fold),
        "n_samples": int(n_samples),
        "notes": notes,
    }
    for k in METRIC_ORDER:
        row[k] = float(metrics[k])
    return row


def format_metrics_oneline(metrics: Dict[str, float]) -> str:
    return (
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
        f"PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} R2={metrics['r2']:.4f} "
        f"Acc={metrics['acc']:.4f} +-1Acc={metrics['pm1_acc']:.4f} "
        f"MacroF1={metrics['macro_f1']:.4f} TailMAE={metrics['tail_mae']:.2f} "
        f"TailF1={metrics['tail_f1']:.4f}"
    )


def evaluate_predictions_and_record(
    *,
    y_true_pm25: np.ndarray,
    y_pred_pm25: np.ndarray,
    y_true_bin: Optional[np.ndarray] = None,
    y_pred_bin: Optional[np.ndarray] = None,
    experiment_id: str,
    model: str,
    seed: int,
    group_mode: str,
    eval_split: str,
    fold: str,
    output_csv: Optional[Path] = None,
    notes: str = "",
) -> Dict[str, float]:
    """Compute full metrics, optionally append to CSV, and return metrics dict."""
    metrics = compute_all_metrics(
        y_true=y_true_pm25,
        y_pred=y_pred_pm25,
        y_true_bin=y_true_bin,
        y_pred_bin=y_pred_bin,
    )
    if output_csv is not None:
        row = build_result_row(
            experiment_id=experiment_id,
            model=model,
            seed=seed,
            group_mode=group_mode,
            eval_split=eval_split,
            fold=fold,
            n_samples=len(y_true_pm25),
            metrics=metrics,
            notes=notes,
        )
        append_result_row(output_csv, row)
    return metrics


def _build_pfrml_for_eval(model_name: str, cfg: Config) -> PFRMLModel:
    backbone_name, pretrained = PFRML_MODEL_SPECS[model_name]
    p = cfg.pfrml
    model = PFRMLModel(
        backbone_name=backbone_name,
        pretrained=False,
        tap_mode=p.tap_mode or "multistage",
        e3b_single_tap=p.e3b_single_tap,
        dim=p.dim,
        num_classes=p.num_classes,
        dcm_fusion=p.dcm_fusion,
        dcm_avg_pool_kernel=p.dcm_avg_pool_kernel,
        transformer_layers=p.transformer_layers,
        transformer_heads=p.transformer_heads,
        transformer_mlp_ratio=p.transformer_mlp_ratio,
        transformer_dropout=p.transformer_dropout,
        use_attention_pooling=p.use_attention_pooling,
        pooling_mode=p.pooling_mode,
        use_physics_gate=p.use_physics_gate,
        haze_patch_size=cfg.model.haze_patch_size,
        haze_omega=cfg.model.haze_omega,
        physics_input_is_normalized=p.physics_input_is_normalized,
        use_prototypes=p.use_prototypes,
        proto_ema_decay=p.proto_ema_decay,
        proto_tau_init=p.proto_tau_init,
        reg_input=p.reg_input,
        reg_hidden_dims=p.reg_hidden_dims,
        reg_dropout=p.reg_dropout,
    )
    model.backbone.pretrained = pretrained
    return model


def _build_fusion_for_eval(model_name: str, cfg: Config) -> PFRMLFusionModel:
    spec = FUSION_MODEL_SPECS[model_name]
    backbone_name = str(spec["backbone_name"])
    pretrained = bool(spec["pretrained"])
    use_prototypes = (
        False
        if model_name == "fusion_r1_mobilenetv3_s"
        else bool(cfg.pfrml.use_prototypes)
    )
    p = cfg.pfrml
    model = PFRMLFusionModel(
        backbone_name=backbone_name,
        pretrained=False,
        tap_mode=p.tap_mode or "multistage",
        e3b_single_tap=p.e3b_single_tap,
        dim=p.dim,
        num_classes=p.num_classes,
        dcm_fusion=p.dcm_fusion,
        dcm_avg_pool_kernel=p.dcm_avg_pool_kernel,
        transformer_layers=p.transformer_layers,
        transformer_heads=p.transformer_heads,
        transformer_mlp_ratio=p.transformer_mlp_ratio,
        transformer_dropout=p.transformer_dropout,
        use_attention_pooling=p.use_attention_pooling,
        pooling_mode=p.pooling_mode,
        use_physics_gate=p.use_physics_gate,
        haze_patch_size=cfg.model.haze_patch_size,
        haze_omega=cfg.model.haze_omega,
        physics_input_is_normalized=p.physics_input_is_normalized,
        use_prototypes=use_prototypes,
        proto_ema_decay=p.proto_ema_decay,
        proto_tau_init=p.proto_tau_init,
        reg_input=p.reg_input,
        reg_hidden_dims=p.reg_hidden_dims,
        reg_dropout=p.reg_dropout,
        metadata_dim=cfg.model.metadata_dim,
        metadata_hidden=cfg.model.metadata_hidden,
        metadata_dropout=cfg.model.metadata_dropout,
        modality_dropout=p.modality_dropout,
    )
    model.backbone.pretrained = pretrained
    return model


def _predict_from_checkpoint(
    ckpt_dir: Path,
    model_name: str,
    df: pd.DataFrame,
    cfg: Config,
    images_dir: Path,
    haze_features: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    metadata_off: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    dev = device if device is not None else resolve_device(cfg.train.device)

    if model_name == "median":
        with (ckpt_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        return model.predict(len(df)), None, ""
    if model_name == "ridge_haze":
        if haze_features is None:
            raise ValueError("haze_features required to evaluate ridge_haze checkpoint.")
        with (ckpt_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        return model.predict(haze_features), None, ""
    if model_name == "metadata_mlp":
        ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
        model = MetadataMLP(
            in_dim=cfg.model.metadata_dim,
            hidden_dims=cfg.model.metadata_hidden,
            dropout=cfg.model.metadata_dropout,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        meta = encode_metadata_features(
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
            df["captured_at"],
        )
        preds: List[np.ndarray] = []
        loader = DataLoader(TensorDataset(torch.from_numpy(meta)), batch_size=512, shuffle=False)
        with torch.no_grad():
            for (batch_meta,) in loader:
                out = model(batch_meta) * cfg.data.aqi_scale
                preds.append(out.numpy())
        return np.clip(np.concatenate(preds, axis=0), 0.0, None), None, ""

    if model_name in PFRML_MODEL_SPECS or model_name in FUSION_MODEL_SPECS:
        ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
        is_fusion = model_name in FUSION_MODEL_SPECS
        if is_fusion:
            eval_model: Union[PFRMLModel, PFRMLFusionModel] = _build_fusion_for_eval(model_name, cfg)
        else:
            eval_model = _build_pfrml_for_eval(model_name, cfg)
        eval_model.load_state_dict(ckpt["state_dict"])
        eval_model = eval_model.to(dev)
        eval_model.eval()

        ds = PM25Dataset(
            df,
            images_dir=images_dir,
            transform=get_eval_transforms(cfg.data.image_size),
            aqi_scale=cfg.data.aqi_scale,
        )
        loader = DataLoader(
            ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )
        pm25_chunks: List[np.ndarray] = []
        bin_chunks: List[np.ndarray] = []
        var_chunks: List[np.ndarray] = []
        with torch.no_grad():
            p_norm = None
            if eval_model.head.use_prototypes and eval_model.head.proto_cls is not None:
                p_norm = eval_model.head.proto_cls._l2_normalize(
                    eval_model.head.proto_cls.prototypes.detach()
                ).float()
            for batch in loader:
                images = batch.image.to(dev)
                if isinstance(eval_model, PFRMLFusionModel):
                    out = eval_model(
                        images,
                        metadata=batch.metadata.to(dev),
                        targets=None,
                        metadata_off=metadata_off,
                    )
                else:
                    out = eval_model(images, targets=None)
                pm25_chunks.append(
                    np.clip(out.reg_pred.float().cpu().numpy() * cfg.data.aqi_scale, 0.0, None)
                )
                if out.logits is not None:
                    bin_chunks.append(
                        out.logits.float().argmax(dim=-1).cpu().numpy().astype(np.int64)
                    )
                if p_norm is not None and out.probs is not None:
                    extra_f = (
                        out.meta_embed.float()
                        if (isinstance(eval_model, PFRMLFusionModel) and out.meta_embed is not None)
                        else None
                    )
                    _, batch_var, _ = eval_model.head.reg_head.predict_prototype_mixture_variance(
                        out.chi.float(), out.probs.float(), p_norm, extra_feat=extra_f
                    )
                    var_chunks.append(batch_var.cpu().numpy())

        mean_proto_var = float(np.mean(np.concatenate(var_chunks, axis=0))) if var_chunks else 0.0
        cls_src = "logits" if bin_chunks else "reg_pred_bins"
        notes = f"reg_src=reg_pred*aqi_scale,cls_src={cls_src},proto_mix_var={mean_proto_var:.6f}"
        if is_fusion and metadata_off:
            notes += ",metadata_off=True"
        return (
            np.concatenate(pm25_chunks, axis=0),
            np.concatenate(bin_chunks, axis=0) if bin_chunks else None,
            notes,
        )

    builders = {
        "resnet18_scratch": lambda: build_resnet18_scratch(head_type="regression"),
        "resnet18_pretrained": lambda: build_resnet18_pretrained(head_type="regression"),
        "mobilenetv3_s_pretrained": lambda: build_mobilenet_v3_small_pretrained(head_type="regression"),
        "mobilenetv3_s_md_neck": lambda: build_mobilenet_v3_small_md_neck(head_type="regression"),
        "mdnet_t_reg": lambda: build_mdnet_t(head_type="regression"),
        "mdnet_t_cls": lambda: build_mdnet_t(head_type="classification", num_classes=cfg.model.num_classes),
    }
    if model_name in builders:
        ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
        model = builders[model_name]()
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(dev)
        model.eval()
        ds = PM25Dataset(
            df,
            images_dir=images_dir,
            transform=get_eval_transforms(cfg.data.image_size),
            aqi_scale=cfg.data.aqi_scale,
        )
        loader = DataLoader(
            ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )
        pm25_chunks = []
        bin_chunks = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch.image.to(dev)).float().cpu()
                if model_name == "mdnet_t_cls":
                    probs = torch.softmax(out, dim=-1).numpy()
                    pm25_chunks.append(probs @ EPA_BIN_MIDPOINTS)
                    bin_chunks.append(out.argmax(dim=-1).numpy().astype(np.int64))
                else:
                    pm25_chunks.append(np.clip(out.numpy() * cfg.data.aqi_scale, 0.0, None))
        y_pred_bin = np.concatenate(bin_chunks, axis=0) if bin_chunks else None
        return np.concatenate(pm25_chunks, axis=0), y_pred_bin, ""

    raise ValueError(f"Unsupported model={model_name!r} in evaluate.py")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved PM25Vision models/predictions.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--group-mode",
        type=str,
        default=None,
        choices=["station", "spatial_cluster"],
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Fold index 0..4, 'all' for 5-fold CV, or 'full' for full-train test eval.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--outputs-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--metadata-off", action="store_true", help="Zero metadata embedding m during evaluation.")
    parser.add_argument("--from-preds", action="store_true", help="Read saved preds.npz instead of running checkpoint.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    overrides: Dict[str, Any] = {}
    if args.outputs_dir is not None:
        overrides["outputs_dir"] = args.outputs_dir
    if args.model is not None:
        overrides["model.name"] = args.model
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.group_mode is not None:
        overrides["data.group_mode"] = args.group_mode
    if args.device is not None:
        overrides["train.device"] = args.device
    if args.modality_dropout is not None:
        overrides["pfrml.modality_dropout"] = float(args.modality_dropout)

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    model_name = args.model or cfg.model.name
    if args.experiment_id:
        experiment_id = args.experiment_id
    elif model_name in FUSION_MODEL_SPECS:
        experiment_id = default_fusion_experiment_id(model_name)
    elif model_name in PFRML_MODEL_SPECS:
        experiment_id = default_pfrml_experiment_id(model_name, args.config, reg_input=cfg.pfrml.reg_input)
    else:
        experiment_id = model_name

    seed = cfg.seed
    group_mode = cfg.data.group_mode
    device = resolve_device(cfg.train.device)

    if args.output_csv is None:
        if experiment_id.startswith("E5_") or model_name in FUSION_MODEL_SPECS:
            default_csv_name = "step5_eval.csv"
        elif model_name in PFRML_MODEL_SPECS:
            default_csv_name = "step4_eval.csv"
        else:
            default_csv_name = "step3_eval.csv"
        output_csv: Optional[Path] = Path(cfg.outputs_dir) / "results" / default_csv_name
    elif args.output_csv == "":
        output_csv = None
    else:
        output_csv = Path(args.output_csv)

    train_df, test_df = load_and_clean_metadata(cfg.data.train_csv, cfg.data.test_csv)
    ckpt_root = Path(cfg.outputs_dir) / "checkpoints" / experiment_id
    rec_exp_id = (
        f"{experiment_id}_meta_off"
        if (args.metadata_off and not experiment_id.endswith("_meta_off"))
        else experiment_id
    )

    if args.fold == "full":
        ckpt_dir = ckpt_root / "full_train"
        preds_name = "preds_meta_off.npz" if args.metadata_off else "preds.npz"
        preds_path = ckpt_dir / preds_name
        notes = ""
        if args.from_preds and preds_path.exists():
            data = np.load(preds_path)
            y_pred = data["y_pred"]
            y_pred_bin = data["y_pred_bin"] if "y_pred_bin" in data else None
            if "proto_mix_var" in data:
                notes = (
                    f"reg_src=reg_pred*aqi_scale,cls_src=logits,"
                    f"proto_mix_var={float(data['proto_mix_var']):.6f}"
                )
        else:
            haze_test = None
            if model_name == "ridge_haze":
                cache = np.load(Path(cfg.outputs_dir) / "cache" / "haze_features_v1.npz")
                haze_test = cache["test_X"]
            y_pred, y_pred_bin, notes = _predict_from_checkpoint(
                ckpt_dir,
                model_name,
                test_df,
                cfg,
                Path(cfg.data.test_images_dir),
                haze_test,
                device,
                metadata_off=bool(args.metadata_off),
            )

        y_true = test_df["pm25"].to_numpy(dtype=np.float64)
        y_bin = test_df["aqi_bin"].to_numpy(dtype=np.int64)
        m_all = evaluate_predictions_and_record(
            y_true_pm25=y_true,
            y_pred_pm25=y_pred,
            y_true_bin=y_bin,
            y_pred_bin=y_pred_bin,
            experiment_id=rec_exp_id,
            model=model_name,
            seed=seed,
            group_mode="full_train",
            eval_split="test",
            fold="test",
            output_csv=output_csv,
            notes=notes,
        )
        print(f"[{rec_exp_id} | test (n={len(y_true)})] {format_metrics_oneline(m_all)}")

        _, gt25_mask = compute_test_distance_mask(train_df, test_df, min_km=25.0)
        m_gt25 = evaluate_predictions_and_record(
            y_true_pm25=y_true[gt25_mask],
            y_pred_pm25=y_pred[gt25_mask],
            y_true_bin=y_bin[gt25_mask],
            y_pred_bin=y_pred_bin[gt25_mask] if y_pred_bin is not None else None,
            experiment_id=rec_exp_id,
            model=model_name,
            seed=seed,
            group_mode="full_train",
            eval_split="test_gt25km",
            fold="test_gt25km",
            output_csv=output_csv,
            notes=notes,
        )
        print(
            f"[{rec_exp_id} | test >25km (n={int(gt25_mask.sum())})] "
            f"{format_metrics_oneline(m_gt25)}"
        )
        return

    _, folds = build_cv_splits(
        train_df,
        n_splits=cfg.data.n_splits,
        group_mode=group_mode,  # type: ignore[arg-type]
        spatial_cluster_km=cfg.data.spatial_cluster_km,
    )
    fold_indices = list(range(len(folds))) if args.fold == "all" else [int(args.fold)]

    haze_train = None
    if not args.from_preds and model_name == "ridge_haze":
        cache = np.load(Path(cfg.outputs_dir) / "cache" / "haze_features_v1.npz")
        haze_train = cache["train_X"]

    fold_metrics: List[Dict[str, float]] = []
    for k in fold_indices:
        _, val_idx = folds[k]
        ckpt_dir = ckpt_root / group_mode / f"fold_{k}"
        eff_max_samples = args.max_samples
        if eff_max_samples is None and (ckpt_dir / "model.pt").exists():
            ckpt_meta = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
            if isinstance(ckpt_meta, dict) and ckpt_meta.get("max_samples") is not None:
                eff_max_samples = int(ckpt_meta["max_samples"])
        if eff_max_samples is not None:
            n_tr = max(1, int(round(eff_max_samples * 0.8)))
            n_va = max(1, eff_max_samples - n_tr)
            val_idx = _subsample_indices(val_idx, n_va)

        val_df = train_df.iloc[val_idx]
        preds_name = "preds_meta_off.npz" if args.metadata_off else "preds.npz"
        preds_path = ckpt_dir / preds_name
        notes = ""
        if args.from_preds and preds_path.exists():
            data = np.load(preds_path)
            y_pred = data["y_pred"]
            y_pred_bin = data["y_pred_bin"] if "y_pred_bin" in data else None
            if "proto_mix_var" in data:
                cls_src = "logits" if y_pred_bin is not None else "reg_pred_bins"
                notes = (
                    f"reg_src=reg_pred*aqi_scale,cls_src={cls_src},"
                    f"proto_mix_var={float(data['proto_mix_var']):.6f}"
                )
        else:
            h_val = haze_train[val_idx] if haze_train is not None else None
            y_pred, y_pred_bin, notes = _predict_from_checkpoint(
                ckpt_dir,
                model_name,
                val_df,
                cfg,
                Path(cfg.data.train_images_dir),
                h_val,
                device,
                metadata_off=bool(args.metadata_off),
            )

        m = evaluate_predictions_and_record(
            y_true_pm25=val_df["pm25"].to_numpy(dtype=np.float64),
            y_pred_pm25=y_pred,
            y_true_bin=val_df["aqi_bin"].to_numpy(dtype=np.int64),
            y_pred_bin=y_pred_bin,
            experiment_id=rec_exp_id,
            model=model_name,
            seed=seed,
            group_mode=group_mode,
            eval_split="val",
            fold=str(k),
            output_csv=output_csv,
            notes=notes,
        )
        fold_metrics.append(m)
        print(f"[{rec_exp_id} | {group_mode} | fold {k}] {format_metrics_oneline(m)}")

    if len(fold_metrics) > 1:
        summary = summarize_fold_metrics(fold_metrics)
        fmt = format_mean_std(summary)
        print(
            f"[{rec_exp_id} | {group_mode} | 5-fold mean +- std] "
            + " ".join(f"{k}={v}" for k, v in fmt.items())
        )


if __name__ == "__main__":
    main()
