"""Training CLI for tabular baselines (E0, E1), image-only baselines (E2, E3, E3b), PFRML models (E4), and Multimodal Fusion models (E5)."""

import argparse
import csv
import math
import os
import pickle
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from evaluate import (
    PFRML_MODEL_NAMES,
    PFRML_MODEL_SPECS,
    append_result_row,
    build_result_row,
    compute_test_distance_mask,
    default_fusion_experiment_id,
    default_pfrml_experiment_id,
    evaluate_predictions_and_record,
    format_metrics_oneline,
    resolve_device,
)
from src.config import Config, load_config
from src.data.cleaning import load_and_clean_metadata
from src.data.dataset import PM25Dataset, encode_metadata_features
from src.data.splits import build_cv_splits
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.features.haze_features import HazeStatisticsExtractor
from src.losses import LabelSmoothingCrossEntropy, RegressionLoss
from src.metrics import METRIC_ORDER, compute_all_metrics, summarize_fold_metrics
from src.models.baselines import (
    HazeRidgeBaseline,
    MedianBaseline,
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
from src.pfrml_losses import PFRMLMultitaskLoss

IMAGE_BASELINE_MODEL_NAMES: Tuple[str, ...] = (
    "resnet18_scratch",
    "resnet18_pretrained",
    "mobilenetv3_s_pretrained",
    "mobilenetv3_s_md_neck",
    "mdnet_t_reg",
    "mdnet_t_cls",
)
IMAGE_MODEL_NAMES: Tuple[str, ...] = (
    *IMAGE_BASELINE_MODEL_NAMES,
    *PFRML_MODEL_NAMES,
    *FUSION_MODEL_NAMES,
)
ALL_MODEL_NAMES: Tuple[str, ...] = ("median", "ridge_haze", "metadata_mlp", *IMAGE_MODEL_NAMES)

# Midpoints of the 6 US EPA AQI bins [0-50, 51-100, 101-150, 151-200, 201-300, 301-600]
EPA_BIN_MIDPOINTS = np.array([25.0, 75.0, 125.0, 175.0, 250.0, 450.0], dtype=np.float64)

CURVE_FIELDNAMES: Tuple[str, ...] = (
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    *METRIC_ORDER,
    "epoch_time_s",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _atomic_torch_save(obj: Any, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_npz_save(target_path: Path, **arrays: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".npz",
        dir=str(target_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _subsample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Select up to `count` evenly-spaced indices from `indices`."""
    if count <= 0 or len(indices) <= count:
        return indices
    pos = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return indices[pos]


def build_image_model(model_name: str, cfg: Config) -> Tuple[nn.Module, str]:
    """Instantiate an image baseline, PFRML, or Multimodal Fusion model and return (model, head_type)."""
    num_classes = int(cfg.model.num_classes)
    if model_name == "resnet18_scratch":
        return build_resnet18_scratch(head_type="regression", num_classes=num_classes), "regression"
    if model_name == "resnet18_pretrained":
        return build_resnet18_pretrained(head_type="regression", num_classes=num_classes), "regression"
    if model_name == "mobilenetv3_s_pretrained":
        return (
            build_mobilenet_v3_small_pretrained(head_type="regression", num_classes=num_classes),
            "regression",
        )
    if model_name == "mobilenetv3_s_md_neck":
        return (
            build_mobilenet_v3_small_md_neck(head_type="regression", num_classes=num_classes),
            "regression",
        )
    if model_name == "mdnet_t_reg":
        return build_mdnet_t(head_type="regression", num_classes=num_classes), "regression"
    if model_name == "mdnet_t_cls":
        return build_mdnet_t(head_type="classification", num_classes=num_classes), "classification"
    if model_name in PFRML_MODEL_SPECS:
        backbone_name, pretrained = PFRML_MODEL_SPECS[model_name]
        p = cfg.pfrml
        model = PFRMLModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
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
        return model, "multitask"
    if model_name in FUSION_MODEL_SPECS:
        spec = FUSION_MODEL_SPECS[model_name]
        backbone_name = str(spec["backbone_name"])
        pretrained = bool(spec["pretrained"])
        p = cfg.pfrml
        use_prototypes = (
            False
            if model_name == "fusion_r1_mobilenetv3_s"
            else bool(p.use_prototypes)
        )
        use_attn = (
            False
            if model_name == "fusion_r1_mobilenetv3_s"
            else bool(p.use_attention_pooling or spec["use_attention_pooling"])
        )
        use_phys = (
            False
            if model_name == "fusion_r1_mobilenetv3_s"
            else bool(p.use_physics_gate or spec["use_physics_gate"])
        )
        fusion_model = PFRMLFusionModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
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
            use_attention_pooling=use_attn,
            pooling_mode=p.pooling_mode,
            use_physics_gate=use_phys,
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
        return fusion_model, "multitask"
    raise ValueError(f"Unknown image model_name={model_name!r}")


def build_fold_loss(
    head_type: str,
    cfg: Config,
    train_bins: Optional[Sequence[int]] = None,
    use_prototypes_override: Optional[bool] = None,
    aux_loss_weight: float = 0.0,
) -> nn.Module:
    """Build loss function for a fold; class weights computed strictly from fold training bins."""
    if head_type == "regression":
        return RegressionLoss(
            loss_type=cfg.loss.reg_loss,
            huber_delta=cfg.loss.huber_delta,
        )
    if head_type == "classification":
        return LabelSmoothingCrossEntropy(eps=cfg.loss.cls_eps)
    if head_type == "multitask":
        use_protos = (
            bool(use_prototypes_override)
            if use_prototypes_override is not None
            else bool(cfg.pfrml.use_prototypes)
        )
        if train_bins is None and use_protos:
            raise ValueError("PFRML multitask loss requires fold-internal `train_bins`.")
        return PFRMLMultitaskLoss.from_loss_config(
            cfg.loss,
            train_targets_or_counts=(
                np.asarray(train_bins, dtype=np.int64) if train_bins is not None else None
            ),
            num_classes=int(cfg.pfrml.num_classes),
            use_prototypes=use_protos,
            aux_loss_weight=float(aux_loss_weight),
        )
    raise ValueError(f"Unsupported head_type={head_type!r}")


@torch.no_grad()
def init_pfrml_prototypes_from_train_fold(
    model: Union[PFRMLModel, PFRMLFusionModel],
    train_df: pd.DataFrame,
    train_images_dir: Path,
    cfg: Config,
    device: torch.device,
) -> None:
    """Initialize PFRML prototypes from class centroids of chi on fold training images.

    Runs strictly on the fold's `train_df` in `model.eval()` mode with
    `get_eval_transforms` (no augmentation) and `torch.no_grad()`.
    """
    if not model.head.use_prototypes or model.head.proto_cls is None:
        return
    was_training = model.training
    model.eval()
    eval_tf = get_eval_transforms(image_size=cfg.data.image_size)
    init_ds = PM25Dataset(
        train_df,
        images_dir=train_images_dir,
        transform=eval_tf,
        aqi_scale=float(cfg.data.aqi_scale),
    )
    num_workers = int(cfg.data.num_workers)
    init_loader = DataLoader(
        init_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=num_workers,
    )
    chi_chunks: List[torch.Tensor] = []
    bin_chunks: List[torch.Tensor] = []
    for batch in init_loader:
        images = batch.image.to(device)
        if isinstance(model, PFRMLFusionModel):
            out = model(images, metadata=batch.metadata.to(device), targets=None)
        else:
            out = model(images, targets=None)
        chi_chunks.append(out.chi.detach().float())
        bin_chunks.append(batch.aqi_bin.to(device))
    if chi_chunks:
        all_chi = torch.cat(chi_chunks, dim=0)
        all_bins = torch.cat(bin_chunks, dim=0)
        model.head.proto_cls.init_prototypes(all_chi, all_bins)
    if was_training:
        model.train()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing LR schedule (per optimizer step)."""
    total_steps = max(int(epochs) * max(int(steps_per_epoch), 1), 1)
    warmup_steps = max(int(warmup_epochs) * max(int(steps_per_epoch), 1), 0)
    if total_steps <= 1:
        warmup_steps = 0
    else:
        warmup_steps = min(warmup_steps, total_steps - 1)
    min_ratio = float(min_lr) / max(float(base_lr), 1e-12)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class _RawImageDataset(Dataset):
    """Minimal pre-normalisation [0, 1] image dataset for haze feature extraction."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, image_size: int = 224) -> None:
        self.filenames = df["filename"].astype(str).tolist()
        self.images_dir = Path(images_dir)
        self.tf = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.images_dir / self.filenames[idx]) as img:
            return self.tf(img.convert("RGB"))


def _extract_haze_matrix(
    df: pd.DataFrame,
    images_dir: Path,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    ds = _RawImageDataset(df, images_dir, image_size=cfg.data.image_size)
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )
    extractor = HazeStatisticsExtractor(
        patch_size=cfg.model.haze_patch_size,
        omega=cfg.model.haze_omega,
        input_is_normalized=False,
    ).to(device)
    extractor.eval()

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = extractor(batch.to(device))
            chunks.append(feats.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def get_or_compute_haze_features(
    cfg: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached [N, 18] haze feature matrices or extract and cache them once."""
    cache_path = Path(cfg.outputs_dir) / "cache" / "haze_features_v1.npz"
    train_ids = train_df["image_id"].to_numpy()
    test_ids = test_df["image_id"].to_numpy()

    if cache_path.exists():
        data = np.load(cache_path)
        if (
            np.array_equal(data["train_image_ids"], train_ids)
            and np.array_equal(data["test_image_ids"], test_ids)
        ):
            return data["train_X"], data["test_X"]

    print("Extracting deterministic 18-D haze statistics (cached for reuse)...")
    train_X = _extract_haze_matrix(train_df, Path(cfg.data.train_images_dir), cfg, device)
    test_X = _extract_haze_matrix(test_df, Path(cfg.data.test_images_dir), cfg, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        train_X=train_X,
        test_X=test_X,
        train_image_ids=train_ids,
        test_image_ids=test_ids,
    )
    return train_X, test_X


def _train_predict_median(
    train_pm25: np.ndarray,
    eval_n: int,
    ckpt_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    model = MedianBaseline().fit(train_pm25)
    preds = model.predict(eval_n)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    _atomic_npz_save(ckpt_dir / "preds.npz", y_pred=preds)
    return preds, None, f"median={model.median_aqi_:.2f}"


def _train_predict_ridge(
    X_train: np.ndarray,
    train_pm25: np.ndarray,
    X_eval: np.ndarray,
    alphas: Sequence[float],
    ckpt_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    model = HazeRidgeBaseline(alphas=alphas).fit(X_train, train_pm25)
    preds = np.clip(model.predict(X_eval), 0.0, None)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    _atomic_npz_save(ckpt_dir / "preds.npz", y_pred=preds)
    return preds, None, f"alpha={model.alpha_:g}"


def _train_predict_metadata_mlp(
    meta_train: np.ndarray,
    train_pm25: np.ndarray,
    meta_eval: np.ndarray,
    cfg: Config,
    seed: int,
    device: torch.device,
    ckpt_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    seed_everything(seed)
    aqi_scale = float(cfg.data.aqi_scale)
    y_train_scaled = (train_pm25 / aqi_scale).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(meta_train.astype(np.float32)),
        torch.from_numpy(y_train_scaled),
    )
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        generator=g,
    )

    model = MetadataMLP(
        in_dim=cfg.model.metadata_dim,
        hidden_dims=cfg.model.metadata_hidden,
        dropout=cfg.model.metadata_dropout,
    ).to(device)

    loss_fn = RegressionLoss(
        loss_type=cfg.loss.reg_loss,
        huber_delta=cfg.loss.huber_delta,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs, 1)
    )

    model.train()
    for _ in range(cfg.train.epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    eval_ds = TensorDataset(torch.from_numpy(meta_eval.astype(np.float32)))
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False)
    pred_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for (bx,) in eval_loader:
            out = model(bx.to(device)) * aqi_scale
            pred_chunks.append(out.cpu().numpy())
    preds = np.clip(np.concatenate(pred_chunks, axis=0), 0.0, None)

    _atomic_torch_save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "seed": seed,
            "epochs": cfg.train.epochs,
        },
        ckpt_dir / "model.pt",
    )
    _atomic_npz_save(ckpt_dir / "preds.npz", y_pred=preds)
    return preds, None, f"epochs={cfg.train.epochs},loss={cfg.loss.reg_loss}"


def _evaluate_image_model(
    model: nn.Module,
    loader: DataLoader,
    head_type: str,
    loss_fn: nn.Module,
    device: torch.device,
    aqi_scale: float,
    use_amp: bool,
    metadata_off: bool = False,
) -> Tuple[float, np.ndarray, Optional[np.ndarray], Optional[float]]:
    """Run validation pass and return (val_loss, y_pred_pm25, y_pred_bin, mean_proto_var)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    pm25_chunks: List[np.ndarray] = []
    bin_chunks: List[np.ndarray] = []
    var_chunks: List[np.ndarray] = []

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_amp and device.type in {"mps", "cuda"}
        else nullcontext()
    )

    with torch.no_grad():
        p_norm = None
        if (
            head_type == "multitask"
            and isinstance(model, (PFRMLModel, PFRMLFusionModel))
            and model.head.use_prototypes
            and model.head.proto_cls is not None
        ):
            p_norm = model.head.proto_cls._l2_normalize(
                model.head.proto_cls.prototypes.detach()
            ).float()

        for batch in loader:
            images = batch.image.to(device)
            targets_aqi = batch.aqi.to(device)
            targets_bin = batch.aqi_bin.to(device)
            bs = images.shape[0]
            with amp_ctx:
                if head_type == "multitask":
                    if isinstance(model, PFRMLFusionModel):
                        out = model(
                            images,
                            metadata=batch.metadata.to(device),
                            targets=None,
                            metadata_off=metadata_off,
                        )
                    else:
                        out = model(images, targets=None)
                    loss, _ = loss_fn(out, targets_bin, targets_aqi)
                else:
                    out = model(images)
                    if head_type == "regression":
                        loss = loss_fn(out.float(), targets_aqi)
                    else:
                        loss = loss_fn(out.float(), targets_bin)
            total_loss += float(loss.item()) * bs
            total_n += bs

            if head_type == "regression":
                pred_pm25 = np.clip(out.float().cpu().numpy() * aqi_scale, 0.0, None)
                pm25_chunks.append(pred_pm25)
            elif head_type == "classification":
                logits = out.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_bins = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                pred_pm25 = probs @ EPA_BIN_MIDPOINTS
                pm25_chunks.append(pred_pm25)
                bin_chunks.append(pred_bins)
            else:
                pred_pm25 = np.clip(out.reg_pred.float().cpu().numpy() * aqi_scale, 0.0, None)
                pm25_chunks.append(pred_pm25)
                if out.logits is not None:
                    pred_bins = out.logits.float().argmax(dim=-1).cpu().numpy().astype(np.int64)
                    bin_chunks.append(pred_bins)
                if (
                    p_norm is not None
                    and out.probs is not None
                    and isinstance(model, (PFRMLModel, PFRMLFusionModel))
                ):
                    extra_f = (
                        out.meta_embed.float()
                        if (isinstance(model, PFRMLFusionModel) and out.meta_embed is not None)
                        else None
                    )
                    _, batch_var, _ = model.head.reg_head.predict_prototype_mixture_variance(
                        out.chi.float(), out.probs.float(), p_norm, extra_feat=extra_f
                    )
                    var_chunks.append(batch_var.cpu().numpy())

    mean_loss = total_loss / max(total_n, 1)
    y_pred_pm25 = np.concatenate(pm25_chunks, axis=0)
    y_pred_bin = np.concatenate(bin_chunks, axis=0) if bin_chunks else None
    mean_proto_var = float(np.mean(np.concatenate(var_chunks, axis=0))) if var_chunks else None
    return mean_loss, y_pred_pm25, y_pred_bin, mean_proto_var


def _prepare_curve_file_for_resume(curve_path: Path, start_epoch: int) -> None:
    """Keep valid curve rows strictly before `start_epoch` when resuming."""
    if start_epoch <= 1 or not curve_path.exists():
        with curve_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(CURVE_FIELDNAMES))
            writer.writeheader()
        return

    kept_rows: List[Dict[str, str]] = []
    with curve_path.open("r", newline="", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            if ep < start_epoch:
                kept_rows.append(row)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{curve_path.name}.",
        suffix=".tmp",
        dir=str(curve_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as tmp_f:
            writer = csv.DictWriter(tmp_f, fieldnames=list(CURVE_FIELDNAMES))
            writer.writeheader()
            for r in kept_rows:
                writer.writerow({k: r.get(k, "") for k in CURVE_FIELDNAMES})
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        os.replace(tmp_path, curve_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _train_predict_image_model(
    *,
    model_name: str,
    experiment_id: str,
    group_mode: str,
    fold_label: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_images_dir: Path,
    eval_images_dir: Path,
    cfg: Config,
    seed: int,
    device: torch.device,
    ckpt_dir: Path,
    resume: bool = False,
    max_samples: Optional[int] = None,
    metadata_off: bool = False,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    str,
    Optional[Tuple[np.ndarray, Optional[np.ndarray], str]],
]:
    seed_everything(seed)
    aqi_scale = float(cfg.data.aqi_scale)
    is_e5 = experiment_id.startswith("E5_")
    is_fusion = model_name in FUSION_MODEL_SPECS

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

    g = torch.Generator().manual_seed(seed)
    num_workers = int(cfg.data.num_workers)
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

    total_epochs = int(cfg.train.epochs)
    warmup_epochs = int(cfg.train.warmup_epochs)
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

    curve_path = (
        Path(cfg.outputs_dir)
        / "curves"
        / f"{experiment_id}_{group_mode}_fold{fold_label}.csv"
    )
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pt"

    y_eval_true = eval_df["pm25"].to_numpy(dtype=np.float64)
    y_eval_bin = eval_df["aqi_bin"].to_numpy(dtype=np.int64)

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    best_ep_mae = float("nan")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_preds: Optional[np.ndarray] = None
    best_pred_bins: Optional[np.ndarray] = None
    best_proto_var: Optional[float] = None

    last_epoch = 0
    last_val_loss = float("nan")
    last_ep_mae = float("nan")
    last_state: Optional[Dict[str, torch.Tensor]] = None
    last_preds: Optional[np.ndarray] = None
    last_pred_bins: Optional[np.ndarray] = None
    last_proto_var: Optional[float] = None

    epoch_times: List[float] = []

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("model_name") == model_name and int(ckpt.get("epochs", -1)) == total_epochs:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_loss = float(ckpt["best_val_loss"])
            best_epoch = int(ckpt["best_epoch"])
            best_ep_mae = float(ckpt.get("best_ep_mae", float("nan")))
            best_state = ckpt["best_state"]
            best_preds = ckpt["best_preds"]
            best_pred_bins = ckpt["best_pred_bins"]
            best_proto_var = ckpt.get("best_proto_var")

            last_epoch = int(ckpt.get("epoch", 0))
            last_val_loss = float(ckpt.get("last_val_loss", best_val_loss))
            last_ep_mae = float(ckpt.get("last_ep_mae", best_ep_mae))
            last_state = ckpt["model_state"]
            last_preds = ckpt.get("last_preds", best_preds)
            last_pred_bins = ckpt.get("last_pred_bins", best_pred_bins)
            last_proto_var = ckpt.get("last_proto_var", best_proto_var)

            epoch_times = [float(x) for x in ckpt.get("epoch_times", [])]
            start_epoch = int(ckpt["epoch"]) + 1
            if "rng_python" in ckpt:
                random.setstate(ckpt["rng_python"])
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])
            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])
            print(
                f"[{experiment_id} f{fold_label}] Resuming from epoch {start_epoch - 1}/{total_epochs}",
                flush=True,
            )

    _prepare_curve_file_for_resume(curve_path, start_epoch)

    if start_epoch <= total_epochs:
        with curve_path.open("a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(CURVE_FIELDNAMES))

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
                                    metadata_off=metadata_off,
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

                # Run init_prototypes once at the end of warmup epochs (epoch 3)
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

                train_loss = running_loss / max(n_seen, 1)
                val_loss, val_preds, val_pred_bins, val_proto_var = _evaluate_image_model(
                    model=model,
                    loader=eval_loader,
                    head_type=head_type,
                    loss_fn=loss_fn,
                    device=device,
                    aqi_scale=aqi_scale,
                    use_amp=use_amp,
                    metadata_off=metadata_off,
                )
                sync_device(device)
                ep_sec = time.perf_counter() - t0
                epoch_times.append(ep_sec)

                val_metrics = compute_all_metrics(
                    y_true=y_eval_true,
                    y_pred=val_preds,
                    y_true_bin=y_eval_bin,
                    y_pred_bin=val_pred_bins,
                )
                cur_lr = float(optimizer.param_groups[0]["lr"])

                row: Dict[str, Any] = {
                    "epoch": epoch,
                    "lr": cur_lr,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time_s": ep_sec,
                }
                for k in METRIC_ORDER:
                    row[k] = float(val_metrics[k])
                writer.writerow(row)
                cf.flush()

                cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                last_epoch = epoch
                last_val_loss = val_loss
                last_ep_mae = float(val_metrics["mae"])
                last_state = cur_state
                last_preds = val_preds.copy()
                last_pred_bins = val_pred_bins.copy() if val_pred_bins is not None else None
                last_proto_var = val_proto_var

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_ep_mae = float(val_metrics["mae"])
                    best_state = cur_state
                    best_preds = val_preds.copy()
                    best_pred_bins = val_pred_bins.copy() if val_pred_bins is not None else None
                    best_proto_var = val_proto_var

                # Per-epoch atomic checkpoint for crash/interrupt resumption
                _atomic_torch_save(
                    {
                        "epoch": epoch,
                        "epochs": total_epochs,
                        "model_name": model_name,
                        "head_type": head_type,
                        "model_state": cur_state,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "best_ep_mae": best_ep_mae,
                        "best_state": best_state,
                        "best_preds": best_preds,
                        "best_pred_bins": best_pred_bins,
                        "best_proto_var": best_proto_var,
                        "last_val_loss": last_val_loss,
                        "last_ep_mae": last_ep_mae,
                        "last_preds": last_preds,
                        "last_pred_bins": last_pred_bins,
                        "last_proto_var": last_proto_var,
                        "epoch_times": epoch_times,
                        "rng_python": random.getstate(),
                        "rng_numpy": np.random.get_state(),
                        "rng_torch": torch.get_rng_state(),
                    },
                    last_ckpt_path,
                )

                if epoch == 1 or epoch % 5 == 0 or epoch == total_epochs:
                    if head_type == "regression":
                        key_stat = f"MAE={val_metrics['mae']:.2f} R2={val_metrics['r2']:.4f}"
                    elif head_type == "classification":
                        key_stat = f"Acc={val_metrics['acc']:.4f} F1={val_metrics['macro_f1']:.4f}"
                    else:
                        key_stat = (
                            f"MAE={val_metrics['mae']:.2f} R2={val_metrics['r2']:.4f} "
                            f"Acc={val_metrics['acc']:.4f} F1={val_metrics['macro_f1']:.4f}"
                        )
                    print(
                        f"[{experiment_id} f{fold_label}] ep {epoch:02d}/{total_epochs:02d} "
                        f"tr={train_loss:.4f} val={val_loss:.4f} {key_stat} ({ep_sec:.1f}s)",
                        flush=True,
                    )

    assert best_state is not None and best_preds is not None
    assert last_state is not None and last_preds is not None
    mean_ep_s = float(np.mean(epoch_times)) if epoch_times else 0.0

    if is_e5:
        # E5_ runs report LAST epoch and save two checkpoints (best-loss & last-epoch)
        _atomic_torch_save(
            {
                "state_dict": best_state,
                "checkpoint_type": "best_loss",
                "model_name": model_name,
                "head_type": head_type,
                "seed": seed,
                "epochs": total_epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_ep_mae": best_ep_mae,
                "best_proto_var": best_proto_var,
                "max_samples": max_samples,
            },
            ckpt_dir / "best_model.pt",
        )
        _atomic_torch_save(
            {
                "state_dict": last_state,
                "checkpoint_type": "last_epoch",
                "model_name": model_name,
                "head_type": head_type,
                "seed": seed,
                "epochs": total_epochs,
                "epoch": last_epoch,
                "val_loss": last_val_loss,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_ep_mae": best_ep_mae,
                "last_proto_var": last_proto_var,
                "max_samples": max_samples,
            },
            ckpt_dir / "model.pt",
        )
        reported_preds = last_preds
        reported_pred_bins = last_pred_bins
        reported_proto_var = last_proto_var
        notes = (
            f"last_ep={last_epoch}/{total_epochs},val_loss={last_val_loss:.4f},"
            f"best_ep={best_epoch}/{total_epochs},best_ep_mae={best_ep_mae:.2f},"
            f"bs={int(cfg.train.batch_size)},lr={cfg.train.lr:g},amp={use_amp},"
            f"sec_per_ep={mean_ep_s:.2f}"
        )
    else:
        # Old experiments keep their old behaviour (best-loss checkpoint & reporting)
        _atomic_torch_save(
            {
                "state_dict": best_state,
                "model_name": model_name,
                "head_type": head_type,
                "seed": seed,
                "epochs": total_epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_proto_var": best_proto_var,
                "max_samples": max_samples,
            },
            ckpt_dir / "model.pt",
        )
        reported_preds = best_preds
        reported_pred_bins = best_pred_bins
        reported_proto_var = best_proto_var
        notes = (
            f"best_ep={best_epoch}/{total_epochs},val_loss={best_val_loss:.4f},"
            f"bs={int(cfg.train.batch_size)},lr={cfg.train.lr:g},amp={use_amp},"
            f"sec_per_ep={mean_ep_s:.2f}"
        )

    if head_type == "multitask":
        pv = reported_proto_var if reported_proto_var is not None else 0.0
        cls_src = "logits" if reported_pred_bins is not None else "reg_pred_bins"
        notes += f",reg_src=reg_pred*aqi_scale,cls_src={cls_src},proto_mix_var={pv:.6f}"
    if is_fusion and metadata_off:
        notes += ",metadata_off=True"

    npz_payload: Dict[str, Any] = {"y_pred": reported_preds}
    if reported_pred_bins is not None:
        npz_payload["y_pred_bin"] = reported_pred_bins
    if reported_proto_var is not None:
        npz_payload["proto_mix_var"] = np.array(reported_proto_var, dtype=np.float64)
    _atomic_npz_save(ckpt_dir / "preds.npz", **npz_payload)

    meta_off_result: Optional[Tuple[np.ndarray, Optional[np.ndarray], str]] = None
    if is_e5 and is_fusion and not metadata_off:
        model.load_state_dict(last_state)
        mo_loss, mo_preds, mo_pred_bins, mo_proto_var = _evaluate_image_model(
            model=model,
            loader=eval_loader,
            head_type=head_type,
            loss_fn=loss_fn,
            device=device,
            aqi_scale=aqi_scale,
            use_amp=use_amp,
            metadata_off=True,
        )
        mo_pv = mo_proto_var if mo_proto_var is not None else 0.0
        mo_cls_src = "logits" if mo_pred_bins is not None else "reg_pred_bins"
        mo_notes = (
            f"last_ep={last_epoch}/{total_epochs},val_loss={mo_loss:.4f},"
            f"best_ep={best_epoch}/{total_epochs},best_ep_mae={best_ep_mae:.2f},"
            f"bs={int(cfg.train.batch_size)},lr={cfg.train.lr:g},amp={use_amp},"
            f"sec_per_ep={mean_ep_s:.2f},"
            f"reg_src=reg_pred*aqi_scale,cls_src={mo_cls_src},proto_mix_var={mo_pv:.6f},"
            f"metadata_off=True"
        )
        mo_payload: Dict[str, Any] = {"y_pred": mo_preds}
        if mo_pred_bins is not None:
            mo_payload["y_pred_bin"] = mo_pred_bins
        if mo_proto_var is not None:
            mo_payload["proto_mix_var"] = np.array(mo_proto_var, dtype=np.float64)
        _atomic_npz_save(ckpt_dir / "preds_meta_off.npz", **mo_payload)
        meta_off_result = (mo_preds, mo_pred_bins, mo_notes)

    return reported_preds, reported_pred_bins, notes, meta_off_result


def run_single_split(
    *,
    model_name: str,
    experiment_id: str,
    cfg: Config,
    seed: int,
    group_mode: str,
    fold_label: str,
    eval_split: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_images_dir: Path,
    eval_images_dir: Path,
    train_haze: Optional[np.ndarray],
    eval_haze: Optional[np.ndarray],
    train_meta: np.ndarray,
    eval_meta: np.ndarray,
    device: torch.device,
    ckpt_dir: Path,
    output_csv: Optional[Path],
    resume: bool = False,
    max_samples: Optional[int] = None,
    metadata_off: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float], Optional[Dict[str, float]]]:
    y_train = train_df["pm25"].to_numpy(dtype=np.float64)
    y_eval = eval_df["pm25"].to_numpy(dtype=np.float64)
    y_eval_bin = eval_df["aqi_bin"].to_numpy(dtype=np.int64)
    pred_bins: Optional[np.ndarray] = None
    meta_off_result: Optional[Tuple[np.ndarray, Optional[np.ndarray], str]] = None

    if model_name == "median":
        preds, pred_bins, notes = _train_predict_median(y_train, len(eval_df), ckpt_dir)
    elif model_name == "ridge_haze":
        assert train_haze is not None and eval_haze is not None
        preds, pred_bins, notes = _train_predict_ridge(
            train_haze, y_train, eval_haze, cfg.train.ridge_alphas, ckpt_dir
        )
    elif model_name == "metadata_mlp":
        preds, pred_bins, notes = _train_predict_metadata_mlp(
            train_meta, y_train, eval_meta, cfg, seed, device, ckpt_dir
        )
    elif model_name in IMAGE_MODEL_NAMES:
        preds, pred_bins, notes, meta_off_result = _train_predict_image_model(
            model_name=model_name,
            experiment_id=experiment_id,
            group_mode=group_mode,
            fold_label=fold_label,
            train_df=train_df,
            eval_df=eval_df,
            train_images_dir=train_images_dir,
            eval_images_dir=eval_images_dir,
            cfg=cfg,
            seed=seed,
            device=device,
            ckpt_dir=ckpt_dir,
            resume=resume,
            max_samples=max_samples,
            metadata_off=metadata_off,
        )
    else:
        raise ValueError(f"Unsupported model={model_name!r}")

    rec_exp_id = (
        f"{experiment_id}_meta_off"
        if (metadata_off and not experiment_id.endswith("_meta_off"))
        else experiment_id
    )
    metrics = evaluate_predictions_and_record(
        y_true_pm25=y_eval,
        y_pred_pm25=preds,
        y_true_bin=y_eval_bin,
        y_pred_bin=pred_bins,
        experiment_id=rec_exp_id,
        model=model_name,
        seed=seed,
        group_mode=group_mode,
        eval_split=eval_split,
        fold=fold_label,
        output_csv=output_csv,
        notes=notes,
    )

    meta_off_metrics: Optional[Dict[str, float]] = None
    if meta_off_result is not None:
        mo_preds, mo_pred_bins, mo_notes = meta_off_result
        meta_off_metrics = evaluate_predictions_and_record(
            y_true_pm25=y_eval,
            y_pred_pm25=mo_preds,
            y_true_bin=y_eval_bin,
            y_pred_bin=mo_pred_bins,
            experiment_id=f"{experiment_id}_meta_off",
            model=model_name,
            seed=seed,
            group_mode=group_mode,
            eval_split=eval_split,
            fold=fold_label,
            output_csv=output_csv,
            notes=mo_notes,
        )

    last_ckpt_path = ckpt_dir / "last.pt"
    if last_ckpt_path.exists():
        last_ckpt_path.unlink()

    return preds, pred_bins, metrics, meta_off_metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PM25Vision models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(ALL_MODEL_NAMES),
    )
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
        help="Fold index 0..4, 'all' for 5-fold CV, or 'full' for full-train + test.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--outputs-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--reg-loss", type=str, default=None, choices=["mse", "huber"])
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--metadata-off", action="store_true", help="Zero metadata embedding m during evaluation.")
    parser.add_argument("--use-amp", action="store_true", help="Enable fp16 autocast on MPS/CUDA.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total train+eval images per fold (e.g., 64 for smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted image training from per-epoch last.pt checkpoint.",
    )
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
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.warmup_epochs is not None:
        overrides["train.warmup_epochs"] = args.warmup_epochs
    if args.lr is not None:
        overrides["train.lr"] = args.lr
    if args.weight_decay is not None:
        overrides["train.weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["data.num_workers"] = args.num_workers
    if args.reg_loss is not None:
        overrides["loss.reg_loss"] = args.reg_loss
    if args.modality_dropout is not None:
        overrides["pfrml.modality_dropout"] = float(args.modality_dropout)
    if args.use_amp:
        overrides["train.use_amp"] = True
    if args.device is not None:
        overrides["train.device"] = args.device

    cfg = load_config(args.config, overrides=overrides)
    model_name = cfg.model.name
    if args.experiment_id:
        experiment_id = args.experiment_id
    elif model_name in FUSION_MODEL_SPECS:
        experiment_id = default_fusion_experiment_id(model_name)
    elif model_name in PFRML_MODEL_SPECS:
        experiment_id = default_pfrml_experiment_id(
            model_name, args.config, reg_input=cfg.pfrml.reg_input
        )
    else:
        experiment_id = model_name

    seed = cfg.seed
    group_mode = cfg.data.group_mode
    if args.output_csv is None:
        if experiment_id.startswith("E5_") or model_name in FUSION_MODEL_SPECS:
            default_csv_name = "fusion_cv_results.csv"
        elif model_name in PFRML_MODEL_SPECS:
            default_csv_name = "pfrml_ablation_cv_results.csv"
        else:
            default_csv_name = "image_models_cv_results.csv"
        output_csv: Optional[Path] = Path(cfg.outputs_dir) / "results" / default_csv_name
    elif args.output_csv == "":
        output_csv = None
    else:
        output_csv = Path(args.output_csv)

    seed_everything(seed)
    device = resolve_device(cfg.train.device)

    train_df, test_df = load_and_clean_metadata(cfg.data.train_csv, cfg.data.test_csv)
    train_meta = encode_metadata_features(
        train_df["latitude"].to_numpy(),
        train_df["longitude"].to_numpy(),
        train_df["captured_at"],
    )
    test_meta = encode_metadata_features(
        test_df["latitude"].to_numpy(),
        test_df["longitude"].to_numpy(),
        test_df["captured_at"],
    )

    train_haze: Optional[np.ndarray] = None
    test_haze: Optional[np.ndarray] = None
    if model_name == "ridge_haze":
        train_haze, test_haze = get_or_compute_haze_features(cfg, train_df, test_df, device)

    ckpt_root = Path(cfg.outputs_dir) / "checkpoints" / experiment_id

    if args.fold == "full":
        full_tr_idx = np.arange(len(train_df))
        full_te_idx = np.arange(len(test_df))
        if args.max_samples is not None:
            n_tr = max(1, int(round(args.max_samples * 0.8)))
            n_te = max(1, args.max_samples - n_tr)
            full_tr_idx = _subsample_indices(full_tr_idx, n_tr)
            full_te_idx = _subsample_indices(full_te_idx, n_te)

        sub_train_df = train_df.iloc[full_tr_idx]
        sub_test_df = test_df.iloc[full_te_idx]
        preds, pred_bins, m_test, m_test_mo = run_single_split(
            model_name=model_name,
            experiment_id=experiment_id,
            cfg=cfg,
            seed=seed,
            group_mode="full_train",
            fold_label="test",
            eval_split="test",
            train_df=sub_train_df,
            eval_df=sub_test_df,
            train_images_dir=Path(cfg.data.train_images_dir),
            eval_images_dir=Path(cfg.data.test_images_dir),
            train_haze=train_haze[full_tr_idx] if train_haze is not None else None,
            eval_haze=test_haze[full_te_idx] if test_haze is not None else None,
            train_meta=train_meta[full_tr_idx],
            eval_meta=test_meta[full_te_idx],
            device=device,
            ckpt_dir=ckpt_root / "full_train",
            output_csv=output_csv,
            resume=args.resume,
            max_samples=args.max_samples,
            metadata_off=bool(args.metadata_off),
        )
        print(f"[{experiment_id} | test (n={len(sub_test_df)})] {format_metrics_oneline(m_test)}")
        if m_test_mo is not None:
            print(
                f"[{experiment_id}_meta_off | test (n={len(sub_test_df)})] "
                f"{format_metrics_oneline(m_test_mo)}"
            )

        _, gt25_mask = compute_test_distance_mask(sub_train_df, sub_test_df, min_km=25.0)
        if np.any(gt25_mask):
            m_gt25 = evaluate_predictions_and_record(
                y_true_pm25=sub_test_df["pm25"].to_numpy(dtype=np.float64)[gt25_mask],
                y_pred_pm25=preds[gt25_mask],
                y_true_bin=sub_test_df["aqi_bin"].to_numpy(dtype=np.int64)[gt25_mask],
                y_pred_bin=pred_bins[gt25_mask] if pred_bins is not None else None,
                experiment_id=experiment_id,
                model=model_name,
                seed=seed,
                group_mode="full_train",
                eval_split="test_gt25km",
                fold="test_gt25km",
                output_csv=output_csv,
                notes="nearest_train_km>25.0",
            )
            print(
                f"[{experiment_id} | test >25km (n={int(gt25_mask.sum())})] "
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

    fold_metrics: List[Dict[str, float]] = []
    fold_meta_off_metrics: List[Dict[str, float]] = []
    for k in fold_indices:
        tr_idx, va_idx = folds[k]
        if args.max_samples is not None:
            n_tr = max(1, int(round(args.max_samples * 0.8)))
            n_va = max(1, args.max_samples - n_tr)
            tr_idx = _subsample_indices(tr_idx, n_tr)
            va_idx = _subsample_indices(va_idx, n_va)
        _, _, m, m_mo = run_single_split(
            model_name=model_name,
            experiment_id=experiment_id,
            cfg=cfg,
            seed=seed,
            group_mode=group_mode,
            fold_label=str(k),
            eval_split="val",
            train_df=train_df.iloc[tr_idx],
            eval_df=train_df.iloc[va_idx],
            train_images_dir=Path(cfg.data.train_images_dir),
            eval_images_dir=Path(cfg.data.train_images_dir),
            train_haze=train_haze[tr_idx] if train_haze is not None else None,
            eval_haze=train_haze[va_idx] if train_haze is not None else None,
            train_meta=train_meta[tr_idx],
            eval_meta=train_meta[va_idx],
            device=device,
            ckpt_dir=ckpt_root / group_mode / f"fold_{k}",
            output_csv=output_csv,
            resume=args.resume,
            max_samples=args.max_samples,
            metadata_off=bool(args.metadata_off),
        )
        fold_metrics.append(m)
        rec_id = (
            f"{experiment_id}_meta_off"
            if (args.metadata_off and not experiment_id.endswith("_meta_off"))
            else experiment_id
        )
        print(f"[{rec_id} | {group_mode} | fold {k}] {format_metrics_oneline(m)}")
        if m_mo is not None:
            fold_meta_off_metrics.append(m_mo)
            print(
                f"[{experiment_id}_meta_off | {group_mode} | fold {k}] "
                f"{format_metrics_oneline(m_mo)}"
            )

    if len(fold_metrics) > 1:
        summary = summarize_fold_metrics(fold_metrics)
        mean_dict = {k: summary[k][0] for k in METRIC_ORDER}
        std_dict = {k: summary[k][1] for k in METRIC_ORDER}
        if output_csv is not None:
            append_result_row(
                output_csv,
                build_result_row(
                    experiment_id=experiment_id,
                    model=model_name,
                    seed=seed,
                    group_mode=group_mode,
                    eval_split="val",
                    fold="cv_mean",
                    n_samples=len(train_df),
                    metrics=mean_dict,
                ),
            )
            append_result_row(
                output_csv,
                build_result_row(
                    experiment_id=experiment_id,
                    model=model_name,
                    seed=seed,
                    group_mode=group_mode,
                    eval_split="val",
                    fold="cv_std",
                    n_samples=len(train_df),
                    metrics=std_dict,
                ),
            )
        print(
            f"[{experiment_id} | {group_mode} | CV mean] {format_metrics_oneline(mean_dict)}"
        )

    if len(fold_meta_off_metrics) > 1:
        mo_summary = summarize_fold_metrics(fold_meta_off_metrics)
        mo_mean_dict = {k: mo_summary[k][0] for k in METRIC_ORDER}
        mo_std_dict = {k: mo_summary[k][1] for k in METRIC_ORDER}
        if output_csv is not None:
            append_result_row(
                output_csv,
                build_result_row(
                    experiment_id=f"{experiment_id}_meta_off",
                    model=model_name,
                    seed=seed,
                    group_mode=group_mode,
                    eval_split="val",
                    fold="cv_mean",
                    n_samples=len(train_df),
                    metrics=mo_mean_dict,
                ),
            )
            append_result_row(
                output_csv,
                build_result_row(
                    experiment_id=f"{experiment_id}_meta_off",
                    model=model_name,
                    seed=seed,
                    group_mode=group_mode,
                    eval_split="val",
                    fold="cv_std",
                    n_samples=len(train_df),
                    metrics=mo_std_dict,
                ),
            )
        print(
            f"[{experiment_id}_meta_off | {group_mode} | CV mean] "
            f"{format_metrics_oneline(mo_mean_dict)}"
        )


if __name__ == "__main__":
    main()
