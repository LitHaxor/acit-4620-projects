"""Minimal YAML configuration loader with typed dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml


@dataclass
class DataConfig:
    """Configuration for the PM25Vision data pipeline."""

    train_csv: str = "datasets/train/metadata.csv"
    test_csv: str = "datasets/test/metadata.csv"
    train_images_dir: str = "datasets/train/images"
    test_images_dir: str = "datasets/test/images"
    image_size: int = 224
    aqi_scale: float = 100.0
    batch_size: int = 32
    num_workers: int = 4
    n_splits: int = 5
    group_mode: str = "station"
    spatial_cluster_km: float = 25.0
    rotation_deg: float = 5.0
    translate_frac: float = 0.02
    scale_range: Tuple[float, float] = (0.98, 1.02)
    horizontal_flip_prob: float = 0.5


@dataclass
class ModelConfig:
    """Configuration for baseline and spine models."""

    name: str = "metadata_mlp"
    head_type: str = "regression"
    num_classes: int = 6
    metadata_dim: int = 5
    metadata_hidden: Tuple[int, int] = (64, 32)
    metadata_dropout: float = 0.0
    haze_patch_size: int = 15
    haze_omega: float = 0.95


@dataclass
class LossConfig:
    """Configuration for loss functions and optional class weighting."""

    cls_eps: float = 0.1
    reg_loss: str = "mse"
    huber_delta: float = 0.5
    beta: float = 0.5
    use_class_weights: bool = False
    class_weight_gamma: float = 0.5
    weight_reg_by_class: bool = False


@dataclass
class TrainConfig:
    """Configuration for model training."""

    epochs: int = 30
    warmup_epochs: int = 3
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 32
    use_amp: bool = False
    device: str = "auto"
    ridge_alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass
class PFRMLHeadConfig:
    """Configuration for `PFRMLModel`, `PFRMLFusionModel`, and architecture toggles."""

    backbone: str = "mdnet_t"
    pretrained: bool = False
    tap_mode: str = "multistage"          # "multistage" or "single"
    e3b_single_tap: str = "md_neck"       # "md_neck" (C=128, stride=32) or "stride16" (C=48)
    dim: int = 224
    num_classes: int = 6
    dcm_fusion: str = "sum"               # "sum" (SPEC Table I) or "concat"
    dcm_avg_pool_kernel: int = 1
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.1
    use_attention_pooling: bool = False   # Toggle #1
    pooling_mode: str = "softmax"         # "softmax" (Paper 16) or "gwap" (Paper 8)
    use_physics_gate: bool = False        # Toggle #2
    physics_input_is_normalized: bool = True
    use_prototypes: bool = True           # R1 ablation toggle (Paper 1 Sec. III-C3, Fig. 7)
    proto_ema_decay: float = 0.9
    proto_tau_init: float = 0.2
    reg_input: str = "chi_proto"          # "chi_proto" (Fig. 7 default) or "chi" (Eq. 5 literal)
    reg_hidden_dims: Tuple[int, ...] = (256,)
    reg_dropout: float = 0.1
    modality_dropout: float = 0.3         # Fusion: per-sample prob of zeroing metadata m in train
    aux_loss_weight: float = 0.5          # Fusion: weight on vision-only aux regression loss on chi


@dataclass
class Config:
    """Top-level configuration container."""

    seed: int = 42
    outputs_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pfrml: PFRMLHeadConfig = field(default_factory=PFRMLHeadConfig)


def load_config(
    config_path: Union[str, Path] = "configs/default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration from a YAML file and apply optional dictionary overrides."""
    path = Path(config_path)
    raw: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw = loaded

    if overrides:
        for k, v in overrides.items():
            if "." in k:
                sec, subk = k.split(".", 1)
                raw.setdefault(sec, {})[subk] = v
            else:
                raw[k] = v

    data_dict = dict(raw.get("data", {}))
    if "scale_range" in data_dict and isinstance(data_dict["scale_range"], list):
        data_dict["scale_range"] = tuple(data_dict["scale_range"])

    model_dict = dict(raw.get("model", {}))
    if "metadata_hidden" in model_dict and isinstance(model_dict["metadata_hidden"], list):
        model_dict["metadata_hidden"] = tuple(model_dict["metadata_hidden"])

    loss_dict = dict(raw.get("loss", {}))

    train_dict = dict(raw.get("train", {}))
    if "ridge_alphas" in train_dict and isinstance(train_dict["ridge_alphas"], list):
        train_dict["ridge_alphas"] = tuple(float(x) for x in train_dict["ridge_alphas"])

    pfrml_dict = dict(raw.get("pfrml", {}))
    if "reg_hidden_dims" in pfrml_dict and isinstance(pfrml_dict["reg_hidden_dims"], list):
        pfrml_dict["reg_hidden_dims"] = tuple(int(x) for x in pfrml_dict["reg_hidden_dims"])

    return Config(
        seed=int(raw.get("seed", 42)),
        outputs_dir=str(raw.get("outputs_dir", "outputs")),
        data=DataConfig(**data_dict),
        model=ModelConfig(**model_dict),
        loss=LossConfig(**loss_dict),
        train=TrainConfig(**train_dict),
        pfrml=PFRMLHeadConfig(**pfrml_dict),
    )
