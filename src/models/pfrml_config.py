"""Configuration dataclass and YAML builder for PFRML and Fusion configs.

Re-exports `PFRMLHeadConfig` from `src.config` and provides `PFRMLFullConfig`,
`load_pfrml_config`, and `build_pfrml_from_config`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from src.config import Config, PFRMLHeadConfig, load_config
from src.models.dcm import PFRMLModel
from src.pfrml_losses import PFRMLMultitaskLoss


@dataclass
class PFRMLFullConfig:
    """Combined base `Config` (`src.config`) and `PFRMLHeadConfig`."""

    base: Config = field(default_factory=Config)
    pfrml: PFRMLHeadConfig = field(default_factory=PFRMLHeadConfig)


def load_pfrml_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> PFRMLFullConfig:
    """Load `configs/pfrml_*.yaml` into `PFRMLFullConfig`."""
    base_cfg = load_config(config_path, overrides=overrides)
    return PFRMLFullConfig(base=base_cfg, pfrml=base_cfg.pfrml)


def build_pfrml_from_config(
    cfg: Union[PFRMLFullConfig, Config],
    train_targets_or_counts: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None,
    backbone_name: Optional[str] = None,
    pretrained: Optional[bool] = None,
) -> Tuple[PFRMLModel, PFRMLMultitaskLoss]:
    """Instantiate `(PFRMLModel, PFRMLMultitaskLoss)` from `PFRMLFullConfig` or `Config`."""
    base_cfg = cfg.base if isinstance(cfg, PFRMLFullConfig) else cfg
    p = cfg.pfrml
    model = PFRMLModel(
        backbone_name=backbone_name if backbone_name is not None else p.backbone,
        pretrained=pretrained if pretrained is not None else p.pretrained,
        tap_mode=p.tap_mode,
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
        haze_patch_size=base_cfg.model.haze_patch_size,
        haze_omega=base_cfg.model.haze_omega,
        physics_input_is_normalized=p.physics_input_is_normalized,
        use_prototypes=p.use_prototypes,
        proto_ema_decay=p.proto_ema_decay,
        proto_tau_init=p.proto_tau_init,
        reg_input=p.reg_input,
        reg_hidden_dims=p.reg_hidden_dims,
        reg_dropout=p.reg_dropout,
    )
    loss_fn = PFRMLMultitaskLoss.from_loss_config(
        base_cfg.loss,
        train_targets_or_counts=train_targets_or_counts,
        num_classes=p.num_classes,
        use_prototypes=p.use_prototypes,
    )
    return model, loss_fn
