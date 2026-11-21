"""Multimodal Fusion Models (Image + Metadata) with Modality Dropout & Vision Aux Loss.

Implements `PFRMLFusionModel` (`fusion_r1_mobilenetv3_s` and `fusion_full_mobilenetv3_s`)
per `docs/SPEC.md` Sec. 3.4:
- Image branch: `BackboneTapWrapper` (`mobilenetv3_s`, pretrained) + `PFRMLHead`
  (DCM + Light Transformer + optional `AttentionPooling` & `PhysicsPriorGate`) -> `chi` (`[B, 224]`).
- Metadata branch: Existing `MetadataMLP` (`5 -> 64 -> 32`) -> `m` (`[B, 32]`).
- Prototype & classification branch (when `use_prototypes=True`): operates strictly on
  image features `chi` only (never sees `m`).
- Regression MLP (`self.head.reg_head`): operates on `[chi; (c_tilde if use_prototypes); m]`
  (`256 -> 256 -> 1` for `fusion_r1_mobilenetv3_s`, `480 -> 256 -> 1` for `fusion_full_mobilenetv3_s`).
- Modality dropout: during `train()` only, zeroes the 32-D embedding `m` per sample with
  probability `p_drop = modality_dropout` (default `0.3`). Never active in `eval()`.
- Auxiliary vision-only regression head (`self.aux_reg_head`): predicts scaled AQI from `chi`
  alone (`224 -> 256 -> 1`), weighted by `aux_loss_weight = 0.5` in the total training loss.
  Gradients from the auxiliary loss flow into `aux_reg_head` and the image backbone (`chi`)
  but are strictly detached from (independent of) `MetadataMLP`.
"""

from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from src.models.baselines import MetadataMLP
from src.models.dcm import (
    BackboneTapWrapper,
    PFRMLHead,
    PFRMLOutput,
    PrototypeGuidedRegressionHead,
)


FUSION_MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "fusion_r1_mobilenetv3_s": {
        "backbone_name": "mobilenetv3_s",
        "pretrained": True,
        "use_prototypes": False,
        "use_attention_pooling": False,
        "use_physics_gate": False,
        "dcm_fusion": "sum",
    },
    "fusion_full_mobilenetv3_s": {
        "backbone_name": "mobilenetv3_s",
        "pretrained": True,
        "use_prototypes": True,
        "use_attention_pooling": True,
        "use_physics_gate": True,
        "dcm_fusion": "sum",
    },
}
FUSION_MODEL_NAMES: Tuple[str, ...] = tuple(FUSION_MODEL_SPECS.keys())


class PFRMLFusionModel(nn.Module):
    """Multimodal Image + Metadata PFRML Fusion Model."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_s",
        pretrained: bool = True,
        tap_mode: str = "multistage",
        e3b_single_tap: str = "md_neck",
        dim: int = 224,
        num_classes: int = 6,
        dcm_fusion: str = "sum",
        dcm_avg_pool_kernel: int = 1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        transformer_dropout: float = 0.1,
        use_attention_pooling: bool = False,
        pooling_mode: str = "softmax",
        use_physics_gate: bool = False,
        haze_patch_size: int = 15,
        haze_omega: float = 0.95,
        physics_input_is_normalized: bool = True,
        use_prototypes: bool = True,
        proto_ema_decay: float = 0.9,
        proto_tau_init: float = 0.2,
        reg_input: str = "chi_proto",
        reg_hidden_dims: Sequence[int] = (256,),
        reg_dropout: float = 0.1,
        metadata_dim: int = 5,
        metadata_hidden: Tuple[int, int] = (64, 32),
        metadata_dropout: float = 0.0,
        modality_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (0.0 <= float(modality_dropout) <= 1.0):
            raise ValueError(f"modality_dropout must be in [0, 1], got {modality_dropout}")

        self.dim = int(dim)
        self.meta_embed_dim = int(metadata_hidden[-1])
        self.modality_dropout = float(modality_dropout)

        self.backbone = BackboneTapWrapper(
            backbone_name=backbone_name,
            pretrained=pretrained,
            tap_mode=tap_mode,
            e3b_single_tap=e3b_single_tap,
        )
        self.head = PFRMLHead(
            in_channels=self.backbone.out_channels,
            feature_stride=self.backbone.out_stride,
            dim=self.dim,
            num_classes=num_classes,
            dcm_fusion=dcm_fusion,
            dcm_avg_pool_kernel=dcm_avg_pool_kernel,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_dropout=transformer_dropout,
            use_attention_pooling=use_attention_pooling,
            pooling_mode=pooling_mode,
            use_physics_gate=use_physics_gate,
            haze_patch_size=haze_patch_size,
            haze_omega=haze_omega,
            physics_input_is_normalized=physics_input_is_normalized,
            use_prototypes=use_prototypes,
            proto_ema_decay=proto_ema_decay,
            proto_tau_init=proto_tau_init,
            reg_input=reg_input,
            reg_hidden_dims=reg_hidden_dims,
            reg_dropout=reg_dropout,
        )
        # Replace head.reg_head with one that accepts extra_dim = 32 (metadata embedding m)
        self.head.reg_head = PrototypeGuidedRegressionHead(
            dim=self.dim,
            hidden_dims=reg_hidden_dims,
            dropout=reg_dropout,
            use_prototypes=use_prototypes,
            reg_input=reg_input,
            extra_dim=self.meta_embed_dim,
        )

        self.metadata_mlp = MetadataMLP(
            in_dim=metadata_dim,
            hidden_dims=metadata_hidden,
            dropout=metadata_dropout,
        )

        # Auxiliary vision-only regression head on chi alone (detached from metadata branch)
        self.aux_reg_head = PrototypeGuidedRegressionHead(
            dim=self.dim,
            hidden_dims=reg_hidden_dims,
            dropout=reg_dropout,
            use_prototypes=False,
            reg_input="chi",
            extra_dim=0,
        )

    def encode_metadata(
        self,
        metadata: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        metadata_off: bool = False,
        m_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 32-D metadata embedding `m`, applying per-sample modality dropout in train mode."""
        if m_override is not None:
            return m_override.to(device=device, dtype=dtype)
        if metadata_off or metadata is None:
            return torch.zeros(batch_size, self.meta_embed_dim, dtype=dtype, device=device)

        m = self.metadata_mlp.forward_features(metadata)
        if self.training and self.modality_dropout > 0.0:
            if self.modality_dropout >= 1.0:
                m = torch.zeros_like(m)
            else:
                drop_mask = torch.rand(m.shape[0], 1, device=m.device) < self.modality_dropout
                m = m.masked_fill(drop_mask, 0.0)
        return m

    def forward(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
        metadata_off: bool = False,
        m_override: Optional[torch.Tensor] = None,
    ) -> PFRMLOutput:
        feats = self.backbone(images)
        fmap = self.head.neck(feats)  # [B, D, 14, 14]
        chi_deep, attn_weights = self.head.transformer(fmap)  # [B, D], [B, 196]
        chi, alpha_gate, phys_scalars = self.head.physics_gate.forward_with_details(
            f_deep=chi_deep,
            images=images,
            raw_images=raw_images,
        )

        m = self.encode_metadata(
            metadata=metadata,
            batch_size=images.shape[0],
            dtype=chi.dtype,
            device=chi.device,
            metadata_off=metadata_off,
            m_override=m_override,
        )

        if self.head.use_prototypes and self.head.proto_cls is not None:
            logits, probs, dist_sq, p_norm = self.head.proto_cls(chi, targets=targets)
            reg_pred = self.head.reg_head(chi, probs, p_norm, extra_feat=m)
        else:
            logits, probs, dist_sq = None, None, None
            reg_pred = self.head.reg_head(chi, extra_feat=m)

        # Vision-only auxiliary regression prediction on chi alone
        aux_reg_pred = self.aux_reg_head(chi)

        return PFRMLOutput(
            logits=logits,
            probs=probs,
            dist_sq=dist_sq,
            reg_pred=reg_pred,
            chi=chi,
            attn_weights=attn_weights,
            alpha_gate=alpha_gate,
            phys_scalars=phys_scalars,
            aux_reg_pred=aux_reg_pred,
            meta_embed=m,
        )
