"""Multitask loss wrapper for PFRML with Toggle #3 long-tail class weighting and multimodal vision aux loss.

Imports `HybridMultitaskLoss`, `RegressionLoss`, and `compute_class_weights`
from `src.losses` without modifying `src/losses.py`.

Citations:
- Paper 1 (PFRML): Wang et al., IEEE TGRS 2026, Sec. II-D Eq. (6) & Sec. III-C5 —
  L = L_cls + beta * L_reg (beta = 0.5).
- Paper 1 Sec. III-C3, Fig. 7 — R1 w/o prototype ablation (`use_prototypes=False`):
  regression-only loss on pooled chi (`cls_loss = 0`).
- Paper 2 (MDNet): Wang et al., IEEE TII 2024, Eq. (2) —
  Label-smoothing Cross-Entropy (eps = 0.1).
- Toggle #3 (Long-tail class weighting): Paper 7 (Akdim et al., IEEE ICCSC 2026,
  Eq. 11), Paper 16 (Abdelsalam & Sayed, IEEE IMSA 2026, Sec. III-D),
  Paper 14 (Byler et al., WACVW 2025, Sec. 4.1), Paper 6 (Mohan et al.,
  IEEE RAIT 2025, WERCS) — dampened inverse-frequency weights w_k ~ N_k^{-gamma}
  computed strictly inside each training fold.
- Multimodal Fusion (`aux_loss_weight = 0.5`): Auxiliary vision-only
  regression loss on `chi` alone (`L_total = L_main + aux_loss_weight * L_aux`),
  whose gradient flows into the image backbone and never into `MetadataMLP`.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.config import LossConfig
from src.losses import HybridMultitaskLoss, RegressionLoss, compute_class_weights


class PFRMLMultitaskLoss(nn.Module):
    """Wrapper around `src.losses.HybridMultitaskLoss` supporting PFRML outputs, R1, Toggle #3, and Fusion Aux Loss."""

    def __init__(
        self,
        beta: float = 0.5,
        cls_eps: float = 0.1,
        reg_loss_type: str = "mse",
        huber_delta: float = 0.5,
        use_class_weights: bool = False,
        class_weight_gamma: float = 0.5,
        weight_reg_by_class: bool = False,
        num_classes: int = 6,
        class_weights: Optional[torch.Tensor] = None,
        train_targets_or_counts: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None,
        use_prototypes: bool = True,
        aux_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_prototypes = bool(use_prototypes)
        self.use_class_weights = bool(use_class_weights) and self.use_prototypes
        self.class_weight_gamma = float(class_weight_gamma)
        self.num_classes = int(num_classes)
        self.aux_loss_weight = float(aux_loss_weight)

        resolved_weights: Optional[torch.Tensor] = None
        if self.use_class_weights:
            if class_weights is not None:
                resolved_weights = class_weights.float()
            elif train_targets_or_counts is not None:
                resolved_weights = compute_class_weights(
                    targets_or_counts=train_targets_or_counts,
                    num_classes=self.num_classes,
                    gamma=self.class_weight_gamma,
                    normalize=True,
                )
            else:
                raise ValueError(
                    "use_class_weights=True requires either `class_weights` or "
                    "fold-internal `train_targets_or_counts`."
                )

        if self.use_prototypes:
            self.hybrid_loss: Optional[HybridMultitaskLoss] = HybridMultitaskLoss(
                beta=beta,
                cls_eps=cls_eps,
                reg_loss_type=reg_loss_type,
                huber_delta=huber_delta,
                class_weights=resolved_weights,
                weight_reg_by_class=weight_reg_by_class if self.use_class_weights else False,
            )
            self.reg_only_loss: Optional[RegressionLoss] = None
        else:
            self.hybrid_loss = None
            self.reg_only_loss = RegressionLoss(
                loss_type=reg_loss_type,
                huber_delta=huber_delta,
            )

        self.aux_reg_loss = RegressionLoss(
            loss_type=reg_loss_type,
            huber_delta=huber_delta,
        )

    @classmethod
    def from_loss_config(
        cls,
        loss_cfg: LossConfig,
        train_targets_or_counts: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None,
        class_weights: Optional[torch.Tensor] = None,
        num_classes: int = 6,
        use_prototypes: bool = True,
        aux_loss_weight: float = 0.0,
    ) -> "PFRMLMultitaskLoss":
        """Construct `PFRMLMultitaskLoss` directly from `src.config.LossConfig`."""
        return cls(
            beta=loss_cfg.beta,
            cls_eps=loss_cfg.cls_eps,
            reg_loss_type=loss_cfg.reg_loss,
            huber_delta=loss_cfg.huber_delta,
            use_class_weights=loss_cfg.use_class_weights,
            class_weight_gamma=loss_cfg.class_weight_gamma,
            weight_reg_by_class=loss_cfg.weight_reg_by_class,
            num_classes=num_classes,
            class_weights=class_weights,
            train_targets_or_counts=train_targets_or_counts,
            use_prototypes=use_prototypes,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(
        self,
        outputs: Union[Any, Tuple[Optional[torch.Tensor], torch.Tensor]],
        target_bins: torch.Tensor,
        target_aqi_scaled: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute `L = L_cls + beta * L_reg (+ aux_loss_weight * L_aux)`."""
        aux_reg_pred: Optional[torch.Tensor] = None
        if isinstance(outputs, tuple):
            logits, reg_pred = outputs
        elif isinstance(outputs, dict):
            logits, reg_pred = outputs.get("logits"), outputs["reg_pred"]
            aux_reg_pred = outputs.get("aux_reg_pred")
        else:
            logits, reg_pred = getattr(outputs, "logits", None), outputs.reg_pred
            aux_reg_pred = getattr(outputs, "aux_reg_pred", None)

        if not self.use_prototypes:
            assert self.reg_only_loss is not None
            reg_l = self.reg_only_loss(reg_pred, target_aqi_scaled)
            total_l = reg_l
            parts: Dict[str, float] = {
                "total_loss": float(reg_l.detach().item()),
                "cls_loss": 0.0,
                "reg_loss": float(reg_l.detach().item()),
            }
        else:
            assert self.hybrid_loss is not None and logits is not None
            total_l, parts = self.hybrid_loss(
                logits=logits,
                reg_preds=reg_pred,
                target_bins=target_bins,
                target_aqi_scaled=target_aqi_scaled,
            )

        if aux_reg_pred is not None and self.aux_loss_weight > 0.0:
            aux_l = self.aux_reg_loss(aux_reg_pred, target_aqi_scaled)
            total_l = total_l + self.aux_loss_weight * aux_l
            parts["aux_reg_loss"] = float(aux_l.detach().item())
            parts["total_loss"] = float(total_l.detach().item())

        return total_l, parts
