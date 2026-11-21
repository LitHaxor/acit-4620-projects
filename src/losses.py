"""Loss functions and long-tail class weighting for PM25Vision.

Implements:
- Label-smoothing Cross-Entropy (eps=0.1; Paper 2 / MDNet, Eq. 2).
- Continuous regression losses on scaled AQI (pm25 / 100.0):
  - MSE loss (Paper 1 / PFRML, Eq. 5).
  - Huber / Smooth L1 loss (Paper 14: Byler et al., WACVW 2025, Sec. 4.1).
- Hybrid multitask loss L = L_cls + beta * L_reg (beta=0.5; Paper 1 / PFRML,
  Sec. II-D Eq. 6 & Sec. III-C5).
- Optional dampened inverse-frequency class weights w_k ~ N_k^-gamma
  (Paper 14 Sec. 4.1; Paper 7 Akdim et al. 2026 Eq. 11; Paper 16; Paper 6).
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    targets_or_counts: Union[np.ndarray, torch.Tensor, Sequence[int]],
    num_classes: int = 6,
    gamma: float = 0.5,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute dampened inverse-frequency class weights w_k ~ N_k^{-gamma}.

    Args:
        targets_or_counts: Either 1D integer bin labels in {0..num_classes-1}
            from the training fold, or a 1D array/tensor of per-class counts of
            length `num_classes`.
        num_classes: Number of discrete AQI classes (default 6).
        gamma: Dampening exponent in [0, 1] (0 -> uniform, 0.5 -> sqrt inverse,
            1.0 -> pure inverse frequency).
        normalize: If True, normalize weights so their mean over present classes
            equals 1.0, preserving loss scale.
        dtype: Output tensor dtype.
    """
    arr = np.asarray(
        targets_or_counts.detach().cpu().numpy()
        if isinstance(targets_or_counts, torch.Tensor)
        else targets_or_counts
    ).ravel()

    if arr.shape == (num_classes,) and np.all(arr >= 0) and (
        not np.all(arr < num_classes) or np.sum(arr) > num_classes
    ):
        counts = arr.astype(np.float64)
    else:
        counts = np.bincount(arr.astype(np.int64), minlength=num_classes)[:num_classes].astype(
            np.float64
        )

    safe_counts = np.maximum(counts, 1.0)
    weights = np.power(safe_counts, -float(gamma))
    if np.any(counts == 0):
        weights[counts == 0] = 0.0

    if normalize:
        present = weights[weights > 0]
        if len(present) > 0:
            weights = weights / np.mean(present)

    return torch.tensor(weights, dtype=dtype)


class LabelSmoothingCrossEntropy(nn.Module):
    """6-class cross-entropy loss with label smoothing and optional class weights."""

    def __init__(
        self,
        eps: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= eps < 1.0):
            raise ValueError(f"Label smoothing eps must be in [0, 1), got {eps}")
        self.eps = float(eps)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = (
            self.class_weights.to(device=logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=weight,
            label_smoothing=self.eps,
        )


class RegressionLoss(nn.Module):
    """Continuous AQI/100 regression loss (MSE or Huber) with optional class weighting."""

    def __init__(
        self,
        loss_type: str = "mse",
        huber_delta: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        lt = loss_type.lower()
        if lt not in {"mse", "huber"}:
            raise ValueError(
                f"Unsupported regression loss_type={loss_type!r}. Expected 'mse' or 'huber'."
            )
        self.loss_type = lt
        self.huber_delta = float(huber_delta)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_bins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = preds.view(-1)
        t = targets.view(-1).to(dtype=p.dtype)
        if self.loss_type == "mse":
            per_sample = F.mse_loss(p, t, reduction="none")
        else:
            per_sample = F.huber_loss(p, t, delta=self.huber_delta, reduction="none")

        if self.class_weights is not None and target_bins is not None:
            w = self.class_weights.to(device=p.device, dtype=p.dtype)[
                target_bins.long().view(-1)
            ]
            return (per_sample * w).mean()
        return per_sample.mean()


class HybridMultitaskLoss(nn.Module):
    """PFRML hybrid loss: L = L_cls + beta * L_reg (Paper 1 Eq. 6, beta=0.5)."""

    def __init__(
        self,
        beta: float = 0.5,
        cls_eps: float = 0.1,
        reg_loss_type: str = "mse",
        huber_delta: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        weight_reg_by_class: bool = False,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.cls_loss_fn = LabelSmoothingCrossEntropy(
            eps=cls_eps,
            class_weights=class_weights,
        )
        self.reg_loss_fn = RegressionLoss(
            loss_type=reg_loss_type,
            huber_delta=huber_delta,
            class_weights=class_weights if weight_reg_by_class else None,
        )

    def forward(
        self,
        logits: torch.Tensor,
        reg_preds: torch.Tensor,
        target_bins: torch.Tensor,
        target_aqi_scaled: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_cls = self.cls_loss_fn(logits, target_bins)
        l_reg = self.reg_loss_fn(reg_preds, target_aqi_scaled, target_bins=target_bins)
        total = l_cls + self.beta * l_reg
        stats = {
            "cls_loss": float(l_cls.detach().item()),
            "reg_loss": float(l_reg.detach().item()),
            "total_loss": float(total.detach().item()),
        }
        return total, stats
