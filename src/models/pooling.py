"""Learnable spatial / token attention pooling over 14x14 tokens (Toggle #1).

Replaces uniform Global Average Pooling (GAP) over the 14x14 = 196 spatial tokens
from the DCM/Transformer to emphasize sky/haze regions and suppress foreground
clutter.

Citations:
- Paper 16 (SkyFuseNet): M. S. Abdelsalam & G. I. Sayed, IEEE IMSA 2026,
  Sec. III / Eq. (2) & Fig. 7 — softmax attention pooling over spatial tokens:
  f = sum_i softmax(W_A t_i) t_i.
- Paper 8 (DCEL-PM2.5): C.-Y. Lee & L. M. Manova, IEEE Sensors J. 2025,
  Eq. (7) — Global Weighted Average Pooling (GWAP) using a sigmoid spatial mask:
  GWAP(d) = sum_{x,y} A_m(x,y) F(x,y,d) / (sum_{x,y} A_m(x,y) + eps).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Single-layer spatial attention gate over sequence tokens [B, N, D].

    When `enabled=False`, falls back to exact uniform Global Average Pooling
    (`tokens.mean(dim=1)`), matching the PFRML baseline.
    """

    def __init__(
        self,
        dim: int = 224,
        enabled: bool = True,
        mode: str = "softmax",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        m = mode.lower()
        if m not in {"softmax", "gwap"}:
            raise ValueError(f"Unsupported pooling mode={mode!r}. Expected 'softmax' or 'gwap'.")
        self.dim = int(dim)
        self.enabled = bool(enabled)
        self.mode = m
        self.eps = float(eps)

        # Single-layer spatial attention gate (Paper 16 Eq. 2 / Paper 8 Eq. 7)
        self.gate = nn.Linear(self.dim, 1) if self.enabled else None

    def forward_with_weights(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool `[B, N, D]` tokens to `[B, D]` and return `[B, N]` spatial weights."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected tokens of shape [B, N, {self.dim}], got {tuple(tokens.shape)}"
            )
        if not self.enabled or self.gate is None:
            n_tokens = tokens.shape[1]
            pooled = tokens.mean(dim=1)
            uniform = tokens.new_full((tokens.shape[0], n_tokens), 1.0 / float(n_tokens))
            return pooled, uniform

        scores = self.gate(tokens)  # [B, N, 1]
        if self.mode == "softmax":
            weights = F.softmax(scores, dim=1)  # [B, N, 1]
        else:
            mask = torch.sigmoid(scores)  # [B, N, 1]
            weights = mask / (mask.sum(dim=1, keepdim=True) + self.eps)

        pooled = (weights * tokens).sum(dim=1)  # [B, D]
        return pooled, weights.squeeze(-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool `[B, N, D]` tokens to `[B, D]`."""
        pooled, _ = self.forward_with_weights(tokens)
        return pooled
