"""In-model atmospheric physics-prior extractor and sigmoid-gated fusion (Toggle #2).

Computes the 4-scalar atmospheric physics descriptor
`[entropy, rms_contrast, dark_channel_transmission t, extinction -ln(t)]`
inside the model using static, differentiable PyTorch ops on pre-normalisation
RGB pixels in `[0, 1]` by reusing `HazeStatisticsExtractor` from
`src.features.haze_features`, projects it to `D`, and fuses it with `F_deep`:

    F_phys = W_p x_phys + b_p
    alpha  = sigmoid(W_g [F_deep; F_phys] + b_g)
    F_out  = alpha * F_deep + (1 - alpha) * F_phys

Citations:
- Paper 7: A. Doha et al., IEEE ICCSC 2026, Sec. 3.3 & 3.5 (Eq. 9) —
  [Entropy, Contrast, Transmission] + sigmoid gate fusion.
- Paper 15: N. Yadav et al., IEEE ICCSAI 2025, Sec. IV / Table II —
  atmospheric extinction coefficient (-ln(t)).
- Paper 2: K. Marzagui et al., IEEE WINCOM 2024, Sec. II-III —
  handcrafted atmospheric haze & visibility priors.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.features.haze_features import HazeStatisticsExtractor


class PhysicsPriorGate(nn.Module):
    """4-scalar physics-prior projection and learned sigmoid-gate fusion.

    When `enabled=False`, returns `f_deep` unchanged (exact baseline output).
    """

    PHYS_DIM: int = 4

    def __init__(
        self,
        dim: int = 224,
        enabled: bool = True,
        patch_size: int = 15,
        omega: float = 0.95,
        input_is_normalized: bool = True,
        use_phys_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.enabled = bool(enabled)
        self.input_is_normalized = bool(input_is_normalized)
        self.use_phys_layernorm = bool(use_phys_layernorm)

        if self.enabled:
            # Keep input_is_normalized=False inside HazeStatisticsExtractor and handle
            # optional unnormalization via static buffers below to avoid dynamic
            # tensor creation at Core ML trace time.
            self.extractor = HazeStatisticsExtractor(
                patch_size=patch_size,
                omega=omega,
                input_is_normalized=False,
            )
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("imagenet_mean", mean, persistent=False)
            self.register_buffer("imagenet_std", std, persistent=False)

            self.phys_norm = nn.LayerNorm(self.PHYS_DIM) if self.use_phys_layernorm else None
            self.phys_proj = nn.Linear(self.PHYS_DIM, self.dim)
            self.gate_proj = nn.Linear(self.dim * 2, self.dim)
        else:
            self.extractor = None
            self.phys_norm = None
            self.phys_proj = None
            self.gate_proj = None

    def _to_pre_norm_rgb(
        self,
        images: Optional[torch.Tensor],
        raw_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if raw_images is not None:
            return torch.clamp(raw_images, 0.0, 1.0)
        if images is None:
            raise ValueError(
                "PhysicsPriorGate requires either `raw_images` (in [0, 1]) or `images`."
            )
        if self.input_is_normalized:
            mean = self.imagenet_mean.to(device=images.device, dtype=images.dtype)
            std = self.imagenet_std.to(device=images.device, dtype=images.dtype)
            return torch.clamp(images * std + mean, 0.0, 1.0)
        return torch.clamp(images, 0.0, 1.0)

    def forward_with_details(
        self,
        f_deep: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return `(f_fused, alpha_gate, phys_scalars)`."""
        if not self.enabled or self.extractor is None:
            return f_deep, None, None

        rgb = self._to_pre_norm_rgb(images=images, raw_images=raw_images)
        phys_scalars = self.extractor.compute_physics_priors(rgb)  # [B, 4]
        phys_in = self.phys_norm(phys_scalars) if self.phys_norm is not None else phys_scalars
        f_phys = self.phys_proj(phys_in)  # [B, D]
        alpha = torch.sigmoid(self.gate_proj(torch.cat([f_deep, f_phys], dim=-1)))  # [B, D]
        f_out = alpha * f_deep + (1.0 - alpha) * f_phys
        return f_out, alpha, phys_scalars

    def forward(
        self,
        f_deep: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse `f_deep [B, D]` with physics priors computed from `[0, 1]` pixels."""
        f_out, _, _ = self.forward_with_details(
            f_deep=f_deep,
            images=images,
            raw_images=raw_images,
        )
        return f_out
