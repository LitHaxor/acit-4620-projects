"""PyTorch atmospheric haze statistics and physics-prior extractor.

Designed for dual use:
1. Step 2 baseline E0 (Ridge regression on per-band haze statistics).
2. Step 4+ in-model physics-prior branch on pre-normalisation RGB pixels [0, 1]
   using purely static, differentiable, Core ML-compatible PyTorch ops.

Citations:
- HSV Saturation & Sobel Gradient Contrast: Paper 1 (PFRML, Wang et al.,
  IEEE TGRS 2026, Sec. III-D Table VIII).
- Dark Channel Prior (DCP) & Transmission: He et al. (TPAMI 2011); Paper 2
  (MDNet, Wang et al., IEEE TII 2024, Sec. II); Paper 7 (Akdim et al.,
  IEEE ICCSC 2026, Sec. 3.3).
- Entropy, RMS Contrast, and Atmospheric Extinction (-ln(t)): Paper 7
  (Sec. 3.3 & 3.5) and Paper 15 (Yadav et al., IEEE ICCSAI 2025, Sec. IV).
- Top (sky/distant) vs. Bottom (ground/near) band contrast cues: Paper 5
  (Chakraborty et al., IJCNN 2024) and Paper 10 (Vahdatpour et al., ICCVW 2025).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import unnormalize_image


class HazeStatisticsExtractor(nn.Module):
    """Differentiable PyTorch extractor for atmospheric haze statistics.

    Expects RGB tensors in [0, 1] of shape [B, 3, H, W] (or [3, H, W]).
    If `input_is_normalized=True`, ImageNet normalization is inverted first.
    """

    BANDS: Tuple[str, ...] = ("full", "top", "bottom")
    CUES: Tuple[str, ...] = ("sat", "grad", "dcp")
    STATS: Tuple[str, ...] = ("mean", "std")

    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        eps: float = 1e-6,
        entropy_bins: int = 32,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd for symmetric padding, got {patch_size}")
        self.patch_size = int(patch_size)
        self.pad = self.patch_size // 2
        self.omega = float(omega)
        self.eps = float(eps)
        self.entropy_bins = int(entropy_bins)
        self.input_is_normalized = bool(input_is_normalized)

        self.feature_names: Tuple[str, ...] = tuple(
            f"{band}_{cue}_{stat}"
            for band in self.BANDS
            for cue in self.CUES
            for stat in self.STATS
        )

        # ITU-R BT.601 luminance weights [1, 3, 1, 1]
        luma_w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma_weights", luma_w, persistent=False)

        # Normalized 3x3 Sobel kernels [2, 1, 3, 3] (divided by 8 so gradient magnitude stays in ~[0, 1])
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        ky = kx.t().contiguous()
        sobel_k = torch.stack([kx, ky], dim=0).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("sobel_kernels", sobel_k, persistent=False)

        # Uniform bin centers in [0, 1] for differentiable soft-histogram entropy
        bin_centers = torch.linspace(0.0, 1.0, self.entropy_bins, dtype=torch.float32)
        self.register_buffer("entropy_centers", bin_centers.view(1, self.entropy_bins, 1, 1), persistent=False)
        self.entropy_sigma = 1.0 / max(self.entropy_bins - 1, 1)

    def _prepare_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] or [3, H, W] tensor, got {tuple(x.shape)}")
        if self.input_is_normalized:
            x = unnormalize_image(x)
        return torch.clamp(x, 0.0, 1.0)

    def compute_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute BT.601 luminance map Y in [0, 1] of shape [B, 1, H, W]."""
        w = self.luma_weights.to(device=rgb.device, dtype=rgb.dtype)
        return (rgb * w).sum(dim=1, keepdim=True)

    def compute_saturation_map(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute HSV saturation map S = (C_max - C_min) / (C_max + eps), shape [B, 1, H, W]."""
        c_max = rgb.amax(dim=1, keepdim=True)
        c_min = rgb.amin(dim=1, keepdim=True)
        return (c_max - c_min) / (c_max + self.eps)

    def compute_gradient_map(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient contrast magnitude map on luminance, shape [B, 1, H, W]."""
        luma = self.compute_luminance(rgb)
        kernels = self.sobel_kernels.to(device=rgb.device, dtype=rgb.dtype)
        # Replicate pad avoids artificial border spikes compared to zero-padding
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        grads = F.conv2d(padded, kernels, padding=0)
        gx = grads[:, 0:1]
        gy = grads[:, 1:2]
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def compute_dark_channel_map(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute patch-wise Dark Channel map via static max-pooling, shape [B, 1, H, W]."""
        min_rgb = rgb.amin(dim=1, keepdim=True)
        # Pad with +1.0 (white) by negating -> pad with -1.0 -> max_pool2d -> negate
        neg = -min_rgb
        padded = F.pad(neg, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        return -F.max_pool2d(padded, kernel_size=self.patch_size, stride=1, padding=0)

    def compute_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute pixel-wise spatial maps for saturation, Sobel gradient, and dark channel."""
        rgb = self._prepare_rgb(x)
        return {
            "sat": self.compute_saturation_map(rgb),
            "grad": self.compute_gradient_map(rgb),
            "dcp": self.compute_dark_channel_map(rgb),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 18 haze statistics: (full, top, bottom) x (sat, grad, dcp) x (mean, std)."""
        maps = self.compute_maps(x)
        h = maps["sat"].shape[2]
        mid = h // 2

        regions = {
            "full": slice(None),
            "top": slice(0, mid),
            "bottom": slice(mid, h),
        }

        feats = []
        for band_name in self.BANDS:
            sl = regions[band_name]
            for cue_name in self.CUES:
                region_map = maps[cue_name][:, :, sl, :]
                mean_val = region_map.mean(dim=(2, 3)).squeeze(-1)
                std_val = region_map.std(dim=(2, 3), unbiased=False).squeeze(-1)
                feats.append(mean_val)
                feats.append(std_val)

        return torch.stack(feats, dim=1)  # [B, 18]

    def compute_physics_priors(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 4 scalar atmospheric physics priors for Step 4+ fusion:
        [entropy, rms_contrast, dark_channel_transmission, extinction (-ln t)].

        Shape: [B, 4]. Pure differentiable static PyTorch ops on pre-norm pixels.
        """
        rgb = self._prepare_rgb(x)
        luma = self.compute_luminance(rgb)
        dcp = self.compute_dark_channel_map(rgb)

        # 1. Differentiable soft-histogram Shannon entropy of luminance
        centers = self.entropy_centers.to(device=rgb.device, dtype=rgb.dtype)
        diff = (luma - centers) / self.entropy_sigma
        weights = torch.exp(-0.5 * diff * diff)
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)
        probs = weights.mean(dim=(2, 3))  # [B, entropy_bins]
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=1)

        # 2. RMS contrast (standard deviation of luminance)
        rms_contrast = luma.std(dim=(2, 3), unbiased=False).squeeze(-1)

        # 3. Mean dark-channel transmission t = clamp(1 - omega * mean(D), 0.05, 1.0)
        dcp_mean = dcp.mean(dim=(2, 3)).squeeze(-1)
        transmission = torch.clamp(1.0 - self.omega * dcp_mean, min=0.05, max=1.0)

        # 4. Atmospheric extinction proxy -ln(t)
        extinction = -torch.log(transmission)

        return torch.stack([entropy, rms_contrast, transmission, extinction], dim=1)
