"""MDNet-T backbone (Paper 2: Wang et al., IEEE TII 2024, Sec. III-A & III-B).

Implements:
- `LayerNorm2d`: channel-wise LayerNorm for `[B, C, H, W]` feature maps.
- `MDBlock`: 3 parallel depthwise (`groups=C`) dilated branches
  `(k=3, d=2, p=2)`, `(k=2, d=2, p=1)`, `(k=1, d=1, p=0)`, summed,
  followed by `GELU`, pointwise `1x1` conv (`C -> C`), `LayerNorm`, and residual.
- `MDNet`: 4-stage hierarchical backbone (`dims=[96, 192, 384, 768]`,
  `depths=[3, 3, 9, 3]` for `MDNet-T`) exposing `forward_stages(x)` for PFRML
  DCM fusion and supporting either `regression` (1 output, AQI/100) or
  `classification` (6-class logits).
- `MobileNetV3MDNeck` (`E3b`): author's own adaptation combining a pretrained
  MobileNetV3-Small backbone with a `1x1` channel reduction (`576 -> 128`) and
  2 `MDBlock` layers (NOT part of Paper 2).
- `compute_mdnet_receptive_field`: theoretical receptive field calculator using
  Paper 2 Eq. (3a)-(3c).
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from torchvision import models as tv_models


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over `[B, C, H, W]` tensors (Core ML friendly)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to [B, H, W, C], normalize over C, and permute back.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MDBlock(nn.Module):
    """Multiscale Dilated Block (Paper 2, Sec. III-A/B, Fig. 1 & Fig. 2).

    Architecture per `SPEC.md` Sec. 3.1:
      1. Three parallel depthwise (`groups=C`) branches:
         - `(k=3, d=2, p=2, s=1)` -> effective kernel `5x5`
         - `(k=2, d=2, p=1, s=1)` -> effective kernel `3x3`
         - `(k=1, d=1, p=0, s=1)` -> effective kernel `1x1`
      2. Element-wise summation of the three branch outputs.
      3. `GELU` activation.
      4. Pointwise `1x1` convolution (`C -> C`).
      5. `LayerNorm` over channel dimension `C`.
      6. Residual shortcut addition with input `x`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.branch_k3 = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, groups=dim
        )
        self.branch_k2 = nn.Conv2d(
            dim, dim, kernel_size=2, stride=1, padding=1, dilation=2, groups=dim
        )
        self.branch_k1 = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=dim
        )
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = LayerNorm2d(dim, eps=1e-6)

    def forward_branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return individual outputs of the 3 parallel depthwise branches."""
        return self.branch_k3(x), self.branch_k2(x), self.branch_k1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3, b2, b1 = self.forward_branches(x)
        y = b3 + b2 + b1
        y = self.act(y)
        y = self.pwconv(y)
        y = self.norm(y)
        return x + y


class MDNet(nn.Module):
    """Multiscale Dilated Network (`MDNet-T` default: depths=[3, 3, 9, 3]).

    Produces 4 hierarchical stage feature maps on `224x224` input:
      - Stage 1: `[B,  96, 56, 56]`
      - Stage 2: `[B, 192, 28, 28]`
      - Stage 3: `[B, 384, 14, 14]`
      - Stage 4: `[B, 768,  7,  7]`
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        head_type: str = "regression",
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("MDNet expects 4 stages for `depths` and `dims`.")
        ht = head_type.lower()
        if ht not in {"regression", "classification"}:
            raise ValueError(
                f"Unsupported head_type={head_type!r}. Expected 'regression' or 'classification'."
            )

        self.depths: Tuple[int, int, int, int] = tuple(int(d) for d in depths)
        self.dims: Tuple[int, int, int, int] = tuple(int(c) for c in dims)
        self.head_type = ht
        self.num_classes = int(num_classes)

        # Stem: 4x4 stride-4 patchify conv + LayerNorm (224 -> 56)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dims[0], kernel_size=4, stride=4),
            LayerNorm2d(self.dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)

        # Inter-stage downsamplers: LayerNorm + 2x2 stride-2 conv
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(self.dims[i], eps=1e-6),
                nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # 4 feature stages of MDBlocks
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[MDBlock(self.dims[i]) for _ in range(self.depths[i])])
                for i in range(4)
            ]
        )

        # Classification / regression head after Global Average Pooling
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        out_dim = 1 if self.head_type == "regression" else self.num_classes
        self.head = nn.Linear(self.dims[-1], out_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale stage feature maps `[c1, c2, c3, c4]`."""
        stage_outputs: List[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stage_outputs.append(x)
        return stage_outputs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled & normalized `[B, 768]` global feature vector."""
        c4 = self.forward_stages(x)[-1]
        pooled = c4.mean(dim=(-2, -1))
        return self.norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict scaled AQI `[B]` (regression) or logits `[B, num_classes]`."""
        feat = self.forward_features(x)
        out = self.head(feat)
        if self.head_type == "regression":
            return out.squeeze(-1)
        return out


def build_mdnet_t(
    head_type: str = "regression",
    num_classes: int = 6,
    in_chans: int = 3,
) -> MDNet:
    """Construct MDNet-T (`depths=[3, 3, 9, 3]`, `dims=[96, 192, 384, 768]`)."""
    return MDNet(
        in_chans=in_chans,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        head_type=head_type,
        num_classes=num_classes,
    )


class MobileNetV3MDNeck(nn.Module):
    """E3b: Pretrained MobileNetV3-Small backbone + Multiscale Dilated (MD) neck.

    NOTE: This architecture is the author's own adaptation of Paper 2's
    (`MDNet`, Wang et al., IEEE TII 2024) multiscale dilated block (`MDBlock`)
    idea onto an ImageNet-pretrained MobileNetV3-Small feature extractor; it is
    NOT part of Paper 2.

    Pipeline on `224x224` RGB images:
      1. Pretrained MobileNetV3-Small `features` -> `[B, 576, 7, 7]`
      2. `1x1` Conv2d reduction (`576 -> 128`) + `LayerNorm2d(128)`
      3. `2` sequential `MDBlock(128)` blocks (each summing the 3 parallel
         dilated depthwise branches `(k=3,d=2)`, `(k=2,d=2)`, `(k=1,d=1)`)
      4. Global average pooling (`[B, 128]`) + `LayerNorm(128)`
      5. Linear regression head (`128 -> 1`, scaled AQI `pm25 / 100.0`)
    """

    def __init__(
        self,
        pretrained: bool = True,
        neck_channels: int = 128,
        num_md_blocks: int = 2,
        head_type: str = "regression",
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        ht = head_type.lower()
        if ht not in {"regression", "classification"}:
            raise ValueError(
                f"Unsupported head_type={head_type!r}. Expected 'regression' or 'classification'."
            )
        self.pretrained = bool(pretrained)
        self.neck_channels = int(neck_channels)
        self.num_md_blocks = int(num_md_blocks)
        self.head_type = ht
        self.num_classes = int(num_classes)

        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
        base = tv_models.mobilenet_v3_small(weights=weights)
        in_channels = int(base.classifier[0].in_features)  # 576
        self.backbone = base.features

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, self.neck_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(self.neck_channels, eps=1e-6),
        )
        self.md_neck = nn.Sequential(
            *[MDBlock(self.neck_channels) for _ in range(self.num_md_blocks)]
        )
        self.norm = nn.LayerNorm(self.neck_channels, eps=1e-6)
        out_dim = 1 if self.head_type == "regression" else self.num_classes
        self.head = nn.Linear(self.neck_channels, out_dim)

        # Initialize only the newly added neck and head; preserve pretrained backbone weights.
        self.reduce.apply(MDNet._init_weights)
        self.md_neck.apply(MDNet._init_weights)
        self.norm.apply(MDNet._init_weights)
        self.head.apply(MDNet._init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract `[B, 128]` pooled & normalized MD-neck features."""
        feat = self.backbone(x)
        feat = self.reduce(feat)
        feat = self.md_neck(feat)
        pooled = feat.mean(dim=(-2, -1))
        return self.norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict scaled AQI `[B]` (regression) or logits `[B, num_classes]`."""
        feat = self.forward_features(x)
        out = self.head(feat)
        if self.head_type == "regression":
            return out.squeeze(-1)
        return out


def build_mobilenet_v3_small_md_neck(
    head_type: str = "regression",
    num_classes: int = 6,
    pretrained: bool = True,
) -> MobileNetV3MDNeck:
    """Construct E3b: pretrained MobileNetV3-S backbone + 128-ch 2-block MD neck.

    Author adaptation of Paper 2's (`MDNet`, Wang et al., IEEE TII 2024)
    `MDBlock` idea; NOT part of Paper 2.
    """
    return MobileNetV3MDNeck(
        pretrained=pretrained,
        neck_channels=128,
        num_md_blocks=2,
        head_type=head_type,
        num_classes=num_classes,
    )


def compute_mdnet_receptive_field(
    depths: Sequence[int] = (3, 3, 9, 3),
    max_branch_k: int = 3,
    max_branch_d: int = 2,
) -> int:
    """Compute theoretical receptive field via Paper 2 Eq. (3a)-(3c).

    Equations:
        (3a) k' = k + (k - 1) * (d - 1)
        (3b) RF_{i+1} = RF_i + (k' - 1) * S_i
        (3c) S_i = prod_{j=1}^{i} Stride_j   (with S_0 = 1, RF_0 = 1)
    """
    k_eff = max_branch_k + (max_branch_k - 1) * (max_branch_d - 1)  # 3 + 2*1 = 5
    rf = 1
    stride_prod = 1

    # Stem: kernel=4, stride=4
    rf = rf + (4 - 1) * stride_prod
    stride_prod *= 4

    for stage_idx, n_blocks in enumerate(depths):
        if stage_idx > 0:
            # Inter-stage downsample: kernel=2, stride=2
            rf = rf + (2 - 1) * stride_prod
            stride_prod *= 2
        for _ in range(int(n_blocks)):
            # MDBlock: max effective kernel k_eff (5x5), stride=1
            rf = rf + (k_eff - 1) * stride_prod
    return rf
