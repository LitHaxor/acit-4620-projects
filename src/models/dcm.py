"""Dimension Consistency Module (DCM), Light Transformer, Prototype Heads & Backbone Interface.

Implements the PFRML spine (Paper 1: Wang et al., IEEE TGRS 2026)
adapted per `docs/SPEC.md` Sec. 3.2 & 3.3:
- `DimensionConsistencyModule`: Projects 4 hierarchical feature maps
  `[C1, C2, C3, C4]` (strides `[4, 8, 16, 32]`) to `D=224` via `1x1` conv,
  spatially aligns them to `14x14` with static ops (`MaxPool2d(4,4)`,
  `MaxPool2d(2,2)`, `Identity`, `Upsample(scale_factor=2, mode='bilinear')`),
  applies `LayerNorm2d` + `AvgPool2d`, and fuses via element-wise summation
  (Paper 1 Table I `DCM Stage 5`, with config toggle for `concat`).
- `SingleFeatureAdapter`: Adapts a single `(B, C, H, W)` feature map (e.g.
  stride-16 `14x14` tokens or E3b's stride-32 `7x7` MD-neck map) to
  `[B, 224, 14, 14]`.
- `LightTransformerEncoder`: Flattens `14x14` -> `196` tokens + prepends
  `[CLS]` (`197 x 224`), adds factored 2D learnable positional encoding, passes
  through a 2-layer `nn.TransformerEncoder` (`enable_nested_tensor=False` for
  Core ML compatibility), and pools the 196 spatial tokens via uniform GAP or
  toggled `AttentionPooling` (Toggle #1).
- `PhysicsPriorGate` (Toggle #2, imported from `src.models.physics_gate`).
- `PrototypeClassificationHead`: `K=6` class prototypes stored as registered
  buffers (checkpoint-saved, gradient-free), class-centroid initialization,
  eval-safe `no_grad` EMA updates, L2-normalized features, squared Euclidean
  distance computed via `matmul` (`2 - 2 cos_sim`, avoiding `torch.cdist`),
  and learnable positive temperature `tau`.
- `PrototypeGuidedRegressionHead`: Single regression MLP `h(.)` supporting:
  1. `reg_input='chi_proto'` (default, PFRML Sec. III-C3 Fig. 7 `"w/ prototype
     (prototype-weighted feature)"`): concatenates `[chi_i; c_tilde_i]` where
     `c_tilde_i = sum_k p_k c'_k` (`2*D = 448`).
  2. `reg_input='chi'` (PFRML Sec. II-D Eq. 5 & Fig. 5 literal variant,
     `y_hat_i = h(chi_i)`): regression MLP sees only `chi_i` (`D = 224`) while
     the prototype/classification branch remains active and trained by
     cross-entropy + EMA prototype updates.
  3. `use_prototypes=False` (R1 ablation, PFRML Sec. III-C3 Fig. 7 `"w/o prototype"`):
     omits prototypes and feeds `chi_i` (`D = 224`) directly to `h(chi_i)`.
  4. Optional `extra_dim` (e.g., 32-D `MetadataMLP` embedding `m` in multimodal
     multimodal fusion, yielding `[chi_i; (c_tilde_i); m]`).
- `BackboneTapWrapper`, `PFRMLHead`, and `PFRMLModel`: Backbone-agnostic
  interface supporting `mobilenetv3_s`, `resnet18`, `mdnet_t`, and `e3b`.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from src.models.mdnet import LayerNorm2d, MobileNetV3MDNeck, MDNet, build_mdnet_t
from src.models.physics_gate import PhysicsPriorGate
from src.models.pooling import AttentionPooling


BACKBONE_SPECS: Dict[str, Dict[str, object]] = {
    "mobilenetv3_s": {
        "stage_channels": (16, 24, 48, 576),
        "single_tap_channel": 48,
        "single_tap_stride": 16,
    },
    "resnet18": {
        "stage_channels": (64, 128, 256, 512),
        "single_tap_channel": 256,
        "single_tap_stride": 16,
    },
    "mdnet_t": {
        "stage_channels": (96, 192, 384, 768),
        "single_tap_channel": 384,
        "single_tap_stride": 16,
    },
    "e3b": {
        "stage_channels": (16, 24, 48, 128),
        # Post-MD-neck feature map is [B, 128, 7, 7] (stride 32).
        # Set e3b_single_tap="stride16" to tap MobileNetV3 features[8] (C=48, stride 16)
        # before the MD neck if strict stride-16 tapping is desired.
        "single_tap_channel": 128,
        "single_tap_stride": 32,
        "stride16_channel": 48,
    },
}


def _make_spatial_aligner(stride: int, target_stride: int = 16) -> nn.Module:
    """Build static spatial aligner mapping feature map stride -> target_stride (14x14 on 224x224)."""
    if stride == target_stride:
        return nn.Identity()
    if stride < target_stride:
        if target_stride % stride != 0:
            raise ValueError(f"Unsupported stride={stride} for target_stride={target_stride}")
        factor = target_stride // stride
        return nn.MaxPool2d(kernel_size=factor, stride=factor)
    if stride % target_stride != 0:
        raise ValueError(f"Unsupported stride={stride} for target_stride={target_stride}")
    factor = stride // target_stride
    return nn.Upsample(scale_factor=float(factor), mode="bilinear", align_corners=False)


class DimensionConsistencyModule(nn.Module):
    """PFRML Dimension Consistency Module (Paper 1 Sec. II-B, Fig. 3, Table I).

    Takes 4 hierarchical feature maps `[C1, C2, C3, C4]` at strides
    `[4, 8, 16, 32]` (`[56x56, 28x28, 14x14, 7x7]` for `224x224` input),
    projects each to `D=224` channels via `1x1` conv, aligns all to `14x14`,
    applies `LayerNorm2d(D)` + static `AvgPool2d`, and fuses via element-wise
    sum (`fusion='sum'`, default per Table I) or concatenation + `1x1` conv
    (`fusion='concat'`).
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        out_channels: int = 224,
        fusion: str = "sum",
        stage_strides: Sequence[int] = (4, 8, 16, 32),
        avg_pool_kernel: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4 or len(stage_strides) != 4:
            raise ValueError("DCM expects 4 stage channel counts and 4 stage strides.")
        f_mode = fusion.lower()
        if f_mode not in {"sum", "concat"}:
            raise ValueError(f"Unsupported DCM fusion={fusion!r}. Expected 'sum' or 'concat'.")
        if avg_pool_kernel not in {1, 3}:
            raise ValueError(f"avg_pool_kernel must be 1 or 3, got {avg_pool_kernel}")

        self.in_channels = tuple(int(c) for c in in_channels)
        self.out_channels = int(out_channels)
        self.fusion = f_mode
        self.stage_strides = tuple(int(s) for s in stage_strides)
        self.avg_pool_kernel = int(avg_pool_kernel)
        pad = self.avg_pool_kernel // 2

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c_in, self.out_channels, kernel_size=1, stride=1, padding=0),
                    _make_spatial_aligner(stride=stride, target_stride=16),
                    LayerNorm2d(self.out_channels, eps=1e-6),
                    nn.AvgPool2d(kernel_size=self.avg_pool_kernel, stride=1, padding=pad),
                )
                for c_in, stride in zip(self.in_channels, self.stage_strides)
            ]
        )

        if self.fusion == "concat":
            self.concat_proj = nn.Sequential(
                nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0),
                LayerNorm2d(self.out_channels, eps=1e-6),
            )
        else:
            self.concat_proj = None

    def forward(self, stages: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse 4 stage feature maps into `[B, out_channels, 14, 14]`."""
        if len(stages) != 4:
            raise ValueError(f"DCM expects 4 stage tensors, got {len(stages)}")
        aligned = [branch(feat) for branch, feat in zip(self.branches, stages)]
        if self.fusion == "sum":
            return aligned[0] + aligned[1] + aligned[2] + aligned[3]
        cat = torch.cat(aligned, dim=1)
        assert self.concat_proj is not None
        return self.concat_proj(cat)


class SingleFeatureAdapter(nn.Module):
    """Adapt a single `(B, C, H, W)` feature map to `[B, 224, 14, 14]`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 224,
        stride: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            _make_spatial_aligner(stride=self.stride, target_stride=16),
            LayerNorm2d(self.out_channels, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LightTransformerEncoder(nn.Module):
    """Lightweight Transformer Encoder over 14x14 spatial tokens + [CLS].

    Per `SPEC.md` Sec. 3.2:
    - Flattens `[B, D, 14, 14]` -> `[B, 196, D]`
    - Prepends `[CLS]` token -> `[B, 197, D]`
    - Adds factored 2D learnable positional encoding (`row` + `col` + `cls`)
    - Runs `nn.TransformerEncoder(..., enable_nested_tensor=False)` (`dim=224`,
      default 2 layers)
    - Pools the 196 spatial tokens via `AttentionPooling` (uniform GAP when
      `use_attention_pooling=False`, learnable spatial attention when `True`)
    """

    def __init__(
        self,
        dim: int = 224,
        grid_size: int = 14,
        num_layers: int = 2,
        nhead: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_attention_pooling: bool = False,
        pooling_mode: str = "softmax",
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for factored 2D positional encoding, got {dim}")
        if dim % nhead != 0:
            raise ValueError(f"dim={dim} must be divisible by nhead={nhead}")

        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.num_tokens = self.grid_size * self.grid_size
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        dim_feedforward = int(round(self.dim * float(mlp_ratio)))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        half_dim = self.dim // 2
        self.row_embed = nn.Parameter(torch.zeros(self.grid_size, half_dim))
        self.col_embed = nn.Parameter(torch.zeros(self.grid_size, half_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.pooling = AttentionPooling(
            dim=self.dim,
            enabled=use_attention_pooling,
            mode=pooling_mode,
        )

    def _build_pos_embed(self) -> torch.Tensor:
        # Static [1, 197, D] positional encoding from 2D row/col embeddings.
        row = self.row_embed.unsqueeze(1).expand(self.grid_size, self.grid_size, -1)
        col = self.col_embed.unsqueeze(0).expand(self.grid_size, self.grid_size, -1)
        pos_2d = torch.cat([row, col], dim=-1).reshape(1, self.num_tokens, self.dim)
        return torch.cat([self.cls_pos_embed, pos_2d], dim=1)

    def forward(self, fmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode `[B, D, 14, 14]` -> `(chi [B, D], attn_weights [B, 196])`."""
        b, c, h, w = fmap.shape
        if c != self.dim or h != self.grid_size or w != self.grid_size:
            raise ValueError(
                f"Expected [{b}, {self.dim}, {self.grid_size}, {self.grid_size}], "
                f"got {tuple(fmap.shape)}"
            )
        spatial_tokens = fmap.flatten(2).transpose(1, 2)  # [B, 196, D]
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, spatial_tokens], dim=1)  # [B, 197, D]
        tokens = tokens + self._build_pos_embed()
        encoded = self.norm(self.encoder(tokens))
        spatial_out = encoded[:, 1:, :]  # [B, 196, D]
        chi, attn_weights = self.pooling.forward_with_weights(spatial_out)
        return chi, attn_weights


class PrototypeClassificationHead(nn.Module):
    """Prototype-based 6-class AQI head (Paper 1 Sec. II-C, Eq. 1-3).

    Features:
    - Prototypes `[K, D]` stored as registered buffers (`persistent=True`) so
      they save/load with checkpoints and receive zero gradients; initialized
      orthogonally so classes start well separated on the unit hypersphere.
    - Class-centroid initialization via `init_prototypes(features, labels)`.
    - Eval-safe `torch.no_grad()` EMA prototype updates during `train()` when
      `targets` are provided.
    - L2-normalized feature & prototype representations; squared Euclidean
      distance computed via `matmul` (`2 - 2 * cos_sim`, avoiding `torch.cdist`).
    - Learnable positive temperature `tau` via `log_tau` parameter.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dim: int = 224,
        ema_decay: float = 0.9,
        tau_init: float = 0.2,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if tau_init <= 0.0:
            raise ValueError(f"tau_init must be > 0, got {tau_init}")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

        init_protos = torch.empty(self.num_classes, self.dim)
        if self.num_classes <= self.dim:
            nn.init.orthogonal_(init_protos)
        else:
            nn.init.normal_(init_protos)
        init_protos = self._l2_normalize(init_protos)

        self.register_buffer("prototypes", init_protos)
        self.register_buffer("prototypes_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("class_counts", torch.zeros(self.num_classes, dtype=torch.float32))

        self.log_tau = nn.Parameter(torch.tensor(math.log(float(tau_init)), dtype=torch.float32))

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Static Core ML-friendly L2 normalization using rsqrt.
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + self.eps)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-3, max=100.0)

    @torch.no_grad()
    def init_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize prototypes from class centroids of `(features, labels)`."""
        x_norm = self._l2_normalize(features.detach())
        lbl = labels.detach().long().view(-1)
        for k in range(self.num_classes):
            mask = lbl == k
            if mask.any():
                centroid = self._l2_normalize(x_norm[mask].mean(dim=0, keepdim=True)).squeeze(0)
                self.prototypes[k].copy_(
                    centroid.to(dtype=self.prototypes.dtype, device=self.prototypes.device)
                )
                self.class_counts[k] = float(mask.sum().item())
        self.prototypes_initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, x_norm: torch.Tensor, targets: torch.Tensor) -> None:
        """Update prototypes in `no_grad` with batch class centroids (train-only).

        Always blends via `ema_decay` starting from orthogonal unit prototypes
        so early untrained batches do not collapse all prototypes together.
        """
        lbl = targets.long().view(-1)
        for k in range(self.num_classes):
            mask = lbl == k
            if mask.any():
                centroid = self._l2_normalize(x_norm[mask].mean(dim=0, keepdim=True)).squeeze(0)
                centroid = centroid.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
                blended = self.ema_decay * self.prototypes[k] + (1.0 - self.ema_decay) * centroid
                self.prototypes[k].copy_(self._l2_normalize(blended.unsqueeze(0)).squeeze(0))
                self.class_counts[k] += float(mask.sum().item())
        if not bool(self.prototypes_initialized.item()):
            self.prototypes_initialized.fill_(True)

    def forward(
        self,
        chi: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(logits [B, K], probs [B, K], dist_sq [B, K], p_norm [K, D])`."""
        x_norm = self._l2_normalize(chi)
        # Clone detached normalized prototypes before optional EMA update so autograd
        # never observes an in-place buffer version bump on a tensor used in forward.
        p_norm = self._l2_normalize(self.prototypes.detach()).clone()

        cos_sim = torch.matmul(x_norm, p_norm.t())  # [B, K]
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0)
        logits = -dist_sq / self.tau
        probs = F.softmax(logits, dim=-1)

        if self.training and targets is not None:
            self._ema_update(x_norm.detach(), targets.detach())

        return logits, probs, dist_sq, p_norm


class PrototypeGuidedRegressionHead(nn.Module):
    """Single regression MLP over `[chi_i; c_tilde_i]` or `chi_i` (with optional metadata `m`).

    Citations (Paper 1: Wang et al., IEEE TGRS 2026, PFRML):
    - **PFRML Sec. III-C3, Fig. 7 (`reg_input='chi_proto'`, default when `use_prototypes=True`):**
      Explicitly labels the prototype-constrained representation `"w/ prototype
      (prototype-weighted feature)"`. Computes `c_tilde_i = sum_k p(y=k|chi_i) c'_k`
      and feeds `[chi_i; c_tilde_i]` (`2*D = 448`) into one single regression MLP `h(.)`.
    - **PFRML Sec. II-D, Eq. (5) & Fig. 5 (`reg_input='chi'`, literal Eq. 5 variant):**
      Writes `y_hat_i = h(chi_i)` where the regression MLP `h(.)` sees only `chi_i`
      (`D = 224`), while the prototype/classification branch still runs in parallel
      (trained by cross-entropy `L_cls` and EMA prototype updates).
    - **PFRML Sec. III-C3, Fig. 7 (`use_prototypes=False`, R1 `"w/o prototype"`):**
      Omits prototypes entirely (`cls_loss = 0`) and feeds `chi_i` (`D = 224`) to `h(chi_i)`.
    - **Multimodal Fusion (`extra_dim > 0`):**
      Appends metadata embedding `m` (`32`-D) to form `[chi_i; (c_tilde_i); m]`.
    """

    def __init__(
        self,
        dim: int = 224,
        hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.1,
        use_prototypes: bool = True,
        reg_input: str = "chi_proto",
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        if reg_input not in {"chi_proto", "chi"}:
            raise ValueError(
                f"Unsupported reg_input={reg_input!r}. Expected 'chi_proto' or 'chi'."
            )
        self.dim = int(dim)
        self.use_prototypes = bool(use_prototypes)
        self.reg_input = str(reg_input)
        self.extra_dim = int(extra_dim)
        self.uses_proto_in_reg = self.use_prototypes and (self.reg_input == "chi_proto")

        base_dim = self.dim * 2 if self.uses_proto_in_reg else self.dim
        in_dim = base_dim + self.extra_dim
        self.in_dim = in_dim

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        chi: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        prototypes_norm: Optional[torch.Tensor] = None,
        extra_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict scaled AQI `[B]` from `chi [B, D]` (plus `c_tilde` and/or `extra_feat`)."""
        parts: List[torch.Tensor] = [chi]
        if self.uses_proto_in_reg:
            if probs is None or prototypes_norm is None:
                raise ValueError(
                    "use_prototypes=True with reg_input='chi_proto' requires `probs` and `prototypes_norm`."
                )
            c_tilde = torch.matmul(probs, prototypes_norm)  # [B, D]
            parts.append(c_tilde)
        if self.extra_dim > 0:
            if extra_feat is None:
                raise ValueError(f"extra_dim={self.extra_dim} requires `extra_feat` tensor.")
            parts.append(extra_feat)
        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.mlp(fused).squeeze(-1)

    @torch.no_grad()
    def predict_prototype_mixture_variance(
        self,
        chi: torch.Tensor,
        probs: torch.Tensor,
        prototypes_norm: torch.Tensor,
        extra_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form predictive variance diagnostic (`SPEC.md` Sec. 1, Paper 17).

        Evaluates the single regression MLP `h([chi_i; c'_k; (m)])` for each
        prototype `k=1..K`, returning `(y_mean [B], var [B], y_per_proto [B, K])`.
        When `uses_proto_in_reg=False` (R1 or `reg_input='chi'`), the regression
        MLP does not depend on `c'_k`, so variance across prototypes is identically zero.
        """
        b, k = probs.shape
        if not self.uses_proto_in_reg:
            y_pred = self.forward(chi, probs, prototypes_norm, extra_feat=extra_feat)
            zeros_var = torch.zeros_like(y_pred)
            return y_pred, zeros_var, y_pred.unsqueeze(1).expand(b, k)

        per_proto_preds = []
        for idx in range(k):
            p_k = prototypes_norm[idx : idx + 1].expand(b, -1)
            parts = [chi, p_k]
            if self.extra_dim > 0:
                if extra_feat is None:
                    raise ValueError(f"extra_dim={self.extra_dim} requires `extra_feat` tensor.")
                parts.append(extra_feat)
            y_k = self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
            per_proto_preds.append(y_k)
        y_stack = torch.stack(per_proto_preds, dim=1)  # [B, K]
        y_mean = (probs * y_stack).sum(dim=1)
        var = (probs * (y_stack - y_mean.unsqueeze(1)) ** 2).sum(dim=1)
        return y_mean, var, y_stack


@dataclass
class PFRMLOutput:
    """Structured output of `PFRMLHead`, `PFRMLModel`, and `PFRMLFusionModel`."""

    logits: Optional[torch.Tensor]
    probs: Optional[torch.Tensor]
    dist_sq: Optional[torch.Tensor]
    reg_pred: torch.Tensor
    chi: torch.Tensor
    attn_weights: torch.Tensor
    alpha_gate: Optional[torch.Tensor] = None
    phys_scalars: Optional[torch.Tensor] = None
    aux_reg_pred: Optional[torch.Tensor] = None
    meta_embed: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> Optional[torch.Tensor]:
        return getattr(self, key)


class PFRMLHead(nn.Module):
    """Backbone-agnostic PFRML head accepting either 4-stage `[C1..C4]` or single `(B, C, H, W)`.

    Integrates:
    - DCM (`fusion='sum'` or `'concat'`) for 4-stage inputs, or `SingleFeatureAdapter`
      for single feature maps `(B, C, H, W)`
    - `LightTransformerEncoder` (`dim=224`, default `2` layers, `enable_nested_tensor=False`)
    - Toggle #1: `AttentionPooling` (`use_attention_pooling`)
    - Toggle #2: `PhysicsPriorGate` (`use_physics_gate`)
    - `PrototypeClassificationHead` (when `use_prototypes=True`; omitted in R1 when `False`)
    - `PrototypeGuidedRegressionHead` (`reg_input='chi_proto'` per PFRML Fig. 7 or
      `reg_input='chi'` per PFRML Eq. 5)
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        feature_stride: int = 16,
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
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.use_prototypes = bool(use_prototypes)
        self.reg_input = str(reg_input)

        if isinstance(in_channels, int):
            self.is_multistage = False
            self.neck: nn.Module = SingleFeatureAdapter(
                in_channels=in_channels,
                out_channels=self.dim,
                stride=feature_stride,
            )
        else:
            self.is_multistage = True
            self.neck = DimensionConsistencyModule(
                in_channels=in_channels,
                out_channels=self.dim,
                fusion=dcm_fusion,
                avg_pool_kernel=dcm_avg_pool_kernel,
            )

        self.transformer = LightTransformerEncoder(
            dim=self.dim,
            grid_size=14,
            num_layers=transformer_layers,
            nhead=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
            use_attention_pooling=use_attention_pooling,
            pooling_mode=pooling_mode,
        )
        self.physics_gate = PhysicsPriorGate(
            dim=self.dim,
            enabled=use_physics_gate,
            patch_size=haze_patch_size,
            omega=haze_omega,
            input_is_normalized=physics_input_is_normalized,
        )
        if self.use_prototypes:
            self.proto_cls: Optional[PrototypeClassificationHead] = PrototypeClassificationHead(
                num_classes=self.num_classes,
                dim=self.dim,
                ema_decay=proto_ema_decay,
                tau_init=proto_tau_init,
            )
        else:
            self.proto_cls = None

        self.reg_head = PrototypeGuidedRegressionHead(
            dim=self.dim,
            hidden_dims=reg_hidden_dims,
            dropout=reg_dropout,
            use_prototypes=self.use_prototypes,
            reg_input=self.reg_input,
        )

    def forward(
        self,
        features: Union[torch.Tensor, Sequence[torch.Tensor]],
        images: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> PFRMLOutput:
        fmap = self.neck(features)  # [B, D, 14, 14]
        chi_deep, attn_weights = self.transformer(fmap)  # [B, D], [B, 196]
        chi, alpha_gate, phys_scalars = self.physics_gate.forward_with_details(
            f_deep=chi_deep,
            images=images,
            raw_images=raw_images,
        )
        if self.use_prototypes and self.proto_cls is not None:
            logits, probs, dist_sq, p_norm = self.proto_cls(chi, targets=targets)
            reg_pred = self.reg_head(chi, probs, p_norm)
        else:
            logits, probs, dist_sq = None, None, None
            reg_pred = self.reg_head(chi)
        return PFRMLOutput(
            logits=logits,
            probs=probs,
            dist_sq=dist_sq,
            reg_pred=reg_pred,
            chi=chi,
            attn_weights=attn_weights,
            alpha_gate=alpha_gate,
            phys_scalars=phys_scalars,
        )


class BackboneTapWrapper(nn.Module):
    """Unified tap extractor for `mobilenetv3_s`, `resnet18`, `mdnet_t`, and `e3b`.

    Supports:
    - `tap_mode='multistage'`: returns `[C1, C2, C3, C4]` at strides `[4, 8, 16, 32]`.
    - `tap_mode='single'`: returns single `(B, C, H, W)` feature map (stride-16
      `14x14` tokens for MobileNetV3-S / ResNet-18 / MDNet-T; post-MD-neck
      `[B, 128, 7, 7]` at stride 32 for E3b by default, or stride-16 `[B, 48, 14, 14]`
      if `e3b_single_tap='stride16'`).
    """

    def __init__(
        self,
        backbone_name: str = "mdnet_t",
        pretrained: bool = False,
        tap_mode: str = "multistage",
        e3b_single_tap: str = "md_neck",
    ) -> None:
        super().__init__()
        name = backbone_name.lower()
        if name not in BACKBONE_SPECS:
            raise ValueError(
                f"Unsupported backbone={backbone_name!r}. "
                f"Expected one of {tuple(BACKBONE_SPECS.keys())}."
            )
        t_mode = tap_mode.lower()
        if t_mode not in {"multistage", "single"}:
            raise ValueError(f"Unsupported tap_mode={tap_mode!r}. Expected 'multistage' or 'single'.")
        if e3b_single_tap not in {"md_neck", "stride16"}:
            raise ValueError(
                f"Unsupported e3b_single_tap={e3b_single_tap!r}. Expected 'md_neck' or 'stride16'."
            )

        self.backbone_name = name
        self.pretrained = bool(pretrained)
        self.tap_mode = t_mode
        self.e3b_single_tap = e3b_single_tap

        if name == "mdnet_t":
            self.backbone: nn.Module = build_mdnet_t()
        elif name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if self.pretrained else None
            self.backbone = tv_models.resnet18(weights=weights)
        elif name == "mobilenetv3_s":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
            self.backbone = tv_models.mobilenet_v3_small(weights=weights).features
        else:  # e3b
            self.backbone = MobileNetV3MDNeck(pretrained=self.pretrained)

        spec = BACKBONE_SPECS[name]
        if self.tap_mode == "multistage":
            self.out_channels: Union[int, Tuple[int, ...]] = spec["stage_channels"]  # type: ignore[assignment]
            self.out_stride = 16
        else:
            if name == "e3b" and self.e3b_single_tap == "stride16":
                self.out_channels = int(spec["stride16_channel"])  # type: ignore[arg-type]
                self.out_stride = 16
            else:
                self.out_channels = int(spec["single_tap_channel"])  # type: ignore[arg-type]
                self.out_stride = int(spec["single_tap_stride"])  # type: ignore[arg-type]

    def _extract_mobilenetv3_stages(
        self, features: nn.Sequential, x: torch.Tensor
    ) -> List[torch.Tensor]:
        c1 = c2 = c3 = c4 = None
        for idx, layer in enumerate(features):
            x = layer(x)
            if idx == 1:
                c1 = x
            elif idx == 3:
                c2 = x
            elif idx == 8:
                c3 = x
            elif idx == 12:
                c4 = x
        assert c1 is not None and c2 is not None and c3 is not None and c4 is not None
        return [c1, c2, c3, c4]

    def _extract_resnet18_stages(self, net: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        c1 = net.layer1(x)
        c2 = net.layer2(c1)
        c3 = net.layer3(c2)
        c4 = net.layer4(c3)
        return [c1, c2, c3, c4]

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.backbone_name == "mdnet_t":
            assert isinstance(self.backbone, MDNet)
            return self.backbone.forward_stages(x)
        if self.backbone_name == "resnet18":
            return self._extract_resnet18_stages(self.backbone, x)
        if self.backbone_name == "mobilenetv3_s":
            assert isinstance(self.backbone, nn.Sequential)
            return self._extract_mobilenetv3_stages(self.backbone, x)

        # E3b: stages 1..3 from MobileNetV3-Small, stage 4 from E3b's MD neck.
        assert isinstance(self.backbone, MobileNetV3MDNeck)
        c1, c2, c3, c4_raw = self._extract_mobilenetv3_stages(self.backbone.backbone, x)
        c4 = self.backbone.md_neck(self.backbone.reduce(c4_raw))
        return [c1, c2, c3, c4]

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        stages = self.forward_stages(x)
        if self.tap_mode == "multistage":
            return stages
        if self.backbone_name == "e3b":
            return stages[2] if self.e3b_single_tap == "stride16" else stages[3]
        # Stride-16 stage output ([B, C, 14, 14] on 224x224)
        return stages[2]


class PFRMLModel(nn.Module):
    """End-to-end Backbone + PFRML Head wrapper."""

    def __init__(
        self,
        backbone_name: str = "mdnet_t",
        pretrained: bool = False,
        tap_mode: str = "multistage",
        e3b_single_tap: str = "md_neck",
        **head_kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = BackboneTapWrapper(
            backbone_name=backbone_name,
            pretrained=pretrained,
            tap_mode=tap_mode,
            e3b_single_tap=e3b_single_tap,
        )
        self.head = PFRMLHead(
            in_channels=self.backbone.out_channels,
            feature_stride=self.backbone.out_stride,
            **head_kwargs,  # type: ignore[arg-type]
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
    ) -> PFRMLOutput:
        feats = self.backbone(images)
        return self.head(
            features=feats,
            images=images,
            raw_images=raw_images,
            targets=targets,
        )

    def count_backbone_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.backbone.parameters()
            if not trainable_only or p.requires_grad
        )

    def count_head_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.head.parameters()
            if not trainable_only or p.requires_grad
        )
