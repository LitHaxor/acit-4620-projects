"""Tabular, metadata, and image-only baseline models.

- E0: `MedianBaseline`, `HazeRidgeBaseline` (18 atmospheric haze statistics).
- E1: `MetadataMLP` (5 -> 64 -> 32 -> 1 over spherical GPS + cyclic month).
- E2: `ImageBaseline` supporting ResNet-18 (from scratch or ImageNet-pretrained)
  and MobileNetV3-Small (ImageNet-pretrained) predicting scaled AQI (pm25 / 100.0)
  or 6-class logits.
"""

from typing import Optional, Sequence, Tuple
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models as tv_models


class MetadataMLP(nn.Module):
    """5 -> 64 -> 32 -> 1 MLP over [x, y, z, sin_month, cos_month] metadata.

    Predicts scaled AQI (pm25 / 100.0). Exposes `forward_features` (32-D)
    for reuse in multimodal fusion (`SPEC.md` Sec. 3.4).
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        layers = [
            nn.Linear(in_dim, h1),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(h1, h2),
                nn.ReLU(),
            ]
        )
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(h2, 1)

    def forward_features(self, metadata: torch.Tensor) -> torch.Tensor:
        """Encode 5D metadata into a 32D representation."""
        return self.encoder(metadata)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """Predict scaled AQI of shape [B]."""
        feat = self.forward_features(metadata)
        return self.head(feat).squeeze(-1)


class MedianBaseline:
    """E0a: Constant median AQI predictor fit on the training fold."""

    def __init__(self) -> None:
        self.median_aqi_: Optional[float] = None

    def fit(self, y_train_aqi: np.ndarray) -> "MedianBaseline":
        self.median_aqi_ = float(np.median(np.asarray(y_train_aqi, dtype=np.float64)))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        if self.median_aqi_ is None:
            raise RuntimeError("MedianBaseline must be fit before calling predict().")
        return np.full(int(n_samples), self.median_aqi_, dtype=np.float64)


class HazeRidgeBaseline:
    """E0b: StandardScaler + RidgeCV on 18 atmospheric haze statistics."""

    def __init__(
        self,
        alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    ) -> None:
        self.alphas = tuple(float(a) for a in alphas)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        )

    def fit(self, X_train: np.ndarray, y_train_aqi: np.ndarray) -> "HazeRidgeBaseline":
        self.pipeline.fit(
            np.asarray(X_train, dtype=np.float64),
            np.asarray(y_train_aqi, dtype=np.float64),
        )
        return self

    @property
    def alpha_(self) -> float:
        return float(self.pipeline.named_steps["ridge"].alpha_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(np.asarray(X, dtype=np.float64))


class ImageBaseline(nn.Module):
    """E2 image-only CNN baseline wrapping ResNet-18 or MobileNetV3-Small."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        head_type: str = "regression",
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        bb = backbone.lower()
        ht = head_type.lower()
        if ht not in {"regression", "classification"}:
            raise ValueError(f"Unsupported head_type={head_type!r}")
        self.backbone_name = bb
        self.pretrained = bool(pretrained)
        self.head_type = ht
        self.num_classes = int(num_classes)
        out_dim = 1 if ht == "regression" else self.num_classes

        if bb == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, out_dim)
            self.net = net
        elif bb == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = tv_models.mobilenet_v3_small(weights=weights)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_features, out_dim)
            self.net = net
        else:
            raise ValueError(
                f"Unsupported E2 backbone={backbone!r}. Expected 'resnet18' or 'mobilenet_v3_small'."
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        if self.head_type == "regression":
            return out.squeeze(-1)
        return out


def build_resnet18_scratch(head_type: str = "regression", num_classes: int = 6) -> ImageBaseline:
    """E2a: ResNet-18 initialized from scratch."""
    return ImageBaseline(
        backbone="resnet18",
        pretrained=False,
        head_type=head_type,
        num_classes=num_classes,
    )


def build_resnet18_pretrained(head_type: str = "regression", num_classes: int = 6) -> ImageBaseline:
    """E2b: ResNet-18 with ImageNet-1K pretrained weights."""
    return ImageBaseline(
        backbone="resnet18",
        pretrained=True,
        head_type=head_type,
        num_classes=num_classes,
    )


def build_mobilenet_v3_small_pretrained(
    head_type: str = "regression",
    num_classes: int = 6,
) -> ImageBaseline:
    """E2c: MobileNetV3-Small with ImageNet-1K pretrained weights."""
    return ImageBaseline(
        backbone="mobilenet_v3_small",
        pretrained=True,
        head_type=head_type,
        num_classes=num_classes,
    )
