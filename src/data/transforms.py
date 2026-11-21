"""Physics-preserving image transforms for air quality estimation.

Following Paper 2 (Wang et al., IEEE TII 2024, Sec. III-C & IV-A):
- Whole images are resized directly to 224x224 without cropping so both the
  upper sky region and lower ground/building reflection region remain intact.
- Colour jitter and grayscale conversions are strictly excluded because colour
  saturation and brightness encode the physical Mie scattering AQI signal.
- Only horizontal flips and gentle geometric transforms (small rotation /
  translation / scaling) are applied during training.
"""

from typing import Tuple
import torch
from torchvision import transforms

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    horizontal_flip_prob: float = 0.5,
    rotation_deg: float = 5.0,
    translate_frac: float = 0.02,
    scale_range: Tuple[float, float] = (0.98, 1.02),
) -> transforms.Compose:
    """Build training transform pipeline with whole-image resize and gentle geometric augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomAffine(
                degrees=rotation_deg,
                translate=(translate_frac, translate_frac),
                scale=scale_range,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build deterministic evaluation transform pipeline (whole-image resize + ImageNet norm)."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization on a [C, H, W] or [B, C, H, W] tensor and clamp to [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
