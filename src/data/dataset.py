"""Lazy-loading PyTorch Dataset for PM25Vision street-level images and metadata."""

import math
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.cleaning import aqi_to_bin_index
from src.data.transforms import get_eval_transforms


class PM25Sample(NamedTuple):
    """Single sample or collated batch returned by PM25Dataset."""

    image: torch.Tensor
    metadata: torch.Tensor
    aqi: torch.Tensor
    aqi_bin: torch.Tensor
    station_id: torch.Tensor


def encode_metadata_features(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    captured_at_series: pd.Series,
) -> np.ndarray:
    """Encode (latitude, longitude, captured_at) into a 5D float32 array:
    [x, y, z, sin_month, cos_month] in [-1, 1].

    - Spherical unit-sphere coordinates:
        x = cos(lat) * cos(lon)
        y = cos(lat) * sin(lon)
        z = sin(lat)
    - Cyclic month encoding for month m in {1..12}:
        sin_month = sin(2 * pi * m / 12)
        cos_month = cos(2 * pi * m / 12)
    """
    lat_rad = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon_rad = np.radians(np.asarray(longitudes, dtype=np.float64))

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    months = pd.to_datetime(captured_at_series).dt.month.to_numpy(dtype=np.float64)
    angle = (2.0 * math.pi * months) / 12.0
    sin_m = np.sin(angle)
    cos_m = np.cos(angle)

    features = np.stack([x, y, z, sin_m, cos_m], axis=1).astype(np.float32)
    return features


class PM25Dataset(Dataset):
    """Lazy-loading PyTorch Dataset for PM25Vision.

    Each item returns a `PM25Sample` (5-tuple):
        (image, metadata, aqi_scaled, aqi_bin, station_id)
    where:
        - `image`: float32 tensor of shape `[3, H, W]` (ImageNet-normalized)
        - `metadata`: float32 tensor of shape `[5]` (`[x, y, z, sin_month, cos_month]`)
        - `aqi`: float32 scalar tensor (`pm25 / aqi_scale`, default `pm25 / 100.0`)
        - `aqi_bin`: int64 scalar tensor in `{0, 1, 2, 3, 4, 5}`
        - `station_id`: int64 scalar tensor
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        aqi_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform if transform is not None else get_eval_transforms()
        self.aqi_scale = float(aqi_scale)

        self.filenames = self.df["filename"].astype(str).tolist()
        self.metadata_array = encode_metadata_features(
            self.df["latitude"].to_numpy(),
            self.df["longitude"].to_numpy(),
            self.df["captured_at"],
        )
        self.aqi_scaled = (self.df["pm25"].to_numpy(dtype=np.float32) / self.aqi_scale).astype(
            np.float32
        )
        if "aqi_bin" in self.df.columns:
            self.aqi_bins = self.df["aqi_bin"].to_numpy(dtype=np.int64)
        else:
            self.aqi_bins = aqi_to_bin_index(self.df["pm25"]).to_numpy(dtype=np.int64)
        self.station_ids = self.df["station_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> PM25Sample:
        img_path = self.images_dir / self.filenames[index]
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            image_tensor = self.transform(rgb)

        return PM25Sample(
            image=image_tensor,
            metadata=torch.from_numpy(self.metadata_array[index]),
            aqi=torch.tensor(self.aqi_scaled[index], dtype=torch.float32),
            aqi_bin=torch.tensor(self.aqi_bins[index], dtype=torch.long),
            station_id=torch.tensor(self.station_ids[index], dtype=torch.long),
        )
