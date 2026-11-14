"""Deterministic data cleaning for PM25Vision metadata.

Performs:
1. Purging of the 18 overlapping `image_id`s present in both train and test
   from the training split to eliminate cross-split image leakage.
2. Deduplication of multi-station `image_id`s by taking the median `pm25` AQI
   across conflicting rows and updating the corresponding 6-class US EPA bin
   (Paper 2, Sec. III-A, Table II; Paper 1, Sec. II-C, Table II).
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd

EPA_BIN_EDGES = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, float("inf")]
EPA_BIN_LABELS = ["0–50", "51–100", "101–150", "151–200", "201–300", "301–600"]
EPA_QUALITY_NAMES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


def aqi_to_bin_index(pm25: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, int]:
    """Map continuous US EPA AQI values to integer bin indices in {0, 1, 2, 3, 4, 5}.

    Thresholds follow US EPA AQI levels (Paper 2, Table II):
      0: 0–50, 1: 51–100, 2: 101–150, 3: 151–200, 4: 201–300, 5: >300.
    """
    if isinstance(pm25, pd.Series):
        return pd.cut(
            pm25,
            bins=EPA_BIN_EDGES,
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
            right=True,
        ).astype(int)
    arr = np.asarray(pm25, dtype=np.float64)
    idx = np.digitize(arr, bins=[50.0, 100.0, 150.0, 200.0, 300.0], right=True)
    if np.ndim(pm25) == 0:
        return int(idx)
    return idx.astype(int)


def dedupe_split_by_median_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows sharing the same `image_id` by median AQI.

    Sorts deterministically by (`image_id`, `station_id`), replaces `pm25`
    with the median AQI across rows sharing `image_id`, and updates `aqi_bin`
    and `pm25_bin` to match the deduplicated median AQI.
    """
    out = df.sort_values(["image_id", "station_id"]).copy()
    out["pm25"] = out.groupby("image_id")["pm25"].transform("median")
    out = out.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)
    out["aqi_bin"] = aqi_to_bin_index(out["pm25"])
    out["pm25_bin"] = out["aqi_bin"].map(dict(enumerate(EPA_BIN_LABELS)))
    return out


def load_and_clean_metadata(
    train_csv: Union[str, Path] = "datasets/train/metadata.csv",
    test_csv: Union[str, Path] = "datasets/test/metadata.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test metadata CSVs, purge train/test overlap from train, and deduplicate.

    Returns:
        (clean_train_df, clean_test_df) with 8,176 and 2,920 unique images respectively.
    """
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)

    # 1. Purge any image_id present in test from train (18 overlapping images)
    overlap_ids = set(train_raw["image_id"]).intersection(set(test_raw["image_id"]))
    train_purged = train_raw[~train_raw["image_id"].isin(overlap_ids)].copy()

    # 2. Deduplicate multi-station images within each split by median AQI
    clean_train = dedupe_split_by_median_aqi(train_purged)
    clean_test = dedupe_split_by_median_aqi(test_raw)

    return clean_train, clean_test
