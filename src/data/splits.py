"""Station-grouped and spatial-cluster cross-validation splits.

Implements 5-fold StratifiedGroupKFold (Paper 1, Sec. III-A3 geographic/station
partitioning principle) stratified by the 6 US EPA AQI bins (`aqi_bin`) and
grouped by either:
- `station`: raw `station_id` (0 station overlap between train and val folds)
- `spatial_cluster`: connected-component spatial clusters of stations within
  `spatial_cluster_km` (default 25.0 km) using Haversine distance, preventing
  same-city nearby station leakage across folds while also guaranteeing 0
  `station_id` overlap.
"""

from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold

GroupMode = Literal["station", "spatial_cluster"]
EARTH_RADIUS_KM = 6371.0


def haversine_distance_matrix_km(coords_deg_a: np.ndarray, coords_deg_b: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise Haversine distance matrix in kilometers between two sets of (lat, lon) in degrees."""
    rad_a = np.radians(np.asarray(coords_deg_a, dtype=np.float64))
    rad_b = rad_a if coords_deg_b is None else np.radians(np.asarray(coords_deg_b, dtype=np.float64))
    lat_a = rad_a[:, 0:1]
    lon_a = rad_a[:, 1:2]
    lat_b = rad_b[:, 0:1].T
    lon_b = rad_b[:, 1:2].T
    dlat = lat_a - lat_b
    dlon = lon_a - lon_b
    a = np.sin(dlat / 2.0) ** 2 + (np.cos(lat_a) @ np.cos(lat_b)) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def compute_val_to_train_nearest_km(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> np.ndarray:
    """Compute Haversine distance (km) from each validation image to its nearest training image."""
    train_coords = df.iloc[train_idx][["latitude", "longitude"]].to_numpy(dtype=np.float64)
    val_coords = df.iloc[val_idx][["latitude", "longitude"]].to_numpy(dtype=np.float64)
    dist_mat = haversine_distance_matrix_km(val_coords, train_coords)
    return dist_mat.min(axis=1)


def assign_spatial_clusters(
    df: pd.DataFrame,
    distance_threshold_km: float = 25.0,
) -> pd.Series:
    """Assign each station_id to a connected spatial cluster within `distance_threshold_km`.

    Computes median (latitude, longitude) per `station_id` so every row belonging
    to a given `station_id` receives the exact same cluster ID.
    """
    station_coords = (
        df.groupby("station_id", as_index=False)[["latitude", "longitude"]]
        .median()
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    if len(station_coords) <= 1:
        cluster_map = {
            int(row["station_id"]): 0 for _, row in station_coords.iterrows()
        }
        return df["station_id"].map(cluster_map)

    dist_km = haversine_distance_matrix_km(station_coords[["latitude", "longitude"]].values)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="single",
        distance_threshold=float(distance_threshold_km),
    )
    labels = clustering.fit_predict(dist_km)
    cluster_map = dict(zip(station_coords["station_id"].astype(int), labels.astype(int)))
    return df["station_id"].astype(int).map(cluster_map)


def build_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_mode: GroupMode = "station",
    spatial_cluster_km: float = 25.0,
) -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
    """Create `n_splits` StratifiedGroupKFold splits over `df`.

    Args:
        df: Cleaned training DataFrame containing `station_id`, `aqi_bin`,
            `latitude`, and `longitude`.
        n_splits: Number of cross-validation folds (default 5).
        group_mode: `"station"` (group by `station_id`) or `"spatial_cluster"`
            (group by single-linkage Haversine station clusters within
            `spatial_cluster_km`).
        spatial_cluster_km: Distance threshold in km when `group_mode == "spatial_cluster"`.

    Returns:
        (df_with_fold_and_group_cols, list_of_(train_idx, val_idx)_tuples)
    """
    out = df.copy().reset_index(drop=True)
    out["spatial_cluster"] = assign_spatial_clusters(out, distance_threshold_km=spatial_cluster_km)

    if group_mode == "station":
        out["group_id"] = out["station_id"].astype(int)
    elif group_mode == "spatial_cluster":
        out["group_id"] = out["spatial_cluster"].astype(int)
    else:
        raise ValueError(f"Unsupported group_mode={group_mode!r}. Expected 'station' or 'spatial_cluster'.")

    groups = out["group_id"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    out["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(out, out["aqi_bin"].values, groups)):
        out.loc[val_idx, "fold"] = fold_idx
        folds.append((train_idx, val_idx))

    return out, folds
