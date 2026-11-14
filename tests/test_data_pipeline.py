"""Pytest suite for Step 1: Data cleaning, station/spatial splits, transforms, and Dataset."""

import numpy as np
import torch
from src.config import load_config
from src.data.cleaning import load_and_clean_metadata
from src.data.dataset import PM25Dataset
from src.data.splits import build_cv_splits, compute_val_to_train_nearest_km
from src.data.transforms import get_train_transforms


def test_clean_counts_and_no_overlap():
    cfg = load_config()
    train_df, test_df = load_and_clean_metadata(cfg.data.train_csv, cfg.data.test_csv)

    assert len(train_df) == 8176
    assert len(test_df) == 2920
    assert len(train_df) + len(test_df) == 11096

    assert train_df["image_id"].is_unique
    assert test_df["image_id"].is_unique
    assert len(set(train_df["image_id"]).intersection(set(test_df["image_id"]))) == 0
    assert len(set(train_df["station_id"]).intersection(set(test_df["station_id"]))) == 0
    assert set(train_df["aqi_bin"].unique()) == {0, 1, 2, 3, 4, 5}
    assert set(test_df["aqi_bin"].unique()) == {0, 1, 2, 3, 4, 5}


def test_splits_station_and_spatial_cluster_modes():
    train_df, _ = load_and_clean_metadata()

    expected_groups = {"station": 2527, "spatial_cluster": 1059}
    for mode, exp_n_groups in expected_groups.items():
        df_split, folds = build_cv_splits(
            train_df,
            n_splits=5,
            group_mode=mode,
            spatial_cluster_km=25.0,
        )
        assert len(folds) == 5
        assert (df_split["fold"] >= 0).all()
        assert df_split["group_id"].nunique() == exp_n_groups

        for tr_idx, val_idx in folds:
            tr_stations = set(df_split.iloc[tr_idx]["station_id"])
            val_stations = set(df_split.iloc[val_idx]["station_id"])
            assert len(tr_stations.intersection(val_stations)) == 0

            val_bins = set(df_split.iloc[val_idx]["aqi_bin"].unique())
            tr_bins = set(df_split.iloc[tr_idx]["aqi_bin"].unique())
            assert val_bins == {0, 1, 2, 3, 4, 5}
            assert tr_bins == {0, 1, 2, 3, 4, 5}


def test_spatial_cluster_no_overlap_and_haversine_separation():
    train_df, _ = load_and_clean_metadata()
    df_station, folds_station = build_cv_splits(
        train_df, n_splits=5, group_mode="station", spatial_cluster_km=25.0
    )
    df_spatial, folds_spatial = build_cv_splits(
        train_df, n_splits=5, group_mode="spatial_cluster", spatial_cluster_km=25.0
    )

    for fold_idx in range(5):
        tr_s, va_s = folds_station[fold_idx]
        tr_c, va_c = folds_spatial[fold_idx]

        # 1. Verify zero spatial_cluster overlap in spatial_cluster mode
        tr_clusters = set(df_spatial.iloc[tr_c]["spatial_cluster"])
        va_clusters = set(df_spatial.iloc[va_c]["spatial_cluster"])
        assert len(tr_clusters.intersection(va_clusters)) == 0

        # 2. Verify val-to-nearest-train Haversine distances (km)
        d_station = compute_val_to_train_nearest_km(df_station, tr_s, va_s)
        d_spatial = compute_val_to_train_nearest_km(df_spatial, tr_c, va_c)

        assert d_station.shape == (len(va_s),)
        assert d_spatial.shape == (len(va_c),)
        assert np.median(d_spatial) > np.median(d_station)
        assert np.min(d_spatial) >= 24.0


def test_dataset_and_transforms_shapes():
    train_df, _ = load_and_clean_metadata()
    ds = PM25Dataset(
        train_df.iloc[:8],
        images_dir="datasets/train/images",
        transform=get_train_transforms(image_size=224),
        aqi_scale=100.0,
    )
    sample = ds[0]
    assert sample.image.shape == (3, 224, 224)
    assert sample.image.dtype == torch.float32
    assert sample.metadata.shape == (5,)
    assert sample.metadata.dtype == torch.float32
    assert torch.all(sample.metadata >= -1.0001) and torch.all(sample.metadata <= 1.0001)
    assert np.isclose(torch.norm(sample.metadata[:3]).item(), 1.0, atol=1e-4)
    assert np.isclose(torch.norm(sample.metadata[3:]).item(), 1.0, atol=1e-4)
    assert sample.aqi.dtype == torch.float32
    assert 0.01 <= sample.aqi.item() <= 5.30
    assert sample.aqi_bin.dtype == torch.long
    assert 0 <= sample.aqi_bin.item() <= 5
    assert sample.station_id.dtype == torch.long
