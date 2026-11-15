"""Unit tests for evaluation metrics, losses, haze features, and baseline models."""

import math
import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
import torch

from src.data.cleaning import aqi_to_bin_index
from src.features.haze_features import HazeStatisticsExtractor
from src.losses import (
    HybridMultitaskLoss,
    LabelSmoothingCrossEntropy,
    RegressionLoss,
    compute_class_weights,
)
from src.metrics import (
    compute_all_metrics,
    paired_wilcoxon_test,
    summarize_fold_metrics,
)
from src.models.baselines import HazeRidgeBaseline, MedianBaseline, MetadataMLP


def test_metrics_match_scipy_sklearn_random_example() -> None:
    rng = np.random.default_rng(123)
    y_true = rng.uniform(5.0, 420.0, size=100)
    y_pred = y_true + rng.normal(0.0, 45.0, size=100)
    y_pred = np.clip(y_pred, 1.0, 500.0)

    yt_b = aqi_to_bin_index(y_true)
    yp_b = aqi_to_bin_index(y_pred)

    got = compute_all_metrics(y_true, y_pred)

    assert got["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred), rel=1e-7)
    assert got["rmse"] == pytest.approx(root_mean_squared_error(y_true, y_pred), rel=1e-7)
    assert got["plcc"] == pytest.approx(stats.pearsonr(y_true, y_pred).statistic, rel=1e-7)
    assert got["srcc"] == pytest.approx(stats.spearmanr(y_true, y_pred).statistic, rel=1e-7)
    assert got["r2"] == pytest.approx(r2_score(y_true, y_pred), rel=1e-7)
    assert got["acc"] == pytest.approx(accuracy_score(yt_b, yp_b), rel=1e-7)
    assert got["pm1_acc"] == pytest.approx(float(np.mean(np.abs(yt_b - yp_b) <= 1)), rel=1e-7)
    assert got["macro_f1"] == pytest.approx(
        f1_score(yt_b, yp_b, labels=[0, 1, 2, 3, 4, 5], average="macro", zero_division=0.0),
        rel=1e-7,
    )

    tail_mask = np.isin(yt_b, [4, 5])
    assert np.any(tail_mask)
    assert got["tail_mae"] == pytest.approx(
        mean_absolute_error(y_true[tail_mask], y_pred[tail_mask]), rel=1e-7
    )
    assert got["tail_f1"] == pytest.approx(
        f1_score(yt_b, yp_b, labels=[4, 5], average="macro", zero_division=0.0),
        rel=1e-7,
    )


def test_fold_summary_and_wilcoxon() -> None:
    folds = [{"mae": 40.0, "r2": 0.1}, {"mae": 42.0, "r2": 0.2}, {"mae": 44.0, "r2": 0.3}]
    summary = summarize_fold_metrics(folds, ddof=1)
    assert summary["mae"][0] == pytest.approx(42.0)
    assert summary["mae"][1] == pytest.approx(2.0)
    w = paired_wilcoxon_test([40.0, 41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0, 49.0])
    assert w["mean_diff"] == pytest.approx(-5.0)
    assert 0.0 < w["p_value"] <= 0.1


def test_ce_at_uniform_logits_is_ln6_and_hybrid_loss() -> None:
    logits = torch.zeros(32, 6)
    targets = torch.tensor([i % 6 for i in range(32)], dtype=torch.long)

    ce_smooth = LabelSmoothingCrossEntropy(eps=0.1)
    loss_smooth = ce_smooth(logits, targets)
    assert loss_smooth.item() == pytest.approx(math.log(6.0), abs=1e-5)

    ce_plain = LabelSmoothingCrossEntropy(eps=0.0)
    assert ce_plain(logits, targets).item() == pytest.approx(math.log(6.0), abs=1e-5)

    # Check class weights w_k ~ N_k^-gamma
    labels = np.array([0] * 100 + [1] * 50 + [2] * 25 + [3] * 10 + [4] * 5 + [5] * 2)
    w = compute_class_weights(labels, num_classes=6, gamma=0.5, normalize=True)
    assert w.shape == (6,)
    assert w[5] > w[4] > w[3] > w[2] > w[1] > w[0]
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)

    # Hybrid loss L = L_cls + beta * L_reg
    reg_preds = torch.full((32,), 1.0)
    reg_targets = torch.full((32,), 2.0)
    hybrid_mse = HybridMultitaskLoss(beta=0.5, cls_eps=0.1, reg_loss_type="mse")
    total, parts = hybrid_mse(logits, reg_preds, targets, reg_targets)
    assert parts["cls_loss"] == pytest.approx(math.log(6.0), abs=1e-5)
    assert parts["reg_loss"] == pytest.approx(1.0, abs=1e-5)
    assert total.item() == pytest.approx(math.log(6.0) + 0.5 * 1.0, abs=1e-5)

    # Huber check
    huber = RegressionLoss(loss_type="huber", huber_delta=0.5)
    assert huber(reg_preds, reg_targets).item() == pytest.approx(0.375, abs=1e-5)


def test_haze_features_physical_properties_and_gradients() -> None:
    extractor = HazeStatisticsExtractor(patch_size=15, input_is_normalized=False)
    assert len(extractor.feature_names) == 18

    # Crisp colourful checkerboard image vs. hazy blended image (I_haze = t * I + (1 - t) * 1.0)
    torch.manual_seed(0)
    grid_x = torch.arange(64).view(1, 1, 64, 1).expand(1, 1, 64, 64)
    grid_y = torch.arange(64).view(1, 1, 1, 64).expand(1, 1, 64, 64)
    checker = (((grid_x // 8) + (grid_y // 8)) % 2).float()
    clear = torch.cat([checker * 0.9 + 0.05, (1.0 - checker) * 0.8 + 0.05, checker * 0.2 + 0.05], dim=1)
    hazy = 0.3 * clear + 0.7 * torch.ones_like(clear)

    clear_req = clear.clone().requires_grad_(True)
    feats_clear = extractor(clear_req)
    feats_hazy = extractor(hazy)
    priors_clear = extractor.compute_physics_priors(clear_req)
    priors_hazy = extractor.compute_physics_priors(hazy)

    assert feats_clear.shape == (1, 18)
    assert priors_clear.shape == (1, 4)

    # Physical sanity: haze reduces saturation (idx 0), reduces gradient (idx 2), increases DCP (idx 4)
    assert feats_hazy[0, 0] < feats_clear[0, 0]
    assert feats_hazy[0, 2] < feats_clear[0, 2]
    assert feats_hazy[0, 4] > feats_clear[0, 4]
    # Transmission (idx 2) drops and extinction (idx 3) rises under haze
    assert priors_hazy[0, 2] < priors_clear[0, 2]
    assert priors_hazy[0, 3] > priors_clear[0, 3]

    # Differentiability check for in-model reuse
    (feats_clear.sum() + priors_clear.sum()).backward()
    assert clear_req.grad is not None
    assert torch.isfinite(clear_req.grad).all()


def test_baseline_models_shapes_and_fit() -> None:
    mlp = MetadataMLP(in_dim=5, hidden_dims=(64, 32))
    x = torch.randn(10, 5)
    assert mlp.forward_features(x).shape == (10, 32)
    assert mlp(x).shape == (10,)

    med = MedianBaseline().fit(np.array([10.0, 20.0, 100.0]))
    assert np.allclose(med.predict(3), 20.0)

    rng = np.random.default_rng(7)
    Xh = rng.normal(size=(40, 18))
    yh = 50.0 + 10.0 * Xh[:, 0]
    ridge = HazeRidgeBaseline(alphas=(0.1, 1.0, 10.0)).fit(Xh, yh)
    assert ridge.predict(Xh).shape == (40,)
