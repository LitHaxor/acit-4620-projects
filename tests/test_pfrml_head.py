"""CPU-only unit tests for PFRML head, backbone interfaces, toggles, R1, and train/eval wiring."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluate import PFRML_MODEL_SPECS, default_pfrml_experiment_id
from scripts.analysis.summarize_last_epoch import discover_curves
from scripts.experiments.queue_pfrml_ablations import build_pfrml_ablation_specs
from src.config import load_config
from src.losses import HybridMultitaskLoss, RegressionLoss, compute_class_weights
from src.models.dcm import (
    BACKBONE_SPECS,
    DimensionConsistencyModule,
    PFRMLHead,
    PFRMLModel,
)
from src.models.pfrml_config import build_pfrml_from_config, load_pfrml_config
from src.models.physics_gate import PhysicsPriorGate
from src.models.pooling import AttentionPooling
from src.pfrml_losses import PFRMLMultitaskLoss
from train import _evaluate_image_model, build_fold_loss, build_image_model

torch.set_num_threads(2)


class TinyRandomBackbone(nn.Module):
    """Tiny 4-stage CNN outputting [56, 28, 14, 7] feature maps for CPU overfit test."""

    def __init__(self, dims=(16, 32, 64, 128)) -> None:
        super().__init__()
        c1, c2, c3, c4 = dims
        self.s1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )  # 56x56
        self.s2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )  # 28x28
        self.s3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )  # 14x14
        self.s4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(c4),
            nn.GELU(),
        )  # 7x7

    def forward(self, x: torch.Tensor):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return [f1, f2, f3, f4]


@pytest.mark.parametrize("backbone_name", ["mobilenetv3_s", "resnet18", "mdnet_t", "e3b"])
@pytest.mark.parametrize("tap_mode", ["multistage", "single"])
def test_backbone_interfaces_and_output_shapes(backbone_name: str, tap_mode: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    model = PFRMLModel(
        backbone_name=backbone_name,
        pretrained=False,
        tap_mode=tap_mode,
        use_attention_pooling=True,
        use_physics_gate=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.logits is not None and out.logits.shape == (2, 6)
    assert out.probs is not None and out.probs.shape == (2, 6)
    assert torch.allclose(out.probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert out.dist_sq is not None and out.dist_sq.shape == (2, 6)
    assert (out.dist_sq >= 0.0).all()
    assert out.reg_pred.shape == (2,)
    assert out.chi.shape == (2, 224)
    assert out.attn_weights.shape == (2, 196)
    assert torch.allclose(out.attn_weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert out.alpha_gate is not None and out.alpha_gate.shape == (2, 224)
    assert out.phys_scalars is not None and out.phys_scalars.shape == (2, 4)

    # Check single-feature-map direct PFRMLHead interface on (B, C, H, W)
    if tap_mode == "single":
        c_tap = BACKBONE_SPECS[backbone_name]["single_tap_channel"]
        stride_tap = BACKBONE_SPECS[backbone_name]["single_tap_stride"]
        h_tap = 224 // int(stride_tap)
        fmap = torch.randn(2, int(c_tap), h_tap, h_tap)
        head = PFRMLHead(in_channels=int(c_tap), feature_stride=int(stride_tap))
        head.eval()
        with torch.no_grad():
            h_out = head(fmap)
        assert h_out.logits is not None and h_out.logits.shape == (2, 6)
        assert h_out.reg_pred.shape == (2,)

        # Prototype-mixture variance diagnostic (SPEC Sec. 1)
        assert head.proto_cls is not None and h_out.probs is not None
        p_norm = head.proto_cls._l2_normalize(head.proto_cls.prototypes)
        y_mean, y_var, y_per_proto = head.reg_head.predict_prototype_mixture_variance(
            h_out.chi, h_out.probs, p_norm
        )
        assert y_mean.shape == (2,)
        assert y_var.shape == (2,)
        assert (y_var >= 0.0).all()
        assert y_per_proto.shape == (2, 6)


def test_e3b_stride16_single_tap_option() -> None:
    x = torch.randn(2, 3, 224, 224)
    model = PFRMLModel(
        backbone_name="e3b",
        pretrained=False,
        tap_mode="single",
        e3b_single_tap="stride16",
    )
    assert model.backbone.out_channels == 48
    assert model.backbone.out_stride == 16
    out = model(x)
    assert out.logits is not None and out.logits.shape == (2, 6)
    assert out.reg_pred.shape == (2,)


@pytest.mark.parametrize("dcm_fusion", ["sum", "concat"])
def test_gradients_reach_all_trainable_parameters(dcm_fusion: str) -> None:
    torch.manual_seed(1)
    x_raw = torch.rand(4, 3, 224, 224)
    y_bin = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    y_aqi = torch.tensor([0.25, 0.75, 1.75, 3.50], dtype=torch.float32)

    tiny_bb = TinyRandomBackbone(dims=(16, 32, 64, 128))
    head = PFRMLHead(
        in_channels=(16, 32, 64, 128),
        dcm_fusion=dcm_fusion,
        use_attention_pooling=True,
        use_physics_gate=True,
        physics_input_is_normalized=False,
    )
    loss_fn = PFRMLMultitaskLoss(use_class_weights=False)

    tiny_bb.train()
    head.train()
    feats = tiny_bb(x_raw)
    out = head(feats, raw_images=x_raw, targets=y_bin)
    loss, stats = loss_fn(out, y_bin, y_aqi)
    assert "total_loss" in stats and torch.isfinite(loss)
    loss.backward()

    for module_name, mod in [("backbone", tiny_bb), ("head", head)]:
        for name, p in mod.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for {module_name}.{name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {module_name}.{name}"
                assert p.grad.abs().sum().item() > 0.0, f"Zero grad for {module_name}.{name}"


def test_prototypes_buffer_eval_safe_and_no_grad() -> None:
    torch.manual_seed(2)
    head = PFRMLHead(in_channels=(16, 32, 64, 128))
    tiny_bb = TinyRandomBackbone(dims=(16, 32, 64, 128))
    assert head.proto_cls is not None

    # Prototypes stored as buffers, not parameters
    bufs = dict(head.named_buffers())
    params = dict(head.named_parameters())
    assert "proto_cls.prototypes" in bufs
    assert "proto_cls.prototypes" not in params
    assert head.proto_cls.prototypes.requires_grad is False

    # Centroid init
    feats_init = torch.randn(12, 224)
    labels_init = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.long)
    head.proto_cls.init_prototypes(feats_init, labels_init)
    assert bool(head.proto_cls.prototypes_initialized.item()) is True
    norms = head.proto_cls.prototypes.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    # Checkpoint round-trip preserves prototypes
    state = head.state_dict()
    head_reloaded = PFRMLHead(in_channels=(16, 32, 64, 128))
    head_reloaded.load_state_dict(state)
    assert head_reloaded.proto_cls is not None
    assert torch.equal(head_reloaded.proto_cls.prototypes, head.proto_cls.prototypes)

    x = torch.randn(4, 3, 224, 224)
    y_bin = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    y_aqi = torch.tensor([0.2, 0.7, 1.2, 1.8], dtype=torch.float32)

    # Eval mode: prototypes MUST NOT update even if targets are passed
    head.eval()
    tiny_bb.eval()
    before_eval = head.proto_cls.prototypes.clone()
    with torch.no_grad():
        _ = head(tiny_bb(x), targets=y_bin)
    assert torch.equal(head.proto_cls.prototypes, before_eval)

    # Train mode without targets: prototypes MUST NOT update
    head.train()
    tiny_bb.train()
    _ = head(tiny_bb(x), targets=None)
    assert torch.equal(head.proto_cls.prototypes, before_eval)

    # Train mode with targets: prototypes DO update, and receive NO gradient
    out = head(tiny_bb(x), targets=y_bin)
    loss, _ = PFRMLMultitaskLoss()(out, y_bin, y_aqi)
    loss.backward()
    assert not torch.equal(head.proto_cls.prototypes, before_eval)
    assert head.proto_cls.prototypes.grad is None
    assert head.proto_cls.log_tau.grad is not None


def test_toggles_off_match_baseline_exactly() -> None:
    torch.manual_seed(3)

    # Toggle #1: AttentionPooling(enabled=False) == uniform GAP
    tokens = torch.randn(3, 196, 224)
    pool_off = AttentionPooling(dim=224, enabled=False)
    pooled, weights = pool_off.forward_with_weights(tokens)
    assert torch.allclose(pooled, tokens.mean(dim=1), atol=1e-6)
    assert torch.allclose(weights, torch.full((3, 196), 1.0 / 196.0), atol=1e-6)

    # Toggle #2: PhysicsPriorGate(enabled=False) == identity
    f_deep = torch.randn(3, 224)
    img = torch.rand(3, 3, 224, 224)
    pg_off = PhysicsPriorGate(dim=224, enabled=False)
    f_out, alpha, phys = pg_off.forward_with_details(f_deep, raw_images=img)
    assert torch.allclose(f_out, f_deep, atol=1e-7)
    assert alpha is None and phys is None

    # End-to-end head with toggles off matches shared-weight baseline exactly
    head_a = PFRMLHead(
        in_channels=(16, 32, 64, 128),
        use_attention_pooling=False,
        use_physics_gate=False,
    )
    head_b = PFRMLHead(
        in_channels=(16, 32, 64, 128),
        use_attention_pooling=False,
        use_physics_gate=False,
    )
    head_b.load_state_dict(head_a.state_dict())
    head_a.eval()
    head_b.eval()
    stages = [
        torch.randn(2, 16, 56, 56),
        torch.randn(2, 32, 28, 28),
        torch.randn(2, 64, 14, 14),
        torch.randn(2, 128, 7, 7),
    ]
    out_a = head_a(stages, raw_images=torch.rand(2, 3, 224, 224))
    out_b = head_b(stages, raw_images=None)
    assert out_a.logits is not None and out_b.logits is not None
    assert torch.allclose(out_a.logits, out_b.logits, atol=1e-6)
    assert torch.allclose(out_a.reg_pred, out_b.reg_pred, atol=1e-6)

    # Toggle #3: PFRMLMultitaskLoss(use_class_weights=False) == HybridMultitaskLoss(class_weights=None)
    logits = torch.randn(8, 6)
    reg_pred = torch.randn(8)
    y_bin = torch.tensor([0, 0, 0, 0, 1, 2, 4, 5], dtype=torch.long)
    y_aqi = torch.tensor([0.2, 0.3, 0.4, 0.45, 0.8, 1.3, 2.5, 4.0], dtype=torch.float32)

    base_loss = HybridMultitaskLoss(beta=0.5, cls_eps=0.1, reg_loss_type="mse", class_weights=None)
    wrap_off = PFRMLMultitaskLoss(use_class_weights=False, beta=0.5, cls_eps=0.1, reg_loss_type="mse")
    l_base, _ = base_loss(logits, reg_pred, y_bin, y_aqi)
    l_off, _ = wrap_off((logits, reg_pred), y_bin, y_aqi)
    assert torch.allclose(l_base, l_off, atol=1e-7)

    wrap_on = PFRMLMultitaskLoss(
        use_class_weights=True,
        class_weight_gamma=0.5,
        train_targets_or_counts=y_bin,
    )
    l_on, _ = wrap_on((logits, reg_pred), y_bin, y_aqi)
    assert not torch.allclose(l_base, l_on)

    # DCM sum vs concat shapes
    dcm_sum = DimensionConsistencyModule(in_channels=(16, 32, 64, 128), fusion="sum")
    dcm_cat = DimensionConsistencyModule(in_channels=(16, 32, 64, 128), fusion="concat")
    assert dcm_sum(stages).shape == (2, 224, 14, 14)
    assert dcm_cat(stages).shape == (2, 224, 14, 14)


def test_overfit_batch_of_16_random_tensors() -> None:
    torch.manual_seed(42)
    b = 16
    x_raw = torch.rand(b, 3, 224, 224)
    y_bin = torch.tensor([i % 6 for i in range(b)], dtype=torch.long)
    y_aqi = 0.3 + 0.6 * y_bin.float()

    backbone = TinyRandomBackbone(dims=(16, 32, 64, 128))
    head = PFRMLHead(
        in_channels=(16, 32, 64, 128),
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.0,
        reg_dropout=0.0,
        use_attention_pooling=True,
        use_physics_gate=True,
        physics_input_is_normalized=False,
    )
    loss_fn = PFRMLMultitaskLoss(
        beta=0.5,
        cls_eps=0.0,
        use_class_weights=True,
        train_targets_or_counts=y_bin,
    )

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)

    backbone.train()
    head.train()
    initial_loss = None
    final_loss = None
    for step in range(50):
        optimizer.zero_grad()
        out = head(backbone(x_raw), raw_images=x_raw, targets=y_bin)
        loss, _ = loss_fn(out, y_bin, y_aqi)
        if step == 0:
            initial_loss = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.1

    backbone.eval()
    head.eval()
    with torch.no_grad():
        eval_out = head(backbone(x_raw), raw_images=x_raw)
        assert eval_out.logits is not None
        acc = float((eval_out.logits.argmax(dim=-1) == y_bin).float().mean().item())
        mae = float((eval_out.reg_pred - y_aqi).abs().mean().item())
    assert acc == 1.0
    assert mae < 0.15


def test_pfrml_yaml_configs_load_and_build() -> None:
    cfg_paths = sorted(Path("configs").glob("pfrml_*.yaml"))
    assert len(cfg_paths) >= 6
    dummy_counts = torch.tensor([100, 80, 50, 30, 15, 5], dtype=torch.float32)
    for path in cfg_paths:
        cfg = load_pfrml_config(path)
        model, loss_fn = build_pfrml_from_config(cfg, train_targets_or_counts=dummy_counts)
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(x)
            l, _ = loss_fn(out, torch.tensor([0]), torch.tensor([0.3]))
        assert torch.isfinite(l)


def test_eval_forward_leaves_prototypes_unchanged_in_train_eval_loop() -> None:
    """Verify prototypes and counts are strictly unchanged during `_evaluate_image_model`."""
    torch.manual_seed(11)
    model = PFRMLModel(
        backbone_name="mdnet_t",
        pretrained=False,
        tap_mode="multistage",
        use_attention_pooling=True,
        use_physics_gate=True,
    )
    loss_fn = PFRMLMultitaskLoss(use_class_weights=False)
    assert model.head.proto_cls is not None

    # First do one training forward with targets so prototypes move away from init
    model.train()
    x_tr = torch.randn(4, 3, 224, 224)
    y_bin_tr = torch.tensor([0, 1, 4, 5], dtype=torch.long)
    _ = model(x_tr, targets=y_bin_tr)

    protos_before = model.head.proto_cls.prototypes.clone()
    counts_before = model.head.proto_cls.class_counts.clone()

    # Build a synthetic evaluation loader with non-trivial bin targets
    fake_batches = [
        SimpleNamespace(
            image=torch.randn(3, 3, 224, 224),
            aqi=torch.tensor([0.3, 1.2, 3.8], dtype=torch.float32),
            aqi_bin=torch.tensor([0, 2, 5], dtype=torch.long),
        ),
        SimpleNamespace(
            image=torch.randn(2, 3, 224, 224),
            aqi=torch.tensor([0.8, 2.5], dtype=torch.float32),
            aqi_bin=torch.tensor([1, 4], dtype=torch.long),
        ),
    ]
    val_loss, val_preds, val_pred_bins, val_proto_var = _evaluate_image_model(
        model=model,
        loader=fake_batches,  # type: ignore[arg-type]
        head_type="multitask",
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        aqi_scale=100.0,
        use_amp=False,
    )

    assert np.isfinite(val_loss)
    assert val_preds.shape == (5,)
    assert val_pred_bins is not None and val_pred_bins.shape == (5,)
    assert val_proto_var is not None and val_proto_var >= 0.0
    assert torch.equal(model.head.proto_cls.prototypes, protos_before)
    assert torch.equal(model.head.proto_cls.class_counts, counts_before)


def test_class_weights_computed_from_training_bins_only() -> None:
    """Verify fold loss computes class weights strictly from fold training bins."""
    cfg = load_config("configs/pfrml_all_toggles.yaml")
    assert cfg.loss.use_class_weights is True

    # Fold training bins have skewed distribution; validation bins have a very different one
    fold_train_bins = np.array([0] * 50 + [1] * 30 + [2] * 15 + [3] * 8 + [4] * 4 + [5] * 2)
    fold_val_bins = np.array([0] * 2 + [1] * 2 + [2] * 2 + [3] * 10 + [4] * 20 + [5] * 40)

    loss_fn = build_fold_loss("multitask", cfg, train_bins=fold_train_bins)
    assert isinstance(loss_fn, PFRMLMultitaskLoss)
    assert loss_fn.hybrid_loss is not None

    expected_train_w = compute_class_weights(
        fold_train_bins,
        num_classes=6,
        gamma=cfg.loss.class_weight_gamma,
        normalize=True,
    )
    wrong_combined_w = compute_class_weights(
        np.concatenate([fold_train_bins, fold_val_bins]),
        num_classes=6,
        gamma=cfg.loss.class_weight_gamma,
        normalize=True,
    )
    actual_w = loss_fn.hybrid_loss.cls_loss_fn.class_weights
    assert actual_w is not None
    assert torch.allclose(actual_w, expected_train_w, atol=1e-6)
    assert not torch.allclose(actual_w, wrong_combined_w, atol=1e-3)


@pytest.mark.parametrize(
    "model_name,expected_backbone,expected_pretrained",
    [
        ("pfrml_mobilenetv3_s", "mobilenetv3_s", True),
        ("pfrml_resnet18", "resnet18", True),
        ("pfrml_mdnet_t", "mdnet_t", False),
        ("pfrml_e3b", "e3b", True),
    ],
)
def test_pfrml_models_build_with_right_pretrained_flag(
    model_name: str,
    expected_backbone: str,
    expected_pretrained: bool,
) -> None:
    """Verify each registered pfrml_* model builds with the required backbone and pretrained flag."""
    cfg = load_config("configs/pfrml_all_toggles.yaml")
    assert PFRML_MODEL_SPECS[model_name] == (expected_backbone, expected_pretrained)

    model, head_type = build_image_model(model_name, cfg)
    assert head_type == "multitask"
    assert isinstance(model, PFRMLModel)
    assert model.backbone.backbone_name == expected_backbone
    assert model.backbone.pretrained is expected_pretrained
    assert model.backbone.tap_mode == "multistage"
    assert model.head.transformer.pooling.enabled is True
    assert model.head.physics_gate.enabled is True
    assert default_pfrml_experiment_id(model_name, "configs/pfrml_all_toggles.yaml") == (
        f"E4_{expected_backbone}_full"
    )


def test_r1_use_prototypes_false_no_buffers_and_regression_only_loss() -> None:
    """Verify R1 (`use_prototypes: false`) uses no prototype buffers and regression-only loss."""
    cfg = load_config("configs/pfrml_regression_only.yaml")
    assert cfg.pfrml.use_prototypes is False

    head = PFRMLHead(in_channels=(16, 24, 48, 576), use_prototypes=False)
    assert head.use_prototypes is False
    assert head.proto_cls is None

    # No prototype buffers exist in head
    buffer_names = [name for name, _ in head.named_buffers()]
    assert not any("proto" in name for name in buffer_names), f"Found unexpected buffers: {buffer_names}"
    assert head.reg_head.mlp[0].in_features == 224

    stages = [
        torch.randn(4, 16, 56, 56),
        torch.randn(4, 24, 28, 28),
        torch.randn(4, 48, 14, 14),
        torch.randn(4, 576, 7, 7),
    ]
    y_bin = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    y_aqi = torch.tensor([0.25, 0.80, 1.70, 3.60], dtype=torch.float32)

    out = head(stages, targets=y_bin)
    assert out.logits is None
    assert out.probs is None
    assert out.dist_sq is None
    assert out.reg_pred.shape == (4,)

    loss_fn = build_fold_loss("multitask", cfg, train_bins=y_bin.numpy())
    assert isinstance(loss_fn, PFRMLMultitaskLoss)
    assert loss_fn.use_prototypes is False
    assert loss_fn.hybrid_loss is None

    total_loss, stats = loss_fn(out, y_bin, y_aqi)
    expected_reg = RegressionLoss(loss_type="mse")(out.reg_pred, y_aqi)
    assert torch.allclose(total_loss, expected_reg, atol=1e-7)
    assert stats["cls_loss"] == 0.0
    assert np.isclose(stats["total_loss"], stats["reg_loss"])

    total_loss.backward()
    for name, p in head.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing grad in R1 head for {name}"


def test_pfrml_queue_specs_and_empty_curve_discovery(tmp_path: Path) -> None:
    """Verify config-major PFRML ablation queue order and empty curve discovery."""
    specs = build_pfrml_ablation_specs(smoke=False)
    assert [s.rung for s in specs] == [
        "R2",
        "R1",
        "R2+all",
        "R2+T1",
        "R2+T2",
        "R2+T3",
        "R2+T4",
        "Extra run A",
        "Extra run B",
    ]
    assert specs[-1].is_extra_b is True
    assert specs[-1].epochs == 60
    assert specs[-2].group_mode == "spatial_cluster"

    empty_curves = tmp_path / "empty_curves"
    empty_curves.mkdir()
    assert discover_curves(empty_curves) == {}
