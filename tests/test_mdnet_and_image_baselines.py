"""Unit tests for MDNet-T backbone, receptive field, image baselines, and scheduler."""

import pytest
import torch

from src.config import load_config
from src.models.baselines import (
    build_mobilenet_v3_small_pretrained,
    build_resnet18_pretrained,
    build_resnet18_scratch,
)
from src.models.mdnet import (
    MDBlock,
    build_mdnet_t,
    compute_mdnet_receptive_field,
)
from train import build_image_model, build_warmup_cosine_scheduler


def test_mdblock_dilated_branches_preserve_spatial_size() -> None:
    for c, size in [(96, 56), (192, 28), (384, 14), (768, 7)]:
        block = MDBlock(c)
        x = torch.randn(2, c, size, size)
        b3, b2, b1 = block.forward_branches(x)
        assert b3.shape == (2, c, size, size)
        assert b2.shape == (2, c, size, size)
        assert b1.shape == (2, c, size, size)
        out = block(x)
        assert out.shape == (2, c, size, size)


def test_mdnet_t_param_count_and_stage_shapes() -> None:
    model_reg = build_mdnet_t(head_type="regression")
    model_cls = build_mdnet_t(head_type="classification", num_classes=6)

    n_params_reg = sum(p.numel() for p in model_reg.parameters())
    n_params_cls = sum(p.numel() for p in model_cls.parameters())

    # Paper 2 Table III reports 4.9M (about 4.88M); verify within ~1.5% of 4.88M
    assert 4_800_000 <= n_params_reg <= 5_000_000
    assert 4_800_000 <= n_params_cls <= 5_000_000

    x = torch.randn(2, 3, 224, 224)
    stages = model_reg.forward_stages(x)
    assert len(stages) == 4
    expected_shapes = [
        (2, 96, 56, 56),
        (2, 192, 28, 28),
        (2, 384, 14, 14),
        (2, 768, 7, 7),
    ]
    for feat, exp in zip(stages, expected_shapes):
        assert feat.shape == exp

    assert model_reg.forward_features(x).shape == (2, 768)
    out_reg = model_reg(x)
    out_cls = model_cls(x)
    assert out_reg.shape == (2,)
    assert out_cls.shape == (2, 6)

    # Backward gradient check
    loss = out_reg.sum() + out_cls.sum()
    loss.backward()
    for name, p in model_reg.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_mdnet_theoretical_receptive_field() -> None:
    assert compute_mdnet_receptive_field((3, 3, 9, 3)) == 1136
    assert compute_mdnet_receptive_field((3, 3, 27, 3)) == 2288


def test_e2_image_baselines_shapes() -> None:
    x = torch.randn(2, 3, 224, 224)
    for builder in (
        build_resnet18_scratch,
        build_resnet18_pretrained,
        build_mobilenet_v3_small_pretrained,
    ):
        m_reg = builder(head_type="regression")
        assert m_reg(x).shape == (2,)
        m_cls = builder(head_type="classification", num_classes=6)
        assert m_cls(x).shape == (2, 6)


def test_warmup_cosine_scheduler_and_model_registry() -> None:
    cfg = load_config("configs/default.yaml")
    for name, exp_head in [
        ("resnet18_scratch", "regression"),
        ("resnet18_pretrained", "regression"),
        ("mobilenetv3_s_pretrained", "regression"),
        ("mdnet_t_reg", "regression"),
        ("mdnet_t_cls", "classification"),
    ]:
        _, ht = build_image_model(name, cfg)
        assert ht == exp_head

    param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = build_warmup_cosine_scheduler(
        opt,
        epochs=10,
        steps_per_epoch=4,
        warmup_epochs=2,
        base_lr=1e-3,
        min_lr=1e-6,
    )
    lrs = []
    for _ in range(40):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-3 / 8.0, rel=1e-5)
    assert lrs[7] == pytest.approx(1e-3, rel=1e-5)
    assert lrs[-1] < 2e-5
    assert lrs[-1] >= 1e-6
