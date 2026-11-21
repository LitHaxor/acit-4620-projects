"""Unit tests for reg_input='chi', multimodal fusion, modality dropout, and vision aux loss."""

import torch

from src.config import load_config
from src.models.fusion import PFRMLFusionModel
from src.models.pfrml_config import build_pfrml_from_config
from src.pfrml_losses import PFRMLMultitaskLoss

torch.set_num_threads(2)


def test_fusion_r1_and_fusion_full_shapes() -> None:
    """Verify forward output shapes and MLP input dimensions for fusion_r1 and fusion_full."""
    bs = 4
    images = torch.randn(bs, 3, 224, 224)
    metadata = torch.randn(bs, 5)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    cfg_r1 = load_config("configs/fusion_regression_only_neck.yaml")
    cfg_r1.pfrml.pretrained = False
    model_r1 = PFRMLFusionModel(
        backbone_name="mobilenetv3_s",
        pretrained=False,
        use_prototypes=False,
        use_attention_pooling=False,
        use_physics_gate=False,
        modality_dropout=0.3,
    )
    model_r1.eval()
    # [chi (224) ; m (32)] = 256
    assert model_r1.head.reg_head.in_dim == 256
    out_r1 = model_r1(images, metadata, targets=None)
    assert out_r1.reg_pred.shape == (bs,)
    assert out_r1.aux_reg_pred is not None and out_r1.aux_reg_pred.shape == (bs,)
    assert out_r1.chi.shape == (bs, 224)
    assert out_r1.meta_embed is not None and out_r1.meta_embed.shape == (bs, 32)
    assert out_r1.logits is None and out_r1.probs is None

    cfg_full = load_config("configs/fusion_pfrml_full.yaml")
    assert cfg_full.data.group_mode == "spatial_cluster"
    assert cfg_r1.data.group_mode == "spatial_cluster"
    model_full = PFRMLFusionModel(
        backbone_name="mobilenetv3_s",
        pretrained=False,
        use_prototypes=True,
        use_attention_pooling=True,
        use_physics_gate=True,
        modality_dropout=0.3,
    )
    model_full.train()
    # [chi (224) ; c_tilde (224) ; m (32)] = 480
    assert model_full.head.reg_head.in_dim == 480
    out_full = model_full(images, metadata, targets=targets)
    assert out_full.reg_pred.shape == (bs,)
    assert out_full.aux_reg_pred is not None and out_full.aux_reg_pred.shape == (bs,)
    assert out_full.logits is not None and out_full.logits.shape == (bs, 6)
    assert out_full.probs is not None and out_full.probs.shape == (bs, 6)
    assert out_full.meta_embed is not None and out_full.meta_embed.shape == (bs, 32)


def test_modality_dropout_only_in_train_mode_and_per_sample() -> None:
    """Modality dropout zeroes entire 32-D m per sample in train mode and never in eval mode."""
    torch.manual_seed(123)
    model = PFRMLFusionModel(
        backbone_name="mobilenetv3_s",
        pretrained=False,
        use_prototypes=False,
        modality_dropout=0.5,
    )
    bs = 64
    metadata = torch.randn(bs, 5) + 2.0

    # Raw metadata embedding without modality dropout
    model.eval()
    with torch.no_grad():
        m_eval = model.encode_metadata(
            metadata=metadata,
            batch_size=bs,
            dtype=torch.float32,
            device=metadata.device,
            metadata_off=False,
        )
        m_raw = model.metadata_mlp.forward_features(metadata)
        # In eval mode, modality dropout is never applied
        assert torch.equal(m_eval, m_raw)
        assert not torch.all(m_raw == 0.0, dim=-1).any()

    # In train mode, modality dropout zeroes entire 32-D vectors per sample
    model.train()
    with torch.no_grad():
        m_train = model.encode_metadata(
            metadata=metadata,
            batch_size=bs,
            dtype=torch.float32,
            device=metadata.device,
            metadata_off=False,
        )
    is_zero_row = torch.all(m_train == 0.0, dim=-1)
    is_exact_raw_row = torch.all(m_train == m_raw, dim=-1)
    # Every sample must be EITHER completely zeroed OR completely untouched (per-sample dropout)
    assert torch.all(is_zero_row | is_exact_raw_row)
    # With B=64 and p_drop=0.5, some samples are dropped and some are kept
    assert int(is_zero_row.sum().item()) > 0
    assert int(is_exact_raw_row.sum().item()) > 0


def test_metadata_off_equals_forward_with_m_zero() -> None:
    """Calling forward with metadata_off=True equals a forward with m = 0."""
    torch.manual_seed(42)
    for use_protos in (False, True):
        model = PFRMLFusionModel(
            backbone_name="mobilenetv3_s",
            pretrained=False,
            use_prototypes=use_protos,
            use_attention_pooling=use_protos,
            use_physics_gate=use_protos,
            modality_dropout=0.3,
        )
        model.eval()
        bs = 4
        images = torch.randn(bs, 3, 224, 224)
        metadata = torch.randn(bs, 5)
        zeros_m = torch.zeros(bs, 32, dtype=images.dtype)

        with torch.no_grad():
            out_meta_off = model(images, metadata=metadata, metadata_off=True)
            out_m_zero = model(images, metadata=metadata, m_override=zeros_m)

        assert torch.allclose(out_meta_off.reg_pred, out_m_zero.reg_pred, atol=1e-6)
        assert out_meta_off.meta_embed is not None
        assert torch.all(out_meta_off.meta_embed == 0.0)


def test_aux_loss_gradient_wrt_metadata_mlp_is_zero() -> None:
    """Aux-loss gradient flows into image backbone/aux head, but is strictly zero w.r.t. MetadataMLP."""
    torch.manual_seed(7)
    model = PFRMLFusionModel(
        backbone_name="mobilenetv3_s",
        pretrained=False,
        use_prototypes=True,
        modality_dropout=0.0,
    )
    model.train()
    bs = 4
    images = torch.randn(bs, 3, 224, 224)
    metadata = torch.randn(bs, 5)
    target_aqi = torch.rand(bs)

    # 1. Backward from auxiliary vision loss alone: MetadataMLP gets zero gradient
    model.zero_grad(set_to_none=True)
    out = model(images, metadata=metadata, targets=None)
    assert out.aux_reg_pred is not None
    aux_loss = ((out.aux_reg_pred - target_aqi) ** 2).mean()
    aux_loss.backward()

    for name, p in model.metadata_mlp.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0.0), (
            f"Expected zero aux-loss grad on MetadataMLP.{name}"
        )
    backbone_grad_norm = sum(
        float(p.grad.abs().sum().item())
        for p in model.backbone.parameters()
        if p.grad is not None
    )
    assert backbone_grad_norm > 0.0

    # 2. In total loss (main + 0.5 * aux), changing aux_loss_weight does not alter MetadataMLP grads
    model.zero_grad(set_to_none=True)
    torch.manual_seed(99)
    out1 = model(images, metadata=metadata, targets=None)
    loss_no_aux = PFRMLMultitaskLoss(use_prototypes=True, aux_loss_weight=0.0)
    l1, _ = loss_no_aux(out1, torch.zeros(bs, dtype=torch.long), target_aqi)
    l1.backward()
    grads_no_aux = {
        k: p.grad.detach().clone()
        for k, p in model.metadata_mlp.encoder.named_parameters()
        if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    torch.manual_seed(99)
    out2 = model(images, metadata=metadata, targets=None)
    loss_with_aux = PFRMLMultitaskLoss(use_prototypes=True, aux_loss_weight=0.5)
    l2, parts = loss_with_aux(out2, torch.zeros(bs, dtype=torch.long), target_aqi)
    assert "aux_reg_loss" in parts
    l2.backward()
    for k, p in model.metadata_mlp.encoder.named_parameters():
        assert p.grad is not None
        assert torch.allclose(p.grad, grads_no_aux[k], atol=1e-6)


def test_prototype_buffers_unchanged_by_eval_forward_in_fusion() -> None:
    """Eval forward in fusion_full_mobilenetv3_s leaves prototype buffers completely unchanged."""
    torch.manual_seed(11)
    model = PFRMLFusionModel(
        backbone_name="mobilenetv3_s",
        pretrained=False,
        use_prototypes=True,
        use_attention_pooling=True,
        use_physics_gate=True,
    )
    assert model.head.proto_cls is not None
    bs = 4
    images = torch.randn(bs, 3, 224, 224)
    metadata = torch.randn(bs, 5)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # Train forward updates prototypes
    model.train()
    _ = model(images, metadata=metadata, targets=targets)
    proto_snapshot = model.head.proto_cls.prototypes.detach().clone()
    counts_snapshot = model.head.proto_cls.class_counts.detach().clone()

    # Eval forward (even if targets are passed or metadata_off=True) must never mutate buffers
    model.eval()
    with torch.no_grad():
        _ = model(images, metadata=metadata, targets=targets, metadata_off=False)
        _ = model(images, metadata=metadata, targets=targets, metadata_off=True)

    assert torch.equal(model.head.proto_cls.prototypes, proto_snapshot)
    assert torch.equal(model.head.proto_cls.class_counts, counts_snapshot)


def test_reg_input_chi_regression_unchanged_when_prototype_probs_perturbed() -> None:
    """With reg_input='chi', regression output h(chi) is unchanged when prototype probs/prototypes change."""
    torch.manual_seed(21)
    cfg = load_config("configs/pfrml_eq5_regression.yaml")
    assert cfg.pfrml.reg_input == "chi"
    assert cfg.pfrml.use_prototypes is True

    model, _ = build_pfrml_from_config(cfg, backbone_name="mobilenetv3_s", pretrained=False)
    model.eval()
    assert model.head.proto_cls is not None
    assert model.head.reg_head.in_dim == 224
    assert model.head.reg_head.uses_proto_in_reg is False

    bs = 4
    images = torch.randn(bs, 3, 224, 224)
    with torch.no_grad():
        out1 = model(images, targets=None)
        assert out1.logits is not None and out1.probs is not None

        # Direct call to reg_head with completely different probability distributions
        probs_a = torch.softmax(torch.randn(bs, 6), dim=-1)
        probs_b = torch.softmax(torch.randn(bs, 6) * 5.0, dim=-1)
        p_norm = model.head.proto_cls.prototypes.detach()
        pred_a = model.head.reg_head(out1.chi, probs_a, p_norm)
        pred_b = model.head.reg_head(out1.chi, probs_b, p_norm)
        assert torch.equal(pred_a, pred_b)

        # Perturb prototypes and temperature log_tau in proto_cls -> logits/probs change, reg_pred does not
        model.head.proto_cls.prototypes.copy_(torch.randn_like(model.head.proto_cls.prototypes))
        model.head.proto_cls.log_tau.fill_(1.5)
        out2 = model(images, targets=None)
        assert out2.probs is not None
        assert not torch.allclose(out1.probs, out2.probs)
        assert torch.equal(out1.reg_pred, out2.reg_pred)
