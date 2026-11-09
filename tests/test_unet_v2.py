"""
Tests for InstanSeg_UNetV2 (factory key: instanseg_unet_v2).

Run with:
    python -m pytest tests/test_unet_v2.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_v2(**kwargs):
    """Factory helper: returns an InstanSeg_UNetV2 in eval mode."""
    from instanseg.utils.models.InstanSeg_UNet_v2 import InstanSeg_UNetV2
    defaults = dict(in_channels=3, out_channels=[[2, 2, 1]])
    defaults.update(kwargs)
    model = InstanSeg_UNetV2(**defaults)
    model.eval()
    return model


def _count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


# ===========================================================================
# TestInstanSegUNetV2ShapeAndForward
# ===========================================================================

class TestInstanSegUNetV2ShapeAndForward:

    def test_standard_input_2_3_256_256(self):
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 5, 256, 256), f"Expected (2,5,256,256) but got {y.shape}"
        assert torch.isfinite(y).all(), "Output contains NaN or Inf"

    def test_single_channel_input_1_1_520_704(self):
        model = _make_v2(in_channels=1, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 1, 520, 704)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 5, 520, 704), f"Expected (1,5,520,704) but got {y.shape}"
        assert torch.isfinite(y).all()

    def test_non_square_input(self):
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 3, 256, 512)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 5, 256, 512), f"Expected (1,5,256,512) but got {y.shape}"

    def test_output_channels_equal_input_spatial_dims(self):
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 3, 128, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape[-2] == x.shape[-2], "Output H != Input H"
        assert y.shape[-1] == x.shape[-1], "Output W != Input W"

    def test_all_outputs_finite(self):
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    def test_forward_and_backward_succeed(self):
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1]])
        model.train()
        x = torch.randn(1, 3, 64, 64, requires_grad=False)
        y = model(x)
        loss = y.sum()
        loss.backward()
        # At least one parameter should have a gradient
        assert any(p.grad is not None for p in model.parameters())


# ===========================================================================
# TestInstanSegUNetV2Configuration
# ===========================================================================

class TestInstanSegUNetV2Configuration:

    def test_int_out_channels(self):
        model = _make_v2(in_channels=3, out_channels=5)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 5, 64, 64), f"Expected (1,5,64,64) but got {y.shape}"

    def test_flat_list_out_channels(self):
        model = _make_v2(in_channels=3, out_channels=[5, 3])
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 8, 64, 64), f"Expected (1,8,64,64) but got {y.shape}"

    def test_nested_list_out_channels(self):
        # Two decoder groups, each producing 5ch → total 10ch
        model = _make_v2(in_channels=3, out_channels=[[2, 2, 1], [2, 2, 1]])
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 10, 64, 64), f"Expected (1,10,64,64) but got {y.shape}"

    def test_norm_group(self):
        model = _make_v2(norm="GROUP")
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_norm_batch(self):
        model = _make_v2(norm="BATCH")
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_norm_none(self):
        model = _make_v2(norm=None)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_act_silu(self):
        model = _make_v2(act="SILU")
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_dropout_zero(self):
        model = _make_v2(dropout=0)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()

    def test_dropout_nonzero(self):
        model = _make_v2(dropout=0.1)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert torch.isfinite(y).all()


# ===========================================================================
# TestInstanSegUNetV2InvalidInputs
# ===========================================================================

class TestInstanSegUNetV2InvalidInputs:

    def test_wrong_tensor_rank(self):
        model = _make_v2()
        x = torch.randn(3, 64, 64)  # 3D — should be 4D
        with pytest.raises(ValueError):
            model(x)

    def test_non_floating_point(self):
        model = _make_v2()
        x = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
        with pytest.raises(ValueError):
            model(x)

    def test_incorrect_input_channels(self):
        model = _make_v2(in_channels=3)
        x = torch.randn(1, 1, 64, 64)  # wrong channels
        with pytest.raises(ValueError):
            model(x)

    def test_spatial_not_divisible_by_8(self):
        model = _make_v2()
        x = torch.randn(1, 3, 33, 256)  # 33 not divisible by 8
        with pytest.raises(ValueError):
            model(x)

    def test_invalid_dropout_probability(self):
        with pytest.raises(ValueError):
            _make_v2(dropout=1.5)

    def test_unsupported_norm(self):
        with pytest.raises(ValueError):
            _make_v2(norm="LAYER_BOGUS")

    def test_unsupported_act(self):
        with pytest.raises(ValueError):
            _make_v2(act="TANH")

    def test_bottleneck_branch_channels_validation(self):
        from instanseg.utils.models.InstanSeg_UNet_v2 import MultiscaleDilatedBottleneck
        with pytest.raises(ValueError, match="positive"):
            MultiscaleDilatedBottleneck(C=3, norm="GROUP", act="SILU")



# ===========================================================================
# TestInstanSegUNetV2Gradients
# ===========================================================================

class TestInstanSegUNetV2Gradients:

    def test_gradients_through_all_components(self):
        from instanseg.utils.models.InstanSeg_UNet_v2 import InstanSeg_UNetV2
        model = InstanSeg_UNetV2(in_channels=3, out_channels=[[2, 2, 1]])
        model.train()
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients reach key components
        components = {
            "enc0": model.enc0,
            "enc1": model.enc1,
            "enc2": model.enc2,
            "enc3": model.enc3,
            "bottleneck": model.bottleneck,
        }
        for name, module in components.items():
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
                if p.requires_grad
            )
            assert has_grad, f"No gradient in component: {name}"

        # Verify gradients reach all decoder groups and their sub-stages
        for gi, dec_group in enumerate(model.decoder_groups):
            for stage_name in ("stage0", "stage1", "stage2", "refinement"):
                stage = getattr(dec_group, stage_name)
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in stage.parameters()
                    if p.requires_grad
                )
                assert has_grad, f"No gradient in decoder_group[{gi}].{stage_name}"

            # Output heads
            for hi, head in enumerate(dec_group.heads):
                has_grad = any(
                    p.grad is not None
                    for p in head.parameters()
                    if p.requires_grad
                )
                assert has_grad, f"No gradient in decoder_group[{gi}].heads[{hi}]"


# ===========================================================================
# TestInstanSegUNetV2ParameterAccounting
# ===========================================================================

class TestInstanSegUNetV2ParameterAccounting:

    def test_parameter_count_report(self):
        from instanseg.utils.models.InstanSeg_UNet import UNet

        baseline = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        v2 = UNet(model_type="instanseg_unet_v2", in_channels=3, out_channels=[[2, 2, 1]])

        b_train, b_non = _count_params(baseline)
        v2_train, v2_non = _count_params(v2)

        print("\n--- Parameter count comparison ---")
        print(f"  Baseline (instanseg_unet): trainable={b_train:,}  non-trainable={b_non:,}")
        print(f"  V2 (instanseg_unet_v2):    trainable={v2_train:,}  non-trainable={v2_non:,}")

        # Both should have a non-trivial number of parameters
        assert b_train > 0
        assert v2_train > 0


# ===========================================================================
# TestInstanSegUNetV2BaselinePreservation
# ===========================================================================

class TestInstanSegUNetV2BaselinePreservation:

    def test_baseline_still_instantiable(self):
        from instanseg.utils.models.InstanSeg_UNet import UNet, InstanSeg_UNet
        model = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, InstanSeg_UNet)

    def test_v2_selectable_by_model_type(self):
        from instanseg.utils.models.InstanSeg_UNet import UNet
        from instanseg.utils.models.InstanSeg_UNet_v2 import InstanSeg_UNetV2
        model = UNet(model_type="instanseg_unet_v2", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, InstanSeg_UNetV2)

    def test_v2_does_not_affect_baseline_state_dict(self):
        from instanseg.utils.models.InstanSeg_UNet import UNet
        baseline_before = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        keys_before = set(baseline_before.state_dict().keys())

        # Instantiate v2 (should not mutate baseline class state)
        _ = UNet(model_type="instanseg_unet_v2", in_channels=3, out_channels=[[2, 2, 1]])

        baseline_after = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        keys_after = set(baseline_after.state_dict().keys())

        assert keys_before == keys_after, "Baseline state dict keys changed after v2 instantiation"


# ===========================================================================
# TestInstanSegUNetV2FactoryAndLoader
# ===========================================================================

class TestInstanSegUNetV2FactoryAndLoader:

    def test_factory_selects_v2(self):
        from instanseg.utils.models.InstanSeg_UNet import UNet
        from instanseg.utils.models.InstanSeg_UNet_v2 import InstanSeg_UNetV2
        model = UNet(model_type="instanseg_unet_v2", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, InstanSeg_UNetV2)
        assert type(model).__name__ == "InstanSeg_UNetV2"

    def test_build_model_from_dict_v2(self):
        from instanseg.utils.model_loader import build_model_from_dict
        from instanseg.utils.models.InstanSeg_UNet_v2 import InstanSeg_UNetV2

        build_dict = {
            "model_str": "instanseg_unet_v2",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "dim_out": 5,
            "dropprob": 0.0,
            "norm": "GROUP",
            "layers": (256, 128, 64, 32),
            "multihead": False,
            "feature_engineering": "0",
        }
        model = build_model_from_dict(build_dict)
        assert isinstance(model, InstanSeg_UNetV2)

        x = torch.randn(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 5, 64, 64), f"Expected (1,5,64,64) but got {y.shape}"
