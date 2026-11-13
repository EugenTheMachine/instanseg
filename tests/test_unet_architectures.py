import pytest
import torch
import torch.nn as nn
from pathlib import Path

from instanseg.utils.models.InstanSeg_UNet import (
    UNet,
    InstanSeg_UNet,
    MedNeXt_UNet,
    MedNeXtBlock2D,
    MedNeXtDownBlock2D,
    MedNeXtUpBlock2D,
    MedNeXtDecoder2D,
    LayerNorm2d,
    ImprovedUNet,
    ImprovedDecoderBlock,
    EfficientUNetB0,
    EfficientEncoder,
    EfficientNetEncoder,
    EfficientDecoderBlock,
    EfficientNetDecoderBlock,
    EfficientDecoder,
    EfficientNetDecoder,
    ConvNeXtUNet,
    ConvNeXtTinyUNet,
    ConvNeXtEncoder,
    ConvNeXtTinyEncoder,
    ConvNeXtDecoderBlock,
    ConvNeXtTinyDecoderBlock,
    ConvNeXtDecoder,
    ConvNeXtTinyDecoder,
)
from instanseg.utils.model_loader import build_model_from_dict
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss


class TestLegacyRegression:
    """
    Regression tests using state-dict/output/gradient references captured
    from the unmodified legacy implementation.
    """

    @pytest.fixture
    def legacy_ref(self):
        ref_path = Path(__file__).parent / "legacy_reference.pt"
        if not ref_path.exists():
            pytest.skip("legacy_reference.pt not found")
        return torch.load(ref_path, weights_only=False)

    def test_legacy_strict_loading_single_head(self, legacy_ref):
        model = UNet(
            model_type="instanseg_unet",
            in_channels=3,
            out_channels=[[2, 2, 1]],
        )
        # Verify strict loading into the refactored legacy route
        incompatible = model.load_state_dict(legacy_ref["single_state_dict"], strict=True)
        assert len(incompatible.missing_keys) == 0
        assert len(incompatible.unexpected_keys) == 0

        # Verify numerical equivalence of forward output in both eval and train modes
        model.eval()
        with torch.no_grad():
            output_eval = model(legacy_ref["input"])
        torch.testing.assert_close(output_eval, legacy_ref["single_output_eval"], rtol=1e-5, atol=1e-5)

        model.train()
        with torch.no_grad():
            output_train = model(legacy_ref["input"])
        torch.testing.assert_close(output_train, legacy_ref["single_output_train"], rtol=1e-5, atol=1e-5)

    def test_legacy_strict_loading_multi_head(self, legacy_ref):
        model = UNet(
            model_type="instanseg_unet",
            in_channels=3,
            out_channels=[[2], [2], [1]],
        )
        incompatible = model.load_state_dict(legacy_ref["multi_state_dict"], strict=True)
        assert len(incompatible.missing_keys) == 0
        assert len(incompatible.unexpected_keys) == 0

        model.eval()
        with torch.no_grad():
            output_eval = model(legacy_ref["input"])
        torch.testing.assert_close(output_eval, legacy_ref["multi_output_eval"], rtol=1e-5, atol=1e-5)

        model.train()
        with torch.no_grad():
            output_train = model(legacy_ref["input"])
        torch.testing.assert_close(output_train, legacy_ref["multi_output_train"], rtol=1e-5, atol=1e-5)

    def test_legacy_gradients_single_head(self, legacy_ref):
        model = UNet(
            model_type="instanseg_unet",
            in_channels=3,
            out_channels=[[2, 2, 1]],
        )
        model.load_state_dict(legacy_ref["single_state_dict"], strict=True)
        model.train()
        output = model(legacy_ref["input"])
        output.sum().backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert name in legacy_ref["single_grads"]
                torch.testing.assert_close(
                    param.grad,
                    legacy_ref["single_grads"][name],
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_legacy_gradients_multi_head(self, legacy_ref):
        model = UNet(
            model_type="instanseg_unet",
            in_channels=3,
            out_channels=[[2], [2], [1]],
        )
        model.load_state_dict(legacy_ref["multi_state_dict"], strict=True)
        model.train()
        output = model(legacy_ref["input"])
        output.sum().backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert name in legacy_ref["multi_grads"]
                torch.testing.assert_close(
                    param.grad,
                    legacy_ref["multi_grads"][name],
                    rtol=1e-5,
                    atol=1e-5,
                )


class TestUNetInterfaceAndFactory:
    """
    Tests for the UNet parent/base class and construction interface.
    """

    def test_factory_selects_instanseg_unet(self):
        model = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, InstanSeg_UNet)
        assert model.model_type == "instanseg_unet"

    def test_factory_selects_efficientunetb0(self):
        model = UNet(model_type="efficientunetb0", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, EfficientUNetB0)
        assert model.model_type == "efficientunetb0"

    def test_factory_case_insensitive(self):
        m1 = UNet(model_type="InstanSeg_UNet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(m1, InstanSeg_UNet)
        m2 = UNet(model_type="EfficientUNetB0", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(m2, EfficientUNetB0)

    def test_factory_rejects_unsupported_model_type(self):
        unsupported = [
            "resnet18",
            "efficientunetb1",
            "efficientunetb2",
            "mobileunetv2",
            "unknown",
            "",
            None,
        ]
        for mt in unsupported:
            with pytest.raises(ValueError):
                UNet(model_type=mt, in_channels=3, out_channels=[[2, 2, 1]])

    def test_direct_construction_backward_compatibility(self):
        m_legacy = InstanSeg_UNet(in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(m_legacy, UNet)
        assert isinstance(m_legacy, InstanSeg_UNet)

        m_eff = EfficientUNetB0(in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(m_eff, UNet)
        assert isinstance(m_eff, EfficientUNetB0)

    def test_no_wrapper_dimension_in_parameter_names(self):
        m_direct = InstanSeg_UNet(in_channels=3, out_channels=[[2, 2, 1]])
        m_factory = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        direct_keys = set(m_direct.state_dict().keys())
        factory_keys = set(m_factory.state_dict().keys())
        assert direct_keys == factory_keys
        for k in factory_keys:
            assert not k.startswith("model.")
            assert not k.startswith("unet.")


class TestInputContract:
    """
    Tests for the input contract of both supported architectures.
    """

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_valid_input_passes_through(self, model_type):
        model = UNet(model_type=model_type, in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 5, 64, 64)

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_rejects_non_4d_input(self, model_type):
        model = UNet(model_type=model_type, in_channels=3, out_channels=[[2, 2, 1]])
        with pytest.raises(ValueError):
            model(torch.randn(3, 64, 64))  # 3D
        with pytest.raises(ValueError):
            model(torch.randn(1, 1, 3, 64, 64))  # 5D

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_rejects_non_floating_point(self, model_type):
        model = UNet(model_type=model_type, in_channels=3, out_channels=[[2, 2, 1]])
        with pytest.raises(ValueError):
            model(torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8))

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_rejects_channel_mismatch(self, model_type):
        model = UNet(model_type=model_type, in_channels=3, out_channels=[[2, 2, 1]])
        with pytest.raises(ValueError):
            model(torch.randn(1, 4, 64, 64))  # 4 channels instead of 3

    def test_instanseg_unet_spatial_divisibility(self):
        # Default 4-block encoder requires divisibility by 2 ** (4 - 1) = 8
        model = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[[2, 2, 1]])
        # Divisible by 8: should succeed
        out = model(torch.randn(1, 3, 32, 40))
        assert out.shape == (1, 5, 32, 40)

        # Not divisible by 8: must reject with ValueError
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 33, 40))
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 32, 35))

    def test_efficientunetb0_spatial_divisibility(self):
        # 5 downsamplings of factor 2 require divisibility by 32
        model = UNet(model_type="efficientunetb0", in_channels=3, out_channels=[[2, 2, 1]])
        # Divisible by 32: should succeed
        out = model(torch.randn(1, 3, 64, 96))
        assert out.shape == (1, 5, 64, 96)

        # Not divisible by 32: must reject with ValueError
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 48, 64))
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 64, 50))


class TestOutputContractAndMultihead:
    """
    Tests for output tensor format, channel counts, and multihead modes.
    """

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_output_restores_exact_spatial_dimensions(self, model_type):
        model = UNet(model_type=model_type, in_channels=3, out_channels=[[2, 2, 1]])
        for h, w in [(64, 64), (64, 128), (128, 64)]:
            x = torch.randn(2, 3, h, w)
            out = model(x)
            assert out.shape == (2, 5, h, w)

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_single_vs_multihead_output_shape(self, model_type):
        dim_coords = 2
        n_sigma = 2
        dim_seeds = 1
        total_channels = dim_coords + n_sigma + dim_seeds

        # Single-head: one decoder emitting coordinate, sigma, and seed channels
        m_single = UNet(
            model_type=model_type,
            in_channels=3,
            out_channels=[[dim_coords, n_sigma, dim_seeds]],
        )
        # Multi-head: three decoders emitting each group independently
        m_multi = UNet(
            model_type=model_type,
            in_channels=3,
            out_channels=[[dim_coords], [n_sigma], [dim_seeds]],
        )

        x = torch.randn(2, 3, 64, 64)
        out_single = m_single(x)
        out_multi = m_multi(x)

        assert out_single.shape == (2, total_channels, 64, 64)
        assert out_multi.shape == (2, total_channels, 64, 64)
        assert isinstance(out_single, torch.Tensor)
        assert isinstance(out_multi, torch.Tensor)

    @pytest.mark.parametrize("model_type", ["instanseg_unet", "efficientunetb0"])
    def test_out_channels_conventions(self, model_type):
        # Integer: single head with that many channels
        m_int = UNet(model_type=model_type, in_channels=3, out_channels=4)
        out_int = m_int(torch.randn(1, 3, 64, 64))
        assert out_int.shape == (1, 4, 64, 64)

        # Flat list: one decoder with multiple output groups
        m_flat = UNet(model_type=model_type, in_channels=3, out_channels=[2, 3])
        out_flat = m_flat(torch.randn(1, 3, 64, 64))
        assert out_flat.shape == (1, 5, 64, 64)

        # List of lists: multiple decoders
        m_nested = UNet(model_type=model_type, in_channels=3, out_channels=[[2], [3]])
        out_nested = m_nested(torch.randn(1, 3, 64, 64))
        assert out_nested.shape == (1, 5, 64, 64)


class TestEfficientUNetB0ModularArchitecture:
    """
    Tests for modularity and components of EfficientUNetB0.
    """

    def test_encoder_component(self):
        enc = EfficientEncoder(in_channels=3)
        assert isinstance(enc, nn.Module)
        assert isinstance(enc, EfficientNetEncoder)

        x = torch.randn(1, 3, 64, 64)
        bottleneck, skips = enc(x)
        assert bottleneck.shape == (1, 320, 2, 2)
        assert len(skips) == 5
        # Skip dimensions from shallow to deep (skips[0] is None to avoid raw-pixel injection):
        assert skips[0] is None
        assert skips[1].shape == (1, 16, 32, 32)  # scale 1/2
        assert skips[2].shape == (1, 24, 16, 16)  # scale 1/4
        assert skips[3].shape == (1, 40, 8, 8)    # scale 1/8
        assert skips[4].shape == (1, 112, 4, 4)   # scale 1/16

    def test_decoder_block_component(self):
        block = EfficientDecoderBlock(
            in_channels=320,
            skip_channels=112,
            out_channels=112,
        )
        assert isinstance(block, nn.Module)
        assert isinstance(block, EfficientNetDecoderBlock)

        x = torch.randn(1, 320, 4, 4)
        skip = torch.randn(1, 112, 8, 8)
        out = block(x, skip)
        assert out.shape == (1, 112, 8, 8)

    def test_decoder_component(self):
        stage_channels = [
            (320, 112, 112),
            (112, 40, 40),
            (40, 24, 24),
            (24, 16, 16),
            (16, 0, 16),
        ]
        dec = EfficientDecoder(stage_channels, out_channels=[2, 2, 1])
        assert isinstance(dec, nn.Module)
        assert isinstance(dec, EfficientNetDecoder)

        bottleneck = torch.randn(1, 320, 2, 2)
        skips = [
            None,
            torch.randn(1, 16, 32, 32),
            torch.randn(1, 24, 16, 16),
            torch.randn(1, 40, 8, 8),
            torch.randn(1, 112, 4, 4),
        ]
        out = dec(bottleneck, skips)
        assert out.shape == (1, 5, 64, 64)

    def test_model_composition(self):
        model = EfficientUNetB0(in_channels=3, out_channels=[[2, 2, 1]])
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "decoders")
        assert isinstance(model.encoder, EfficientEncoder)
        assert isinstance(model.decoder, EfficientDecoder)

    def test_custom_in_channels(self):
        model = EfficientUNetB0(in_channels=1, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)


class TestModelLoaderAndLossIntegration:
    """
    Tests end-to-end integration with model_loader and InstanSegLoss.
    """

    def test_build_model_from_dict_instanseg_unet(self):
        cfg = {
            "model_str": "instanseg_unet",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "norm": "BATCH",
            "dropprob": 0.0,
        }
        model = build_model_from_dict(cfg)
        assert isinstance(model, InstanSeg_UNet)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)

    def test_build_model_from_dict_efficientunetb0(self):
        cfg = {
            "model_str": "efficientunetb0",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "norm": "BATCH",
            "dropprob": 0.0,
        }
        model = build_model_from_dict(cfg)
        assert isinstance(model, EfficientUNetB0)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)

    def test_build_mobilenet_with_pretrained_backbone(self, monkeypatch):
        import torchvision.models as torchvision_models

        original_mobilenet_v2 = torchvision_models.mobilenet_v2
        requested_weights = []

        def fake_mobilenet_v2(*, weights):
            requested_weights.append(weights)
            return original_mobilenet_v2(weights=None)

        monkeypatch.setattr(torchvision_models, "mobilenet_v2", fake_mobilenet_v2)
        cfg = {
            "model_str": "mobilenetv2_unet",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "norm": "BATCH",
            "dropprob": 0.0,
            "pretrained_backbone": True,
        }

        model = build_model_from_dict(cfg)
        output = model(torch.rand(1, 3, 64, 64))

        assert requested_weights == [torchvision_models.MobileNet_V2_Weights.DEFAULT]
        assert model.encoder.use_imagenet_normalization
        assert output.shape == (1, 5, 64, 64)

        cfg["dim_in"] = 1
        grayscale_model = build_model_from_dict(cfg)
        grayscale_output = grayscale_model(torch.rand(1, 1, 64, 64))
        assert requested_weights[-1] == torchvision_models.MobileNet_V2_Weights.DEFAULT
        assert grayscale_output.shape == (1, 5, 64, 64)

        cfg["pretrained_backbone"] = False
        scratch_model = build_model_from_dict(cfg)
        assert requested_weights[-1] is None
        assert not scratch_model.encoder.use_imagenet_normalization

    def test_build_model_from_dict_dim_in_zero_or_none(self):
        for dim in [0, None]:
            cfg = {
                "model_str": "efficientunetb0",
                "dim_in": dim,
                "dim_coords": 2,
                "n_sigma": 2,
                "dim_seeds": 1,
                "multihead": False,
                "layers": [32, 64, 128, 256],
                "norm": "BATCH",
            }
            model = build_model_from_dict(cfg)
            assert model.in_channels == 3
            out = model(torch.randn(1, 3, 64, 64))
            assert out.shape == (1, 5, 64, 64)

    def test_pixel_classifier_and_loss_forward(self):
        cfg = {
            "model_str": "efficientunetb0",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "norm": "BATCH",
        }
        model = build_model_from_dict(cfg)
        loss_fn = InstanSegLoss(
            binary_loss_fn_str="lovasz_hinge",
            seed_loss_fn="l1_distance",
            device=torch.device("cpu"),
            n_sigma=2,
            dim_coords=2,
            dim_seeds=1,
        )
        model = loss_fn.initialize_pixel_classifier(model, MLP_width=5)
        assert hasattr(model, "pixel_classifier")

        # Test forward pass through model
        x = torch.randn(1, 3, 64, 64)
        pred = model(x)
        assert pred.shape == (1, 5, 64, 64)

        # Create dummy target instance labels (1, 1, 64, 64)
        targets = torch.zeros((1, 1, 64, 64), dtype=torch.int32)
        targets[0, 0, 10:25, 10:25] = 1
        targets[0, 0, 35:50, 35:50] = 2

        loss = loss_fn(pred, targets)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        loss.backward()
        # Verify gradients flowed back into backbone
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestConvNeXtUNetModularArchitecture:
    """
    Tests for modularity and components of ConvNeXtUNet.
    """

    def test_factory_selects_convnext_unet(self):
        model = UNet(model_type="convnext_unet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, ConvNeXtUNet)
        assert model.model_type == "convnext_unet"

        m_alias = UNet(model_type="convnext_tiny", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(m_alias, ConvNeXtUNet)

    def test_encoder_component(self):
        enc = ConvNeXtEncoder(in_channels=3)
        assert isinstance(enc, nn.Module)
        assert isinstance(enc, ConvNeXtTinyEncoder)

        x = torch.randn(1, 3, 64, 64)
        bottleneck, skips = enc(x)
        assert bottleneck.shape == (1, 768, 2, 2)
        assert len(skips) == 5
        assert skips[0] is None
        assert skips[1] is None
        assert skips[2].shape == (1, 96, 16, 16)
        assert skips[3].shape == (1, 192, 8, 8)
        assert skips[4].shape == (1, 384, 4, 4)

    def test_decoder_block_component(self):
        block = ConvNeXtDecoderBlock(
            in_channels=768,
            skip_channels=384,
            out_channels=384,
        )
        assert isinstance(block, nn.Module)
        assert isinstance(block, ConvNeXtTinyDecoderBlock)

        x = torch.randn(1, 768, 4, 4)
        skip = torch.randn(1, 384, 8, 8)
        out = block(x, skip)
        assert out.shape == (1, 384, 8, 8)

    def test_decoder_component(self):
        stage_channels = [
            (768, 384, 384),
            (384, 192, 192),
            (192, 96, 96),
            (96, 0, 48),
            (48, 0, 32),
        ]
        dec = ConvNeXtDecoder(stage_channels, out_channels=[2, 2, 1])
        assert isinstance(dec, nn.Module)
        assert isinstance(dec, ConvNeXtTinyDecoder)

        bottleneck = torch.randn(1, 768, 2, 2)
        skips = [
            None,
            None,
            torch.randn(1, 96, 16, 16),
            torch.randn(1, 192, 8, 8),
            torch.randn(1, 384, 4, 4),
        ]
        out = dec(bottleneck, skips)
        assert out.shape == (1, 5, 64, 64)

    def test_model_composition(self):
        model = ConvNeXtUNet(in_channels=3, out_channels=[[2, 2, 1]])
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "decoders")
        assert isinstance(model.encoder, ConvNeXtEncoder)
        assert isinstance(model.decoder, ConvNeXtDecoder)

    def test_custom_in_channels(self):
        model = ConvNeXtUNet(in_channels=1, out_channels=[[2, 2, 1]])
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)

    def test_convnext_unet_spatial_divisibility(self):
        model = ConvNeXtUNet(in_channels=3, out_channels=[[2, 2, 1]])
        out = model(torch.randn(1, 3, 64, 96))
        assert out.shape == (1, 5, 64, 96)
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 48, 64))

    def test_build_model_from_dict_convnext_unet(self):
        cfg = {
            "model_str": "convnext_unet",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "norm": "BATCH",
            "dropprob": 0.0,
        }
        model = build_model_from_dict(cfg)
        assert isinstance(model, ConvNeXtUNet)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)


class TestImprovedUNet:
    """
    Unit tests for ImprovedUNet and ImprovedDecoderBlock.
    """

    def test_factory_selects_improved_unet(self):
        model = UNet(model_type="improved_unet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, InstanSeg_UNet)
        assert model.model_type == "improved_unet"

    def test_direct_construction_and_inheritance(self):
        from instanseg.utils.models.InstanSeg_UNet import ImprovedUNet, ImprovedDecoderBlock, Instanseg_Unet
        model = ImprovedUNet(in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, InstanSeg_UNet)
        assert isinstance(model, Instanseg_Unet)
        assert issubclass(ImprovedUNet, InstanSeg_UNet)
        assert model.model_type == "improved_unet"

        # Verify decoders use ImprovedDecoderBlock
        for decoder in model.decoders:
            for block in decoder.decoder:
                assert isinstance(block, ImprovedDecoderBlock)

    def test_forward_pass_single_head(self):
        model = UNet(model_type="improved_unet", in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64)

    def test_forward_pass_multi_head(self):
        model = UNet(model_type="improved_unet", in_channels=3, out_channels=[[2], [2], [1]])
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64)

    def test_spatial_divisibility(self):
        model = UNet(model_type="improved_unet", in_channels=3, out_channels=[[2, 2, 1]])
        out = model(torch.randn(1, 3, 32, 40))
        assert out.shape == (1, 5, 32, 40)
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 33, 40))

    def test_build_model_from_dict_improved_unet(self):
        cfg = {
            "model_str": "improved_unet",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "norm": "BATCH",
            "dropprob": 0.0,
        }
        model = build_model_from_dict(cfg)
        from instanseg.utils.models.InstanSeg_UNet import ImprovedUNet
        assert isinstance(model, ImprovedUNet)
        assert isinstance(model, InstanSeg_UNet)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)



class TestMedNeXtUNet:
    """
    Unit tests for MedNeXt-2D architecture (MedNeXt_UNet) and building blocks.
    """

    def test_layer_norm_2d(self):
        norm = LayerNorm2d(32)
        x = torch.randn(2, 32, 16, 16)
        out = norm(x)
        assert out.shape == (2, 32, 16, 16)
        # Verify mean ~ 0 and variance ~ 1 along channel dimension
        assert torch.allclose(out.mean(dim=1), torch.zeros(2, 16, 16), atol=1e-4)

    def test_mednext_block_2d(self):
        block = MedNeXtBlock2D(in_channels=32, out_channels=32, exp_ratio=2, k_size=7)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    def test_mednext_down_block_2d(self):
        down = MedNeXtDownBlock2D(in_channels=32, out_channels=64, exp_ratio=2, k_size=7)
        x = torch.randn(2, 32, 32, 32)
        out = down(x)
        assert out.shape == (2, 64, 16, 16)

    def test_mednext_up_block_2d(self):
        up = MedNeXtUpBlock2D(in_channels=64, skip_channels=32, out_channels=32, exp_ratio=2)
        x = torch.randn(2, 64, 16, 16)
        skip = torch.randn(2, 32, 32, 32)
        out = up(x, skip)
        assert out.shape == (2, 32, 32, 32)

    def test_mednext_decoder_2d(self):
        decoder = MedNeXtDecoder2D(layers=[128, 64, 32], out_channels=[2, 2, 1], exp_ratio=2)
        x = torch.randn(2, 128, 8, 8)
        skips = [torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16)]
        out = decoder(x, skips)
        assert out.shape == (2, 5, 32, 32)

    def test_factory_selects_mednext_unet(self):
        model = UNet(model_type="mednext_unet", in_channels=3, out_channels=[[2, 2, 1]])
        assert isinstance(model, UNet)
        assert isinstance(model, MedNeXt_UNet)
        assert model.model_type == "mednext_unet"

    def test_forward_pass_single_head(self):
        model = UNet(model_type="mednext_unet", in_channels=3, out_channels=[[2, 2, 1]])
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64)

    def test_forward_pass_multi_head(self):
        model = UNet(model_type="mednext_unet", in_channels=3, out_channels=[[2], [2], [1]])
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64)

    def test_spatial_divisibility(self):
        model = UNet(model_type="mednext_unet", in_channels=3, out_channels=[[2, 2, 1]])
        out = model(torch.randn(1, 3, 32, 40))
        assert out.shape == (1, 5, 32, 40)
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 33, 40))

    def test_deep_variant(self):
        shallow = MedNeXt_UNet(in_channels=3, out_channels=[[2, 2, 1]], block_depth=1)
        deep = MedNeXt_UNet(in_channels=3, out_channels=[[2, 2, 1]], block_depth=2)
        p_shallow = sum(p.numel() for p in shallow.parameters())
        p_deep = sum(p.numel() for p in deep.parameters())
        assert p_deep > p_shallow
        x = torch.randn(1, 3, 32, 32)
        assert deep(x).shape == (1, 5, 32, 32)

    def test_build_model_from_dict_mednext(self):
        cfg = {
            "model_str": "mednext_unet",
            "dim_in": 3,
            "dim_coords": 2,
            "n_sigma": 2,
            "dim_seeds": 1,
            "multihead": False,
            "layers": [32, 64, 128, 256],
            "exp_ratio": 2,
            "k_size": 7,
        }
        model = build_model_from_dict(cfg)
        assert isinstance(model, MedNeXt_UNet)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5, 64, 64)
