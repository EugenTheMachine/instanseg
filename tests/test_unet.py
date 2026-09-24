import pytest
import torch
import torch.nn as nn
from instanseg.utils.models.UNet import (
    UNet,
    InstanSeg_UNet,
    EfficientUNetB0,
    EfficientUNetB1,
    EfficientUNetB2,
    EfficientUNetB3,
    EfficientUNetV2S,
    MobileUNetV2,
    MobileUNetV3S,
    MobileUNetV3L,
    RegNetUNetY400mf,
    RegNetUNetY800mf,
    ResNetUNet18,
)
from instanseg.utils.model_loader import build_model_from_dict


SUPPORTED_TYPES = [
    "instanseg_unet",
    "efficientunetb0",
    "efficientunetb1",
    "efficientunetb2",
    "efficientunetb3",
    "efficientunetv2s",
    "mobileunetv2",
    "mobileunetv3s",
    "mobileunetv3l",
    "regnetunety400mf",
    "regnetunety800mf",
    "resnetunet18",
]


def test_unet_inheritance():
    """Verify all UNet variants inherit from base UNet class."""
    classes = [
        InstanSeg_UNet,
        EfficientUNetB0,
        EfficientUNetB1,
        EfficientUNetB2,
        EfficientUNetB3,
        EfficientUNetV2S,
        MobileUNetV2,
        MobileUNetV3S,
        MobileUNetV3L,
        RegNetUNetY400mf,
        RegNetUNetY800mf,
        ResNetUNet18,
    ]
    for cls in classes:
        assert issubclass(cls, UNet), f"{cls.__name__} does not inherit from UNet"


def test_unet_invalid_type():
    """Verify invalid model_type raises ValueError."""
    with pytest.raises(ValueError):
        UNet(model_type="invalid_model_type_xyz")


@pytest.mark.parametrize("mtype", SUPPORTED_TYPES)
def test_unet_instantiation_and_forward(mtype):
    """Verify instantiation via UNet factory and forward output shape."""
    out_channels = [[2, 2, 1]]
    model = UNet(
        model_type=mtype,
        in_channels=3,
        out_channels=out_channels,
    )
    assert isinstance(model, UNet)
    
    x = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    # 2 + 2 + 1 = 5 channels
    assert out.shape == (1, 5, 128, 128), f"Model {mtype} produced output shape {out.shape}, expected (1, 5, 128, 128)"


@pytest.mark.parametrize("mtype", SUPPORTED_TYPES)
def test_build_model_from_dict_integration(mtype):
    """Verify build_model_from_dict works with each UNet model_type."""
    cfg = {
        "model_name": mtype,
        "dim_in": 3,
        "dim_coords": 2,
        "n_sigma": 2,
        "dim_seeds": 1,
        "cells_and_nuclei": False,
        "multihead": False,
        "layers": (32, 64, 128, 256),
        "norm": "BATCH",
        "dropout": 0.0,
    }
    model = build_model_from_dict(cfg)
    assert isinstance(model, UNet)
    x = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5, 128, 128)
