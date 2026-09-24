from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Union, Optional, Dict, Any, Tuple


def conv_norm_act(in_channels, out_channels, sz, norm="BATCH", act="ReLU", depthwise=False):
    if norm is None or norm == "None":
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels, eps=1e-5, track_running_stats=False, affine=True)
    else:
        raise ValueError("Norm must be None, batch or instance")

    if act is None or act == "None":
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    else:
        raise ValueError("Act must be None, ReLU or Mish")

    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2, groups=in_channels),
            norm_layer,
            act_layer,
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
            norm_layer,
            act_layer,
        )


class GenericDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "BATCH",
        act: str = "ReLU",
    ):
        super().__init__()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = proj + self.conv2(x + self.conv_skip(skip))
        else:
            x = proj + self.conv2(x)
        x = x + self.conv3(x)
        return x


class GenericDecoder(nn.Module):
    def __init__(
        self,
        stage_channels: List[int],
        out_channels: Union[int, List[int]],
        norm: str = "BATCH",
        act: str = "ReLU",
    ):
        super().__init__()
        decoder_blocks = []
        in_ch = stage_channels[0]
        for skip_ch in stage_channels[1:]:
            out_ch = skip_ch
            decoder_blocks.append(GenericDecoderBlock(in_ch, skip_ch, out_ch, norm=norm, act=act))
            in_ch = out_ch
        self.decoder = nn.ModuleList(decoder_blocks)

        if isinstance(out_channels, int):
            out_channels = [out_channels]

        self.final_blocks = nn.ModuleList([
            conv_norm_act(
                in_ch,
                out_c,
                1,
                norm=norm if (norm is not None and norm.lower() != "instance") else None,
                act=None,
            )
            for out_c in out_channels
        ])

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor], target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        for block, skip in zip(self.decoder, skips[::-1]):
            x = block(x, skip)
        out = torch.cat([fb(x) for fb in self.final_blocks], dim=1)
        if target_size is not None and out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="nearest")
        return out


class UNet(nn.Module):
    """
    Main base class for U-Net models.
    Supports factory instantiation via model_type string parameter.
    """

    def __new__(cls, model_type: str = "instanseg_unet", *args, **kwargs):
        if cls is UNet:
            if not isinstance(model_type, str):
                raise ValueError(f"model_type must be a string, got {type(model_type)}")
            mtype = model_type.lower()
            mapping = {
                "instanseg_unet": InstanSeg_UNet,
                "efficientunetb0": EfficientUNetB0,
                "efficientunetb1": EfficientUNetB1,
                "efficientunetb2": EfficientUNetB2,
                "efficientunetb3": EfficientUNetB3,
                "efficientunetv2s": EfficientUNetV2S,
                "mobileunetv2": MobileUNetV2,
                "mobileunetv3s": MobileUNetV3S,
                "mobileunetv3l": MobileUNetV3L,
                "regnetunety400mf": RegNetUNetY400mf,
                "regnetunety800mf": RegNetUNetY800mf,
                "resnetunet18": ResNetUNet18,
            }
            if mtype not in mapping:
                raise ValueError(
                    f"Unsupported model_type: '{model_type}'. "
                    f"Supported model_types are: {list(mapping.keys())}"
                )
            target_cls = mapping[mtype]
            return target_cls(*args, **kwargs)
        return super().__new__(cls)


# ---------------------------------------------------------------------------
# 0. InstanSeg_UNet (Original Custom U-Net)
# ---------------------------------------------------------------------------
from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet as OriginalInstanSeg_UNet


class InstanSeg_UNet(UNet, OriginalInstanSeg_UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        layers: List[int] = [256, 128, 64, 32],
        norm: str = "BATCH",
        dropout: float = 0.0,
        act: str = "ReLU",
        **kwargs,
    ):
        OriginalInstanSeg_UNet.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            layers=layers,
            norm=norm,
            dropout=dropout,
            act=act,
        )


def _adapt_first_conv(module: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if module.in_channels == in_channels:
        return module
    new_conv = nn.Conv2d(
        in_channels,
        module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        bias=(module.bias is not None),
    )
    with torch.no_grad():
        if in_channels < 3:
            new_conv.weight.copy_(module.weight[:, :in_channels, :, :])
        else:
            new_conv.weight[:, :3, :, :].copy_(module.weight)
    return new_conv


# ---------------------------------------------------------------------------
# 1. ResNetUNet18
# ---------------------------------------------------------------------------
class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.stem(x)
        s1 = self.layer1(self.maxpool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s4, [s0, s1, s2, s3]


class ResNet18DecoderBlock(GenericDecoderBlock):
    pass


class ResNet18Decoder(GenericDecoder):
    pass


class ResNetUNet18(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = ResNet18Encoder(in_channels=in_channels)
        stage_channels = [512, 256, 128, 64, 64]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            ResNet18Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# Helper function for Sequential Features Backbones
# ---------------------------------------------------------------------------
class SequentialFeatureEncoder(nn.Module):
    def __init__(
        self,
        features: nn.Sequential,
        skip_stage_indices: List[int],
        in_channels: int = 3,
    ):
        super().__init__()
        if hasattr(features[0], "0") and isinstance(features[0][0], nn.Conv2d):
            features[0][0] = _adapt_first_conv(features[0][0], in_channels)
        elif isinstance(features[0], nn.Conv2d):
            features[0] = _adapt_first_conv(features[0], in_channels)
        self.features = features
        self.skip_indices = skip_stage_indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in self.skip_indices:
                skips.append(out)
        bottleneck = skips.pop()
        return bottleneck, skips


# ---------------------------------------------------------------------------
# 2. EfficientUNetB0
# ---------------------------------------------------------------------------
class EfficientNetB0Encoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.efficientnet_b0(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 3, 5, 7], in_channels=in_channels)


class EfficientNetB0DecoderBlock(GenericDecoderBlock):
    pass


class EfficientNetB0Decoder(GenericDecoder):
    pass


class EfficientUNetB0(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = EfficientNetB0Encoder(in_channels=in_channels)
        stage_channels = [320, 112, 40, 24, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            EfficientNetB0Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 3. EfficientUNetB1
# ---------------------------------------------------------------------------
class EfficientNetB1Encoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.efficientnet_b1(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 3, 5, 7], in_channels=in_channels)


class EfficientNetB1DecoderBlock(GenericDecoderBlock):
    pass


class EfficientNetB1Decoder(GenericDecoder):
    pass


class EfficientUNetB1(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = EfficientNetB1Encoder(in_channels=in_channels)
        stage_channels = [320, 112, 40, 24, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            EfficientNetB1Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 4. EfficientUNetB2
# ---------------------------------------------------------------------------
class EfficientNetB2Encoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.efficientnet_b2(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 3, 5, 7], in_channels=in_channels)


class EfficientNetB2DecoderBlock(GenericDecoderBlock):
    pass


class EfficientNetB2Decoder(GenericDecoder):
    pass


class EfficientUNetB2(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = EfficientNetB2Encoder(in_channels=in_channels)
        stage_channels = [352, 120, 48, 24, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            EfficientNetB2Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 5. EfficientUNetB3
# ---------------------------------------------------------------------------
class EfficientNetB3Encoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.efficientnet_b3(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 3, 5, 7], in_channels=in_channels)


class EfficientNetB3DecoderBlock(GenericDecoderBlock):
    pass


class EfficientNetB3Decoder(GenericDecoder):
    pass


class EfficientUNetB3(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = EfficientNetB3Encoder(in_channels=in_channels)
        stage_channels = [384, 136, 48, 32, 40]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            EfficientNetB3Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 6. EfficientUNetV2S
# ---------------------------------------------------------------------------
class EfficientNetV2SEncoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.efficientnet_v2_s(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 3, 5, 6], in_channels=in_channels)


class EfficientNetV2SDecoderBlock(GenericDecoderBlock):
    pass


class EfficientNetV2SDecoder(GenericDecoder):
    pass


class EfficientUNetV2S(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = EfficientNetV2SEncoder(in_channels=in_channels)
        stage_channels = [256, 160, 64, 48, 24]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            EfficientNetV2SDecoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 7. MobileUNetV2
# ---------------------------------------------------------------------------
class MobileNetV2Encoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.mobilenet_v2(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 2, 4, 11, 17], in_channels=in_channels)


class MobileNetV2DecoderBlock(GenericDecoderBlock):
    pass


class MobileNetV2Decoder(GenericDecoder):
    pass


class MobileUNetV2(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = MobileNetV2Encoder(in_channels=in_channels)
        stage_channels = [320, 96, 32, 24, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            MobileNetV2Decoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 8. MobileUNetV3S
# ---------------------------------------------------------------------------
class MobileNetV3SEncoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.mobilenet_v3_small(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 1, 3, 8, 11], in_channels=in_channels)


class MobileNetV3SDecoderBlock(GenericDecoderBlock):
    pass


class MobileNetV3SDecoder(GenericDecoder):
    pass


class MobileUNetV3S(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = MobileNetV3SEncoder(in_channels=in_channels)
        stage_channels = [96, 48, 24, 16, 16]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            MobileNetV3SDecoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 9. MobileUNetV3L
# ---------------------------------------------------------------------------
class MobileNetV3LEncoder(SequentialFeatureEncoder):
    def __init__(self, in_channels: int = 3):
        m = models.mobilenet_v3_large(weights=None)
        super().__init__(m.features, skip_stage_indices=[0, 3, 6, 12, 15], in_channels=in_channels)


class MobileNetV3LDecoderBlock(GenericDecoderBlock):
    pass


class MobileNetV3LDecoder(GenericDecoder):
    pass


class MobileUNetV3L(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = MobileNetV3LEncoder(in_channels=in_channels)
        stage_channels = [160, 112, 40, 24, 16]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            MobileNetV3LDecoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 10. RegNetUNetY400mf
# ---------------------------------------------------------------------------
class RegNetY400mfEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        m = models.regnet_y_400mf(weights=None)
        if hasattr(m.stem, "0") and isinstance(m.stem[0], nn.Conv2d):
            m.stem[0] = _adapt_first_conv(m.stem[0], in_channels)
        self.stem = m.stem
        self.trunk = m.trunk_output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.stem(x)          # 1/2 res, 32 ch
        s1 = self.trunk[0](s0)     # 1/4 res, 48 ch
        s2 = self.trunk[1](s1)     # 1/8 res, 104 ch
        s3 = self.trunk[2](s2)     # 1/16 res, 208 ch
        s4 = self.trunk[3](s3)     # 1/32 res, 440 ch (bottleneck)
        return s4, [s0, s1, s2, s3]


class RegNetY400mfDecoderBlock(GenericDecoderBlock):
    pass


class RegNetY400mfDecoder(GenericDecoder):
    pass


class RegNetUNetY400mf(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = RegNetY400mfEncoder(in_channels=in_channels)
        stage_channels = [440, 208, 104, 48, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            RegNetY400mfDecoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)


# ---------------------------------------------------------------------------
# 11. RegNetUNetY800mf
# ---------------------------------------------------------------------------
class RegNetY800mfEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        m = models.regnet_y_800mf(weights=None)
        if hasattr(m.stem, "0") and isinstance(m.stem[0], nn.Conv2d):
            m.stem[0] = _adapt_first_conv(m.stem[0], in_channels)
        self.stem = m.stem
        self.trunk = m.trunk_output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.stem(x)          # 1/2 res, 32 ch
        s1 = self.trunk[0](s0)     # 1/4 res, 64 ch
        s2 = self.trunk[1](s1)     # 1/8 res, 144 ch
        s3 = self.trunk[2](s2)     # 1/16 res, 320 ch
        s4 = self.trunk[3](s3)     # 1/32 res, 784 ch (bottleneck)
        return s4, [s0, s1, s2, s3]


class RegNetY800mfDecoderBlock(GenericDecoderBlock):
    pass


class RegNetY800mfDecoder(GenericDecoder):
    pass


class RegNetUNetY800mf(UNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = 1,
        norm: str = "BATCH",
        act: str = "ReLU",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.encoder = RegNetY800mfEncoder(in_channels=in_channels)
        stage_channels = [784, 320, 144, 64, 32]
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]
        self.decoders = nn.ModuleList([
            RegNetY800mfDecoder(stage_channels, oc, norm=norm, act=act)
            for oc in out_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        bottleneck, skips = self.encoder(x)
        return torch.cat([d(bottleneck, skips, target_size=target_size) for d in self.decoders], dim=1)
