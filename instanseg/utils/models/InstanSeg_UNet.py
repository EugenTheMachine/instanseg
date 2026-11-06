from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
import torchvision.models as models


# =====================================================================
# 1. BASE HELPER FUNCTIONS & PARENT BASE CLASS
# =====================================================================

def conv_norm_act(
    in_channels: int,
    out_channels: int,
    sz: int,
    norm: str = "BATCH",
    act: str = "GELU",
    depthwise: bool = False,
) -> nn.Sequential:
    """Helper module combining Convolution, Normalization, and Activation."""
    if norm is None or norm == "None":
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels, eps=1e-5, track_running_stats=False, affine=True)
    elif norm.lower() in ("layer", "group"):
        # GroupNorm with 1 group acts as LayerNorm over Spatial/Channel dimensions
        norm_layer = nn.GroupNorm(1, out_channels, eps=1e-5)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    if act is None or act == "None":
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    elif act.lower() in ("silu", "swish"):
        act_layer = nn.SiLU(inplace=True)
    elif act.lower() == "gelu":
        act_layer = nn.GELU()
    else:
        raise ValueError(f"Unsupported act: {act}")

    groups = in_channels if depthwise else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            sz,
            padding=sz // 2,
            groups=groups,
            bias=(norm is None or norm == "None"),
        ),
        norm_layer,
        act_layer,
    )


class UNet(nn.Module):
    """Factory Base Class supporting instanseg_unet, efficientunetb0, and convnext_unet."""

    def __new__(cls, *args, **kwargs):
        if cls is UNet:
            model_type = kwargs.get("model_type", None)
            if model_type is None and len(args) > 0 and isinstance(args[0], str):
                model_type = args[0]

            if model_type is None or not isinstance(model_type, str):
                raise ValueError("model_type must be specified as 'instanseg_unet', 'efficientunetb0', or 'convnext_unet'")

            mt_lower = model_type.lower().replace("-", "_")
            if mt_lower == "instanseg_unet":
                target_cls = InstanSeg_UNet
            elif mt_lower == "efficientunetb0":
                target_cls = EfficientUNetB0
            elif mt_lower in ("convnext_unet", "convnext_tiny", "convnext"):
                target_cls = ConvNeXtUNet
            else:
                raise ValueError(
                    f"Unsupported model_type: '{model_type}'. Expected 'instanseg_unet', 'efficientunetb0', or 'convnext_unet'."
                )

            return super(UNet, target_cls).__new__(target_cls)
        return super().__new__(cls)

    def _validate_input_tensor(self, x: torch.Tensor, divisor: int):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected input to be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor in NCHW layout, got shape {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise ValueError(f"Expected floating-point input tensor, got {x.dtype}")
        if hasattr(self, "in_channels") and self.in_channels is not None:
            if x.shape[1] != self.in_channels:
                raise ValueError(f"Input channels ({x.shape[1]}) does not match model in_channels ({self.in_channels})")
        h, w = x.shape[-2], x.shape[-1]
        if h % divisor != 0 or w % divisor != 0:
            raise ValueError(f"Input spatial dimensions ({h}, {w}) must be divisible by {divisor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =====================================================================
# 2. CONVNEXT ENCODER
# =====================================================================

class ConvNeXtEncoder(nn.Module):
    """
    Encoder based on torchvision.models.convnext_tiny backbone.
    Extracts 4 feature stages (H/4, H/8, H/16, H/32).
    """

    def __init__(self, in_channels: int = 3, weights=None):
        super().__init__()
        self.in_channels = in_channels
        backbone = models.convnext_tiny(weights=weights)

        # Modify first stem layer if input channels != 3
        if in_channels != 3:
            orig_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )

        # ConvNeXt stage decomposition:
        # stage0: Stem (4x4, stride 4) + ConvNeXt Blocks -> (96, H/4, W/4)
        # stage1: Downsample 1 (2x2, stride 2) + Blocks   -> (192, H/8, W/8)
        # stage2: Downsample 2 (2x2, stride 2) + Blocks   -> (384, H/16, W/16)
        # stage3: Downsample 3 (2x2, stride 2) + Bottleneck -> (768, H/32, W/32)
        self.stage0 = nn.Sequential(backbone.features[0], backbone.features[1])
        self.stage1 = nn.Sequential(backbone.features[2], backbone.features[3])
        self.stage2 = nn.Sequential(backbone.features[4], backbone.features[5])
        self.stage3 = nn.Sequential(backbone.features[6], backbone.features[7])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        s0 = self.stage0(x)   # (N, 96, H/4, W/4)
        s1 = self.stage1(s0)  # (N, 192, H/8, W/8)
        s2 = self.stage2(s1)  # (N, 384, H/16, W/16)
        s3 = self.stage3(s2)  # (N, 768, H/32, W/32) - Bottleneck

        # Ordered skips matching 5-stage decoder (shallowest-first):
        # [H (None), H/2 (None), H/4 (s0), H/8 (s1), H/16 (s2)]
        # Reversed in decoder: [s2, s1, s0, None, None]
        skips = [None, None, s0, s1, s2]
        return s3, skips


# =====================================================================
# 3. CONVNEXT DECODER BLOCK & DECODER
# =====================================================================

class ConvNeXtDecoderBlock(nn.Module):
    """
    Dual-residual decoder block compatible with missing/optional skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "BATCH",
        act: str = "GELU",
    ):
        super().__init__()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        
        if skip_channels and skip_channels > 0:
            self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        else:
            self.conv_skip = None

        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)

        if skip is not None and self.conv_skip is not None:
            x = proj + self.conv2(x + self.conv_skip(skip))
        else:
            x = proj + self.conv2(x)

        x = x + self.conv4(self.conv3(x))
        return x


class ConvNeXtDecoder(nn.Module):
    """Sequential decoder containing 5 ConvNeXtDecoderBlocks and multi-head final blocks."""

    def __init__(
        self,
        stage_channels: List[Tuple[int, int, int]],
        out_channels: List[int],
        norm: str = "BATCH",
        act: str = "GELU",
    ):
        super().__init__()
        self.decoder = nn.ModuleList([
            ConvNeXtDecoderBlock(in_ch, skip_ch, out_ch, norm=norm, act=act)
            for in_ch, skip_ch, out_ch in stage_channels
        ])

        final_norm = norm if (norm is not None) and norm.lower() != "instance" else None
        self.final_block = nn.ModuleList([
            conv_norm_act(stage_channels[-1][2], out_channel, 1, norm=final_norm, act=None)
            for out_channel in out_channels
        ])

    def forward(self, x: torch.Tensor, skips: List[Optional[torch.Tensor]]) -> torch.Tensor:
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)
            
        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
        return x


# =====================================================================
# 4. FULL CONVNEXT-TINY UNET MODEL CLASS
# =====================================================================

class ConvNeXtUNet(UNet):
    """
    U-Net combining a ConvNeXt-Tiny backbone with modular dual-residual multi-head decoders.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = [[2, 2, 1]],
        norm: str = "BATCH",
        dropout: float = 0.0,
        act: str = "GELU",
        weights=None,
        model_type: str = "convnext_unet",
        **kwargs,
    ):
        super().__init__()
        if isinstance(in_channels, str) and in_channels.lower() in ("instanseg_unet", "efficientunetb0", "convnext_unet"):
            in_channels = out_channels if out_channels is not None else 3
            out_channels = kwargs.pop("out_channels", [[2, 2, 1]])

        if in_channels == 0 or in_channels is None:
            in_channels = 3

        self.in_channels = in_channels
        self.model_type = "convnext_unet"
        self.dropout = dropout

        self.encoder = ConvNeXtEncoder(in_channels=in_channels, weights=weights)

        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels, (list, tuple)) and len(out_channels) > 0 and isinstance(out_channels[0], int):
            out_channels = [list(out_channels)]

        # Decoder block channel configs: (in_channels, skip_channels, out_channels)
        # Stage 0: 768 -> 384 (skip s2: 384, H/32 -> H/16)
        # Stage 1: 384 -> 192 (skip s1: 192, H/16 -> H/8)
        # Stage 2: 192 -> 96  (skip s0: 96,  H/8  -> H/4)
        # Stage 3: 96  -> 48  (skip None: 0, H/4  -> H/2)
        # Stage 4: 48  -> 32  (skip None: 0, H/2  -> H)
        stage_channels = [
            (768, 384, 384),
            (384, 192, 192),
            (192, 96, 96),
            (96, 0, 48),
            (48, 0, 32),
        ]

        self.decoders = nn.ModuleList([
            ConvNeXtDecoder(stage_channels, out_channel, norm=norm, act=act)
            for out_channel in out_channels
        ])

        if len(self.decoders) == 1:
            self.decoder = self.decoders[0]
        else:
            self.decoder = self.decoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_tensor(x, divisor=32)
        bottleneck, skips = self.encoder(x)
        return torch.cat([decoder(bottleneck, skips) for decoder in self.decoders], dim=1)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm="BATCH",
        act="ReLU",
        shallow=False,
    ):
        super().__init__()

        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x + self.conv_skip(skip))
        x = x + self.conv4(self.conv3(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True,
        norm="BATCH",
        act="ReLU",
        shallow=False,
    ):
        super().__init__()

        if pool:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = nn.Identity()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers, out_channels, norm, act):
        super().__init__()

        self.decoder = nn.ModuleList(
            [DecoderBlock(layers[i], layers[i + 1], layers[i + 1], norm=norm, act=act) for i in range(len(layers) - 1)]
        )

        self.final_block = nn.ModuleList(
            [
                conv_norm_act(
                    layers[-1],
                    out_channel,
                    1,
                    norm=norm if (norm is not None) and norm.lower() != "instance" else None,
                    act=None,
                )
                for out_channel in out_channels
            ]
        )

    def forward(self, x, skips):
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)

        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
        return x


class InstanSeg_UNet(UNet):
    def __init__(
        self,
        in_channels=3,
        out_channels=[[2, 2, 1]],
        layers=[256, 128, 64, 32],
        norm="BATCH",
        dropout=0,
        act="ReLu",
        model_type="instanseg_unet",
        **kwargs,
    ):
        super().__init__()
        # Handle positional model_type if passed through UNet constructor
        if isinstance(in_channels, str) and in_channels.lower() in ("instanseg_unet", "efficientunetb0"):
            in_channels = out_channels if out_channels is not None else 3
            out_channels = layers if layers is not None else [[2, 2, 1]]
            layers = [256, 128, 64, 32]

        if in_channels == 0 or in_channels is None:
            in_channels = 3
        self.in_channels = in_channels
        self.model_type = "instanseg_unet"
        self.dropout = dropout

        layers = list(layers)[::-1]
        self.encoder = nn.ModuleList(
            [EncoderBlock(in_channels, layers[0], pool=False, norm=norm, act=act)]
            + [EncoderBlock(layers[i], layers[i + 1], norm=norm, act=act) for i in range(len(layers) - 1)]
        )
        layers = layers[::-1]

        # out_channels should be a list of lists [[2,2,1],[2,2,1]] means two decoders, each with 3 output blocks.
        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]

        self.decoders = nn.ModuleList([Decoder(layers, out_channel, norm, act) for out_channel in out_channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        divisor = 2 ** (len(self.encoder) - 1)
        self._validate_input_tensor(x, divisor=divisor)
        skips = []
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder) - 1:
                skips.append(x)

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)


class EfficientEncoder(nn.Module):
    """
    Encoder based on torchvision.models.efficientnet_b0 backbone,
    with classification head and final pooling/linear layers removed
    to preserve spatial latent representation for segmentation.
    """

    def __init__(self, in_channels: int = 3, weights=None):
        super().__init__()
        self.in_channels = in_channels
        backbone = models.efficientnet_b0(weights=weights)

        if in_channels != 3:
            orig = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=orig.bias is not None,
            )

        # Retain stages up to bottleneck (features 0 through 7)
        self.stage0 = backbone.features[0]  # stem: ConvNormAct 3->32, stride 2 (H/2)
        self.stage1 = backbone.features[1]  # MBConv: 32->16, stride 1 (H/2)
        self.stage2 = backbone.features[2]  # MBConv: 16->24, stride 2 (H/4)
        self.stage3 = backbone.features[3]  # MBConv: 24->40, stride 2 (H/8)
        self.stage4 = backbone.features[4]  # MBConv: 40->80, stride 2 (H/16)
        self.stage5 = backbone.features[5]  # MBConv: 80->112, stride 1 (H/16)
        self.stage6 = backbone.features[6]  # MBConv: 112->192, stride 2 (H/32)
        self.stage7 = backbone.features[7]  # MBConv: 192->320, stride 1 (H/32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Skips ordered from shallowest to deepest
        s0 = self.stage0(x)   # H/2, 32
        s1 = self.stage1(s0)  # H/2, 16
        s2 = self.stage2(s1)  # H/4, 24
        s3 = self.stage3(s2)  # H/8, 40
        s4 = self.stage4(s3)  # H/16, 80
        s5 = self.stage5(s4)  # H/16, 112
        s6 = self.stage6(s5)  # H/32, 192
        s7 = self.stage7(s6)  # H/32, 320 (bottleneck)

        # NOTE: x (raw input) is intentionally NOT used as a skip connection.
        # Injecting raw pixel values as a skip at full resolution would let
        # conv_skip learn to amplify unprocessed signals, overwhelming the
        # deep semantic features from the bottleneck and causing training
        # divergence. The last decoder stage upsamples without a skip instead.
        skips = [None, s1, s2, s3, s5]
        return s7, skips


class EfficientDecoderBlock(nn.Module):
    """
    Decoder block mirroring corresponding encoder stages.
    Upsamples input by 2x and fuses with skip connection.
    """

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
        if skip_channels and skip_channels > 0:
            self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        else:
            self.conv_skip = None
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        if skip is not None and self.conv_skip is not None:
            x = proj + self.conv2(x + self.conv_skip(skip))
        else:
            x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x


class EfficientDecoder(nn.Module):
    """
    Decoder consisting of sequential decoder blocks mirroring the backbone network stages.
    """

    def __init__(
        self,
        stage_channels: List[Tuple[int, int, int]],
        out_channels: List[int],
        norm: str = "BATCH",
        act: str = "ReLU",
    ):
        super().__init__()
        self.decoder = nn.ModuleList([
            EfficientDecoderBlock(in_ch, skip_ch, out_ch, norm=norm, act=act)
            for in_ch, skip_ch, out_ch in stage_channels
        ])
        final_norm = norm if (norm is not None) and norm.lower() != "instance" else None
        self.final_block = nn.ModuleList([
            conv_norm_act(stage_channels[-1][2], out_channel, 1, norm=final_norm, act=None)
            for out_channel in out_channels
        ])

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)
        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
        return x


class EfficientUNetB0(UNet):
    """
    U-Net model combining torchvision EfficientNet-B0 encoder backbone
    with a custom modular decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = [[2, 2, 1]],
        norm: str = "BATCH",
        dropout: float = 0.0,
        act: str = "ReLU",
        model_type: str = "efficientunetb0",
        **kwargs,
    ):
        super().__init__()
        # Handle positional model_type if passed through UNet constructor
        if isinstance(in_channels, str) and in_channels.lower() in ("instanseg_unet", "efficientunetb0"):
            in_channels = out_channels if out_channels is not None else 3
            out_channels = kwargs.pop("out_channels", [[2, 2, 1]])

        if in_channels == 0 or in_channels is None:
            in_channels = 3
        self.in_channels = in_channels
        self.model_type = "efficientunetb0"
        self.dropout = dropout

        self.encoder = EfficientEncoder(in_channels=in_channels)

        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels, (list, tuple)) and len(out_channels) > 0 and isinstance(out_channels[0], int):
            out_channels = [list(out_channels)]

        # Decoder block configurations: (in_channels, skip_channels, out_channels)
        # Stage 0: 320 -> 112 (skip s5: 112ch, H/32 -> H/16)
        # Stage 1: 112 -> 40  (skip s3:  40ch, H/16 -> H/8)
        # Stage 2: 40  -> 24  (skip s2:  24ch, H/8  -> H/4)
        # Stage 3: 24  -> 16  (skip s1:  16ch, H/4  -> H/2)
        # Stage 4: 16  -> 16  (no skip: raw x excluded, H/2 -> H)
        stage_channels = [
            (320, 112, 112),
            (112, 40, 40),
            (40, 24, 24),
            (24, 16, 16),
            (16, 0, 16),
        ]

        self.decoders = nn.ModuleList([
            EfficientDecoder(stage_channels, out_channel, norm=norm, act=act)
            for out_channel in out_channels
        ])

        if len(self.decoders) == 1:
            self.decoder = self.decoders[0]
        else:
            self.decoder = self.decoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_tensor(x, divisor=32)
        bottleneck, skips = self.encoder(x)
        return torch.cat([decoder(bottleneck, skips) for decoder in self.decoders], dim=1)


# Aliases for modular components
EfficientNetEncoder = EfficientEncoder
EfficientNetDecoderBlock = EfficientDecoderBlock
EfficientNetDecoder = EfficientDecoder
ConvNeXtTinyUNet = ConvNeXtUNet
ConvNeXtTinyEncoder = ConvNeXtEncoder
ConvNeXtTinyDecoderBlock = ConvNeXtDecoderBlock
ConvNeXtTinyDecoder = ConvNeXtDecoder


if __name__ == "__main__":
    net = UNet(
        model_type="instanseg_unet",
        in_channels=3,
        out_channels=[5, 3],
    )
    print("InstanSeg_UNet output shape:", net(torch.randn(1, 3, 256, 256)).shape)

    eff_net = UNet(
        model_type="efficientunetb0",
        in_channels=3,
        out_channels=[[2, 2, 1]],
    )
    print("EfficientUNetB0 output shape:", eff_net(torch.randn(1, 3, 256, 256)).shape)

    conv_net = UNet(
        model_type="convnext_unet",
        in_channels=3,
        out_channels=[[2, 2, 1]],
    )
    print("ConvNeXtUNet output shape:", conv_net(torch.randn(1, 3, 256, 256)).shape)
