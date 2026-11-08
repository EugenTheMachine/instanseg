from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from instanseg.utils.models.InstanSeg_UNet import UNet


class LayerNorm2d(nn.Module):
    """Channels-first 2D Layer Normalization operating on tensor layout (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MedNeXtBlock2D(nn.Module):
    """
    Core MedNeXt Block utilizing a 7x7 Depthwise Conv, LayerNorm2d, and 1x1 Inverted Bottleneck Expansion.
    """

    def __init__(self, in_channels: int, out_channels: int, exp_ratio: int = 2, k_size: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = int(out_channels * exp_ratio)

        # Large kernel depthwise convolution
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=k_size,
            padding=k_size // 2,
            groups=in_channels,
        )
        self.norm = LayerNorm2d(in_channels)

        # Inverted bottleneck pointwise convolutions
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        # Residual shortcut projection if channel counts mismatch
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + res


class MedNeXtDownBlock2D(nn.Module):
    """Downsampling Block leveraging strided depthwise convolutions."""

    def __init__(self, in_channels: int, out_channels: int, exp_ratio: int = 2, k_size: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = int(out_channels * exp_ratio)

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=k_size,
            stride=2,
            padding=k_size // 2,
            groups=in_channels,
        )
        self.norm = LayerNorm2d(in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + res


class MedNeXtUpBlock2D(nn.Module):
    """Upsampling Block combining feature interpolation and skip connection fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        exp_ratio: int = 2,
        block_depth: int = 1,
    ):
        super().__init__()
        self.conv_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        blocks = [
            MedNeXtBlock2D(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                exp_ratio=exp_ratio,
            )
        ]
        for _ in range(block_depth - 1):
            blocks.append(
                MedNeXtBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    exp_ratio=exp_ratio,
                )
            )
        if len(blocks) == 1:
            self.fusion_block = blocks[0]
        else:
            self.fusion_block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_in(x)
        skip = self.conv_skip(skip)

        fused = torch.cat([x, skip], dim=1)
        return self.fusion_block(fused)


class MedNeXtDecoder2D(nn.Module):
    """Multi-stage decoder processing multi-scale feature maps into task prediction heads."""

    def __init__(
        self,
        layers: List[int],
        out_channels: List[int],
        exp_ratio: int = 2,
        block_depth: int = 1,
    ):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                MedNeXtUpBlock2D(
                    in_channels=layers[i],
                    skip_channels=layers[i + 1],
                    out_channels=layers[i + 1],
                    exp_ratio=exp_ratio,
                    block_depth=block_depth,
                )
                for i in range(len(layers) - 1)
            ]
        )

        self.final_block = nn.ModuleList(
            [nn.Conv2d(layers[-1], out_c, kernel_size=1) for out_c in out_channels]
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for layer, skip in zip(self.decoder_blocks, skips[::-1]):
            x = layer(x, skip)

        return torch.cat([head(x) for head in self.final_block], dim=1)


class MedNeXt_UNet(UNet):
    """
    MedNeXt-2D implementation configured for InstanSeg multi-head instance segmentation.
    Default parameters match the target parameter budget of standard UNet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = [[2, 2, 1]],
        layers: List[int] = [256, 128, 64, 32],
        exp_ratio: int = 2,
        k_size: int = 7,
        block_depth: int = 1,
        model_type: str = "mednext_unet",
        **kwargs,
    ):
        super().__init__()
        # Handle positional or string model_type if passed through UNet constructor
        if isinstance(in_channels, str) and in_channels.lower().replace("-", "_") in (
            "instanseg_unet",
            "improved_unet",
            "efficientunetb0",
            "convnext_unet",
            "mednext_unet",
            "mednext",
            "mednext_instanseg",
            "mednext_instanseg_deep",
        ):
            model_type = in_channels
            in_channels = out_channels if out_channels is not None else 3
            out_channels = layers if layers is not None else [[2, 2, 1]]
            layers = [256, 128, 64, 32]

        if in_channels == 0 or in_channels is None:
            in_channels = 3
        self.in_channels = in_channels
        self.model_type = model_type

        # Support deep variant
        if "deep" in model_type.lower():
            block_depth = max(block_depth, 2)
        self.block_depth = block_depth
        self.exp_ratio = exp_ratio
        self.k_size = k_size

        # Order layers shallowest-to-deepest: [32, 64, 128, 256]
        layers_list = list(layers)
        if len(layers_list) > 1 and layers_list[0] > layers_list[-1]:
            enc_channels = layers_list[::-1]
        else:
            enc_channels = layers_list

        self.enc_channels = enc_channels

        # Stem Stage (512x512 resolution)
        stem_layers = [
            nn.Conv2d(in_channels, enc_channels[0], kernel_size=3, padding=1),
            LayerNorm2d(enc_channels[0]),
            MedNeXtBlock2D(enc_channels[0], enc_channels[0], exp_ratio=exp_ratio, k_size=k_size),
        ]
        for _ in range(block_depth - 1):
            stem_layers.append(
                MedNeXtBlock2D(enc_channels[0], enc_channels[0], exp_ratio=exp_ratio, k_size=k_size)
            )
        self.stem = nn.Sequential(*stem_layers)

        # Downsampling Encoder Path
        encoder_stages = []
        for i in range(len(enc_channels) - 1):
            stage_blocks = [
                MedNeXtDownBlock2D(
                    enc_channels[i], enc_channels[i + 1], exp_ratio=exp_ratio, k_size=k_size
                )
            ]
            for _ in range(block_depth - 1):
                stage_blocks.append(
                    MedNeXtBlock2D(
                        enc_channels[i + 1], enc_channels[i + 1], exp_ratio=exp_ratio, k_size=k_size
                    )
                )
            if len(stage_blocks) == 1:
                encoder_stages.append(stage_blocks[0])
            else:
                encoder_stages.append(nn.Sequential(*stage_blocks))
        self.encoder_stages = nn.ModuleList(encoder_stages)

        # Format output channels for multi-head setup
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels, (list, tuple)) and len(out_channels) > 0 and isinstance(out_channels[0], int):
            out_channels = [list(out_channels)]

        dec_layers = enc_channels[::-1]  # [256, 128, 64, 32]
        self.decoders = nn.ModuleList(
            [
                MedNeXtDecoder2D(
                    dec_layers,
                    head_out,
                    exp_ratio=exp_ratio,
                    block_depth=block_depth,
                )
                for head_out in out_channels
            ]
        )

        if len(self.decoders) == 1:
            self.decoder = self.decoders[0]
        else:
            self.decoder = self.decoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_tensor(x, divisor=8)
        skips = []
        x = self.stem(x)
        skips.append(x)  # Skip 0: (B, 32, 512, 512)

        for stage in self.encoder_stages[:-1]:
            x = stage(x)
            skips.append(x)  # Skip 1: (B, 64, 256, 256), Skip 2: (B, 128, 128, 128)

        x = self.encoder_stages[-1](x)  # Bottleneck: (B, 256, 64, 64)

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)


# Aliases
MedNeXtUNet = MedNeXt_UNet
mednext_unet = MedNeXt_UNet
MedNeXt_InstanSeg = MedNeXt_UNet
MedNeXt_InstanSeg_Deep = MedNeXt_UNet
