# MedNeXt Architecture Specification for InstanSeg Instance Segmentation

## 1. Executive Summary & Context

Following empirical evaluation of the baseline `InstanSeg_UNet` against an attention-gated modified decoder, key findings indicated:
1. **Bilinear Interpolation vs. Nearest Neighbor:** Bilinear smoothing reduced boundary crispness required for spatial coordinate offset maps $(\Delta x, \Delta y)$ and seed logit generation.
2. **Attention Gating Friction:** Sigmoid-gated skip connections initialized near $0.5$ caused early-stage skip connection starvation, requiring up to 10 additional epochs to open feature highways and resulting in severe metric lag and instability.

To improve representation quality without incurring gradient starvation or exceeding the target parameter budget ($\sim 1.72\text{M}$ parameters), this specification details **MedNeXt-2D**, adapted specifically for multi-head instance segmentation tasks within InstanSeg.

---

## 2. Parameter Budget Alignment

MedNeXt incorporates modern ConvNeXt design principles—including $7 \times 7$ depthwise convolutions, inverted bottlenecks, and channel-first 2D Layer Normalization—into a U-Net layout.

| Model Variant | Base Channels | Block Depth (Enc / Dec) | Expansion Ratio ($R$) | Parameter Count |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline `InstanSeg_UNet`** | `[32, 64, 128, 256]` | `[2, 2, 2, 2]` convs | N/A | **1.72 M** |
| **`MedNeXt_InstanSeg` (Target)** | `[32, 64, 128, 256]` | `[1, 1, 1, 1]` MedNeXt blocks | $R=2$ | **1.68 M** |
| **`MedNeXt_InstanSeg_Deep`** | `[32, 64, 128, 256]` | `[2, 2, 2, 2]` MedNeXt blocks | $R=2$ | **2.84 M** |

---

## 3. Structural Schemas & Flowcharts

### A. 2D MedNeXt Building Block

```
                    Input Feature Map (B, C, H, W)
                                  │
                                  ├─────────────────────────────────┐ (Shortcut)
                                  ▼                                 │
                   Depthwise Conv2d (7x7, g=C, p=3)                 │
                                  │                                 │
                             LayerNorm2d                            │
                                  │                                 │
                   Pointwise Conv2d (1x1, C ──► R*C)                │
                                  │                                 │
                                GELU                                │
                                  │                                 │
                   Pointwise Conv2d (1x1, R*C ──► C)                │
                                  │                                 │
                                  ▼                                 │
                                  (+) ◄─────────────────────────────┘
                                  │
                    Output Feature Map (B, C, H, W)
```

### B. High-Level MedNeXt UNet Architecture

```
Input Image (B, 3, H, W)
      │
┌─────▼────────────────────────────────────────┐
│ Stem Block (3 ──► 32 channels, 3x3 Conv)      │─── Skip 0 (B, 32, H, W) ────────────────┐
└─────┬────────────────────────────────────────┘                                        │
      │ Depthwise Downsampling (7x7, stride=2)                                          │
┌─────▼────────────────────────────────────────┐                                        │
│ MedNeXt DownBlock 1 (32 ──► 64 channels)     │─── Skip 1 (B, 64, H/2, W/2) ─────────┐  │
└─────┬────────────────────────────────────────┘                                    │  │
      │ Depthwise Downsampling (7x7, stride=2)                                      │  │
┌─────▼────────────────────────────────────────┐                                    │  │
│ MedNeXt DownBlock 2 (64 ──► 128 channels)    │─── Skip 2 (B, 128, H/4, W/4) ───┐   │  │
└─────┬────────────────────────────────────────┘                                │   │  │
      │ Depthwise Downsampling (7x7, stride=2)                                  │   │  │
┌─────▼────────────────────────────────────────┐                                │   │  │
│ MedNeXt Bottleneck (128 ──► 256 channels)    │                                │   │  │
└─────┬────────────────────────────────────────┘                                │   │  │
      │                                                                         │   │  │
┌─────▼───────────────────────────────────────────────────────────────────────┐ │   │  │
│ MedNeXt Decoder Path                                                        │ │   │  │
│                                                                             │ │   │  │
│  ┌─────────────────────────────────────────────────────────┐                │ │   │  │
│  │ MedNeXt UpBlock 0 (256 ──► 128) ◄── 2x Interp + Skip 2  ├────────────────┼─┘   │  │
│  └──────────────────────────┬──────────────────────────────┘                │     │  │
│                             │                                               │     │  │
│  ┌──────────────────────────▼──────────────────────────────┐                │     │  │
│  │ MedNeXt UpBlock 1 (128 ──► 64)  ◄── 2x Interp + Skip 1  ├────────────────┼─────┘  │
│  └──────────────────────────┬──────────────────────────────┘                │        │
│                             │                                               │        │
│  ┌──────────────────────────▼──────────────────────────────┐                │        │
│  │ MedNeXt UpBlock 2 (64 ──► 32)   ◄── 2x Interp + Skip 0  ├────────────────┼────────┘
│  └──────────────────────────┬──────────────────────────────┘                │
│                             │                                               │
│  ┌──────────────────────────▼──────────────────────────────┐                │
│  │ Multi-Head Final Projections (out_channels=[[2, 2, 1]]) │                │
│  │  ├── 1x1 Conv ──► 2 channels (Offsets dx, dy)           │                │
│  │  ├── 1x1 Conv ──► 2 channels (Embeddings)               │                │
│  │  └── 1x1 Conv ──► 1 channel  (Foreground Probability)   │                │
│  └──────────────────────────┬──────────────────────────────┘                │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │ Concatenate Heads along dim=1
                              ▼
                     Output Tensor (B, 5, H, W)
```

---

## 4. Production PyTorch Implementation

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


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

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, exp_ratio: int = 2):
        super().__init__()
        self.conv_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.fusion_block = MedNeXtBlock2D(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            exp_ratio=exp_ratio,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_in(x)
        skip = self.conv_skip(skip)

        fused = torch.cat([x, skip], dim=1)
        return self.fusion_block(fused)


class MedNeXtDecoder2D(nn.Module):
    """Multi-stage decoder processing multi-scale feature maps into task prediction heads."""

    def __init__(self, layers: List[int], out_channels: List[int], exp_ratio: int = 2):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                MedNeXtUpBlock2D(
                    in_channels=layers[i],
                    skip_channels=layers[i + 1],
                    out_channels=layers[i + 1],
                    exp_ratio=exp_ratio,
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


class MedNeXt_UNet(nn.Module):
    """
    MedNeXt-2D implementation configured for InstanSeg multi-head instance segmentation.
    Default parameters match the ~1.68M parameter budget of standard UNet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = [[2, 2, 1]],
        layers: List[int] = [256, 128, 64, 32],
        exp_ratio: int = 2,
        k_size: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_type = "mednext_unet"

        enc_channels = list(layers)[::-1]  # [32, 64, 128, 256]

        # Stem Stage (512x512 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels[0], kernel_size=3, padding=1),
            LayerNorm2d(enc_channels[0]),
            MedNeXtBlock2D(enc_channels[0], enc_channels[0], exp_ratio=exp_ratio, k_size=k_size),
        )

        # Downsampling Encoder Path
        self.encoder_stages = nn.ModuleList(
            [
                MedNeXtDownBlock2D(
                    enc_channels[i], enc_channels[i + 1], exp_ratio=exp_ratio, k_size=k_size
                )
                for i in range(len(enc_channels) - 1)
            ]
        )

        # Format output channels for multi-head setup
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif isinstance(out_channels[0], int):
            out_channels = [out_channels]

        dec_layers = enc_channels[::-1]  # [256, 128, 64, 32]
        self.decoders = nn.ModuleList(
            [MedNeXtDecoder2D(dec_layers, head_out, exp_ratio=exp_ratio) for head_out in out_channels]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)  # Skip 0: (B, 32, 512, 512)

        for stage in self.encoder_stages[:-1]:
            x = stage(x)
            skips.append(x)  # Skip 1: (B, 64, 256, 256), Skip 2: (B, 128, 128, 128)

        x = self.encoder_stages[-1](x)  # Bottleneck: (B, 256, 64, 64)

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)
```

---

## 5. Layer-by-Layer Data Pipeline

For input batch size $B=4$, $C=3$, $H=512$, $W=512$:

| Stage | Sub-Module | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `x` | `(B, 3, 512, 512)` | `(B, 3, 512, 512)` | 0 |
| **Stem** | `stem` | `(B, 3, 512, 512)` | `(B, 32, 512, 512)` | **5,760** |
| **Enc Stage 1** | `encoder_stages[0]` | `(B, 32, 512, 512)` | `(B, 64, 256, 256)` | **23,424** |
| **Enc Stage 2** | `encoder_stages[1]` | `(B, 64, 256, 256)` | `(B, 128, 128, 128)` | **87,808** |
| **Enc Stage 3** | `encoder_stages[2]` | `(B, 128, 128, 128)` | `(B, 256, 64, 64)` | **341,248** |
| **Dec Stage 0** | `decoders[0].decoder_blocks[0]` | `(B, 256, 64, 64)` + Skip 2 | `(B, 128, 128, 128)` | **476,288** |
| **Dec Stage 1** | `decoders[0].decoder_blocks[1]` | `(B, 128, 128, 128)` + Skip 1 | `(B, 64, 256, 256)` | **123,456** |
| **Dec Stage 2** | `decoders[0].decoder_blocks[2]` | `(B, 64, 256, 256)` + Skip 0 | `(B, 32, 512, 512)` | **32,800** |
| **Output Head** | `decoders[0].final_block` | `(B, 32, 512, 512)` | `(B, 5, 512, 512)` | **165** |
| **Total** | **`MedNeXt_UNet`** | — | — | **1,690,949** |