"""
InstanSeg_UNetV2 - Residual U-Net with Multiscale Dilated Bottleneck.

Architecture for layers=[256, 128, 64, 32] (reversed internally to [32, 64, 128, 256]):

    Input
      -> residual encoder block (32ch, H×W) ------------ skip H
      -> MaxPool 2×2
      -> residual encoder block (64ch, H/2×W/2) --------- skip H/2
      -> MaxPool 2×2
      -> residual encoder block (128ch, H/4×W/4) -------- skip H/4
      -> MaxPool 2×2
      -> residual encoder block (256ch, H/8×W/8)
      -> multiscale dilated residual bottleneck (256ch, H/8×W/8)
      -> upsample(nearest) + 1×1 proj + skip 1×1 proj + add + residual block (128ch, H/4×W/4)
      -> upsample(nearest) + 1×1 proj + skip 1×1 proj + add + residual block (64ch, H/2×W/2)
      -> upsample(nearest) + 1×1 proj + skip 1×1 proj + add + residual block (32ch, H×W)
      -> full-resolution refinement block (32ch)
      -> output heads (1×1 convs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from instanseg.utils.models.InstanSeg_UNet import UNet


# =========================================================================
# Normalization and Activation factories
# =========================================================================

def _groupnorm_groups(channels: int) -> int:
    """Return the largest divisor of *channels* that is ≤ 8 (minimum 2)."""
    for g in range(8, 0, -1):
        if channels % g == 0:
            return g
    return 1  # fallback: only reached if channels == 1


def make_norm(channels: int, norm) -> nn.Module:
    """
    Create a normalization layer for *channels* feature maps.

    Supported values for *norm*:
        ``"BATCH"``    → ``nn.BatchNorm2d``
        ``"GROUP"``    → ``nn.GroupNorm`` (groups = largest divisor of channels ≤ 8)
        ``"INSTANCE"`` → ``nn.InstanceNorm2d(affine=True)``
        ``None`` / ``"None"`` → ``nn.Identity``

    Raises
    ------
    ValueError
        For any other (unsupported) string.
    """
    if norm is None or (isinstance(norm, str) and norm.lower() == "none"):
        return nn.Identity()
    if not isinstance(norm, str):
        raise ValueError(f"Unsupported norm: {norm!r}. Expected 'BATCH', 'GROUP', 'INSTANCE', or None.")
    n = norm.upper()
    if n == "BATCH":
        return nn.BatchNorm2d(channels)
    elif n == "GROUP":
        num_groups = _groupnorm_groups(channels)
        return nn.GroupNorm(num_groups, channels)
    elif n == "INSTANCE":
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        raise ValueError(f"Unsupported norm: {norm!r}. Expected 'BATCH', 'GROUP', 'INSTANCE', or None.")


def make_act(act) -> nn.Module:
    """
    Create an activation layer.

    Supported values for *act*:
        ``"RELU"``           → ``nn.ReLU(inplace=True)``
        ``"GELU"``           → ``nn.GELU()``
        ``"SILU"``/``"SWISH"`` → ``nn.SiLU(inplace=True)``
        ``"MISH"``           → ``nn.Mish(inplace=True)``
        ``None``/``"None"``  → ``nn.Identity``

    Raises
    ------
    ValueError
        For any other (unsupported) string.
    """
    if act is None or (isinstance(act, str) and act.lower() == "none"):
        return nn.Identity()
    if not isinstance(act, str):
        raise ValueError(f"Unsupported act: {act!r}. Expected 'RELU', 'GELU', 'SILU'/'SWISH', 'MISH', or None.")
    a = act.upper()
    if a == "RELU":
        return nn.ReLU(inplace=True)
    elif a == "GELU":
        return nn.GELU()
    elif a in ("SILU", "SWISH"):
        return nn.SiLU(inplace=True)
    elif a == "MISH":
        return nn.Mish(inplace=True)
    else:
        raise ValueError(f"Unsupported act: {act!r}. Expected 'RELU', 'GELU', 'SILU'/'SWISH', 'MISH', or None.")


# =========================================================================
# Residual Block
# =========================================================================

class ResidualBlock(nn.Module):
    """
    Two-layer residual block with optional channel projection on the shortcut.

    Main branch:
        Conv(3×3) → Norm → Act → Conv(3×3) → Norm
        (no activation *before* the residual addition)

    Shortcut:
        Identity           if ``in_ch == out_ch``
        Conv(1×1) + Norm   if ``in_ch != out_ch``

    Activation is applied *after* the residual addition.
    """

    def __init__(self, in_ch: int, out_ch: int, norm, act):
        super().__init__()
        # Main branch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = make_norm(out_ch, norm)
        self.act = make_act(act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(out_ch, norm)

        # Shortcut branch
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                make_norm(out_ch, norm),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + residual)


# =========================================================================
# Multiscale Dilated Bottleneck
# =========================================================================

class BottleneckBranch(nn.Module):
    """
    Single branch of the multiscale dilated bottleneck.

    Applies:
        depthwise Conv(3×3, dilation=dilation) → Norm → Act
        pointwise Conv(1×1) → Norm → Act
    """

    def __init__(self, in_ch: int, pw_out: int, dilation: int, norm, act):
        super().__init__()
        # Depthwise conv: groups = in_ch → each channel convolved independently
        self.dw_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=3,
            padding=dilation, dilation=dilation,
            groups=in_ch, bias=False,
        )
        self.dw_norm = make_norm(in_ch, norm)
        self.dw_act = make_act(act)
        # Pointwise conv: projects to pw_out channels
        self.pw_conv = nn.Conv2d(in_ch, pw_out, kernel_size=1, bias=False)
        self.pw_norm = make_norm(pw_out, norm)
        self.pw_act = make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.dw_norm(x)
        x = self.dw_act(x)
        x = self.pw_conv(x)
        x = self.pw_norm(x)
        x = self.pw_act(x)
        return x


class MultiscaleDilatedBottleneck(nn.Module):
    """
    Three parallel dilated depth-separable branches (dilation 1, 2, 4) whose
    outputs are concatenated and projected back to ``C`` channels, then added
    to the input (residual connection).

    For ``C=256``: each branch outputs ``C//4=64`` channels, so the concat
    tensor is ``[N, 192, H, W]`` and the projection restores it to
    ``[N, 256, H, W]``.

    Optional spatial dropout (``nn.Dropout2d``) is applied after the
    projection but before the residual addition + activation.
    """

    def __init__(self, C: int, norm, act, dropout: float = 0.0):
        super().__init__()
        branch_ch = C // 4  # 64 for C=256
        if branch_ch <= 0:
            raise ValueError(f"Bottleneck branch channels (C // 4) must be positive, got {branch_ch} for C={C}")

        self.branch_d1 = BottleneckBranch(C, branch_ch, dilation=1, norm=norm, act=act)
        self.branch_d2 = BottleneckBranch(C, branch_ch, dilation=2, norm=norm, act=act)
        self.branch_d4 = BottleneckBranch(C, branch_ch, dilation=4, norm=norm, act=act)

        # Project concatenated branches back to C channels
        self.project = nn.Conv2d(branch_ch * 3, C, kernel_size=1, bias=False)
        self.norm = make_norm(C, norm)

        # Spatial dropout at bottleneck (Identity when dropout == 0)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.act = make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b3 = self.branch_d4(x)
        multi_scale = torch.cat([b1, b2, b3], dim=1)   # [N, 3*branch_ch, H, W]
        multi_scale = self.project(multi_scale)          # [N, C, H, W]
        multi_scale = self.norm(multi_scale)
        multi_scale = self.dropout(multi_scale)          # spatial dropout before residual
        return self.act(x + multi_scale)


# =========================================================================
# Decoder Stage
# =========================================================================

class V2DecoderStage(nn.Module):
    """
    Single decoder stage:
        1. Upsample 2× (nearest neighbour)
        2. 1×1 project upsampled features to ``out_channels``
        3. 1×1 project skip to ``out_channels``
        4. Add projected tensors
        5. Apply ResidualBlock(out_channels → out_channels)
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm, act):
        super().__init__()
        self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.block = ResidualBlock(out_channels, out_channels, norm, act)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_proj(x) + self.skip_proj(skip)
        return self.block(x)


# =========================================================================
# Decoder Group (one full decoder path)
# =========================================================================

class V2DecoderGroup(nn.Module):
    """
    One complete decoder path consisting of:
        3 × V2DecoderStage
        Full-resolution refinement block
        Output heads (1×1 conv per channel group)

    Parameters
    ----------
    out_channels_list : list[int]
        Number of output channels for each head, e.g. ``[2, 2, 1]``.
    norm, act :
        Normalization/activation factories forwarded to all sub-modules.
    """

    # Encoder channel sizes (fixed by the architecture)
    _ENC_CHANNELS = [32, 64, 128, 256]  # stages 0-3

    def __init__(self, out_channels_list: List[int], enc_channels: List[int] = [32, 64, 128, 256], norm="GROUP", act="SILU"):
        super().__init__()
        # stage0: bottleneck (enc_channels[3]) → enc_channels[2]  |  skip from enc2 (enc_channels[2])
        # stage1: enc_channels[2] → enc_channels[1]              |  skip from enc1 (enc_channels[1])
        # stage2: enc_channels[1] → enc_channels[0]              |  skip from enc0 (enc_channels[0])
        self.stage0 = V2DecoderStage(enc_channels[3], enc_channels[2], enc_channels[2], norm, act)
        self.stage1 = V2DecoderStage(enc_channels[2], enc_channels[1], enc_channels[1], norm, act)
        self.stage2 = V2DecoderStage(enc_channels[1], enc_channels[0], enc_channels[0], norm, act)

        # Full-resolution refinement block
        self.refinement = nn.Sequential(
            nn.Conv2d(enc_channels[0], enc_channels[0], kernel_size=3, padding=1, bias=False),
            make_norm(enc_channels[0], norm),
            make_act(act),
        )

        # Output heads: one 1×1 conv per element in out_channels_list
        self.heads = nn.ModuleList([
            nn.Conv2d(enc_channels[0], oc, kernel_size=1)
            for oc in out_channels_list
        ])

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck : Tensor[N, 256, H/8, W/8]
        skips : [skip_32, skip_64, skip_128]
            Encoder feature maps at H×W (32ch), H/2×W/2 (64ch), H/4×W/4 (128ch).
        """
        x = self.stage0(bottleneck, skips[2])   # skip_128: H/4
        x = self.stage1(x, skips[1])            # skip_64:  H/2
        x = self.stage2(x, skips[0])            # skip_32:  H
        x = self.refinement(x)
        return torch.cat([head(x) for head in self.heads], dim=1)


# =========================================================================
# InstanSeg_UNetV2
# =========================================================================

class InstanSeg_UNetV2(UNet):
    """
    Residual U-Net (V2) with a multiscale dilated bottleneck.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default: 3).
    out_channels : int | list[int] | list[list[int]]
        Output channel specification.  Three formats are accepted:
        * ``int``              e.g. ``5``           → ``[[5]]``
        * ``list[int]``        e.g. ``[5, 3]``      → ``[[5, 3]]`` (single decoder)
        * ``list[list[int]]``  e.g. ``[[2,2,1],[2,2,1]]`` → two decoder groups
    layers : list[int]
        Encoder channel widths in *descending* order (default: ``[256, 128, 64, 32]``).
        The list is reversed internally so the first element is the deepest width.
    norm : str | None
        Normalization type: ``"GROUP"`` (default), ``"BATCH"``, ``"INSTANCE"``, or ``None``.
    dropout : float
        Spatial dropout probability at the bottleneck (must be in ``[0, 1)``).
    act : str | None
        Activation function: ``"SILU"`` (default), ``"RELU"``, ``"GELU"``, ``"MISH"``,
        or ``None``.
    model_type : str
        Factory key (default: ``"instanseg_unet_v2"``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Union[int, List[int], List[List[int]]] = [[2, 2, 1]],
        layers: List[int] = [256, 128, 64, 32],
        norm: str = "GROUP",
        dropout: float = 0.0,
        act: str = "SILU",
        model_type: str = "instanseg_unet_v2",
        **kwargs,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Handle positional model_type forwarding from UNet.__new__
        # (mirrors the pattern used by InstanSeg_UNet / EfficientUNetB0 etc.)
        # ------------------------------------------------------------------
        if isinstance(in_channels, str) and in_channels.lower() in ("instanseg_unet_v2",):
            in_channels = out_channels if out_channels is not None else 3
            out_channels = layers if layers is not None else [[2, 2, 1]]
            layers = [256, 128, 64, 32]

        if in_channels == 0 or in_channels is None:
            in_channels = 3

        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Eagerly validate norm and act so construction fails fast
        make_norm(32, norm)   # test with an arbitrary channel count
        make_act(act)

        # ------------------------------------------------------------------
        # Store hyper-parameters
        # ------------------------------------------------------------------
        self.in_channels = in_channels
        self.model_type = model_type
        self.dropout = dropout

        # ------------------------------------------------------------------
        # Build encoder
        # ``layers`` is given as [256, 128, 64, 32] (descending order).
        # We reverse it so index 0 = shallowest = 32ch.
        # ------------------------------------------------------------------
        enc_channels = list(layers)[::-1]   # [32, 64, 128, 256]

        # Encoder stages (MaxPool is built into stages 1-3)
        self.enc0 = ResidualBlock(in_channels, enc_channels[0], norm, act)
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(enc_channels[0], enc_channels[1], norm, act),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(enc_channels[1], enc_channels[2], norm, act),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(enc_channels[2], enc_channels[3], norm, act),
        )

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = MultiscaleDilatedBottleneck(
            C=enc_channels[3],   # 256
            norm=norm,
            act=act,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Normalise out_channels into list[list[int]]
        # ------------------------------------------------------------------
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        elif (
            isinstance(out_channels, (list, tuple))
            and len(out_channels) > 0
            and isinstance(out_channels[0], int)
        ):
            out_channels = [list(out_channels)]
        # else: already list[list[int]]

        # ------------------------------------------------------------------
        # Build one decoder group per element in out_channels
        # ------------------------------------------------------------------
        self.decoder_groups = nn.ModuleList([
            V2DecoderGroup(out_ch_list, enc_channels=enc_channels, norm=norm, act=act)
            for out_ch_list in out_channels
        ])
        self.decoders = self.decoder_groups
        if len(self.decoders) == 1:
            self.decoder = self.decoders[0]
        else:
            self.decoder = self.decoders

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate: 3 downsampling ops → divisor = 8
        self._validate_input_tensor(x, divisor=8)

        # Encode
        s0 = self.enc0(x)    # [N, 32,  H,   W  ]
        s1 = self.enc1(s0)   # [N, 64,  H/2, W/2]
        s2 = self.enc2(s1)   # [N, 128, H/4, W/4]
        s3 = self.enc3(s2)   # [N, 256, H/8, W/8]

        # Bottleneck
        bn = self.bottleneck(s3)   # [N, 256, H/8, W/8]

        # Skips: [32ch@H, 64ch@H/2, 128ch@H/4]
        skips = [s0, s1, s2]

        # Decode and concatenate all decoder groups along channel axis
        return torch.cat([dec(bn, skips) for dec in self.decoder_groups], dim=1)


Instanseg_Unet_v2 = InstanSeg_UNetV2
instanseg_unet_v2 = InstanSeg_UNetV2
InstanSeg_UnetV2 = InstanSeg_UNetV2

__all__ = [
    "InstanSeg_UNetV2",
    "Instanseg_Unet_v2",
    "instanseg_unet_v2",
    "InstanSeg_UnetV2",
    "ResidualBlock",
    "BottleneckBranch",
    "MultiscaleDilatedBottleneck",
    "V2DecoderStage",
    "V2DecoderGroup",
    "make_norm",
    "make_act",
]

