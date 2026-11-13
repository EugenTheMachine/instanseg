```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder extracting multi-scale skip features at strides 2, 4, 8, 16, and 32."""

    def __init__(
        self,
        weights: Optional[MobileNet_V2_Weights] = MobileNet_V2_Weights.DEFAULT,
        freeze: bool = False,
    ):
        super().__init__()
        backbone = mobilenet_v2(weights=weights)
        features = backbone.features

        # Group MobileNetV2 feature layers by spatial downsampling stages
        self.stage1 = nn.Sequential(*features[0:2])    # Out: (B, 16,   H/2,  W/2)
        self.stage2 = nn.Sequential(*features[2:4])    # Out: (B, 24,   H/4,  W/4)
        self.stage3 = nn.Sequential(*features[4:7])    # Out: (B, 32,   H/8,  W/8)
        self.stage4 = nn.Sequential(*features[7:14])   # Out: (B, 96,   H/16, W/16)
        self.stage5 = nn.Sequential(*features[14:19])  # Out: (B, 1280, H/32, W/32)

        # Expose channel counts for clean decoder wiring
        self.out_channels = (16, 24, 32, 96, 1280)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip1 = self.stage1(x)
        skip2 = self.stage2(skip1)
        skip3 = self.stage3(skip2)
        skip4 = self.stage4(skip3)
        bottleneck = self.stage5(skip4)
        return bottleneck, skip4, skip3, skip2, skip1


class DecoderBlock(nn.Module):
    """Upsampling decoder block supporting optional skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Guard against spatial mismatch if input H or W is not divisible by 32
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UnetV2(nn.Module):
    """U-Net segmentation model with a MobileNetV2 encoder backbone."""

    def __init__(
        self,
        out_channels: int = 1,
        dropout: float = 0.2,
        weights: Optional[MobileNet_V2_Weights] = MobileNet_V2_Weights.DEFAULT,
        freeze_encoder: bool = False,
        apply_sigmoid: bool = True,
    ):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

        # Encoder (MobileNetV2)
        self.encoder = MobileNetV2Encoder(weights=weights, freeze=freeze_encoder)
        c1, c2, c3, c4, c5 = self.encoder.out_channels  # (16, 24, 32, 96, 1280)

        # Decoder (5 upsampling stages to restore H/32 -> H and W/32 -> W)
        self.decoder5 = DecoderBlock(in_channels=c5, skip_channels=c4, out_channels=96, dropout=dropout)
        self.decoder4 = DecoderBlock(in_channels=96, skip_channels=c3, out_channels=32, dropout=dropout)
        self.decoder3 = DecoderBlock(in_channels=32, skip_channels=c2, out_channels=24, dropout=dropout)
        self.decoder2 = DecoderBlock(in_channels=24, skip_channels=c1, out_channels=16, dropout=dropout)
        self.decoder1 = DecoderBlock(in_channels=16, skip_channels=0, out_channels=16, dropout=dropout)

        # Final segmentation head
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        bottleneck, skip4, skip3, skip2, skip1 = self.encoder(x)

        # Decoder path
        x = self.decoder5(bottleneck, skip4)  # H/16 x W/16
        x = self.decoder4(x, skip3)           # H/8  x W/8
        x = self.decoder3(x, skip2)           # H/4  x W/4
        x = self.decoder2(x, skip1)           # H/2  x W/2
        x = self.decoder1(x, skip=None)       # H    x W

        # Final projection
        x = self.final_conv(x)
        if self.apply_sigmoid:
            x = self.sigmoid(x)
        return x
    ```