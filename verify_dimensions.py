#!/usr/bin/env python
"""Verify dimensional compatibility between MobileNetV2 encoder and decoder"""

import torch
from instanseg.utils.models.InstanSeg_UNet import MobileNetV2Encoder

print("=" * 70)
print("MobileNetV2 Encoder Dimension Verification")
print("=" * 70)

# Create encoder
encoder = MobileNetV2Encoder(in_channels=3)

# Test with 512x512 input
x = torch.randn(1, 3, 512, 512)

print(f"\nInput shape: {tuple(x.shape)}")

# Forward pass
bottleneck, skips = encoder(x)

print(f"\nBottleneck shape: {tuple(bottleneck.shape)}")
print(f"  Channels: {bottleneck.shape[1]}")
print(f"  Spatial: {bottleneck.shape[2]}x{bottleneck.shape[3]}")

print(f"\nNumber of skips: {len(skips)}")
for i, skip in enumerate(skips):
    print(f"  Skip {i}: {tuple(skip.shape)} - {skip.shape[1]} channels at {skip.shape[2]}x{skip.shape[3]}")

print("\n" + "=" * 70)
print("EfficientNet (for comparison)")
print("=" * 70)

from instanseg.utils.models.InstanSeg_UNet import EfficientEncoder

eff_encoder = EfficientEncoder(in_channels=3)
eff_bottleneck, eff_skips = eff_encoder(x)

print(f"\nBottleneck shape: {tuple(eff_bottleneck.shape)}")
print(f"  Channels: {eff_bottleneck.shape[1]}")

print(f"\nNumber of skips: {len(eff_skips)}")
for i, skip in enumerate(eff_skips):
    if skip is not None:
        print(f"  Skip {i}: {tuple(skip.shape)} - {skip.shape[1]} channels")
    else:
        print(f"  Skip {i}: None")

print("\n" + "=" * 70)
print("MISMATCH DETECTED!")
print("=" * 70)
print("\nMobileNetV2 encoder outputs are DIFFERENT from EfficientNet:")
print(f"  MobileNetV2 bottleneck: {bottleneck.shape[1]} channels")
print(f"  EfficientNet bottleneck: {eff_bottleneck.shape[1]} channels")
print(f"\nCannot use EfficientDecoder directly with MobileNetV2!")
print("Need custom stage_channels configuration!")
