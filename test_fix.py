#!/usr/bin/env python
"""Test MobileNetV2_UNet output dimensions"""

import torch
from instanseg.utils.models.InstanSeg_UNet import MobileNetV2_UNet

print("Testing MobileNetV2_UNet dimensions...")

model = MobileNetV2_UNet(in_channels=3, out_channels=[[2, 2, 1]])

# Test various input sizes (all divisible by 32)
test_sizes = [256, 512, 1024]

for size in test_sizes:
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (1, 5, size, size)
    actual_shape = tuple(output.shape)
    
    if actual_shape == expected_shape:
        print(f"✓ {size}x{size}: {actual_shape} - CORRECT")
    else:
        print(f"✗ {size}x{size}: Expected {expected_shape}, got {actual_shape} - FAIL")
        exit(1)

print("\n✓ All dimension tests passed!")
print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
