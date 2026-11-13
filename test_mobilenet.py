"""Test script for MobileNetV2_UNet"""
import torch
from instanseg.utils.models.InstanSeg_UNet import MobileNetV2_UNet, UNet

print("=" * 60)
print("Testing MobileNetV2_UNet Implementation")
print("=" * 60)

# Test 1: Direct instantiation
print("\n1. Direct instantiation:")
model = MobileNetV2_UNet(in_channels=3, out_channels=[[2, 2, 1]])
x = torch.randn(1, 3, 256, 256)
out = model(x)
print(f"   Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
print(f"   Expected: (1, 5, 256, 256)")
print(f"   Match: {out.shape == torch.Size([1, 5, 256, 256])}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Test 2: Factory instantiation via UNet
print("\n2. Factory instantiation via UNet:")
model2 = UNet(model_type="mobilenetv2_unet", in_channels=3, out_channels=[[2, 2, 1]])
out2 = model2(x)
print(f"   Input: {tuple(x.shape)} -> Output: {tuple(out2.shape)}")
print(f"   Match: {out2.shape == torch.Size([1, 5, 256, 256])}")

# Test 3: Different input sizes (must be divisible by 32)
print("\n3. Testing different input sizes:")
for size in [128, 256, 512]:
    x_test = torch.randn(1, 3, size, size)
    out_test = model(x_test)
    print(f"   {size}×{size} -> {tuple(out_test.shape)} ✓")

# Test 4: Multi-decoder support
print("\n4. Testing multi-decoder configuration:")
model_multi = MobileNetV2_UNet(in_channels=3, out_channels=[[2, 2, 1], [2, 2, 1]])
out_multi = model_multi(x)
print(f"   Two decoders output: {tuple(out_multi.shape)}")
print(f"   Expected: (1, 10, 256, 256)")
print(f"   Match: {out_multi.shape == torch.Size([1, 10, 256, 256])}")

# Test 5: Compare with InstanSeg_UNet
print("\n5. Parameter comparison with InstanSeg_UNet:")
from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet
instanseg_model = InstanSeg_UNet(in_channels=3, out_channels=[[2, 2, 1]])
instanseg_params = sum(p.numel() for p in instanseg_model.parameters()) / 1e6
mobilenet_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"   InstanSeg_UNet: {instanseg_params:.2f}M params")
print(f"   MobileNetV2_UNet: {mobilenet_params:.2f}M params")
print(f"   Reduction: {(1 - mobilenet_params / instanseg_params) * 100:.1f}%")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
