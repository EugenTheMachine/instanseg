# MobileNetV2_UNet Dimension Mismatch Fix

## Problem

The model was outputting 256×256 tensors when the input was 512×512, causing this error during training:

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (512) at non-singleton dimension 2
```

## Root Cause

The MobileNetV2 encoder has **5 downsampling stages** (H/32 bottleneck) but only provides **4 skip connections**:
- skip1 at H/2 (16 channels)
- skip2 at H/4 (24 channels)  
- skip3 at H/8 (32 channels)
- skip4 at H/16 (96 channels)

The decoder was configured with **5 upsampling stages** but the forward method used:

```python
for layer, skip in zip(self.decoder, skips[::-1]):
    x = layer(x, skip)
```

This `zip()` stops when the shorter sequence (skips) is exhausted, so the 5th decoder stage (H/2 → H) never executed, leaving the output at H/2 (256×256) instead of H (512×512).

## Solution

Modified `MobileNetV2Decoder.forward()` to pad the skips list with `None` values:

```python
def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
    # Process through decoder stages with skip connections
    # Pad skips with None for stages without skip connections
    # skips = [skip1@H/2, skip2@H/4, skip3@H/8, skip4@H/16]
    # After reverse and padding: [skip4, skip3, skip2, skip1, None]
    padded_skips = skips[::-1] + [None] * (len(self.decoder) - len(skips))
    
    for layer, skip in zip(self.decoder, padded_skips):
        x = layer(x, skip)
    
    # Apply multi-head final convolutions and concatenate
    x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
    return x
```

## Verification

The fix ensures:
1. All 5 decoder stages execute
2. The final stage (H/2 → H) receives `skip=None` as designed
3. Output dimensions match input dimensions: 512×512 → 512×512

## Files Modified

- `instanseg/utils/models/InstanSeg_UNet.py` - Fixed `MobileNetV2Decoder.forward()` method

## Status

✅ **FIXED** - The model now correctly outputs full-resolution tensors matching the input size.

The training should now proceed without dimension mismatch errors.
