#!/usr/bin/env python
"""Verification script for MobileNetV2_UNet integration"""

print("=" * 70)
print("MobileNetV2_UNet Integration Verification")
print("=" * 70)

# Test 1: Direct import
print("\n[1/5] Testing direct import from InstanSeg_UNet...")
try:
    from instanseg.utils.models.InstanSeg_UNet import MobileNetV2_UNet, UNet
    print("✓ Direct import successful")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 2: Factory pattern
print("\n[2/5] Testing UNet factory instantiation...")
try:
    import torch
    model = UNet(model_type="mobilenetv2_unet", in_channels=3, out_channels=[[2, 2, 1]])
    print(f"✓ Factory instantiation successful")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Forward pass
print("\n[3/5] Testing forward pass...")
try:
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (1, 5, 256, 256), f"Shape mismatch: expected (1,5,256,256), got {output.shape}"
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: Model loader integration
print("\n[4/5] Testing model_loader.py integration...")
try:
    from instanseg.utils.model_loader import build_model_from_dict
    
    build_dict = {
        "model_str": "mobilenetv2_unet",
        "dim_in": 3,
        "dim_coords": 2,
        "n_sigma": 2,
        "dim_seeds": 1,
        "dim_out": 5,
        "norm": "BATCH",
        "dropprob": 0.0,
        "multihead": False,
        "layers": [256, 128, 64, 32]
    }
    
    model2 = build_model_from_dict(build_dict)
    print(f"✓ Model loader integration successful")
    print(f"  Model type: {type(model2).__name__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: InstanSegModel integration (simulated)
print("\n[5/5] Testing InstanSegModel compatibility...")
try:
    # We can't fully test InstanSegModel without a dataset, but we can check the config
    from instanseg import InstanSegModel
    
    # This creates the model object but doesn't train
    model_obj = InstanSegModel(model_name='mobilenetv2_unet')
    print(f"✓ InstanSegModel instantiation successful")
    print(f"  Model name set: {model_obj.cfg.get('model', {}).get('model_name', 'N/A')}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✓ ALL VERIFICATION TESTS PASSED!")
print("=" * 70)
print("\nThe MobileNetV2_UNet is ready for training.")
print("You can now run Cell 10 in training_kaggle.ipynb")
