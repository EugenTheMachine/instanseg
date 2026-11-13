#!/usr/bin/env python
"""Quick verification that MobileNetV2_UNet is ready"""

def main():
    print("=" * 70)
    print("MobileNetV2_UNet Quick Verification")
    print("=" * 70)
    
    # Test 1: Import
    print("\n[1/4] Testing imports...")
    try:
        from instanseg.utils.models.InstanSeg_UNet import MobileNetV2_UNet, UNet
        from instanseg.utils.model_loader import build_model_from_dict
        from instanseg import InstanSegModel
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Instantiation
    print("\n[2/4] Testing model instantiation...")
    try:
        import torch
        model = MobileNetV2_UNet(in_channels=3, out_channels=[[2, 2, 1]])
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model created: {params:.2f}M parameters")
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        return False
    
    # Test 3: Dimension test
    print("\n[3/4] Testing output dimensions...")
    try:
        test_sizes = [256, 512]
        for size in test_sizes:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            expected = (1, 5, size, size)
            if out.shape != torch.Size(expected):
                print(f"✗ {size}x{size}: Expected {expected}, got {tuple(out.shape)}")
                return False
            print(f"✓ {size}x{size}: {tuple(out.shape)} ✓")
    except Exception as e:
        print(f"✗ Dimension test failed: {e}")
        return False
    
    # Test 4: Model loader
    print("\n[4/4] Testing model_loader integration...")
    try:
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
        }
        model2 = build_model_from_dict(build_dict)
        print("✓ Model loader integration working")
    except Exception as e:
        print(f"✗ Model loader failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
    print("=" * 70)
    print("\nMobileNetV2_UNet is ready for training!")
    print("Run Cell 10 in training_kaggle.ipynb to start training.")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
