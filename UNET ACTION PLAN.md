# UNET ACTION PLAN

This document outlines the detailed step-by-step action plan for implementing the U-Net architecture family described in `UNET CHANGELOG.md`.

---

## 1. Overview & Architectural Goals

1. **Unified Base & Factory Class (`UNet`)**:
   - Create a general `UNet(nn.Module)` base class.
   - All concrete U-Net variants will inherit directly from `UNet`.
   - Implement `UNet.__new__(cls, model_type="instanseg_unet", *args, **kwargs)` as a factory method that instantiates the corresponding concrete subclass based on the lowercased `model_type` string.
   - Raise a `ValueError` for any unsupported `model_type` string.

2. **Equal-Level Concrete U-Net Classes**:
   - Every U-Net variant will be implemented as a separate class at the exact same level as `InstanSeg_UNet`, all inheriting from `UNet`:
     - `InstanSeg_UNet` (Original custom InstanSeg U-Net)
     - `EfficientUNetB0` (`torchvision.models.efficientnet_b0`)
     - `EfficientUNetB1` (`torchvision.models.efficientnet_b1`)
     - `EfficientUNetB2` (`torchvision.models.efficientnet_b2`)
     - `EfficientUNetB3` (`torchvision.models.efficientnet_b3`)
     - `EfficientUNetV2S` (`torchvision.models.efficientnet_v2_s`)
     - `MobileUNetV2` (`torchvision.models.mobilenet_v2`)
     - `MobileUNetV3S` (`torchvision.models.mobilenet_v3_small`)
     - `MobileUNetV3L` (`torchvision.models.mobilenet_v3_large`)
     - `RegNetUNetY400mf` (`torchvision.models.regnet_y_400mf`)
     - `RegNetUNetY800mf` (`torchvision.models.regnet_y_800mf`)
     - `ResNetUNet18` (`torchvision.models.resnet18`)

3. **Input/Output Dimensionality Compatibility**:
   - **Input Shape**: `(B, in_channels, H, W)`
   - **Output Shape**: `(B, out_channels_total, H, W)` matching spatial dimensions `(H, W)` exactly. Target size upsampling will ensure consistency regardless of backbone downsampling ratios.
   - **Multi-Head Output Support**: All variants will support list-based `out_channels` (e.g., `[[2, 2, 1]]` for `cells` or `[[2, 2, 1], [2, 2, 1]]` for `cells_and_nuclei`) producing output shapes identical to `InstanSeg_UNet` (e.g., `(B, 5, H, W)`).

---

## 2. Modular Class Design (Per Backbone)

Each external backbone architecture will follow a modular 4-component design:

1. **Encoder Class**:
   - Wraps a `torchvision.models` backbone (without final classification head/pooling).
   - Adapts the first convolutional layer (`_adapt_first_conv`) if `in_channels != 3`.
   - Returns `(bottleneck_tensor, skip_connections_list)`.

2. **Decoder Block Class (`GenericDecoderBlock`)**:
   - Uses standard U-Net **Concatenation** (`torch.cat([upsampled_x, skip], dim=1)`) followed by Conv-Norm-Act blocks (preventing unscaled logit explosion).

3. **Decoder Class (`GenericDecoder`)**:
   - Consists of sequential `GenericDecoderBlock` instances mirroring the encoder stages.
   - Includes output 1x1 Conv heads with standard initialization (`std=0.01`, zero bias).
   - Applies target-size interpolation if spatial dimensions differ slightly.

4. **U-Net Model Class**:
   - Inherits directly from `UNet`.
   - Instantiates the encoder and multi-head decoders in `__init__`.
   - Performs feature extraction and multi-head decoding in `forward`.

---

## 3. Detailed Action Plan Steps

### Step 1: Implement `UNet` Base Class & `InstanSeg_UNet`
- Create `instanseg/utils/models/UNet.py`.
- Define `UNet(nn.Module)` with factory dispatching in `__new__`.
- Define `InstanSeg_UNet(UNet)` preserving its exact original `EncoderBlock` and `Decoder` architecture and attribute names (`self.encoder`, `self.decoders`).
- Ensure `layers` array ordering (`[32, 64, 128, 256]`) is handled consistently without double-inversion.

### Step 2: Implement Decoder Components & First-Conv Adaptation
- Implement `_adapt_first_conv` for handling arbitrary `in_channels`.
- Implement `GenericDecoderBlock` (concatenation + Conv-Norm-Act) and `GenericDecoder` (multi-head decoding + target-size matching).

### Step 3: Implement All 11 Backbone Subclasses
- Implement encoder and model classes for all 11 external backbones:
  - `ResNetUNet18`
  - `EfficientUNetB0`, `EfficientUNetB1`, `EfficientUNetB2`, `EfficientUNetB3`, `EfficientUNetV2S`
  - `MobileUNetV2`, `MobileUNetV3S`, `MobileUNetV3L`
  - `RegNetUNetY400mf`, `RegNetUNetY800mf`

### Step 4: Update `model_loader.py` Integration
- Update `build_model_from_dict` in `instanseg/utils/model_loader.py` to route all supported U-Net model types (`model_str` / `model_name`) through `UNet(model_type=...)`.

### Step 5: Comprehensive Local Testing
- Create and execute `tests/test_unet.py`:
  - Test inheritance (`issubclass(cls, UNet)`).
  - Test factory instantiation (`UNet(model_type=...)`).
  - Test `ValueError` for unsupported model types.
  - Test forward output shape `(1, 5, 128, 128)` for all 12 variants.
  - Test `build_model_from_dict` integration for all 12 variants.
- Execute local training tests (`test_dataloader_efficiency.py`, `test_reproducibility.py`, `test_resume_training.py`, `test_train_eval_wrapper.py`) with `imgsz=128` and `batch_size=1`.

### Step 6: Remote Kaggle Verification
- Commit and push implementation to GitHub.
- Trigger Kaggle remote GPU training with `InstanSeg_UNet`.
- Perform an initial 150-second status check for execution errors.
- Monitor run through completion (~2000s) and verify that final quality metrics match historical baseline numbers.

---

## 4. Verification Checklist

- [ ] All 12 U-Net classes inherit directly from `UNet`.
- [ ] `UNet(model_type=...)` instantiates any of the 12 U-Net classes correctly.
- [ ] `UNet` raises `ValueError` for invalid `model_type`.
- [ ] Output shapes match `(B, C_out, H, W)` identically across all 12 backbones.
- [ ] `tests/test_unet.py` passes 26/26 tests cleanly.
- [ ] All 4 local training tests pass cleanly.
- [ ] Remote Kaggle GPU training runs without errors and converges.
