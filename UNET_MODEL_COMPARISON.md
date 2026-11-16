# `instanseg_unet` vs `improved_unet`

## Summary

These are closely related variants, not different encoder families. `ImprovedUNet`
subclasses `InstanSeg_UNet` and reuses its encoder, model construction, output
heads, and input checks. Its main changes are in the decoder: bilinear instead
of nearest-neighbor upsampling, plus an attention gate on every encoder skip.
The default decoder activation also changes from ReLU to GELU.

## Differences, one by one

1. **Decoder block selection.** `InstanSeg_UNet` builds standard `DecoderBlock`s;
   `ImprovedUNet` overrides the decoder-block class with `ImprovedDecoderBlock`.
   The inherited constructor uses that class to build each decoder stage, so
   the encoder remains the same while the decoder behavior changes.
   [Implementation](./instanseg/utils/models/InstanSeg_UNet.py#L313)
   · [Improved variant](./instanseg/utils/models/InstanSeg_UNet.py#L472)

2. **Upsampling.** The standard decoder uses 2× nearest-neighbor interpolation.
   The improved decoder uses 2× bilinear interpolation with
   `align_corners=False`. This changes how intermediate feature values are
   reconstructed between scales; it does not change the number of decoder
   stages or their target resolutions.
   [Standard block](./instanseg/utils/models/InstanSeg_UNet.py#L338)
   · [Improved block](./instanseg/utils/models/InstanSeg_UNet.py#L382)

3. **Skip-connection fusion.** The standard block projects a skip with a 1×1
   convolution, adds it to the decoder features, and refines the sum with
   convolutions. The improved block first projects both feature maps, predicts
   a single-channel spatial gate from their concatenation, and scales the skip
   by `0.5 + 0.5 × gate` before adding it. Because the sigmoid gate is in
   `[0, 1]`, the skip contribution is retained at 50–100% of its projected
   value rather than being fully suppressed. This adds a learned, spatially
   varying way to modulate skip information.
   [Standard block](./instanseg/utils/models/InstanSeg_UNet.py#L328)
   · [Gate and fusion](./instanseg/utils/models/InstanSeg_UNet.py#L359)

4. **Decoder refinement.** Both blocks retain the projected decoder residual
   and the subsequent convolutional residual refinement. The improved block
   also retains the standard block's optional “shallow” behavior: when enabled,
   its `conv3` refinement is replaced with an identity. Thus attention and
   interpolation are the principal structural changes; the overall residual
   decoder pattern is shared.
   [Standard block](./instanseg/utils/models/InstanSeg_UNet.py#L338)
   · [Improved block](./instanseg/utils/models/InstanSeg_UNet.py#L382)

5. **Default activation.** Direct `InstanSeg_UNet` construction defaults to
   `ReLu` (normalized by the helper to ReLU); `ImprovedUNet` defaults to GELU.
   Both otherwise use the inherited encoder and decoder normalization setting,
   which defaults to batch normalization. A caller that explicitly supplies an
   activation can make this difference disappear.
   [Base defaults](./instanseg/utils/models/InstanSeg_UNet.py#L475)
   · [Improved defaults](./instanseg/utils/models/InstanSeg_UNet.py#L537)

## What stays the same

- **Encoder and scale layout:** both use the same residual-style `EncoderBlock`
  stack. With the default four widths `[256, 128, 64, 32]`, the encoder has
  three downsampling steps and the decoder has three upsampling steps. The
  model checks that input height and width are divisible by 8.
- **Predictions:** both support the same output-channel/head configuration,
  produce full-resolution outputs, and concatenate multiple decoder heads
  along the channel dimension.
- **Configuration route:** `UNet(model_type=...)` and
  `build_model_from_dict(...)` route both names to the corresponding class.
  Their shared output contract is covered by the focused architecture tests.
  [Factory](./instanseg/utils/models/InstanSeg_UNet.py#L60)
  · [Model builder](./instanseg/utils/model_loader.py#L129)
  · [Tests](./tests/test_unet_architectures.py#L632)

## Practical implications

Choose `instanseg_unet` for the baseline decoder with nearest upsampling and
unmodulated additive skips. Choose `improved_unet` when you want skip features
to be spatially gated and prefer bilinear upsampling/GELU defaults. The
improved model adds one small gate convolution per decoder stage, so its
decoder has additional parameters and operations. Although the encoder and
output structure are shared, the decoder has extra gate parameters; do not
assume a checkpoint from one model type will load strictly into the other.

`kaggle_train.py` is a Kaggle execution/monitoring script, not where these model
architectures are defined. The model implementations and model-type routing
are in `instanseg/utils/models/InstanSeg_UNet.py` and
`instanseg/utils/model_loader.py`.

## `unet_v3`

`unet_v3` is the baseline `instanseg_unet` architecture with GELU in place of
the baseline's default ReLU activation. It keeps the same encoder, standard
nearest-neighbor decoder, additive skip fusion, and output heads; it does not
use `improved_unet`'s attention gates or bilinear upsampling. As with the other
variants, an explicitly supplied activation overrides the default.

## `unet_v4`

`unet_v4` keeps the baseline encoder, activation defaults, nearest-neighbor
upsampling, residual decoder, and output heads. At every decoder stage, it
projects the encoder skip into two parallel paths: an unmodulated baseline
path and a second path with its own projection and the `improved_unet`-style
spatial attention gate. The gated path is scaled by `0.5 + 0.5 × gate`, then
the two paths are averaged before fusion with decoder features. This gives
both paths equal weights while keeping the combined skip signal on roughly
the scale of a single baseline skip. Both projections use the same
normalization and activation settings and map to the decoder's channel and
spatial dimensions.

## `unet_v5`

`unet_v5` retains the baseline encoder, activation defaults, output heads, and
overall model structure, but replaces each baseline decoder block with
`ImprovedDecoderBlock`. Its decoder therefore uses bilinear upsampling and
attention-gated skip fusion, while the encoder still uses the baseline's
default ReLU activation. Unlike `improved_unet`, this model intentionally
keeps the baseline activation default; all other model settings remain those
of `instanseg_unet` unless explicitly overridden.
