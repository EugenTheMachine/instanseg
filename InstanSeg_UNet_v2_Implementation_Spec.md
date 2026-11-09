# InstanSeg_UNet v2 --- Implementation Specification

**Goal:** Implement a compact revision of the existing InstanSeg_UNet
for brightfield microscopy cell instance segmentation. Improve feature
extraction and provide a straightforward residual optimization path
without substantially increasing model size.

**Implementation rule:** Add this as a separate model variant
(`InstanSeg_UNetV2`, factory key `instanseg_unet_v2`). Do not modify or
replace the existing baseline. The original model must remain available
for controlled comparisons.

## 1. Scope

Implement all of the following:

1.  Replace the existing custom four-convolution encoder/decoder blocks
    with conventional two-convolution residual blocks.
2.  Add a lightweight multiscale dilated bottleneck using three
    depthwise-separable branches.
3.  Add one full-resolution refinement block before the output heads.
4.  Use GroupNorm with 8 groups and SiLU as the recommended defaults.
5.  Retain nearest-neighbor upsampling, max-pooling downsampling,
    existing output-head semantics, and the existing input/output
    interface.

Do **not** add attention gates. Do not change losses, targets,
optimizer, scheduler, augmentations, data loading, or evaluation code.

## 2. Compatibility requirements

Preserve the constructor concepts and defaults where applicable:

-   `in_channels`
-   `out_channels`
-   `layers` (default `[256, 128, 64, 32]`)
-   `norm`
-   `dropout`
-   `act`
-   `model_type`

Preserve output configuration behavior:

-   Integer, e.g. `5`: one output head of width 5.
-   Flat list, e.g. `[5, 3]`: one decoder with two output heads.
-   Nested list, e.g. `[[2, 2, 1], [2, 2, 1]]`: multiple decoder groups;
    concatenate their outputs along channels, as in the baseline.

Output spatial dimensions must equal input spatial dimensions. With
three downsampling operations, require height and width divisible by 8.
Preserve the baseline's channel ordering and output semantics.

The new model must support the same input tensor contract as the
baseline: floating-point NCHW tensor with the configured number of input
channels.

## 3. Target topology

For `layers=[256, 128, 64, 32]`, use the reversed encoder widths:

  Stage          Resolution   Channels Operation
  ------------ ------------ ---------- ------------------------------------
  Encoder 0           H × W         32 Residual block, no downsampling
  Encoder 1             H/2         64 MaxPool 2×2 + residual block
  Encoder 2             H/4        128 MaxPool 2×2 + residual block
  Encoder 3             H/8        256 MaxPool 2×2 + residual block
  Bottleneck            H/8        256 Multiscale dilated residual module

Decoder:

  Stage       Input → output resolution     Output channels   Skip channels
  ----------- --------------------------- ----------------- ---------------
  Decoder 0   H/8 → H/4                                 128             128
  Decoder 1   H/4 → H/2                                  64              64
  Decoder 2   H/2 → H                                    32              32

After the final decoder stage, apply the full-resolution refinement
block and then the existing 1×1 output head(s).

``` text
Input
  -> residual encoder block (32, H) ---------------- skip H
  -> max-pool
  -> residual encoder block (64, H/2) ------------- skip H/2
  -> max-pool
  -> residual encoder block (128, H/4) ------------ skip H/4
  -> max-pool
  -> residual encoder block (256, H/8)
  -> multiscale dilated residual bottleneck (256)
  -> upsample + skip fusion + residual block (128, H/4)
  -> upsample + skip fusion + residual block (64, H/2)
  -> upsample + skip fusion + residual block (32, H)
  -> full-resolution refinement (32)
  -> existing output head(s)
```

## 4. Normalization and activation

### 4.1 Normalization

Recommended default: **GroupNorm with 8 groups**.

The normalization factory must support:

-   `BATCH`: `nn.BatchNorm2d`
-   `GROUP`: `nn.GroupNorm`
-   `INSTANCE`: `nn.InstanceNorm2d(..., affine=True)`
-   `None` / `"None"`: `nn.Identity`

For GroupNorm, choose 8 groups when divisible; otherwise choose the
largest positive divisor of the channel count no greater than 8. Ensure
`num_groups` divides `num_channels`. Do not use `GroupNorm(1, C)` as the
default; the intention is multi-group normalization.

Keep BatchNorm selectable for controlled experiments.

### 4.2 Activation

Recommended default: **SiLU** (`nn.SiLU(inplace=True)`).

Retain existing supported choices if already part of the public API:
ReLU, GELU, SiLU/Swish, Mish, and None/Identity. Use the selected
activation consistently in v2, except that the second convolution's
normalization must remain linear immediately before residual addition.

## 5. Conventional two-convolution residual block

Use this block in both the encoder and decoder instead of the existing
four-convolution custom blocks.

### 5.1 Main branch

1.  3×3 convolution, `in_channels -> out_channels`, padding 1.
2.  Normalization.
3.  Activation.
4.  3×3 convolution, `out_channels -> out_channels`, padding 1.
5.  Normalization, with **no activation before the residual addition**.

### 5.2 Shortcut branch

-   Identity if input and output channel counts match and spatial
    dimensions are unchanged.
-   Otherwise, 1×1 projection convolution `in_channels -> out_channels`,
    followed by normalization.
-   Projection must not change spatial dimensions.
-   Use `bias=False` when followed by normalization.
-   Do not apply an activation to the shortcut before addition.

### 5.3 Merge

``` python
residual = self.shortcut(x)

x = self.conv1(x)
x = self.norm1(x)
x = self.act(x)

x = self.conv2(x)
x = self.norm2(x)

x = self.act(x + residual)
```

### 5.4 Encoder downsampling

-   Encoder stage 0: residual block without downsampling.
-   Encoder stages 1--3: `nn.MaxPool2d(kernel_size=2, stride=2)`
    followed by the residual block.
-   Keep max pooling for this first v2 experiment. Do not simultaneously
    replace it with strided convolutions or anti-aliased downsampling.

### 5.5 Decoder upsampling and skip fusion

At each decoder stage:

1.  Upsample decoder features by 2 using
    `F.interpolate(..., mode="nearest")`.
2.  Project upsampled decoder features to `out_channels` with a 1×1
    convolution.
3.  Project the corresponding encoder skip tensor to `out_channels` with
    a 1×1 convolution.
4.  Add the projected tensors.
5.  Apply the two-convolution residual block at `out_channels`.

Do not use attention gates, sigmoid skip weighting, or bilinear
upsampling. Ensure decoder and skip spatial dimensions match. Do not
silently crop or discard image regions; input validation should ensure
compatible dimensions.

## 6. Lightweight multiscale dilated bottleneck

Insert this module after the final encoder block and before the decoder.
It must preserve both channel count and spatial dimensions.

**Input/output:** `[N, C, H/8, W/8]` → `[N, C, H/8, W/8]`; default
`C=256`.

### 6.1 Three parallel branches

Each branch receives the same input:

  ------------------------------------------------------------------------------------
  Branch           Operator                           Dilation Intended scale
  ---------------- --------------------- --------------------- -----------------------
  1                3×3                                       1 Local cell
                   depthwise-separable                         structures/boundaries
                   convolution                                 

  2                3×3                                       2 Intermediate-scale
                   depthwise-separable                         structures
                   convolution                                 

  3                3×3                                       4 Broader cell/cluster
                   depthwise-separable                         context
                   convolution                                 
  ------------------------------------------------------------------------------------

Each branch consists of:

1.  3×3 depthwise convolution: `in_channels=C`, `out_channels=C`,
    `groups=C`, `padding=dilation`, `dilation=dilation`.
2.  Normalization.
3.  Activation.
4.  1×1 pointwise convolution: `C -> branch_channels`.
5.  Normalization and activation.

Use `branch_channels = C // 4` by default (64 when C=256). Validate that
this produces a positive channel count.

### 6.2 Aggregation and residual

1.  Concatenate the three branch outputs along channels.
2.  Project `3 * branch_channels -> C` with a 1×1 convolution.
3.  Apply normalization.
4.  Add the original bottleneck input.
5.  Apply the configured activation.

``` python
b1 = self.branch_d1(x)
b2 = self.branch_d2(x)
b3 = self.branch_d4(x)

multi_scale = torch.cat([b1, b2, b3], dim=1)
multi_scale = self.project(multi_scale)
multi_scale = self.norm(multi_scale)

out = self.act(x + multi_scale)
```

Do not concatenate multiscale features directly into the decoder. The
bottleneck output must remain `C` channels. Use depthwise-separable
rather than full-width dense 3×3 convolutions in the branches. Do not
add a global-pooling branch in this version. Keep dilation rates 1, 2,
and 4.

## 7. Full-resolution refinement block

Add one refinement block after the last decoder stage and before output
heads.

Default width is 32 channels:

``` python
Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
Normalization(32)
SiLU
```

Respect the model's configured normalization and activation rather than
hardcoding them if those are configurable.

The refinement must: - Preserve H×W resolution and channel count. - Run
once after all decoder stages. - Be shared by the output heads belonging
to the same decoder path. - Not change output channel count, order, or
meaning. - Not add a raw-input full-resolution skip connection.

For multiple independent decoder groups, keep the refinement associated
with each decoder path immediately before that path's output heads.

## 8. Output heads and multiple decoder groups

Retain baseline semantics:

1.  Each output group has one 1×1 convolution per requested output
    width.
2.  Concatenate the heads within that group along the channel dimension.
3.  If multiple decoder groups exist, concatenate group outputs along
    the channel dimension.

Do not collapse multiple decoder groups into one or alter the
meaning/order of `out_channels`.

## 9. Dropout

The baseline stores `dropout` but does not currently use it. Implement
dropout in v2 only at the bottleneck, using `nn.Dropout2d`.

-   If `dropout == 0`, omit dropout or use an identity module.
-   If `dropout > 0`, apply the configured probability at a clearly
    documented bottleneck-only location.
-   Validate `0 <= dropout < 1`.
-   Do not add dropout throughout encoder and decoder blocks.
-   Apply it consistently; do not vary placement across output heads.

A suitable placement is after multiscale aggregation/projection and
before the bottleneck residual output activation. Document the exact
placement in code.

## 10. Initialization

Use PyTorch defaults or the project's existing consistent initialization
policy. If implementing explicit initialization:

-   Convolution weights: Kaiming initialization appropriate to the
    selected activation.
-   Normalization weights: ones.
-   Normalization biases: zeros.

Do not add attention-gate initialization. Do not overwrite pretrained
weights; this architecture has no pretrained encoder. Avoid unnecessary
custom initialization.

## 11. Required tests and acceptance criteria

### 11.1 Shape and forward tests

Test at least:

-   `[2, 3, 256, 256]`
-   `[1, 1, 520, 704]` if single-channel input is supported
-   A non-square input

Verify: - Output H and W equal input H and W. - Output channel count
equals the sum of configured output-head widths. - All outputs are
finite. - Forward and backward passes succeed.

### 11.2 Configuration tests

Test: - Integer `out_channels`. - Flat-list `out_channels`. -
Nested-list `out_channels` with multiple decoder groups. -
`norm="GROUP"`, `norm="BATCH"`, and `norm=None`. - `act="SILU"`. -
`dropout=0` and a nonzero dropout value.

Invalid-input tests: - Wrong tensor rank. - Non-floating-point input. -
Incorrect input channel count. - Spatial dimensions not divisible by
8. - Invalid dropout probability. - Unsupported normalization or
activation.

### 11.3 Gradient tests

Backpropagate a synthetic scalar loss and verify finite gradients
reach: - First encoder block. - Bottleneck. - Every decoder stage. -
Refinement block. - Output heads.

### 11.4 Parameter accounting

Report: - Baseline trainable and non-trainable parameter counts. - v2
trainable and non-trainable parameter counts. - Absolute and percentage
parameter-count change.

Measure actual counts; do not assert an unverified parameter estimate.

### 11.5 Baseline preservation

Verify: - Existing `instanseg_unet` remains instantiable. - Baseline
state-dict keys and forward behavior are not unintentionally changed. -
v2 is selectable through an explicit model identifier. - Existing
training code can select either model without changing loss or target
format.

## 12. Controlled evaluation protocol

This section is guidance for the later experiment, not permission to
modify training code during this implementation.

Compare baseline and v2 with identical: - Train/validation/test split. -
Preprocessing and augmentation. - Loss function. - Optimizer and
scheduler initially. - Batch size and epoch budget. -
Checkpoint-selection criterion.

Use paired seeds and, if compute permits, at least three random seeds.
Record training loss, validation loss, training F1, validation F1, best
epoch, test F1, parameter count, training time, and peak memory.

Do not tune against the test set. Do not claim better F1 or convergence
until evaluated on the actual microscopy dataset.

## 13. Explicitly out of scope

Do not include these in v2:

-   Attention gates or spatial/channel attention.
-   Bilinear interpolation.
-   Transposed convolutions or learned upsampling.
-   Strided-convolution or anti-aliased downsampling.
-   Transformer blocks.
-   ASPP global-pooling branch.
-   Pretrained backbones.
-   Changes to loss, optimizer, scheduler, or augmentation.
-   Raw-input full-resolution skip connections.
-   Changes to instance-segmentation target representation.

Evaluate these only as separate, isolated future experiments.

## 14. Deliverables

The implementation agent must provide:

1.  New v2 model implementation.
2.  Explicit factory/model-type mapping for `instanseg_unet_v2`.
3.  Unit tests for all required shapes, configurations, and gradients.
4.  Summary of modified and newly created files.
5.  Measured parameter-count comparison against baseline.
6.  Minimal instantiation and forward-pass example.
7.  Confirmation that the original baseline remains available and
    unchanged.

**Success criterion:** A correct, tested implementation that preserves
the baseline and is ready for controlled training comparison. Higher F1
or better convergence is an empirical hypothesis, not an implementation
guarantee.
