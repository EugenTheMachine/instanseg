# UNET CHANGELOG

## GENERAL DESCRIPTION

Within this task, preserve the existing `InstanSeg_UNet` and add exactly one new architecture option: `EfficientUNetB0`, based on the torchvision EfficientNet-B0 backbone with a custom decoder. Other architecture options are explicitly out of scope for now.

## REQUIREMENTS

### GENERAL REQUIREMENTS

### BACKWARD-COMPATIBILITY AND TRAINING CONTRACT

The existing `InstanSeg_UNet` is the behavioral baseline and must remain unchanged in behavior. The only new architecture in scope is `EfficientUNetB0`; adding it must not change the default architecture or the training/inference contract of `InstanSeg_UNet`. Do not implement or register any other new backbone architecture in this task.

#### Input contract

* Both supported architecture options accept a floating-point image tensor in `NCHW` layout: `[batch, channels, height, width]`, with two spatial dimensions. Preserve the existing preprocessing and pass image values through to the model without adding an input transform.
* `channels` must match the resolved `dim_in` used to construct the model. Preserve the existing model-builder behavior that resolves `dim_in` of `0` or `None` to `3`.
* The `instanseg_unet` route must preserve the existing layer, normalization, activation, and dropout arguments and their defaults. In particular, do not silently change the current `InstanSeg_UNet` block structure or make its currently unused `dropout` argument affect the forward pass.
* The current `InstanSeg_UNet` has one initial, non-pooling encoder block followed by one 2x max-pooling operation for each remaining encoder block; each decoder block upsamples by 2. Its input height and width must therefore be divisible by `2 ** (number_of_encoder_blocks - 1)` (8 for the default four-block encoder) for skip tensors to align. Do not silently crop, pad, or otherwise alter the legacy input to hide a mismatch; reject unsupported spatial sizes clearly. The current training crop size of 512 satisfies this constraint.

#### Output contract

* Return one raw `torch.Tensor`, not a tuple, list, or mapping, in `NCHW` layout. Its batch size and spatial dimensions must match the input; `EfficientUNetB0` must restore the exact input height and width.
* For InstanSeg, output channels must total `dim_coords + n_sigma + dim_seeds` and remain concatenated in this exact order: coordinate channels, sigma channels, then seed channels. Preserve the raw head values and current final-layer normalization behavior; do not add a sigmoid, softmax, threshold, or channel reordering in the model. The loss and inference code apply their existing channel-specific transforms downstream.
* Preserve the existing `multihead` behavior: with `multihead=False`, one decoder emits the coordinate, sigma, and seed channel groups; with `multihead=True`, three independent decoders emit those groups in coordinate/sigma/seed order. Both modes return the same total channel count and tensor shape.
* Preserve the existing `out_channels` convention for direct `InstanSeg_UNet` construction: an integer means one output group; a flat list means one decoder with those output groups; a list of lists means multiple decoders whose outputs are concatenated in list order.

#### Legacy model and verification requirements

* `UNet(model_type="instanseg_unet")` must select the existing `InstanSeg_UNet` behavior and defaults; `UNet(model_type="efficientunetb0")` must select `EfficientUNetB0`. These are the only supported U-Net model types in this task; all other values must raise `ValueError`. Keep direct `InstanSeg_UNet` construction working for existing callers; the general class/factory must not add wrapper dimensions to the legacy model's parameter names or otherwise break strict loading of existing checkpoints.
* For the legacy model, preserve the forward computation, module/parameter/buffer shapes and names, initialization behavior, and training-time behavior. Do not replace the default legacy model with a torchvision backbone. `EfficientUNetB0` may have different internal computation, but must honor the input and output contracts above.
* Add regression tests using a state-dict/output reference captured from the unmodified legacy implementation: load that state dict strictly into the refactored legacy route, compare outputs and parameter gradients for identical weights and inputs, and check output shapes and channel-group ordering for both `multihead` settings. Cover the spatial divisibility constraint and reject unsupported `model_type` values with `ValueError`. Run the tests required below, including the existing training/checkpoint tests, before considering the refactor complete.

Here is a general step-by-step workflow that you must follow when completing this task. You can only proceed to the next step after you have successfully completed the current step and verified that it is actually fully accomplished. You must not proceed to the next step if you have not verified that the current step is fully accomplished:

1. Analyze this spec thoroughly and carefully to make sure that you have read everything and did not miss something.
2. Analyze the existing codebase to understand the best way to implement the code changes.
3. Plan your work step-by-step to ensure that you implement every requirement into the current codebase in the most efficient way and without breaking it.
4. Implement every requirement by implementing the corresponding code changes.
5. After you have implemented in code every requirement from this file, create the corresponding tests which cover every requirement, so that you can ensure that everything works properly, as expected, and does not cause errors.
6. Run all the tests to ensure everything works as expected. If you see that some tests do not pass, correct the corresponding code parts and re-run the tests again. Repeat this process until every test passes successfully. If at some point you face the execution error which is not related to the incorrect code you produced, but related to the fact that the code executor cannot be reached, stop any actions immediately and tell me explciity that "tests cannot be executed because of some internal error". Do not proceed until I explicitly tell you to do so and await my instructions.
7. After you have executed and passed all the tests, verify the scope of the work you did with what is required in this file to ensure you actually implemented everything required. If something is missing, implement and/or test it and then verify again. Keep verifying until you ensure that you actually verified and tested everything required.
8. After successfully passing full verification from the previous step, explicitly tell me that "all the requirements have been properly implemented and tested. The task is completed".

Also note that the new `EfficientUNetB0` implementation must be modular for better structuring and easier code comprehension. In scope for this task, the resulting code should contain the following components (named accordingly for the EfficientNet-B0 backbone):

* encoder (an already existing backbone from the torchvision library, but without one or several final layers, so that the latent representation would be sufficient for further segmentation);
* decoder block class (it must mirror corresponding encoder blocks from the backbone network);
* decoder class (it must consist of several sequential decoder blocks and fully mirror the backbone network);
* `EfficientUNetB0` model class (it must consist of the encoder and decoder objects; by initializing the instance of this class we should initialize the full U-Net model).

Also, create the `UNet` class as the shared parent/base class and general construction interface for the only two architectures in scope: the existing `InstanSeg_UNet` and the new `EfficientUNetB0`. Both architecture classes must derive from `UNet`; it should define their shared U-Net interface and accept `model_type` to select either `"instanseg_unet"` or `"efficientunetb0"`. Use this interface instead of direct `InstanSeg_UNet` construction when creating the InstanSeg model and in other existing U-Net construction sites. Keep direct `InstanSeg_UNet` construction available for backward compatibility. Do not add factory options for architectures outside this scope.

### 1. Implement the `EfficientUNetB0` class

You must implement the `EfficientUNetB0` class, based on the `torchvision.models.efficientnet_b0` backbone from torchvision.

### 2. Implement the `UNet` class

You must implement the `UNet` class as the shared parent/base class and construction interface, and adapt both `InstanSeg_UNet` and `EfficientUNetB0` as child classes. Ensure that the `UNet` construction interface accepts the configurable `model_type` parameter with only the following values (any other value must trigger `ValueError`):

* "instanseg_unet" - must initialize the current implementation of the `InstanSeg_UNet` class;
* "efficientunetb0" - must initialize the current implementation of the `EfficientUNetB0` class;
