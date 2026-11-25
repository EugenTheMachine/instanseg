# BORDER EMBEDDINGS

## Status

This document is an implementation specification for adding a new border-directed embedding component to InstanSeg. It is based on the current codebase behavior, not on existing `.md` files in the repository.

## 1. Current Embedding Analysis

### 1.1 Scope

The primary model path analyzed is the InstanSeg U-Net family:

- `instanseg/utils/models/InstanSeg_UNet.py` - U-Net factory and default `InstanSeg_UNet` implementation.
- `instanseg/utils/model_loader.py` - builds models from config dictionaries and assembles output channel layout.
- `instanseg/utils/loss/instanseg_loss.py` - `InstanSeg` training loss, postprocessing, and `InstanSeg_Torchscript` export/inference wrapper.
- `instanseg/instanseg_model.py` - training/evaluation orchestration; constructs `InstanSeg` loss and reads seed/output channels.
- `instanseg/scripts/train.py`, `instanseg/scripts/test.py`, `instanseg/utils/utils.py` - argument/config plumbing and TorchScript export.

Other model variants (`InstanSeg_UNet_v2`, `MedNeXt_UNet`, MobileNet/EfficientNet/ConvNeXt UNets) share the same output-channel contract because they are built through `model_loader.build_model_from_dict` and consumed by `InstanSeg` loss/postprocessing.

### 1.2 Current Output Channel Contract

`model_loader.build_model_from_dict` defines the raw model output channels as:

- Single-head: `out_channels = [[dim_coords, n_sigma, dim_seeds]]`
- Multihead: `out_channels = [[dim_coords], [n_sigma], [n_seeds]]`

Defaults are `dim_coords=2`, `n_sigma=2`, `dim_seeds=1`, so the default output has 5 channels.

In `InstanSeg_UNet`, decoders/final blocks concatenate all final head outputs along channel dim:

```py
x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)
```

The channel order is therefore:

1. `dim_coords` spatial embedding channels
2. `n_sigma` sigma/scale channels
3. `dim_seeds` seed channels

This layout is consumed by `instanseg.utils.loss.instanseg_loss.InstanSeg`:

```py
self.dim_out = self.dim_coords + self.n_sigma + self.dim_seeds
```

### 1.3 How Current Center Embeddings Are Created

#### Model input

The UNet receives an image tensor `x` with shape `N,C,H,W`. `UNet._validate_input_tensor` requires NCHW, floating point, matching `in_channels`, and spatial dims divisible by `2^(len(encoder)-1)`.

#### Model computation

`InstanSeg_UNet.forward`:

1. Validates input.
2. Runs encoder blocks, collecting skip features.
3. Runs one or more decoders over the deepest feature and reversed skips.
4. Concatenates decoder outputs into final prediction tensor `N, dim_out, H, W`.

The network itself does not explicitly compute object centers. It learns raw output channels; semantic meaning is imposed by loss and postprocessing.

#### Coordinate grid

`generate_coordinate_map(mode="linear", spatial_dim=2, height, width, device)` creates `xxyy` with shape `spatial_dim,H,W`:

```py
xx = torch.linspace(0, width * 64 / 256, width, ...)
yy = torch.linspace(0, height * 64 / 256, height, ...)
xxyy = torch.cat((xx, yy), 0)
```

This is a fixed linear coordinate system in model space, scaled by `64/256` per pixel. It is not learned.

### 1.4 How Current Center Embeddings Are Calculated

For prediction tensor `prediction_b` with shape `C_out,H,W` per batch item:

```py
xxyy = generate_coordinate_map(..., spatial_dim=self.dim_coords, ...)
spatial_emb = (torch.sigmoid(prediction_b[:, 0:self.dim_coords]) - 0.5) * 8 + xxyy
sigma = prediction_b[:, self.dim_coords:self.dim_coords + self.n_sigma]
seed_map = prediction_b[:, -self.dim_seeds:]
```

Interpretation:

- Raw coord channels are logits/deltas in sigmoid space.
- `sigmoid(...)-0.5)*8` maps them to approximately `[-4, 4]` in grid units before adding `xxyy`.
- After adding `xxyy`, each pixel's `spatial_emb[:, y, x]` is an absolute coordinate prediction in the same coordinate system as the grid. It is trained to represent the instance location/center assignment used by the instance grouping loss.
- `sigma` are per-pixel scale parameters consumed by feature engineering.
- `seed_map` is a seedness/objectness map used to detect instance seeds.

The seed map is interpreted differently depending on `dim_seeds`:

- `dim_seeds == 1`: postprocessing uses `mask_map = ((prediction_i[-1]) / 15) + 0.5` and also as binary mask; training seed loss uses raw `seed_map` against transformed targets depending on configured `seed_loss_fn`.
- `dim_seeds > 1`: channel `-2` is treated as seed distance-like map (`/15 + 0.5`), channel `-1` as binary foreground map after sigmoid; foreground threshold masks seed candidates.

### 1.5 How Current Embeddings Are Used in Loss

`InstanSeg.forward(prediction, instances, ...)` computes loss per batch/image:

1. Builds `xxyy`.
2. Splits prediction into `spatial_emb`, `sigma`, `seed_map`.
3. Computes seed loss via `self.seed_loss(seed_map, instance, mask)`.
4. If instance pixels exist, remaps labels, creates one-hot labels, finds local maxima on `sigmoid(seed_map[0]) * onehot_labels.sum(0)` as training centroids.
5. Samples embedding vectors at centroid locations:

   ```py
   centres = spatial_emb[:, centroids[0], centroids[1]].detach().T
   ```

6. Calls `compute_crops(spatial_emb, centres, sigma, centroids, feature_engineering, pixel_classifier, window_size=...)`.
7. Builds per-instance target crops from `onehot_labels` and applies `binary_loss_fn(dist, crop.float(), sigma=sigma[0])`.

The instance grouping signal comes from feature engineering:

```py
x = torch.cat([x, sigma])[:, mesh_grid_flat[0], mesh_grid_flat[1]]
x = rearrange(x, '(E) (C H W) -> C (E) H W', ...)
x[:, :E] -= c_shaped
```

Where `E == dim_coords`. For each candidate instance center `c`, the model sees per-pixel relative embedding `emb_pixel - emb_center`, plus sigma channels. `ProbabilityNet` is an MLP over this per-pixel feature vector and outputs instance membership logits for the crop.

`ProbabilityNet` input size is initialized as:

```py
MLP_input_dim = self.feature_engineering_width + self.n_sigma - 2 + self.dim_coords
```

With default `feature_engineering="0"`, `feature_engineering_width==2`, so default input is `2 + n_sigma - 2 + dim_coords = n_sigma + dim_coords`.

### 1.6 How Current Embeddings Are Used in Postprocessing

`InstanSeg.postprocessing(...)` performs:

1. Slices `prediction_i = prediction[:dim_out, :, :]`.
2. Builds `xxyy`.
3. Computes `fields = (sigmoid(prediction_i[0:dim_coords]) - 0.5) * 8` and then `fields = fields + xxyy`.
4. Reads `sigma` and `mask_map`/`binary_map` from seed channels.
5. Finds seed peaks by `torch_peak_local_max(mask_map, neighbourhood_size=peak_distance, minimum_value=seed_threshold)`.
6. Optionally caps seeds by top seed score.
7. Gets embeddings at seed locations: `fields_at_centroids = fields[:, local_centroids_idx[:,0], local_centroids_idx[:,1]]`.
8. Runs `compute_crops(fields, fields_at_centroids.T, sigma, ...)` to get per-instance crop logits and crop coordinates.
9. Merges sparse crop predictions with `merge_sparse_predictions(...)` into final instance label map.

`InstanSeg_Torchscript.forward` duplicates this logic for export/runtime. It must be kept behaviorally aligned.

### 1.7 Where Channel Layout Is Configured/Plumbed

| Area | File(s) | Current behavior | Border-mode impact |
|---|---|---|---|
| Model args/config | `instanseg/scripts/train.py`, `instanseg/instanseg_model.py`, `model_loader.read_model_args_from_csv` | reads/passes `dim_coords`, `n_sigma`, `dim_seeds` | add `embedding_mode` to config, CLI/model dict, validation |
| Model construction | `instanseg/utils/model_loader.py::build_model_from_dict` | computes `out_channels` from dim coords/sigma/seeds | compute out_channels based on mode and border channel count |
| UNet output contract | `instanseg/utils/models/InstanSeg_UNet.py` and other UNet variants | returns concatenated head channels | no semantic change in UNet except accepting/propagating mode metadata if needed; output size changes |
| Loss/postprocess | `instanseg/utils/loss/instanseg_loss.py::InstanSeg` | slices by fixed order and uses center embedding | add mode-aware slicing, border target/loss, border-aware feature engineering/postprocess |
| TorchScript wrapper | `instanseg/utils/loss/instanseg_loss.py::InstanSeg_Torchscript` | duplicates slicing/postprocess | mirror mode-aware behavior exactly |
| Export/load | `instanseg/utils/utils.py`, `model_loader.load_model_weights` | exports/loads n_sigma/feature args; initializes pixel classifier | persist and restore `embedding_mode`; include border dims in classifier init |
| Evaluation/diagnostics | `instanseg/instanseg_model.py` | reads seed map as `out[i, -1]` and assumes layout | use mode-aware seed channel accessor |
| Tests | `tests/test_unet_architectures.py`, `tests/test_unet_v2.py` | hardcode 5-channel expectations | add mode coverage; keep center defaults passing |

## 2. Proposed Border Embedding

### 2.1 Goal

Add a new embedding component whose per-pixel vector points to the closest border of the cell instance to which that pixel belongs. This is an additional/alternative spatial embedding to the existing center-pointing embedding.

### 2.2 High-Level Semantics

For a pixel `p` inside instance `I(p)`:

- **Center embedding (existing)** predicts the instance center/anchor coordinate used by InstanSeg's current grouping mechanism.
- **Border embedding (new)** predicts the nearest point `b(p)` on the boundary of `I(p)`.

For background pixels (no instance), border embedding has no instance boundary target. It should follow the same masking/ignore rules as existing instance-specific supervision: where no positive instance label exists, instance embedding supervision is not applied except through existing seed/background behavior.

### 2.3 Representation

The border embedding must inherit the representation logic of center embeddings:

- Same dimensionality as center coordinates: `dim_coords` channels (default 2).
- Same coordinate system as `generate_coordinate_map`.
- Same raw-to-embedding transform:

```py
border_emb = (torch.sigmoid(pred_border) - 0.5) * 8 + xxyy
```

- Same per-pixel vector semantics: absolute coordinate in the scaled grid space.
- Same feature-engineering interface: can be gathered at seed/centroid locations and converted into relative features by subtraction.

The border vector itself can be derived as:

```py
border_offset = border_emb - xxyy   # predicted vector from pixel grid coord to nearest border point
```

This is analogous to how center embedding can be interpreted as pointing from grid location to predicted center location.

### 2.4 Ground Truth Definition

For each labeled instance, compute the nearest boundary point for every pixel inside that instance.

Recommended definition:

1. For instance mask `M_i`, obtain instance boundary `B_i` (pixels of `M_i` adjacent to background or another instance, or morphological boundary).
2. For every pixel `p in M_i`, define target border point `T_border(p)` as argmin over `q in B_i` of Euclidean distance `||p-q||` in image pixel coordinates, then convert to the same scaled coordinate space used by `xxyy`.
3. If multiple boundary points are equally close, use a deterministic tie-breaker (e.g., smallest row, then smallest col).

Implementation note: exact EDT/boundary-distance transforms or precomputed nearest-index maps are acceptable if they produce equivalent nearest-boundary targets. The spec requires nearest boundary, not a learned proxy.

### 2.5 Channel Layout

Introduce mode-aware spatial embedding layout. Keep `sigma` and `seed` semantics unchanged.

Let `E = dim_coords`.

| Mode | Channels | Layout |
|---|---|---|
| `center` | `E + n_sigma + dim_seeds` | `[center(E), sigma(n_sigma), seed(dim_seeds)]` |
| `border` | `E + n_sigma + dim_seeds` | `[border(E), sigma(n_sigma), seed(dim_seeds)]` |
| `combined` | `2*E + n_sigma + dim_seeds` | `[center(E), border(E), sigma(n_sigma), seed(dim_seeds)]` |

Defaults remain backward compatible: `embedding_mode="center"`, `dim_coords=2`, `n_sigma=2`, `dim_seeds=1`.

### 2.6 Model Initialization Argument

Add `embedding_mode` to model initialization/configuration surfaces.

Required accepted values:

- `"center"` - current behavior, center-pointed vectors only.
- `"border"` - new border-pointed vectors only.
- `"combined"` - both center-pointed and border-pointed vectors.

Primary surfaces:

- `InstanSeg` loss/postprocess class (`instanseg/utils/loss/instanseg_loss.py`): `__init__(..., embedding_mode="center", ...)`.
- `build_model_from_dict` and model dictionaries: read/write `embedding_mode`.
- Training/config plumbing: `instanseg_model.py` default config, CLI args where model dims are exposed, saved experiment metadata.
- TorchScript wrapper: `InstanSeg_Torchscript.__init__(..., embedding_mode="center", ...)`.

The UNet model itself only needs to emit the correct number of channels. It may optionally accept/store `embedding_mode` for metadata, but channel count is the functional contract.

Validation rules:

- Reject unknown values with clear `ValueError`.
- Default missing value to `"center"` for backward compatibility when loading old configs/checkpoints where safe.
- Persist `embedding_mode` in saved model dictionaries/checkpoints/config exports.

## 3. Required Behavioral Changes

### 3.1 Model Loader / Output Channels

In `model_loader.build_model_from_dict`:

- Read `embedding_mode = build_model_dictionary.get("embedding_mode", "center")`.
- Compute spatial channel count:

```py
spatial_channels = dim_coords * (2 if embedding_mode == "combined" else 1)
```

- Build output heads using `spatial_channels` in place of current center-only `dim_coords` for the spatial head.
- For multihead models, the spatial head becomes `[spatial_channels]` (or `[center, border]` split only if intentionally designed; simplest is keep one spatial head with `spatial_channels`).

Also update `read_model_args_from_csv` to parse/persist `embedding_mode`.

### 3.2 Loss Class (`InstanSeg`)

Add mode-aware properties:

- `self.embedding_mode`
- `self.dim_center = dim_coords if mode in {"center","combined"} else 0`
- `self.dim_border = dim_coords if mode in {"border","combined"} else 0`
- `self.dim_spatial = self.dim_center + self.dim_border`
- `self.dim_out = self.dim_spatial + self.n_sigma + self.dim_seeds`

Add helper(s) for slicing prediction:

```py
center_raw = pred[:, :dim_coords]                       # center/combined
border_raw = pred[:, dim_coords:2*dim_coords]           # combined
border_raw = pred[:, :dim_coords]                       # border
sigma = pred[:, dim_spatial:dim_spatial+n_sigma]
seed = pred[:, -dim_seeds:]
```

Do not hardcode `0:self.dim_coords` or `-self.dim_seeds` except through mode-aware helpers.

#### Center mode

No behavioral change.

#### Border mode

- Border embedding replaces center embedding everywhere the current spatial embedding is used.
- `fields`/`spatial_emb` are computed from border channels.
- Seed detection remains seed-map based as today.
- Centroid sampling still uses seed peaks; `c = emb[:, seed_y, seed_x]` samples border embedding at seed location.
- Feature engineering receives border embedding and sigma, same subtraction logic.
- Loss supervision for the spatial embedding uses nearest-border targets.

#### Combined mode

- Compute both center and border embeddings.
- Seed detection remains unchanged.
- Instance proposals/crops should use center embedding as the primary grouping field for backward-compatible behavior, unless a deliberate alternative is documented and tested. Recommended default: center embedding drives crop selection/grouping; border embedding is included as auxiliary per-pixel feature.
- Feature engineering should concatenate relative center and/or border features according to mode. Recommended combined feature:

```text
[center_emb - center_at_seed, border_emb - border_at_seed, sigma]
```

This preserves center logic and gives the pixel classifier border context.

Update `ProbabilityNet` input dimension calculation accordingly.

### 3.3 Feature Engineering

Current feature engineering assumes the first `E` channels are the spatial embedding and subtracts `c_shaped` over those `E` channels. For combined mode, `E` becomes `2*dim_coords` if both embeddings are passed as spatial features.

Required changes:

- Make feature width / embedding width mode-aware.
- Ensure `feature_engineering_generator` and any hardcoded width logic accept the new layout.
- Update `initialize_pixel_classifier` to compute correct `MLP_input_dim`.
- Keep old `feature_engineering_function` values working for center mode; add explicit support or compatible handling for border/combined.

### 3.4 Postprocessing / TorchScript

Both `InstanSeg.postprocessing` and `InstanSeg_Torchscript.forward` must:

- Parse `embedding_mode`.
- Slice center/border/sigma/seed by mode.
- Apply identical coordinate transform to any spatial embedding channel group.
- Use the same seed logic as current implementation.
- Use mode-appropriate fields for crop/proposal generation.
- Preserve existing defaults and output shape `1,H,W` (or batch-equivalent internal handling in TorchScript path).

TorchScript constraints: avoid Python-only dynamic constructs that break tracing/scripting; keep tensor operations explicit and shape-stable.

### 3.5 Training / Config / Metadata

Add `embedding_mode` to:

- `instanseg_model.py` default config and `_override_config_kwargs` mapping.
- `build_model_from_dict` / `read_model_args_from_csv` / `load_model_weights` metadata flow as needed.
- `scripts/train.py` CLI arguments and saved config.
- Any model metadata export (`create_bioimageio_model.py`, TorchScript export metadata) so inference can reconstruct the correct channel meaning.

Backward compatibility:

- Missing `embedding_mode` => `"center"`.
- Existing checkpoints/configs with 5-channel default layout must load unchanged.

### 3.6 Tests

Add/adjust tests to cover:

- Model output channel counts for all three modes.
- `model_loader.build_model_from_dict` accepts `embedding_mode` and builds correct `out_channels`.
- `InstanSeg` loss forward accepts mode-specific predictions and computes finite loss.
- Postprocessing handles mode-specific channel layouts.
- Combined mode uses both embedding sets; center mode remains unchanged.
- TorchScript wrapper, if tested in repo, can be constructed/traced for supported modes.

## 4. Implementation Notes / Risks

- **Semantic compatibility:** `border` mode changes what spatial embeddings mean. Seed detection is still driven by seed map, not by border vectors.
- **Center sampling:** in current code, centers are seed peak locations and also embedding values at those locations. For border-only mode, sampling border embedding at seed location is mechanically possible but semantically different; document and test this behavior.
- **Performance:** nearest-border target generation can be expensive if done naively per image; prefer vectorized EDT/nearest-index implementations and cache where possible.
- **Channel ordering:** this is a wire format. Any change must be synchronized across model, loss, postprocess, TorchScript, export, and tests.
- **Old checkpoints:** default-to-center loading is required to avoid breaking existing models.

## 5. Acceptance Criteria

Implementation is complete when:

1. `embedding_mode` is accepted in model/config initialization with values `center`, `border`, `combined`.
2. Default behavior (`center`) is unchanged for existing configs/models.
3. Border mode produces border-directed spatial embeddings with the same representation/transform logic as center embeddings.
4. Combined mode produces both embeddings and uses them as specified inputs to downstream inference/loss stages.
5. All channel slicing, feature engineering, probability classifier input sizing, postprocessing, and TorchScript paths are mode-aware.
6. Tests cover all three modes and old center-only behavior.
