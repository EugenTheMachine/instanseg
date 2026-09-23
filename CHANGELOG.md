# Changelog

All notable changes to InstanSeg are documented in this file.

## [2.0.1] — 2026-06-11 — Deprecation Sweep

### Summary

Completed the deprecation coverage started in 2.0.0. All remaining modules and
classes that deal with nucleus segmentation, biological-marker metadata, or
legacy augmentation config now emit `DeprecationWarning` on import or
construction. A new test file (`refactoring_tests/test_deprecations.py`)
validates every warning path.

---

### Deprecated

#### `instanseg/utils/biological_utils.py` — whole module

- A module-level `DeprecationWarning` is now raised on every import.  
  Message: *"instanseg.utils.biological_utils is deprecated … InstanSeg now
  performs cell segmentation only."*
- All functions remain callable for backwards compatibility:
  `resolve_cell_and_nucleus_boundaries`, `get_intersection_over_union`,
  `nc_heatmap`, `get_nonnucleated_cell_ids`, `get_nucleated_cell_ids`,
  `get_multinucleated_cell_ids`, `keep_only_largest_nucleus_per_cell`,
  `get_features_by_location`, `get_nc_ratio`, `violin_plot_feature_location`,
  `show_umap_and_cluster`.

#### `instanseg/utils/augmentation_config.py` — whole module

- A module-level `DeprecationWarning` is now raised on every import.  
  Message: *"instanseg.utils.augmentation_config is deprecated … Use
  instanseg.utils.preprocessing.build_augmentation_pipeline … markers_info is
  no longer used by the active pipeline."*
- `get_augmentation_dict` and `markers_info` / `markers_info_gpt` remain
  accessible for backwards compatibility.

#### `InstanSeg` loss class — `cells_and_nuclei=True`

- `instanseg.utils.loss.instanseg_loss.InstanSeg(cells_and_nuclei=True)` now
  emits `DeprecationWarning` at construction time.  
  Message: *"cells_and_nuclei=True is deprecated in InstanSeg loss …
  preserved only for loading pre-trained joint models."*
- Passing `cells_and_nuclei=False` (the new default) is silent.

#### `InstanSeg_Torchscript` — `cells_and_nuclei=True`

- `instanseg.utils.postprocessing.InstanSeg_Torchscript(cells_and_nuclei=True)`
  now emits `DeprecationWarning` at construction time.  
  Message: *"cells_and_nuclei=True is deprecated in InstanSeg_Torchscript …
  the forward pass will return the cells channel only."*
- Passing `cells_and_nuclei=False` is silent.

---

### Added

- **`refactoring_tests/test_deprecations.py`** — 18 unit tests covering every
  deprecation warning path:
  - `biological_utils` import warning
  - `augmentation_config` import warning
  - `augmentations` import warning
  - `get_marker_location` call warning
  - `InstanSeg(cells_and_nuclei=True)` warning
  - `InstanSeg_Torchscript(cells_and_nuclei=True)` warning
  - `Segmentation_Dataset(cells_and_nuclei=True)` warning
  - `Segmentation_Dataset(metadata=...)` warning
  - `eval_small_image(target="nuclei")` warning
  - `eval_medium_image(target="nuclei")` warning

---

### Tests

```
refactoring_tests/test_deprecations.py   18 tests added
refactoring_tests/test_preprocessing.py  27 passed (unchanged)
refactoring_tests/test_postprocessing.py 22 passed (unchanged)
tests/test_exported_models.py             4 passed (unchanged)
tests/test_instanseg.py                   4 passed (unchanged)
tests/utils/test_tiling.py                1 passed (unchanged)
```

---

## [2.0.0] — 2026-06-11 — Cell-Segmentation Refactor

### Summary

Major refactor focused on simplifying the codebase to perform **cell segmentation only**. All nucleus/nucleolus segmentation paths, biological-marker metadata handling, and legacy hand-rolled augmentations have been deprecated.

---

### Added

- **`instanseg/utils/preprocessing.py`** — New centralised preprocessing module:
  - `normalize_percentile` — per-channel percentile normalisation (replaces `Augmentations.normalize`).
  - `to_tensor` — converts NumPy or Torch image to `(C, H, W)` float32 tensor (replaces `Augmentations.to_tensor`).
  - `resize_image` — bilinear/nearest resize with optional label tensor.
  - `rescale_to_pixel_size` — pixel-size-aware rescaling (replaces `Augmentations.torch_rescale`).
  - `build_augmentation_pipeline` — builds a standard **albumentations** pipeline (`RandomBrightnessContrast`, `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate`).
  - `apply_augmentation_pipeline` — helper to apply a pipeline to a numpy image and optional mask.

- **`refactoring_tests/test_preprocessing.py`** — 27 unit tests covering all functions in `preprocessing.py`.
- **`refactoring_tests/test_postprocessing.py`** — 22 unit tests covering core postprocessing helpers (`filter_small_blobs`, `find_all_local_maxima`, `torch_peak_local_max`, `centre_crop`, `convert`, `find_connected_components`, `generate_coordinate_map`).

---

### Deprecated

#### Segmentation targets other than cells

- `cells_and_nuclei=True` in `InstanSeg_Torchscript`, `Segmentation_Dataset`, and `Augmentations` now emits `DeprecationWarning`. The flag is preserved for backwards-compatible **model weight loading** only; inference always returns the cell channel.
- `eval_small_image(target="nuclei")` and `eval_medium_image(target="nuclei")` now emit `DeprecationWarning`. Requesting `"all_outputs"` now returns **cells only** for joint models.
- `resolve_cell_and_nucleus_boundaries` call in `InstanSeg_Torchscript.postprocessing` is no longer triggered for new usage.

#### Biological marker / metadata support

- `get_marker_location()` in `augmentations.py` — emits `DeprecationWarning`.
- `extract_nucleus_and_cytoplasm_channels()` — emits `DeprecationWarning`; relies on deprecated `subcellular_location` metadata.
- `Segmentation_Dataset(metadata=...)` — `metadata` parameter now emits `DeprecationWarning` and is ignored.
- `Augmentations.__call__` metadata-based channel routing (`nuclei_channels`, `channel_names`, `subcellular_location`) — effectively disabled; `cells_and_nuclei` is hardcoded to `False`.

#### Legacy augmentation class

- **`instanseg/utils/augmentations.py`** — The entire module and `Augmentations` class are deprecated. A module-level `DeprecationWarning` is raised on import.
  - `Augmentations.to_tensor` → use `preprocessing.to_tensor`
  - `Augmentations.normalize` → use `preprocessing.normalize_percentile`
  - `Augmentations.torch_rescale` → use `preprocessing.rescale_to_pixel_size`
  - `Augmentations.extract_hematoxylin_stain` → deprecated (no replacement; stain separation is not part of the cell-segmentation pipeline)
  - `Augmentations.normalize_HE_stains` → use albumentations color-jitter transforms

---

### Changed

- **`instanseg/utils/utils.py`**
  - `display_as_grid` now uses `preprocessing.to_tensor` instead of `Augmentations`.
  - `display_colourized` now uses `preprocessing.to_tensor` / `preprocessing.normalize_percentile`.
  - `export_to_torchscript` now uses `preprocessing.to_tensor`, `normalize_percentile`, and `rescale_to_pixel_size`.
  - `cells_and_nuclei = model_dict.get('cells_and_nuclei', False)` — safe dict access with default.

- **`instanseg/utils/AI_utils.py` — `Segmentation_Dataset`**
  - Removed `self.Augmenter` (deprecated `Augmentations` instance).
  - Removed `self.metadata` dict.
  - `self.transform` is now built via `preprocessing.build_augmentation_pipeline`.
  - `__getitem__` no longer looks up `self.metadata`.

- **`instanseg/inference_class.py`**
  - `eval_small_image`: for `cells_and_nuclei` models, `target="all_outputs"` and `target="cells"` now both route to `target_segmentation = torch.tensor([0, 1])` (cells only).
  - `eval_medium_image`: same routing change; `output_dimension` is always 1.

- **`instanseg/utils/loss/instanseg_loss.py`**
  - Imported postprocessing helpers directly from `instanseg.utils.postprocessing` (no local duplicates).

---

### Fixed

- `display_colourized` no longer triggers a `DeprecationWarning` from `Augmentations` for its own import.

---

### Tests

All original tests continue to pass:

```
tests/test_instanseg.py::test_inference_brightfield   PASSED
tests/test_instanseg.py::test_inference_fluoro        PASSED
tests/test_instanseg.py::test_image_readers_brightfield PASSED
tests/test_instanseg.py::test_image_readers_fluoro    PASSED
```

New refactoring tests:

```
refactoring_tests/test_preprocessing.py   27 passed
refactoring_tests/test_postprocessing.py  22 passed
```
