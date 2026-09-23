# TASK: Migrate InstanSeg from seed-based to border-based predictions

## Task goal

Move the InstanSeg instance segmentation pipeline from a seed/centroid-based embedding representation to a border-distance and border-vector based representation. The goal is to restructure both model supervision and postprocessing so that predicted outputs encode the nearest object boundary distance and the directional vector to the closest border, rather than centroid offsets and a separate seed/EDT map.

## Brief theoretical description

Current InstanSeg works by predicting three signal types:
- predicted spatial embeddings (`fields`) that map each pixel toward an instance centroid,
- a pixel-wise uncertainty/scale embedding (`sigma`),
- a seed/EDT-like map whose local maxima identify object instance centers.

The crop-based classifier then uses these embeddings to decide whether pixels belong to a candidate object centered at a detected seed.

In the proposed border-based approach, the network should instead predict:
- a border distance map, where each pixel’s value is its L2 distance to the nearest pixel not belonging to the same instance,
- a nearest-border vector field, where each pixel points to the nearest border pixel (background or another instance).

From a theoretical standpoint, this changes the instance representation from center-oriented object embedding to boundary-oriented object encoding. Candidate instance seeds can still be found by maxima of the border-distance map (medial axis peaks), but the local mask classifier must learn membership from border-based features rather than centroid-relative offsets.

## Technical task breakdown

### Atomic task 1: Define new structural outputs and supervision

- Update model output semantics to include a border distance channel and a border vector channel.
- Modify or extend loss-target generation to build:
  - ground-truth border distance maps for each instance,
  - nearest-border displacement vectors for each pixel inside instances.
- Note: this may require implementing a new helper in `instanseg/utils/pytorch_utils.py` or a similar utility module, adjacent to `instance_wise_edt(...)`.

### Atomic task 2: Update model loss and training pipeline

- Modify `InstanSeg.__init__` in `instanseg/utils/loss/instanseg_loss.py` to support the new output channel layout.
- Add a border vector loss term in `InstanSeg.forward()` to supervise the new nearest-border vector predictions.
- Adapt `InstanSeg.update_seed_loss(...)` or add a new seed-loss variant so that `border_dist` is supervised as a true EDT/border-distance target instead of the current centroid-seed semantics.

### Atomic task 3: Rework inference/postprocessing semantics

- In `instanseg/utils/postprocessing.py`:
  - change `InstanSeg_Torchscript.forward()` to interpret the raw prediction channels as border vectors + border distances instead of `fields + sigma + seed_map`,
  - update the `mask_map` generation line so it uses the predicted border-distance map for candidate detection,
  - keep peak detection via `torch_peak_local_max(...)`, but ensure it runs on border-distance peaks.

### Atomic task 4: Replace feature engineering for border embeddings

- Add one or more new feature-engineering functions in `instanseg/utils/postprocessing.py`, for example:
  - `feature_engineering_border(...)` or `feature_engineering_border_and_dist(...)`.
- These functions should compute features from predicted border vectors/distances and the border-based candidate point, instead of subtracting centroid embeddings.
- Update `feature_engineering_generator(...)` to expose the new border-based feature options.
- Update all `compute_crops(...)` and `postprocessing()` calls to use the new border-based feature engineering.

### Atomic task 5: Adapt candidate crop construction

- In `compute_crops(...)` and `InstanSeg.postprocessing()`, verify that crop creation still makes sense for medial-axis seeds derived from border distances.
- Ensure that the seed/center coordinates used for cropping correspond to peaks of the border-distance map.
- If necessary, add a small utility to convert border-distance peaks into strong crop anchors.

### Atomic task 6: Maintain or simplify sigma logic

- Decide whether `sigma` is still needed. If the new model uses pure border embeddings, then:
  - set `n_sigma = 0`,
  - remove or repurpose `sigma` from feature engineering,
  - update `InstanSeg.__init__`, `InstanSeg_Torchscript.__init__`, and any `feature_engineering_*` functions accordingly.
- If `sigma` is retained as an auxiliary signal, explicitly document the new meaning and wire it into the new feature generation.

### Atomic task 7: Refactor merge/postprocessing conflict resolution if needed

- Review `merge_sparse_predictions(...)` and `convert(...)` in `instanseg/utils/postprocessing.py`.
- Confirm whether instance merging needs adjustment for border-based crop outputs.
- Ensure the object overlap and NMS-like logic remains valid when the underlying candidate masks are produced from border-based features.

### Atomic task 8: Add tests and validation coverage

- Add unit tests for new border target generation and loss behavior.
- Add tests that validate the new feature engineering output shapes and semantics.
- Add integration tests for the new postprocessing path if possible.
- If code is not yet implemented for a new component, document the missing helper in `TASK.md` so it is clear what must be created.

## Notes on missing code / implementation gaps

- A helper to compute nearest-border vectors from instance masks is not currently present in the codebase. It must be implemented, likely in `instanseg/utils/pytorch_utils.py`.
- A new border-based feature engineering function is also missing and must be added to `instanseg/utils/postprocessing.py`.
- The current `seed_map` interpretation in `InstanSeg_Torchscript.forward()` and `InstanSeg.postprocessing()` is centroid/EDT-based; it must be rewritten to operate on border-distance peaks.

## Summary of files to change

- `instanseg/utils/loss/instanseg_loss.py`
  - `InstanSeg.__init__`
  - `InstanSeg.forward()`
  - `InstanSeg.postprocessing()`
  - `update_seed_loss()` or new loss methods

- `instanseg/utils/postprocessing.py`
  - `InstanSeg_Torchscript.forward()`
  - `compute_crops()`
  - `feature_engineering(...)` and new border feature functions
  - `feature_engineering_generator(...)`
  - `torch_peak_local_max(...)` usage site(s)
  - optional merge helpers if needed

- `instanseg/utils/pytorch_utils.py`
  - new utility for nearest-border distance + vector target generation

- tests / refactoring_tests
  - add coverage for the new border-based prediction flow and new helper functions
