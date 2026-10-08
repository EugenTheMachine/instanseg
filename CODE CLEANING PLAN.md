# Code Cleaning Plan for InstanSeg

This document outlines a step-by-step plan to clean and refactor the **InstanSeg** codebase. The goal is to eliminate redundant/dead code, remove leftover debugging statements, standardize logging, resolve architectural antipatterns, and clean up repository clutter.

---

## 1. Executive Summary

During the codebase audit, several areas needing cleanup were identified:
- **Debugging statements & breakpoints** (`import pdb`, `pdb.set_trace()`, `breakpoint()`) left in production files.
- **Raw `print()` statements and swallowed exceptions** (`print(e); pass`) scattered across modules instead of using standard logging.
- **Hardcoded secrets/tokens** (`kaggle_token.txt` in root).
- **Dead & legacy functions** (e.g., `torch_peak_local_max_LEGACY()`, `find_connected_components_legacy()`, unused feature engineering variants, and unintegrated experimental methods).
- **Orphaned folders and test artifacts** (`refactoring_tests/` containing only empty `__pycache__`, and temporary test runs in `runs/`).
- **Code duplication** (e.g., `has_pixel_classifier_model()` defined in multiple modules).
- **Test suite antipatterns** (manipulating `sys.path` dynamically inside test functions).
- **Documentation clutter** (20 fragmented `.md` changelogs and task files in the project root).

---

## 2. Phase 1: Removal of Debug Code, Stale Prints & Security Risks

### 1.1 Remove Debuggers and Breakpoints
- [ ] **[instanseg_loss.py](file:///d:/Документы/HW/instanseg/instanseg/utils/loss/instanseg_loss.py)**:
  - Remove `import pdb` (line 3) and `pdb.set_trace()` (line 1144).
- [ ] **[pytorch_utils.py](file:///d:/Документы/HW/instanseg/instanseg/utils/pytorch_utils.py)**:
  - Remove `import pdb` (line 308) and commented out `# pdb.set_trace()` (line 505).
- [ ] **[data_loader.py](file:///d:/Документы/HW/instanseg/instanseg/utils/data_loader.py)**:
  - Remove commented-out `# breakpoint()` statements (lines 127, 182, 192).

### 1.2 Replace Raw `print()` Calls & Exception Swallowing with Logging
- [ ] **[inference_class.py](file:///d:/Документы/HW/instanseg/instanseg/inference_class.py)**:
  - Replace `except Exception as e: print(e)` exception swallowing (lines 108, 127, 138, 156, 202, 213, 226) with proper exception propagation or structured `logger.error()` / `warnings.warn()`.
  - Convert raw stdout `print()` calls for status output (lines 228, 401, 417, 425, 807) to use `instanseg.utils.logger.get_logger()`.
  - Remove commented-out debug print at line 731 (`# print(new_tile.shape, shape[-2:])`).
- [ ] **[scripts/train.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/train.py)**, **[scripts/test.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/test.py)**, and **[scripts/inference.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/inference.py)**:
  - Replace raw `print()` statements in main loops with structured logger calls.
- [ ] **[utils.py](file:///d:/Документы/HW/instanseg/instanseg/utils/utils.py)**:
  - Replace raw `print()` calls inside helper functions (e.g. lines 179, 182, 192, 318, 372, 441, 461) with logger or warning calls.

### 1.3 Remove Security Risk Files
- [ ] **[kaggle_token.txt](file:///d:/Документы/HW/instanseg/kaggle_token.txt)**:
  - Remove `kaggle_token.txt` from repository root. Ensure sensitive tokens are passed via environment variables (`KAGGLE_API_TOKEN`) and kept out of Git.

---

## 3. Phase 2: Elimination of Dead Code, Obsolete Functions & Orphaned Artifacts

### 2.1 Clean Up Orphaned Directories and Local Run Artifacts
- [ ] **Orphaned `refactoring_tests/`**:
  - Remove `refactoring_tests/` folder (contains no source `.py` files, only obsolete `__pycache__`).
- [ ] **Transient Run Directories**:
  - Remove test run directories in `runs/` (`test_reproducibility_run1`, `test_reproducibility_run2`, `test_resume_single_run`, `test_resume_split_run`, `test_train_eval_wrapper_exp`) and ensure `runs/` is listed in `.gitignore`.

### 2.2 Purge Obsolete and Legacy Functions
- [ ] **[instanseg_loss.py](file:///d:/Документы/HW/instanseg/instanseg/utils/loss/instanseg_loss.py)**:
  - Delete legacy peak detection `torch_peak_local_max_LEGACY()` (lines 114-168).
  - Delete legacy connected components solver `find_connected_components_legacy()` (lines 240-263).
  - Delete unused/prototype feature engineering variants (`feature_engineering_slow`, `feature_engineering_2`, `feature_engineering_3`, `feature_engineering_10`, `guide_function`, `MyBlock`, `ConvProbabilityNet`).
  - Delete dead class `IdentityTransform` (lines 1164-1169).
  - Remove commented-out code blocks (e.g. lines 299-304, 946-953, 1149-1153, 1160, 1192-1193, 1354).
- [ ] **[inference_class.py](file:///d:/Документы/HW/instanseg/instanseg/inference_class.py)**:
  - Remove unfinished experimental method `_cluster_instances_by_mean_channel_intensity()` (lines 844-916).
  - Clean out commented-out reader branches (lines 251-260).
- [ ] **[AI_utils.py](file:///d:/Документы/HW/instanseg/instanseg/utils/AI_utils.py)**:
  - Remove unused interactive GUI function `plot_loss()` (lines 219-234) and unused gradient checkers (`check_max_grad()`, `check_min_grad()`, `check_mean_grad()`).
  - Delete unused global variables `global_step` and `global_step_test`.
- [ ] **[metrics.py](file:///d:/Документы/HW/instanseg/instanseg/utils/metrics.py)**:
  - Move test helper function `test_matching_dataset_torch()` and `_check_is_equal()` into a dedicated test module under `tests/` instead of keeping them in production code.
  - Remove unused branch `use_stardist = False` (lines 167-172) and commented-out panoptic quality lines.
- [ ] **[scripts/test.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/test.py)**:
  - Delete commented-out hardcoded local user path (lines 327-342).

---

## 4. Phase 3: Deduplication & Code Structure Refactoring

### 3.1 Consolidate Duplicate Utilities
- [ ] **Unify `has_pixel_classifier_model()`**:
  - Remove duplicate implementation from [instanseg_loss.py](file:///d:/Документы/HW/instanseg/instanseg/utils/loss/instanseg_loss.py#L319) and import the unified version from [model_loader.py](file:///d:/Документы/HW/instanseg/instanseg/utils/model_loader.py#L190).
- [ ] **Unify `_move_channel_axis()`**:
  - Merge redundant implementations between [utils.py](file:///d:/Документы/HW/instanseg/instanseg/utils/utils.py#L99) and [visualization.py](file:///d:/Документы/HW/instanseg/instanseg/utils/visualization.py#L267).

### 3.2 Clean Up Imports and Function Scoping
- [ ] **[instanseg_model.py](file:///d:/Документы/HW/instanseg/instanseg/instanseg_model.py)**:
  - Remove unused local import `from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model` in `train()` and `eval()`.
  - Deduplicate identical `InstanSegLoss` initialization logic across `_init_model_from_checkpoint()`, `train()`, and `eval()`.
- [ ] **[AI_utils.py](file:///d:/Документы/HW/instanseg/instanseg/utils/AI_utils.py)**:
  - Remove duplicate `import torch` at lines 1 and 2.
- [ ] **Top-level Imports**:
  - Move inline/nested imports inside functions (e.g. `import os`, `from skimage import io` inside `save_output()`, `import zarr` inside `eval_whole_slide_image()`) to top-level module scope where appropriate.
- [ ] **Script Global Variable Refactoring**:
  - In [scripts/train.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/train.py) and [scripts/test.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/test.py), remove use of `global device, method, iou_threshold, args, optimizer, scheduler`. Pass explicit state objects instead.
  - Fix variable naming in [scripts/inference.py](file:///d:/Документы/HW/instanseg/instanseg/scripts/inference.py#L71) where `parser = parser.parse_args()` overwrites the module-level ArgumentParser instance.

### 3.3 Fix Test Suite Antipatterns
- [ ] **[tests/test_instanseg.py](file:///d:/Документы/HW/instanseg/tests/test_instanseg.py)**:
  - Remove `sys.path = sys.path[1:]` from test functions. Use Pytest configuration (`pytest.ini` / `conftest.py`) or standard package imports.

---

## 5. Phase 4: Documentation & Project Root Reorganization

### 5.1 Reorganize Project Root Markdown Files
Currently, 20 separate `.md` files clutter the project root:
`CRITICAL TASK.md`, `EMBEDDING-CHANGELOG.md`, `EMBEDDINGS.md`, `ENGINEERING MODIFICATIONS.md`, `INFO.md`, `KAGGLE-CHANGELOG.md`, `LORA-CHANGELOG.md`, `RE-IMPLEMENTATION.md`, `README technical.md`, `REFACTORING.MD`, `TASK.md`, `TEST_DESCRIPTIONS.md`, `THRESHOLD_FIX_COMPLETE.md`, `TRAIN MODIFICATIONS.md`, `TRAINING FORMALIZATION.md`, `UNET CHANGELOG.md`, `UNET OVERVIEW.md`, `CONFIG DESCRIPTION.md`.

- [ ] Consolidate task-specific notes and historic changelogs into a `docs/` subdirectory or merge them into `CHANGELOG.md` and `README.md`.
- [ ] Maintain only essential project-level markdown files (`README.md`, `CHANGELOG.md`, `LICENSE`) in the repository root.

### 5.2 Standardize Jupyter Notebook Locations
- [ ] Move `training_kaggle.ipynb` from root into the `notebooks/` directory alongside all other Jupyter notebooks.

---

## 6. Execution & Verification Roadmap

| Step | Action | Verification |
| :--- | :--- | :--- |
| **Step 1** | Remove debugging statements (`pdb`, `breakpoint()`), swallowed exceptions, and hardcoded `kaggle_token.txt`. | Run code search for `pdb`, `breakpoint()`, and `print(`. |
| **Step 2** | Purge dead functions (`torch_peak_local_max_LEGACY`, legacy connected components, unused feature engineering helpers, experimental methods). | Run existing pytest suite to ensure no active code depended on them. |
| **Step 3** | Delete orphaned folders (`refactoring_tests/`) and temporary run outputs in `runs/`. | Verify clean git status and directory listing. |
| **Step 4** | Deduplicate functions (`has_pixel_classifier_model`, `_move_channel_axis`), fix script global state, clean up `sys.path` in tests. | Run full `pytest` suite. |
| **Step 5** | Reorganize project root markdown files into `docs/` and move `training_kaggle.ipynb` into `notebooks/`. | Inspect project root directory structure. |

---
