
<p align="center">
  <img src="https://github.com/ThibautGoldsborough/instanseg_thibaut/blob/main/assets/instanseg_logo.png?raw=True" alt="Instanseg Logo" width="25%">
</p>



## Overview

InstanSeg is a pytorch-based **cell segmentation** pipeline for fluorescence and brightfield microscopy images. This README provides instructions for setting up the environment, installing dependencies, and using the provided tools and models.

> **v2.0 — Breaking changes:** InstanSeg now exclusively performs **cell segmentation**. Nucleus-only and combined nucleus+cell segmentation targets are deprecated (see [Migration Guide](#migration-guide-v1x--v20) below). Augmentation is handled via [albumentations](https://albumentations.ai); the legacy `Augmentations` class is deprecated.

## Why should I use InstanSeg?

1. InstanSeg is freely available and open source.
2. It's faster than other cell segmentation methods… sometimes much faster.
3. It's highly accurate at whole-cell segmentation across fluorescence and brightfield imaging modalities.
4. InstanSeg can be entirely compiled in TorchScript - including postprocessing! This means it's not only easy to use in Python but also works with LibTorch alone. This allows you to run InstanSeg directly in QuPath!
5. InstanSeg uses a standard [albumentations](https://albumentations.ai) augmentation pipeline — easy to customise and extend.
6. We plan to release more InstanSeg models trained on public datasets. If there's a cell segmentation dataset under a permissive open license (e.g. CC0 or CC-BY) that we missed, let us know, and we may be able to increase our InstanSeg model zoo.
 

## InstanSeg has its own QuPath extension!

InstanSeg is introduced in the [QuPath pre-release v0.6.0-rc2](https://github.com/qupath/qupath/releases/tag/v0.6.0-rc2), so you can start using InstanSeg immediately. You can find the QuPath extension source code [in its GitHub repository](https://github.com/qupath/qupath-extension-instanseg).

## How to cite InstanSeg:

If you use InstanSeg for nucleus segmentation of brightfield histology images, please cite:

> Goldsborough, T. et al. (2024) 'InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation'. _arXiv_. Available at: https://doi.org/10.48550/arXiv.2408.15954.

If you use InstanSeg for nucleus and/or cell segmentation in fluorescence images, please cite:

> Goldsborough, T. et al. (2024) 'A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg'. _bioRxiv_, p. 2024.09.04.611150. Available at: https://doi.org/10.1101/2024.09.04.611150.



<p align="center">
  <img src="https://github.com/ThibautGoldsborough/instanseg_thibaut/blob/main/assets/instanseg_main_figure.png?raw=True" alt="Instanseg Main Figure" width="50%">
</p>

## Table of Contents

- [Overview](#overview)
- [Why should I use InstanSeg?](#why-should-i-use-instanseg)
- [InstanSeg has its own QuPath extension!](#instanseg-has-its-own-qupath-extension)
- [How to cite InstanSeg:](#how-to-cite-instanseg)
- [Table of Contents](#table-of-contents)
- [Migration Guide v1.x → v2.0](#migration-guide-v1x--v20)
- [Project Structure](#project-structure)
- [Installing using pip](#installing-using-pip)
  - [GPU Version (CUDA) for Windows and Linux](#gpu-version-cuda-for-windows-and-linux)
  - [Setup Repository](#setup-repository)
- [Usage](#usage)
  - [Download Datasets](#download-datasets)
  - [Training Models](#training-models)
  - [Testing Models](#testing-models)
  - [Using InstanSeg for inference](#using-instanseg-for-inference)
  - [Model versioning](#model-versioning)

---

## Migration Guide v1.x → v2.0

### Inference

No changes required for standard cell segmentation. If your code used nucleus or joint outputs, update as shown:

```python
# v1.x — requested nucleus channel (now deprecated)
labeled_output = instanseg.eval_small_image(image, pixel_size, target="nuclei")
# v1.x — requested both channels (now returns cells only)
labeled_output = instanseg.eval_small_image(image, pixel_size, target="all_outputs")

# v2.0 — cells only (default)
labeled_output = instanseg.eval_small_image(image, pixel_size)
labeled_output = instanseg.eval_small_image(image, pixel_size, target="cells")  # explicit
```

### Augmentation

The `Augmentations` class (`instanseg/utils/augmentations.py`) is deprecated. Use the new preprocessing module:

```python
# v1.x
from instanseg.utils.augmentations import Augmentations
aug = Augmentations()
tensor, _ = aug.to_tensor(image, normalize=True)
tensor, _ = aug.torch_rescale(tensor, labels, current_pixel_size=0.5, requested_pixel_size=0.25)

# v2.0
from instanseg.utils.preprocessing import to_tensor, rescale_to_pixel_size, build_augmentation_pipeline
tensor = to_tensor(image, normalize=True)
tensor = rescale_to_pixel_size(tensor, current_pixel_size=0.5, requested_pixel_size=0.25)

# Training augmentation pipeline (albumentations-based):
pipeline = build_augmentation_pipeline()
```

### Biological utilities

`instanseg/utils/biological_utils.py` (N/C ratio, nucleus-cell IoU, marker subcellular location, UMAP clustering) is deprecated and emits `DeprecationWarning` on import. There is no replacement in the active pipeline — pin to v1.x if you need these utilities.

### Training

`cells_and_nuclei=True` on `InstanSeg` (loss class), `InstanSeg_Torchscript`, and `Segmentation_Dataset` now emits `DeprecationWarning`. Pre-trained joint models continue to load and run correctly; only the cells channel is used at inference time.

---

## Project Structure

```
instanseg/
├── instanseg/
│   ├── inference_class.py        # Main public API — InstanSeg class
│   ├── model.py                  # InstanSegModel training wrapper
│   └── utils/
│       ├── preprocessing.py      # Image preprocessing (normalize, resize, augment)
│       ├── postprocessing.py     # Post-processing (peak finding, NMS, label merging)
│       ├── AI_utils.py           # Training loop, Segmentation_Dataset
│       ├── utils.py              # Visualization, export, device helpers
│       ├── pytorch_utils.py      # Low-level torch helpers
│       ├── tiling.py             # Sliding-window / WSI tiling inference
│       ├── metrics.py            # AP / F1 metrics
│       └── loss/
│           ├── instanseg_loss.py    # InstanSeg loss function
│           └── lovasz_losses.py     # Lovász surrogate losses
├── tests/                        # Integration tests (require model weights)
├── refactoring_tests/            # Unit tests for refactored modules
│   ├── test_preprocessing.py     # 27 tests — preprocessing.py
│   ├── test_postprocessing.py    # 22 tests — postprocessing.py
│   └── test_deprecations.py      # 18 tests — all deprecation warnings
└── instanseg/scripts/
    ├── train.py
    ├── test.py
    └── inference.py
```

---

## Installing using pip

For a minimal installation:
```bash
pip install instanseg-torch
```

If you want all the requirements used for training:

```bash
pip install instanseg-torch[full]
```

You can get started immediately by calling the InstanSeg class:

```python
from instanseg import InstanSeg
instanseg_brightfield = InstanSeg("brightfield_nuclei", image_reader="tiffslide", verbosity=1)

labeled_output = instanseg_brightfield.eval(
    image="../instanseg/examples/HE_example.tif",
    save_output=True,
    save_overlay=True,
)
```

Alternatively, if you want more control over the intermediate steps:

```python
image_array, pixel_size = instanseg_brightfield.read_image("../instanseg/examples/HE_example.tif")

labeled_output, image_tensor = instanseg_brightfield.eval_small_image(image_array, pixel_size)

display = instanseg_brightfield.display(image_tensor, labeled_output)

from instanseg.utils.utils import show_images
show_images(image_tensor, display, colorbar=False, titles=["Normalized Image", "Image with segmentation"])
```

### GPU Version (CUDA) for Windows and Linux

If you intend to use GPU acceleration and CUDA, follow these additional steps:

4. Uninstall existing PyTorch and reinstall with CUDA support:
    ```bash
    micromamba remove pytorch torchvision monai
    micromamba install pytorch==2.1.1 torchvision==0.16.1 monai=1.3.0 pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia
    pip install cupy-cuda12x
    ```

5. Check if CUDA is available:
    ```bash
    python -c "import torch; print('CUDA is available') if torch.cuda.is_available() else print('CUDA is not available')"
    ```

The repository may work with older versions of CUDA. Replace "12.1" and "12" with the required version.

### Setup Repository

3. Build repository:
    ```bash
    pip install -e .
    ```

## Usage

### Download Datasets

To download public datasets and example images, follow the instructions under **instanseg/notebooks/load_datasets.ipynb**

To train InstanSeg on your own dataset, extend the **instanseg/notebooks/load_datasets.ipynb** with one of the templates provided.

### Training Models

To train models using InstanSeg, use the **train.py** script under the scripts folder.

Training accepts an `embedding_mode` setting for the coordinate embedding and instance separation strategy. Supported values are `center-seed`, `border-seed`, `center-cluster`, `border-cluster`, `combined-center`, and `combined-cluster`. Set it in YAML as `training.embedding_mode` or pass it through the Python training wrapper as an override.

For example, to train InstanSeg on the TNBC_2018 dataset over 250 epochs at a pixel resolution of 0.25 microns/pixel:
```bash
cd instanseg/scripts
python train.py -data segmentation_dataset.pth -source "[TNBC_2018]" --num_epochs 250 --experiment_str my_first_instanseg --requested_pixel_size 0.25
```

To train a channel-invariant InstanSeg on the CPDMI_2023 dataset (cell segmentation):
```bash
cd instanseg/scripts
python train.py -data segmentation_dataset.pth -source "[CPDMI_2023]" --num_epochs 250 --experiment_str my_first_instanseg --channel_invariant True --requested_pixel_size 0.5
```

> **Deprecated:** The `-target NC` (nuclei+cells) training flag is deprecated. Only cell segmentation (`-target C`) is supported going forward.

Each epoch should take approximately 1 to 3 minutes to complete (with mps or cuda support).

For more options and configurations, refer to the parser arguments in the train.py file.

### Testing Models

To test trained models and obtain F1 metrics, use the following command:
```bash
python test.py --model_folder my_first_instanseg -test_set Validation --optimize_hyperparameters True
python test.py --model_folder my_first_instanseg -test_set Test --params best_params
```

### Using InstanSeg for inference

```bash
python inference.py --model_folder my_first_instanseg --image_path ../examples
```
Replace "../examples" with the path to your images. If InstanSeg cannot read the image pixel size from the image metadata, the user is required to provide a `--pixel_size` parameter. InstanSeg provides (limited) support for whole slide images (WSIs). For more options and configurations, refer to the parser arguments in the inference.py file.

### Model versioning

Links to different model versions are stored in `instanseg/models/model-index.json`. When releasing new models, add entries to this JSON file, optionally removing any previous versions that shouldn't be available in future versions.

An example entry looks like this:

```json
{
  "name": "[MODEL_NAME]",
  "url": "https://github.com/instanseg/instanseg/releases/download/[RELEASE_NAME]/[MODEL_NAME].zip",
  "version": "0.1.0",
  "license": "Apache-2.0"
}
```
