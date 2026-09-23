# Configuration Description

This file describes the project configuration parameters used by `config.yaml` and the training wrapper.

## meta

- `data_dir`
  - Description: Root path to the dataset directory.
  - Example: `./data/livecell`

- `experiment_name`
  - Description: Name of the experiment. Used to create `runs/<experiment_name>`.
  - Example: `test_experiment`

## model

- `model_name`
  - Description: Fully qualified model backend name used by the training pipeline.
  - Examples:
    - `maskrcnn-resnet50_fpn`
    - `InstanSeg_UNet`
    - `efficientunetb0`
    - `efficientunetb1`
    - `efficientunetb2`
    - `efficientunetb3`
    - `efficientunetv2s`
    - `mobileunetv2`
    - `mobileunetv3s`
    - `mobileunetv3l`
    - `regnetunety400mf`
    - `regnetunety800mf`
    - `resnetunet18`
  - Notes: This value is used to construct the model. For UNet-style models, the project will use the InstanSeg UNet builder.

- `dropout`
  - Description: Dropout probability for the model backbone or UNet layers.
  - Example: `0.0`

- `freeze_backbone_epochs`
  - Description: Number of initial epochs during which the backbone is frozen.
  - Example: `1`

- `num_classes`
  - Description: Number of segmentation classes predicted by the model.
  - Example: `2`

- `embedding_mode`
  - Description: Embedding/instance-separation strategy used by InstanSeg UNet models.
  - Valid values:
    - `center-seed`
    - `border-seed`
    - `center-cluster`
    - `border-cluster`
    - `combined-center`
    - `combined-cluster`
  - Example: `center-seed`

## optimizer

- `backbone_lr_mult`
  - Description: Learning rate multiplier for backbone parameters when fine-tuning.
  - Example: `0.1`

- `learning_rate`
  - Description: Base learning rate for optimizer.
  - Example: `0.001`

- `momentum`
  - Description: Momentum value for SGD optimizer.
  - Example: `0.9`

- `weight_decay`
  - Description: Weight decay (L2 regularization) value.
  - Example: `0.0001`

## preprocessing

- `imgsz`
  - Description: Resize image input size used for training and inference.
  - Example: `128`

- `seed`
  - Description: Seed value for deterministic data loading, shuffling, and training.
  - Example: `42`

- `train_data_ratio`
  - Description: Fraction of training data to use.
  - Valid range: `0.0 < train_data_ratio <= 1.0`
  - Example: `1.0`

## training

- `batch_size`
  - Description: Batch size used by training and evaluation dataloaders.
  - Example: `2`

- `epochs`
  - Description: Total number of training epochs.
  - Example: `2`

- `patience`
  - Description: Number of non-improving validation epochs before early stopping.
  - Example: `2`

- `resume`
  - Description: Whether to resume training from the latest checkpoint.
  - Valid values: `true`, `false`
  - Example: `false`

- `test_ratio`
  - Description: Fraction of full dataset reserved for test split when using a single dataset root.
  - Example: `0.2`

- `val_ratio`
  - Description: Fraction of the training dataset reserved for validation when using a single dataset root.
  - Example: `0.2`

- `val_interval`
  - Description: Number of epochs between validation evaluations.
  - Example: `1`

- `warmup_epochs`
  - Description: Number of epochs used for learning rate warmup.
  - Example: `1`

- `save_kgl_ckp`
  - Description: Whether to save and optionally upload Kaggle checkpoint bundles during training.
  - Valid values: `true`, `false`
  - Example: `false`

- `kgl_best_ckp_path`
  - Description: Kaggle Models handle or local path where the best checkpoint bundle should be stored.
  - Example: `username/instanseg_best`

- `kgl_last_ckp_path`
  - Description: Kaggle Models handle or local path where the latest checkpoint bundle should be stored.
  - Example: `username/instanseg_last`

- `kgl_ckp_freq`
  - Description: Frequency in epochs for saving Kaggle checkpoint bundles.
  - Example: `1`

- `kgl_creds_path`
  - Description: Path to the Kaggle credentials file used for checkpoint uploads.
  - Example: `kaggle_token.txt`

## additional runtime config

The project also accepts these internal config values if provided in `config.yaml` or via overrides:

- `num_workers`
  - Description: Number of dataloader workers.
  - Example: `4`

- `data_dir`
  - Description: Alternative root dataset directory if placed under a different path.
  - Example: `./data`
