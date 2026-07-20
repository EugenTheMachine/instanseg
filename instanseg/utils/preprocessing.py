"""
preprocessing.py
----------------
Preprocessing utilities for InstanSeg cell segmentation.

This module centralises image preprocessing operations used before model
inference or during data loading.  The functions operate on both NumPy arrays
and PyTorch tensors.

Functions consolidated from:
  - instanseg/utils/augmentations.py  (normalize, to_tensor, torch_rescale)
  - instanseg/utils/utils.py          (percentile_normalize, _move_channel_axis)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from instanseg.utils.utils import _move_channel_axis, percentile_normalize


# ---------------------------------------------------------------------------
# Core image normalisation
# ---------------------------------------------------------------------------

def normalize_percentile(
    image: Union[np.ndarray, torch.Tensor],
    percentile: float = 0.1,
    subsampling_factor: int = 1,
    epsilon: float = 1e-3,
) -> Union[np.ndarray, torch.Tensor]:
    """Per-channel percentile normalisation to the range [0, 1].

    Parameters
    ----------
    image:             2-D or 3-D array/tensor (C, H, W) or (H, W) or (H, W, C).
    percentile:        lower percentile (symmetric clip: *percentile* … 100-*percentile*).
    subsampling_factor: spatial subsampling used when computing percentile statistics.
    epsilon:           minimum denominator to avoid divide-by-zero.

    Returns
    -------
    Normalised image of the same type and shape as the input.
    """
    return percentile_normalize(image, percentile=percentile,
                                subsampling_factor=subsampling_factor, epsilon=epsilon)


def to_tensor(
    image: Union[np.ndarray, torch.Tensor],
    normalize: bool = False,
    percentile: float = 0.1,
) -> torch.Tensor:
    """Convert an image (numpy or torch) to a (C, H, W) float32 tensor.

    Parameters
    ----------
    image:      Input image.  Accepted shapes: (H, W), (H, W, C), (C, H, W).
    normalize:  If True, apply percentile normalisation after conversion.
    percentile: Percentile for normalisation (used only when *normalize* is True).

    Returns
    -------
    (C, H, W) float32 tensor.
    """
    if isinstance(image, np.ndarray):
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32)
        out = torch.tensor(_move_channel_axis(image), dtype=torch.float32)
    elif isinstance(image, torch.Tensor):
        out = _move_channel_axis(image).float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    out = out.squeeze()
    out = _move_channel_axis(torch.atleast_3d(out))

    if normalize:
        out = normalize_percentile(out, percentile=percentile)

    return out


def resize_image(
    image: torch.Tensor,
    size: Tuple[int, int],
    labels: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Resize an image (and optionally its labels) to *size* using bilinear/nearest interpolation.

    Parameters
    ----------
    image:  (C, H, W) float32 tensor.
    size:   Target spatial size (H, W).
    labels: Optional (C, H, W) or (H, W) integer label tensor.

    Returns
    -------
    Resized image, and optionally the resized labels.
    """
    from torchvision.transforms import Resize
    import torchvision

    resized = Resize(size=list(size), antialias=True)(image)
    if labels is None:
        return resized
    resized_labels = Resize(
        size=list(size),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )(labels)
    return resized, resized_labels


def rescale_to_pixel_size(
    image: torch.Tensor,
    current_pixel_size: float,
    requested_pixel_size: float,
    labels: Optional[torch.Tensor] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    modality: str = "Brightfield",
    crop: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Scale an image so that its effective pixel size matches *requested_pixel_size*.

    Parameters
    ----------
    image:                (C, H, W) float32 tensor.
    current_pixel_size:   Physical size of one pixel in the input image (same unit as *requested*).
    requested_pixel_size: Target physical pixel size after rescaling.
    labels:               Optional label tensor to rescale in tandem.
    target_shape:         If given, the output will be cropped / padded to this (H, W) shape.
    modality:             "Brightfield" or "Fluorescence" — determines padding fill value.
    crop:                 If True (and *target_shape* is given) apply a random crop.

    Returns
    -------
    Rescaled image (and labels if provided).
    """
    from torchvision.transforms import Resize, RandomCrop
    import torchvision

    scale = current_pixel_size / requested_pixel_size
    shape = (torch.tensor(image.shape[-2:]) * scale).int().tolist()

    resized = Resize(size=shape, antialias=True)(image)

    if labels is not None:
        resized_labels = Resize(
            size=shape,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )(labels)
    else:
        resized_labels = None

    if target_shape is None or not crop:
        return (resized, resized_labels) if labels is not None else resized

    # Pad to at least target_shape
    while np.any(np.array(resized[0].shape) < target_shape[0]):
        pad = int((target_shape[0] - min(resized[0].shape)) / 2) + 3
        pad = int(torch.tensor([pad, resized.shape[1], resized.shape[2]]).min() - 1)
        fill = float(image.max()) if modality == "Brightfield" else float(resized.min())
        resized = torch.nn.functional.pad(resized, (pad, pad, pad, pad), mode="constant", value=fill)
        if resized_labels is not None:
            resized_labels = torch.nn.functional.pad(
                resized_labels, (pad, pad, pad, pad), mode="constant",
                value=min(int(resized_labels.min()), 0),
            ).to(labels.dtype)

    cropper = RandomCrop(size=target_shape)
    i, j, h, w = cropper.get_params(resized, output_size=target_shape)
    out_image = resized[:, i : i + h, j : j + w].float()

    if resized_labels is not None:
        out_labels = resized_labels[:, i : i + h, j : j + w]
        return out_image, out_labels

    return out_image


# ---------------------------------------------------------------------------
# Albumentations-based augmentation pipeline
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(
    bright_limit: float = 0.1,
    contrast_limit: float = 0.1,
    bright_prob: float = 0.5,
    flip_prob: float = 0.5,
    scale_limit: Tuple[float, float] = (-0.2, 0.2),
    rotate_prob: float = 0.4,
    seed: Optional[int] = None,
) -> A.Compose:
    """Build a standard albumentations augmentation pipeline for cell segmentation.

    The pipeline covers:
    - Random brightness / contrast
    - Horizontal and vertical flips
    - Shift-scale-rotate

    Parameters
    ----------
    bright_limit, contrast_limit: magnitude limits for brightness / contrast jitter.
    bright_prob:   probability of applying brightness/contrast augmentation.
    flip_prob:     probability of each flip augmentation.
    scale_limit:   (min, max) relative scale change for ShiftScaleRotate.
    rotate_prob:   probability of applying shift-scale-rotate.
    seed:          optional random seed for reproducibility.

    Returns
    -------
    An :class:`albumentations.Compose` transform that accepts keyword arguments
    ``image`` (H, W, C) and ``mask`` (H, W).
    """
    transforms = [
        A.RandomBrightnessContrast(
            brightness_limit=bright_limit,
            contrast_limit=contrast_limit,
            p=bright_prob,
        ),
        A.HorizontalFlip(p=flip_prob),
        A.VerticalFlip(p=flip_prob),
        A.ShiftScaleRotate(
            scale_limit=list(scale_limit),
            rotate_limit=45,
            shift_limit=0.1,
            p=rotate_prob,
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
        ),
        ToTensorV2(),
    ]
    return A.Compose(
        transforms,
        additional_targets={"mask": "mask"},
        seed=seed,
    )


def apply_augmentation_pipeline(
    pipeline: A.Compose,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply an albumentations pipeline to an image (and optional mask).

    Parameters
    ----------
    pipeline: A pre-built :class:`albumentations.Compose` pipeline.
    image:    (H, W, C) or (C, H, W) uint8 / float32 NumPy array.
    mask:     Optional (H, W) integer mask.
    seed:     If provided, set the pipeline seed before calling.

    Returns
    -------
    (image_tensor, mask_tensor) — C-first tensors; mask_tensor is None if no mask given.
    """
    # Ensure channels-last for albumentations
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.ascontiguousarray(np.transpose(image, (1, 2, 0)))

    if seed is not None:
        pipeline.set_random_seed(seed)

    if mask is not None:
        result = pipeline(image=image, mask=mask)
        return result["image"], result["mask"]

    result = pipeline(image=image)
    return result["image"], None
