import sys
import tempfile
import os
import numpy as np
import torch
import cv2
import tifffile
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from instanseg.utils.data_loader import read_directory_dataset, get_loaders


def create_dummy_dataset(tmp_dir: Path):
    train_img_dir = tmp_dir / "train" / "images"
    train_msk_dir = tmp_dir / "train" / "masks"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_msk_dir.mkdir(parents=True, exist_ok=True)

    for i in range(4):
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint16)
        mask[20:50, 20:50] = 1
        mask[60:90, 60:90] = 2

        tifffile.imwrite(train_img_dir / f"image_{i}.tif", img)
        cv2.imwrite(str(train_msk_dir / f"mask_{i}.png"), mask)


def test_dataloader_efficiency():
    print("=== Running test_dataloader_efficiency ===")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        create_dummy_dataset(tmp_dir)

        splits = read_directory_dataset(data_dir=tmp_dir, train_data_ratio=1.0, seed=42)
        train_imgs, train_msks, train_meta = splits["train"]
        val_imgs, val_msks, val_meta = splits["val"]

        num_workers = 0 if os.name == "nt" else 2

        args = type(
            "Args",
            (),
            {
                "dim_in": 3,
                "transform_intensity": 0.5,
                "requested_pixel_size": None,
                "mean_object_diameter": None,
                "augmentation_type": "minimal",
                "tile_size": 128,
                "cells_and_nuclei": False,
                "target_segmentation": "N",
                "channel_invariant": False,
                "length_of_epoch": len(train_imgs),
                "weight": False,
                "batch_size": 1,
                "num_workers": num_workers,
                "seed": 42,
            },
        )()

        train_loader, test_loader = get_loaders(
            train_imgs, train_msks, val_imgs, val_msks, train_meta, val_meta, args
        )

        assert train_loader.num_workers == num_workers, f"Expected num_workers={num_workers}, got {train_loader.num_workers}"
        assert train_loader.persistent_workers == (num_workers > 0), f"persistent_workers expected {num_workers > 0}"

        for images, labels, _ in train_loader:
            assert isinstance(images, torch.Tensor), "Images batch must be torch.Tensor"
            assert isinstance(labels, torch.Tensor), "Labels batch must be torch.Tensor"
            assert images.shape[0] == 1, f"Expected batch size 1, got {images.shape[0]}"
            assert images.shape[-2:] == (128, 128), f"Expected H,W (128,128), got {images.shape[-2:]}"
            break

    print("[OK] test_dataloader_efficiency passed successfully.", flush=True)


if __name__ == "__main__":
    test_dataloader_efficiency()

