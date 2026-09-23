import sys
import tempfile
import torch
import numpy as np
import cv2
import tifffile
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from instanseg import InstanSegModel


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


def test_resume_training():
    print("=== Running test_resume_training ===")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        data_dir = Path(tmp_dir_str)
        create_dummy_dataset(data_dir)

        # 1. Single continuous run for 2 epochs
        exp_single = "test_resume_single_run"
        model_single = InstanSegModel()
        history_single = model_single.train(
            epochs=2,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=exp_single,
            seed=42,
        )

        # 2. Split run Part 1: train for 1 epoch
        exp_split = "test_resume_split_run"
        model_part1 = InstanSegModel()
        history_part1 = model_part1.train(
            epochs=1,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=exp_split,
            seed=42,
        )
        assert len(history_part1) == 1, "Part 1 should complete exactly 1 epoch"

        # 3. Split run Part 2: resume training up to total 2 epochs
        model_part2 = InstanSegModel()
        history_part2 = model_part2.train(
            epochs=2,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=exp_split,
            resume=True,
            seed=42,
        )
        assert len(history_part2) == 2, "Part 2 should have 2 total epochs recorded in metrics history"

        # Compare metric histories between single continuous run and split resumed run
        for idx, (h_single, h_split) in enumerate(zip(history_single, history_part2)):
            assert abs(h_single["train_loss"] - h_split["train_loss"]) < 1e-5, (
                f"Epoch {idx+1} train loss mismatch: single={h_single['train_loss']}, resumed={h_split['train_loss']}"
            )
            if h_single["val_loss"] != "N/A":
                assert abs(h_single["val_loss"] - h_split["val_loss"]) < 1e-5, (
                    f"Epoch {idx+1} val loss mismatch: single={h_single['val_loss']}, resumed={h_split['val_loss']}"
                )

        # Compare final model state dicts
        ckp_single = torch.load(Path("runs") / exp_single / "last.pt", map_location="cpu", weights_only=False)
        ckp_split = torch.load(Path("runs") / exp_split / "last.pt", map_location="cpu", weights_only=False)


        sd_single = ckp_single["model_state_dict"]
        sd_split = ckp_split["model_state_dict"]

        assert set(sd_single.keys()) == set(sd_split.keys()), "State dict key mismatch"
        for k in sd_single.keys():
            assert torch.allclose(sd_single[k], sd_split[k], atol=1e-5), f"Model weight mismatch at layer {k}"

    print("[OK] test_resume_training passed successfully.", flush=True)


if __name__ == "__main__":
    test_resume_training()

