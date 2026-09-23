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


def test_reproducibility():
    print("=== Running test_reproducibility ===")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        data_dir = Path(tmp_dir_str)
        create_dummy_dataset(data_dir)

        # Run 1
        model_1 = InstanSegModel()
        exp_name_1 = "test_reproducibility_run1"
        history_1 = model_1.train(
            epochs=2,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=exp_name_1,
            seed=42,
        )

        # Run 2
        model_2 = InstanSegModel()
        exp_name_2 = "test_reproducibility_run2"
        history_2 = model_2.train(
            epochs=2,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=exp_name_2,
            seed=42,
        )

        # Compare metric histories
        assert len(history_1) == len(history_2), "Metric histories length mismatch"
        for epoch_idx, (h1, h2) in enumerate(zip(history_1, history_2)):
            assert abs(h1["train_loss"] - h2["train_loss"]) < 1e-5, (
                f"Epoch {epoch_idx+1} train loss mismatch: {h1['train_loss']} vs {h2['train_loss']}"
            )
            if h1["val_loss"] != "N/A":
                assert abs(h1["val_loss"] - h2["val_loss"]) < 1e-5, (
                    f"Epoch {epoch_idx+1} val loss mismatch: {h1['val_loss']} vs {h2['val_loss']}"
                )

        # Compare model weight tensors
        ckp1 = torch.load(Path("runs") / exp_name_1 / "last.pt", map_location="cpu", weights_only=False)
        ckp2 = torch.load(Path("runs") / exp_name_2 / "last.pt", map_location="cpu", weights_only=False)


        sd1 = ckp1["model_state_dict"]
        sd2 = ckp2["model_state_dict"]

        assert set(sd1.keys()) == set(sd2.keys()), "State dict key mismatch"
        for k in sd1.keys():
            t1 = sd1[k]
            t2 = sd2[k]
            assert torch.allclose(t1, t2, atol=1e-5), f"Model weight mismatch at layer {k}"

    print("[OK] test_reproducibility passed successfully.", flush=True)


if __name__ == "__main__":
    test_reproducibility()

