import sys
import tempfile
import yaml
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


def test_train_eval_wrapper():
    print("=== Running test_train_eval_wrapper ===")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        data_dir = Path(tmp_dir_str)
        create_dummy_dataset(data_dir)

        model = InstanSegModel()
        experiment_name = "test_train_eval_wrapper_exp"

        metrics_history = model.train(
            epochs=2,
            imgsz=128,
            batch_size=1,
            data_dir=str(data_dir),
            experiment_name=experiment_name,
            train_data_ratio=1.0,
            learning_rate=0.002,
        )

        assert isinstance(metrics_history, list), "train() must return metrics history list"
        assert len(metrics_history) == 2, f"Expected 2 epochs recorded, got {len(metrics_history)}"

        exp_config_path = Path("runs") / experiment_name / "config.yaml"
        assert exp_config_path.exists(), f"Config artifact must exist at {exp_config_path}"

        with open(exp_config_path, "r", encoding="utf-8") as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["preprocessing"]["imgsz"] == 128, "Overridden imgsz 128 was not saved"
        assert saved_config["optimizer"]["learning_rate"] == 0.002, "Overridden learning_rate 0.002 was not saved"

        # Evaluate model
        eval_metrics = model.eval(subset="test", data_dir=str(data_dir))
        assert "loss" in eval_metrics, "Evaluation metrics must include loss"
        assert "f1" in eval_metrics, "Evaluation metrics must include f1"
        assert "precision" in eval_metrics, "Evaluation metrics must include precision"
        assert "recall" in eval_metrics, "Evaluation metrics must include recall"
        assert "accuracy" in eval_metrics, "Evaluation metrics must include accuracy"

    print("[OK] test_train_eval_wrapper passed successfully.", flush=True)


if __name__ == "__main__":
    test_train_eval_wrapper()

