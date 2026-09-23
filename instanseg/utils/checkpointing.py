import os
import csv
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = "last.pt",
    best_filename: str = "best.pt",
):
    """Saves model state to last.pt and optionally best.pt."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_filepath = checkpoint_dir / best_filename
        torch.save(state, best_filepath)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Loads model checkpoint and restores weights, optimizer, and scheduler states."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        # Fallback if raw state dict saved
        model.load_state_dict(checkpoint, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_config_yaml(config_dict: Dict[str, Any], save_path: Path):
    """Saves config dictionary as YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_metrics_csv(metrics_history: list, csv_path: Path):
    """Saves metrics history list of dicts to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_history:
        return

    fieldnames = list(metrics_history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_history:
            writer.writerow(row)


def get_rng_states() -> Dict[str, Any]:
    """Captures Python, NumPy, PyTorch CPU & CUDA random state dicts."""
    states = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]):
    """Restores Python, NumPy, PyTorch CPU & CUDA random state dicts."""
    if "python_rng" in states:
        random.setstate(states["python_rng"])
    if "numpy_rng" in states:
        np.random.set_state(states["numpy_rng"])
    if "torch_rng" in states:
        torch_rng = states["torch_rng"]
        if isinstance(torch_rng, torch.Tensor):
            torch_rng = torch_rng.cpu().to(torch.uint8)
        torch.set_rng_state(torch_rng)
    if "torch_cuda_rng" in states and torch.cuda.is_available():
        cuda_rngs = states["torch_cuda_rng"]
        cleaned_cuda_rngs = []
        for c_rng in cuda_rngs:
            if isinstance(c_rng, torch.Tensor):
                c_rng = c_rng.cpu().to(torch.uint8)
            cleaned_cuda_rngs.append(c_rng)
        torch.cuda.set_rng_state_all(cleaned_cuda_rngs)
