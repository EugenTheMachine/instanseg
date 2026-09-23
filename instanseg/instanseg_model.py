import os
import copy
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from tqdm.auto import tqdm

from instanseg.utils.logger import get_logger
from instanseg.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_config_yaml,
    save_metrics_csv,
    get_rng_states,
    set_rng_states,
)
from instanseg.utils.data_loader import read_directory_dataset, get_loaders
from instanseg.utils.metrics import compute_dataset_metrics
from instanseg.utils.model_loader import build_model_from_dict
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss
from instanseg.utils.AI_utils import train_epoch


def seed_everything(seed: int = 42):
    """Fixes all random seeds across Python, NumPy, PyTorch CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_default_config() -> Dict[str, Any]:
    """Loads default configuration dictionary."""
    default_config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "meta": {"data_dir": "../data", "experiment_name": None},
        "preprocessing": {"imgsz": 512, "train_data_ratio": 1.0, "seed": 42},
        "training": {
            "epochs": 10,
            "patience": 5,
            "batch_size": 4,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "val_interval": 1,
            "warmup_epochs": 2,
            "resume": False,
        },
        "model": {
            "model_name": "InstanSeg_UNet",
            "num_classes": 2,
            "freeze_backbone_epochs": 0,
            "dropout": 0.0,
            "embedding_mode": "center-seed",
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "backbone_lr_mult": 1.0,
        },
    }


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens nested config dict into key-value pairs for model builders."""
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
    # Add defaults expected by InstanSeg_UNet / loss builder
    flat["model_str"] = flat.get("model_name", "InstanSeg_UNet")
    flat["dim_in"] = flat.get("dim_in", 3)
    flat["dim_out"] = flat.get("dim_out", 6)
    flat["dropprob"] = flat.get("dropout", 0.0)
    flat["layers"] = flat.get("layers", (32, 64, 128, 256))
    flat["cells_and_nuclei"] = flat.get("cells_and_nuclei", False)
    flat["target_segmentation"] = flat.get("target_segmentation", "N")
    flat["multihead"] = flat.get("multihead", False)
    flat["dim_seeds"] = flat.get("dim_seeds", 1)
    flat["dim_coords"] = flat.get("dim_coords", 2)
    flat["n_sigma"] = flat.get("n_sigma", 2)
    flat["norm"] = flat.get("norm", "BATCH")
    flat["channel_invariant"] = flat.get("channel_invariant", False)
    return flat


class InstanSegModel:
    """
    Unified high-level class for training and evaluating InstanSeg models.
    """

    def __init__(
        self,
        config_path_or_checkpoint: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.config = _load_default_config()
        self.checkpoint_path: Optional[Path] = None

        if config_path_or_checkpoint is not None:
            path = Path(config_path_or_checkpoint)
            if path.suffix in [".pt", ".pth"] or path.is_dir():
                self.checkpoint_path = path if path.is_file() else (path / "checkpoints" / "last.pt")
                if not self.checkpoint_path.exists() and path.is_dir():
                    self.checkpoint_path = path / "last.pt"
                # If adjacent config.yaml exists, load it
                config_file = (path if path.is_dir() else path.parent) / "config.yaml"
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        loaded_cfg = yaml.safe_load(f)
                        if isinstance(loaded_cfg, dict):
                            self.config.update(loaded_cfg)
            elif path.suffix in [".yaml", ".yml"]:
                with open(path, "r", encoding="utf-8") as f:
                    loaded_cfg = yaml.safe_load(f)
                    if isinstance(loaded_cfg, dict):
                        self.config.update(loaded_cfg)

        self._override_config_kwargs(kwargs)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._init_model_from_checkpoint(self.checkpoint_path)

    def _override_config_kwargs(self, kwargs: Dict[str, Any]):
        """Helper to override config sections with flat kwarg values."""
        section_mapping = {
            "data_dir": ("meta", "data_dir"),
            "experiment_name": ("meta", "experiment_name"),
            "imgsz": ("preprocessing", "imgsz"),
            "train_data_ratio": ("preprocessing", "train_data_ratio"),
            "seed": ("preprocessing", "seed"),
            "epochs": ("training", "epochs"),
            "patience": ("training", "patience"),
            "batch_size": ("training", "batch_size"),
            "val_ratio": ("training", "val_ratio"),
            "test_ratio": ("training", "test_ratio"),
            "val_interval": ("training", "val_interval"),
            "warmup_epochs": ("training", "warmup_epochs"),
            "resume": ("training", "resume"),
            "model_name": ("model", "model_name"),
            "num_classes": ("model", "num_classes"),
            "freeze_backbone_epochs": ("model", "freeze_backbone_epochs"),
            "dropout": ("model", "dropout"),
            "embedding_mode": ("model", "embedding_mode"),
            "learning_rate": ("optimizer", "learning_rate"),
            "weight_decay": ("optimizer", "weight_decay"),
            "momentum": ("optimizer", "momentum"),
            "backbone_lr_mult": ("optimizer", "backbone_lr_mult"),
            "optimizer_type": ("optimizer", "type"),
            "type": ("optimizer", "type"),
        }
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in section_mapping:
                sec, key = section_mapping[k]
                self.config[sec][key] = v
            else:
                # Store unmapped kwargs in model section or meta section
                self.config["model"][k] = v

    def _validate_config(self):
        """Validates configuration parameters."""
        tdr = float(self.config["preprocessing"]["train_data_ratio"])
        if not (0.0 < tdr <= 1.0):
            raise ValueError(f"train_data_ratio must be between 0.0 and 1.0, got {tdr}")

        vr = float(self.config["training"]["val_ratio"])
        if not (0.0 <= vr <= 1.0):
            raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {vr}")

        tr = float(self.config["training"]["test_ratio"])
        if not (0.0 <= tr <= 1.0):
            raise ValueError(f"test_ratio must be between 0.0 and 1.0, got {tr}")

    def _init_model_from_checkpoint(self, ckp_path: Path):
        """Initializes model architecture and loads weights from checkpoint."""
        flat_cfg = _flatten_config(self.config)
        self.model = build_model_from_dict(flat_cfg, random_seed=flat_cfg.get("seed", 42))
        instanseg_loss_instance = InstanSegLoss(
            binary_loss_fn_str="lovasz_hinge",
            seed_loss_fn="l1_distance",
            device=self.device,
            n_sigma=flat_cfg.get("n_sigma", 2),
            cells_and_nuclei=flat_cfg.get("cells_and_nuclei", False),
            window_size=flat_cfg.get("window_size", 128),
            dim_coords=flat_cfg.get("dim_coords", 2),
            dim_seeds=flat_cfg.get("dim_seeds", 1),
            bg_weight=None,
        )
        self.model = instanseg_loss_instance.initialize_pixel_classifier(
            self.model, MLP_width=flat_cfg.get("mlp_width", 5)
        )
        load_checkpoint(ckp_path, self.model, device=self.device)
        self.model.to(self.device)

    def train(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Runs full training pipeline.
        Overridden keyword arguments dynamically update model config and saved artifacts.
        """
        self._override_config_kwargs(kwargs)
        self._validate_config()

        seed = int(self.config["preprocessing"]["seed"])
        seed_everything(seed)

        exp_name = self.config["meta"]["experiment_name"] or "instanseg_experiment"
        exp_dir = Path("runs") / exp_name
        checkpoints_dir = exp_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(log_file=exp_dir / "train.log")

        # Save actual merged configuration to artifact directory
        save_config_yaml(self.config, exp_dir / "config.yaml")

        data_dir = self.config["meta"]["data_dir"]
        train_data_ratio = float(self.config["preprocessing"]["train_data_ratio"])
        val_ratio = float(self.config["training"]["val_ratio"])
        test_ratio = float(self.config["training"]["test_ratio"])

        # Load datasets
        dataset_splits = read_directory_dataset(
            data_dir=data_dir,
            train_data_ratio=train_data_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        logger.info("Data loaded successfully")

        train_imgs, train_msks, train_meta = dataset_splits["train"]
        val_imgs, val_msks, val_meta = dataset_splits["val"]
        test_imgs, test_msks, test_meta = dataset_splits["test"]

        # Build args object for get_loaders and model training
        flat_cfg = _flatten_config(self.config)
        args = type("Args", (), flat_cfg)()
        setattr(args, "rng_seed", seed)
        setattr(args, "seed", seed)
        setattr(args, "tile_size", self.config["preprocessing"]["imgsz"])
        batch_size = int(self.config["training"]["batch_size"])
        accumulation_steps = max(1, 4 // max(1, batch_size))
        setattr(args, "batch_size", batch_size)
        setattr(args, "accumulation_steps", accumulation_steps)
        setattr(args, "num_workers", 0)
        setattr(args, "transform_intensity", 0.5)
        setattr(args, "requested_pixel_size", None)
        setattr(args, "mean_object_diameter", None)
        setattr(args, "augmentation_type", "minimal")
        setattr(args, "weight", False)
        setattr(args, "length_of_epoch", len(train_imgs))
        setattr(args, "clip", 20.0)
        setattr(args, "on_cluster", False)

        train_loader, val_loader = get_loaders(
            train_imgs, train_msks, val_imgs, val_msks, train_meta, val_meta, args
        )
        logger.info("Augmentation pipeline initialized")
        logger.info(f"Config: {self.config}")

        # Instantiate model architecture
        self.model = build_model_from_dict(flat_cfg, random_seed=seed)

        # Loss function
        instanseg_loss_instance = InstanSegLoss(
            binary_loss_fn_str="lovasz_hinge",
            seed_loss_fn="l1_distance",
            device=self.device,
            n_sigma=flat_cfg.get("n_sigma", 2),
            cells_and_nuclei=flat_cfg.get("cells_and_nuclei", False),
            window_size=flat_cfg.get("window_size", 128),
            dim_coords=flat_cfg.get("dim_coords", 2),
            dim_seeds=flat_cfg.get("dim_seeds", 1),
            bg_weight=None,
        )

        from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model
        self.model = instanseg_loss_instance.initialize_pixel_classifier(
            self.model, MLP_width=flat_cfg.get("mlp_width", 5)
        )

        self.model.to(self.device)

        def loss_fn(*args_fn, **kwargs_fn):
            return instanseg_loss_instance.forward(*args_fn, **kwargs_fn)

        lr = float(self.config["optimizer"]["learning_rate"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        optimizer_type = str(self.config["optimizer"].get("type", "AdamW"))

        if optimizer_type.lower() == "sgd":
            momentum = float(self.config["optimizer"].get("momentum", 0.9))
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        start_epoch = 1
        best_val_loss = float("inf")
        patience_counter = 0
        metrics_history = []

        resume_flag = self.config["training"]["resume"]
        if resume_flag:
            ckp_file = None
            if isinstance(resume_flag, str) and Path(resume_flag).exists():
                ckp_file = Path(resume_flag)
            elif (checkpoints_dir / "last.pt").exists():
                ckp_file = checkpoints_dir / "last.pt"
            elif (exp_dir / "last.pt").exists():
                ckp_file = exp_dir / "last.pt"
            elif self.checkpoint_path and self.checkpoint_path.exists():
                ckp_file = self.checkpoint_path

            if ckp_file and ckp_file.exists():
                ckp = load_checkpoint(ckp_file, self.model, optimizer=optimizer, device=self.device)
                if "rng_states" in ckp:
                    set_rng_states(ckp["rng_states"])
                start_epoch = ckp.get("epoch", 0) + 1
                best_val_loss = ckp.get("best_val_loss", float("inf"))
                patience_counter = ckp.get("patience_counter", 0)
                metrics_history = ckp.get("metrics_history", [])
                logger.info(f"Loaded checkpoint from {ckp_file}. Resuming model training from epoch {start_epoch}")
            else:
                logger.info("Resume requested but no checkpoint found. Starting model training from scratch.")
        else:
            logger.info("Model initialized from scratch")

        total_epochs = int(self.config["training"]["epochs"])
        patience_limit = int(self.config["training"]["patience"])
        val_interval = int(self.config["training"]["val_interval"])
        warmup_epochs = int(self.config["training"].get("warmup_epochs", 2))

        for epoch in range(start_epoch, total_epochs + 1):
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                current_lr = lr * (epoch / float(warmup_epochs))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                logger.info(f"Warmup Epoch {epoch}/{warmup_epochs}: Learning rate set to {current_lr:.6f}")
            elif warmup_epochs > 0 and epoch == warmup_epochs + 1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            logger.info(f"EPOCH {epoch}/{total_epochs}")

            # Train single epoch
            train_loss, _ = train_epoch(
                train_model=self.model,
                train_device=self.device,
                train_dataloader=train_loader,
                train_loss_fn=loss_fn,
                train_optimizer=optimizer,
                args=args,
            )

            is_val_epoch = (epoch % val_interval == 0) or (epoch == total_epochs)
            val_loss = None
            val_metrics = {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0}

            if is_val_epoch:
                self.model.eval()
                val_losses = []
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for img_b, lbl_b, _ in tqdm(val_loader, desc="Validation", disable=False):
                        img_b = img_b.to(self.device)
                        lbl_b = lbl_b.to(self.device)
                        out = self.model(img_b)
                        loss = loss_fn(out, lbl_b.clone()).mean()
                        val_losses.append(loss.detach().cpu().item())

                        # Generate predictions for metrics matching
                        if isinstance(out, list):
                            out = out[0]
                        for i in range(out.shape[0]):
                            pred_mask = instanseg_loss_instance.postprocessing(out[i])
                            val_preds.append(pred_mask)
                            val_targets.append(lbl_b[i])

                val_loss = float(np.mean(val_losses))
                val_metrics = compute_dataset_metrics(val_targets, val_preds)

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_loss if val_loss is not None else "N/A",
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "accuracy": val_metrics["accuracy"],
                "f1": val_metrics["f1"],
            }
            metrics_history.append(epoch_record)
            save_metrics_csv(metrics_history, exp_dir / "metrics.csv")

            logger.info(
                f"Epoch {epoch} Results - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss if val_loss is not None else 'N/A'} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.4f} | "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Checkpoint saving & early stopping logic
            is_best = False
            if is_val_epoch and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "metrics_history": metrics_history,
                "rng_states": get_rng_states(),
                "config": self.config,
            }

            save_checkpoint(checkpoint_state, is_best, checkpoints_dir, filename="last.pt", best_filename="best.pt")
            save_checkpoint(checkpoint_state, is_best, exp_dir, filename="last.pt", best_filename="best.pt")

            if is_val_epoch and patience_counter >= patience_limit:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final Test Set Evaluation
        best_ckp = checkpoints_dir / "best.pt"
        if not best_ckp.exists():
            best_ckp = exp_dir / "last.pt"
        if best_ckp.exists():
            load_checkpoint(best_ckp, self.model, device=self.device)

        test_data_split = dataset_splits["test"]
        test_imgs, test_msks, test_meta = test_data_split
        _, test_loader = get_loaders(
            test_imgs, test_msks, test_imgs, test_msks, test_meta, test_meta, args
        )

        self.model.eval()
        test_losses = []
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for img_b, lbl_b, _ in tqdm(test_loader, desc="Test Evaluation", disable=False):
                img_b = img_b.to(self.device)
                lbl_b = lbl_b.to(self.device)
                out = self.model(img_b)
                loss = loss_fn(out, lbl_b.clone()).mean()
                test_losses.append(loss.detach().cpu().item())

                if isinstance(out, list):
                    out = out[0]
                for i in range(out.shape[0]):
                    pred_mask = instanseg_loss_instance.postprocessing(out[i])
                    test_preds.append(pred_mask)
                    test_targets.append(lbl_b[i])

        avg_test_loss = float(np.mean(test_losses))
        test_metrics = compute_dataset_metrics(test_targets, test_preds)

        logger.info(
            f"TEST RESULTS - Loss: {avg_test_loss:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f}"
        )

        return metrics_history

    def eval(self, subset: str = "test", **kwargs) -> Dict[str, float]:
        """
        Runs evaluation pipeline on the validation or test set.
        Returns dictionary of averaged test loss and quality metrics.
        """
        self._override_config_kwargs(kwargs)
        seed = int(self.config["preprocessing"]["seed"])
        seed_everything(seed)

        data_dir = self.config["meta"]["data_dir"]
        dataset_splits = read_directory_dataset(
            data_dir=data_dir,
            train_data_ratio=float(self.config["preprocessing"]["train_data_ratio"]),
            val_ratio=float(self.config["training"]["val_ratio"]),
            test_ratio=float(self.config["training"]["test_ratio"]),
            seed=seed,
        )

        split_data = dataset_splits.get(subset, dataset_splits["test"])
        eval_imgs, eval_msks, eval_meta = split_data

        flat_cfg = _flatten_config(self.config)
        args = type("Args", (), flat_cfg)()
        setattr(args, "rng_seed", seed)
        setattr(args, "seed", seed)
        setattr(args, "tile_size", self.config["preprocessing"]["imgsz"])
        setattr(args, "batch_size", self.config["training"]["batch_size"])
        setattr(args, "num_workers", 0)
        setattr(args, "transform_intensity", 0.5)
        setattr(args, "requested_pixel_size", None)
        setattr(args, "mean_object_diameter", None)
        setattr(args, "augmentation_type", "minimal")
        setattr(args, "weight", False)
        setattr(args, "length_of_epoch", len(eval_imgs))

        _, eval_loader = get_loaders(eval_imgs, eval_msks, eval_imgs, eval_msks, eval_meta, eval_meta, args)

        instanseg_loss_instance = InstanSegLoss(
            binary_loss_fn_str="lovasz_hinge",
            seed_loss_fn="l1_distance",
            device=self.device,
            n_sigma=flat_cfg.get("n_sigma", 2),
            cells_and_nuclei=flat_cfg.get("cells_and_nuclei", False),
            window_size=flat_cfg.get("window_size", 128),
            dim_coords=flat_cfg.get("dim_coords", 2),
            dim_seeds=flat_cfg.get("dim_seeds", 1),
            bg_weight=None,
        )

        from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model
        if self.model is None:
            self.model = build_model_from_dict(flat_cfg, random_seed=seed)
            if self.checkpoint_path and self.checkpoint_path.exists():
                load_checkpoint(self.checkpoint_path, self.model, device=self.device)

        self.model = instanseg_loss_instance.initialize_pixel_classifier(
            self.model, MLP_width=flat_cfg.get("mlp_width", 5)
        )
        self.model.to(self.device)
        self.model.eval()



        def loss_fn(*args_fn, **kwargs_fn):
            return instanseg_loss_instance.forward(*args_fn, **kwargs_fn)

        eval_losses = []
        eval_preds = []
        eval_targets = []

        with torch.no_grad():
            for img_b, lbl_b, _ in tqdm(eval_loader, desc=f"Evaluation ({subset})", disable=False):
                img_b = img_b.to(self.device)
                lbl_b = lbl_b.to(self.device)
                out = self.model(img_b)
                loss = loss_fn(out, lbl_b.clone()).mean()
                eval_losses.append(loss.detach().cpu().item())

                if isinstance(out, list):
                    out = out[0]
                for i in range(out.shape[0]):
                    pred_mask = instanseg_loss_instance.postprocessing(out[i])
                    eval_preds.append(pred_mask)
                    eval_targets.append(lbl_b[i])

        avg_loss = float(np.mean(eval_losses))
        metrics = compute_dataset_metrics(eval_targets, eval_preds)
        metrics["loss"] = avg_loss

        return metrics
