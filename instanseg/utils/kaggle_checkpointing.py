"""Utilities for saving and restoring training checkpoints via Kaggle Models."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

BUNDLE_ARCHIVE_NAME = "checkpoint_bundle.tar.gz"
CHECKPOINT_FILE_NAME = "checkpoint.pt"
MANIFEST_FILE_NAME = "bundle_manifest.json"
DEFAULT_FRAMEWORK = "pyTorch"


def _is_kaggle_model_handle(path: str) -> bool:
    parts = path.strip("/").split("/")
    if len(parts) != 4:
        return False
    if Path(path).drive or path.startswith((".", "~")):
        return False
    if os.name == "nt" and "\\" in path:
        return False
    return all(part.strip() for part in parts)


def validate_kaggle_checkpoint_config(config: Dict[str, Any]) -> None:
    """Validate Kaggle checkpoint settings when ``save_kgl_ckp`` is enabled."""
    if not config.get("save_kgl_ckp", False):
        return

    for key in ("kgl_best_ckp_path", "kgl_last_ckp_path", "kgl_creds_path"):
        if config.get(key) is None:
            raise ValueError(f"{key} must be set when save_kgl_ckp is True")

    creds_path = Path(config["kgl_creds_path"])
    if not creds_path.is_file():
        raise ValueError(f"kgl_creds_path does not point to an existing file: {creds_path}")

    try:
        load_kaggle_credentials(creds_path)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        raise ValueError(f"Invalid Kaggle credentials file at {creds_path}: {exc}") from exc

    freq = config.get("kgl_ckp_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"kgl_ckp_freq must be a positive integer, but got {freq}")

    for key in ("kgl_best_ckp_path", "kgl_last_ckp_path"):
        path_value = config[key]
        if not _is_kaggle_model_handle(str(path_value)):
            ensure_checkpoint_path(path_value)


def ensure_checkpoint_path(path_value: Union[str, Path]) -> Path:
    """Create a writable checkpoint directory when needed."""
    if path_value is None:
        raise ValueError("Checkpoint path must not be None")

    path = Path(path_value)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Checkpoint path exists but is not a directory: {path}")
        if not os.access(path, os.W_OK):
            raise ValueError(f"Checkpoint directory is not writable: {path}")
        return path

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Could not create checkpoint directory at {path}: {exc}") from exc

    if not os.access(path, os.W_OK):
        raise ValueError(f"Checkpoint directory is not writable: {path}")
    return path


def load_kaggle_credentials(creds_path: Union[str, Path]) -> Dict[str, str]:
    """Load Kaggle credentials from a JSON file or a raw token file."""
    creds_path = Path(creds_path)
    text = creds_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Kaggle credentials file is empty")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Accept raw token files containing only the Kaggle API token string.
        if text and not text.startswith(('{', '[')):
            return {"token": text}
        raise ValueError("Kaggle credentials file must contain a JSON object or a raw Kaggle API token")

    if isinstance(payload, str):
        return {"token": payload}

    if not isinstance(payload, dict):
        raise ValueError("Kaggle credentials file must contain a JSON object")

    if "username" in payload and "key" in payload:
        return {"username": str(payload["username"]), "key": str(payload["key"])}

    token = payload.get("token") or payload.get("KAGGLE_API_TOKEN")
    if token:
        return {"token": str(token)}

    raise ValueError(
        "Kaggle credentials must contain either {'username', 'key'} or {'token'}/{'KAGGLE_API_TOKEN'}"
    )


def setup_kaggle_credentials(creds_path: Union[str, Path]) -> Dict[str, str]:
    """Configure process environment for Kaggle API access."""
    creds = load_kaggle_credentials(creds_path)
    if "token" in creds:
        os.environ["KAGGLE_API_TOKEN"] = creds["token"]
    else:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = kaggle_dir / "kaggle.json"
        with open(legacy_path, "w", encoding="utf-8") as legacy_file:
            json.dump({"username": creds["username"], "key": creds["key"]}, legacy_file)
        os.chmod(legacy_path, 0o600)
    return creds


def resolve_kaggle_model_handle(
    path_value: Union[str, Path],
    creds: Dict[str, str],
    experiment_name: str,
    checkpoint_kind: str,
) -> str:
    """Resolve the Kaggle Models handle for a checkpoint destination."""
    path_str = str(path_value)
    if _is_kaggle_model_handle(path_str):
        return path_str.strip("/")

    username = creds.get("username")
    if not username:
        username = os.environ.get("KAGGLE_USERNAME", "kaggle-user")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment_name)
    slug = slug.strip("-_") or "instanseg-experiment"
    return f"{username}/{slug}/{DEFAULT_FRAMEWORK}/{checkpoint_kind}"


def write_checkpoint_bundle(
    destination_dir: Path,
    checkpoint: Dict[str, Any],
    experiment_dir: Path,
    checkpoint_kind: str,
    model_handle: Optional[str] = None,
) -> Path:
    """Materialize a resumable checkpoint bundle in ``destination_dir``."""
    destination_dir = ensure_checkpoint_path(destination_dir)
    checkpoint_path = destination_dir / CHECKPOINT_FILE_NAME
    torch.save(checkpoint, checkpoint_path)

    for artifact_name in ("config.yaml", "metrics.csv", "train.log"):
        source = experiment_dir / artifact_name
        if source.exists():
            shutil.copy2(source, destination_dir / artifact_name)

    manifest = {
        "checkpoint_kind": checkpoint_kind,
        "model_handle": model_handle,
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "files": sorted(p.name for p in destination_dir.iterdir() if p.is_file()),
    }
    manifest_path = destination_dir / MANIFEST_FILE_NAME
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    archive_path = destination_dir / BUNDLE_ARCHIVE_NAME
    with tarfile.open(archive_path, "w:gz") as archive:
        for file_path in destination_dir.iterdir():
            if file_path.name == BUNDLE_ARCHIVE_NAME:
                continue
            archive.add(file_path, arcname=file_path.name)
    return archive_path


def decompress_checkpoint_bundle(
    archive_or_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Restore bundle files from an archive or an existing bundle directory."""
    source = Path(archive_or_dir)
    if output_dir is None:
        output_dir = source.parent / f"{source.stem}_extracted"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        for item in source.iterdir():
            target = output_path / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return output_path

    if source.suffixes[-2:] == [".tar", ".gz"] or source.suffix == ".tgz":
        with tarfile.open(source, "r:gz") as archive:
            archive.extractall(path=output_path)
        return output_path

    raise ValueError(f"Unsupported checkpoint bundle source: {source}")


def load_checkpoint_from_bundle(bundle_path: Union[str, Path], map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a training checkpoint dictionary from a bundle directory or archive."""
    bundle_path = Path(bundle_path)
    if bundle_path.is_dir():
        checkpoint_file = bundle_path / CHECKPOINT_FILE_NAME
        if not checkpoint_file.exists():
            extracted = decompress_checkpoint_bundle(bundle_path)
            checkpoint_file = extracted / CHECKPOINT_FILE_NAME
    else:
        extracted = decompress_checkpoint_bundle(bundle_path)
        checkpoint_file = extracted / CHECKPOINT_FILE_NAME

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found in bundle: {bundle_path}")

    checkpoint = torch.load(checkpoint_file, map_location=map_location or "cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict in bundle, got {type(checkpoint)}")
    return checkpoint


def upload_checkpoint_bundle(
    bundle_dir: Path,
    model_handle: str,
    creds_path: Union[str, Path],
    version_notes: str,
) -> None:
    """Upload a checkpoint bundle directory to Kaggle Models."""
    setup_kaggle_credentials(creds_path)
    import kagglehub

    kagglehub.model_upload(
        model_handle,
        str(bundle_dir),
        version_notes=version_notes,
        license_name="Apache 2.0",
    )


class KaggleCheckpointManager:
    """Coordinates local bundle creation and optional Kaggle Models uploads."""

    def __init__(self, config: Dict[str, Any], experiment_dir: Path, logger: Any = None):
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger
        self.enabled = bool(config.get("save_kgl_ckp", False))
        self.creds: Optional[Dict[str, str]] = None
        self.best_path = config.get("kgl_best_ckp_path")
        self.last_path = config.get("kgl_last_ckp_path")
        self.creds_path = config.get("kgl_creds_path")
        self.frequency = int(config.get("kgl_ckp_freq", 1))

        if self.enabled:
            validate_kaggle_checkpoint_config(config)
            self.creds = load_kaggle_credentials(self.creds_path)

    def _get_bundle_dir(self, path_value: Union[str, Path], checkpoint_kind: str) -> Path:
        path_str = str(path_value)
        if _is_kaggle_model_handle(path_str):
            return ensure_checkpoint_path(self.experiment_dir / ".kgl_checkpoint_bundles" / checkpoint_kind)
        return ensure_checkpoint_path(path_value)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)
        else:
            print(message)

    def should_save_this_epoch(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        return (epoch + 1) % self.frequency == 0

    def save_checkpoints(
        self,
        checkpoint: Dict[str, Any],
        epoch: int,
        is_best: bool,
    ) -> None:
        if not self.enabled or not self.should_save_this_epoch(epoch):
            return

        experiment_name = self.config.get("experiment_name") or "default_experiment"
        last_dir = self._get_bundle_dir(self.last_path, "last")
        last_handle = resolve_kaggle_model_handle(self.last_path, self.creds, experiment_name, "last")
        write_checkpoint_bundle(
            last_dir,
            checkpoint,
            self.experiment_dir,
            checkpoint_kind="last",
            model_handle=last_handle,
        )
        self._log(f"Kaggle last checkpoint bundle saved to {last_dir}")
        try:
            upload_checkpoint_bundle(
                last_dir,
                last_handle,
                self.creds_path,
                version_notes=f"last checkpoint after epoch {epoch + 1}",
            )
            self._log(f"Uploaded last checkpoint bundle to Kaggle model {last_handle}")
        except Exception as exc:
            self._log(f"Warning: failed to upload last checkpoint to Kaggle: {exc}")

        if is_best:
            best_dir = self._get_bundle_dir(self.best_path, "best")
            best_handle = resolve_kaggle_model_handle(self.best_path, self.creds, experiment_name, "best")
            write_checkpoint_bundle(
                best_dir,
                checkpoint,
                self.experiment_dir,
                checkpoint_kind="best",
                model_handle=best_handle,
            )
            self._log(f"Kaggle best checkpoint bundle saved to {best_dir}")
            try:
                upload_checkpoint_bundle(
                    best_dir,
                    best_handle,
                    self.creds_path,
                    version_notes=f"best checkpoint after epoch {epoch + 1}",
                )
                self._log(f"Uploaded best checkpoint bundle to Kaggle model {best_handle}")
            except Exception as exc:
                self._log(f"Warning: failed to upload best checkpoint to Kaggle: {exc}")
