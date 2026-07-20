from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.export as torch_export
import yaml
from torch.utils.data import DataLoader

from instanseg.scripts.train import (
    build_maskrcnn,
    collect_image_mask_pairs,
    get_maskrcnn_targets,
    load_config,
    maskrcnn_to_labeled_mask,
    safe_average,
    seed_everything,
    seed_worker,
    train_model,
)
from instanseg.utils.AI_utils import Segmentation_Dataset, collate_fn
from instanseg.utils.metrics import compute_yolo_style_metrics
from instanseg.utils.model_loader import build_model_from_dict
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss, ProbabilityNet
from instanseg.utils.utils import _choose_device
from instanseg.utils.embedding_modes import embedding_vector_channels, validate_embedding_mode


class InstanSegModel:
    def __init__(
        self,
        checkpoint_or_model_type: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        verbosity: int = 1,
        **kwargs: Any,
    ):
        self.config_path = Path(config_path) if config_path is not None else None
        cfg, flat, config_path = load_config(self.config_path)
        self.config_path = config_path
        flat.update(kwargs)
        validate_embedding_mode(flat.get('embedding_mode', 'center-seed'))
        seed_everything(flat.get('seed', 42))

        self.config = flat
        self.model_name = flat.get('model_name', 'maskrcnn-resnet50_fpn')
        self.verbosity = verbosity
        self.device = torch.device(_choose_device(device, verbose=(verbosity != 0)))
        self.experiment_name = flat.get('experiment_name') or 'default_experiment'
        self.experiment_dir = Path('runs') / self.experiment_name
        self.checkpoint_path = None
        self.exported_model_path = None
        self.exported_model_format = None
        self.is_exported_model = False
        self.onnx_input_name = None
        self.onnx_output_names = None
        self._loss_method = None
        self.postprocessing_fn = None

        if checkpoint_or_model_type is not None:
            candidate = Path(checkpoint_or_model_type)
            if candidate.exists() and candidate.suffix.lower() in {'.pth', '.pt'}:
                export_format = self._get_export_format_from_metadata(candidate)
                if export_format:
                    self._load_exported_model(candidate, export_format)
                elif candidate.suffix.lower() == '.pt' and self._is_torchscript_file(candidate):
                    self._load_exported_model(candidate, 'torchscript')
                else:
                    self.checkpoint_path = candidate
            elif candidate.exists() and candidate.suffix.lower() == '.onnx':
                self._load_exported_model(candidate, 'onnx')
            elif candidate.exists() and candidate.suffix.lower() == '.xml':
                self._load_exported_model(candidate, 'openvino')
            elif checkpoint_or_model_type.lower().endswith(('.pth', '.pt')) and candidate.exists():
                self.checkpoint_path = candidate
            else:
                flat['model_name'] = checkpoint_or_model_type

        if self.is_exported_model:
            metadata = self._load_export_metadata(self.exported_model_path)
            if metadata:
                self.config.update(metadata)
                self.model_name = self.config.get('model_name', self.model_name)
            return

        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._apply_checkpoint_config(flat)

        self.model_name = flat.get('model_name', 'maskrcnn-resnet50_fpn')
        self.model = self._build_model(flat)

        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._load_checkpoint(self.checkpoint_path)

    def _apply_checkpoint_config(self, flat: Dict[str, Any]) -> None:
        config_candidates = []
        if self.checkpoint_path.parent.name == 'checkpoints':
            config_candidates.append(self.checkpoint_path.parent.parent / 'config.yaml')
        config_candidates.append(self.experiment_dir / 'config.yaml')

        for config_path in config_candidates:
            if not config_path.exists():
                continue
            with open(config_path, 'r', encoding='utf-8') as config_file:
                saved = yaml.safe_load(config_file) or {}
            if isinstance(saved, dict):
                flat.update(saved)
            return

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if checkpoint.get('model_name'):
                flat['model_name'] = checkpoint['model_name']
                return
            state = checkpoint.get('model_state_dict', checkpoint)
        else:
            state = checkpoint

        if any(key.startswith('encoder.') for key in state.keys()):
            flat['model_name'] = 'InstanSeg_UNet'
        elif any(key.startswith('backbone.') for key in state.keys()):
            flat['model_name'] = 'maskrcnn-resnet50_fpn'

    def _build_model(self, flat: Dict[str, Any]) -> torch.nn.Module:
        model_name = flat.get('model_name', 'maskrcnn-resnet50_fpn')
        num_classes = flat.get('num_classes', 2)
        imgsz = flat.get('imgsz', 512)
        dropout = flat.get('dropout', 0.0)

        if model_name == 'maskrcnn-resnet50_fpn':
            model = build_maskrcnn(num_classes, imgsz=imgsz)
            self.postprocessing_fn = None
            self._loss_method = None
        else:
            dim_in = int(flat.get('dim_in', 1) or 1)
            args_dict = {
                'model_str': model_name,
                'dropprob': dropout,
                'dim_in': dim_in,
                'dim_coords': 2,
                'n_sigma': 4,
                'norm': 'BATCH',
                'layers': [32, 64, 128, 256],
                'multihead': True,
                'dim_out': embedding_vector_channels(flat.get('embedding_mode', 'center-seed'), 2) + 4 + 1,
            }
            model = build_model_from_dict(args_dict)
            method = InstanSegLoss(
                binary_loss_fn_str='lovasz_hinge',
                seed_loss_fn='l1_distance',
                device=self.device,
                n_sigma=4,
                cells_and_nuclei=False,
                to_centre=False,
                window_size=256,
                tile_size=512,
                dim_coords=2,
                multi_centre=True,
                embedding_mode=flat.get('embedding_mode', 'center-seed'),
            )
            method.initialize_pixel_classifier(model, MLP_width=5)
            self._loss_method = method
            self.postprocessing_fn = method.postprocessing

        model.to(self.device)
        return model

    def _load_export_metadata(self, path: Path) -> Dict[str, Any]:
        metadata_path = path.with_suffix(path.suffix + '.json')
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                return json.load(metadata_file) or {}
        except Exception:
            return {}

    def _save_export_metadata(self, export_path: Path, metadata: Dict[str, Any]) -> None:
        metadata_path = export_path.with_suffix(export_path.suffix + '.json')
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file)

    def _is_torchscript_file(self, path: Path) -> bool:
        try:
            torch.jit.load(str(path), map_location='cpu')
            return True
        except Exception:
            return False

    def _get_export_format_from_metadata(self, path: Path) -> Optional[str]:
        metadata = self._load_export_metadata(path)
        if not metadata:
            return None
        export_format = metadata.get('export_format')
        return str(export_format).lower() if export_format else None

    def _load_exported_model(self, path: Path, format_: str) -> None:
        self.exported_model_path = path
        self.exported_model_format = format_.lower()
        self.is_exported_model = True

        metadata = self._load_export_metadata(path)
        if metadata:
            self.config.update(metadata)
            self.model_name = self.config.get('model_name', self.model_name)

        if self.exported_model_format == 'torchscript':
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message='The given buffer is not writable, and PyTorch does not support non-writable tensors.*',
                        category=UserWarning,
                    )
                    with open(str(path), 'rb') as model_file:
                        loaded = torch_export.load(model_file)
                self.model = loaded.module()
            except Exception:
                self.model = torch.jit.load(str(path), map_location=self.device)
            self.model.to(self.device)
        elif self.exported_model_format == 'onnx':
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    'ONNX runtime is required to load ONNX exports. Install onnxruntime.'
                ) from exc
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(path), providers=providers)
            self.model = session
            self.onnx_input_name = session.get_inputs()[0].name
            self.onnx_output_names = [output.name for output in session.get_outputs()]
        elif self.exported_model_format == 'openvino':
            try:
                import openvino.runtime as ov
            except ImportError as exc:
                raise ImportError(
                    'OpenVINO runtime is required to load OpenVINO exports. Install openvino.'
                ) from exc
            core = ov.Core()
            model = core.read_model(str(path))
            device_name = 'CPU'
            if self.device.type == 'cuda':
                device_name = 'GPU'
            self.model = core.compile_model(model, device_name)
        else:
            raise ValueError(
                f'Unsupported exported model format {self.exported_model_format}. '
                'Supported formats are onnx, torchscript, openvino.'
            )

        if self.model_name != 'maskrcnn-resnet50_fpn':
            self._initialize_exported_loss_method()

    def _initialize_exported_loss_method(self) -> None:
        if self._loss_method is not None:
            return
        method = InstanSegLoss(
            binary_loss_fn_str='lovasz_hinge',
            seed_loss_fn='l1_distance',
            device=self.device,
            n_sigma=4,
            cells_and_nuclei=False,
            to_centre=False,
            window_size=256,
            tile_size=512,
            dim_coords=2,
            multi_centre=True,
            embedding_mode=self.config.get('embedding_mode', 'center-seed'),
        )
        mlp_input_dim = method.feature_engineering_width + method.n_sigma - 2 + method.embedding_vector_dim
        method.pixel_classifier = ProbabilityNet(mlp_input_dim, width=5).to(self.device)
        self._loss_method = method
        self.postprocessing_fn = lambda prediction: method.postprocessing(
            prediction,
            classifier=method.pixel_classifier,
            device=self.device,
        )

    def _prepare_image_tensor(self, image: Union[str, Path, np.ndarray, torch.Tensor], imgsz: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f'Input image not found: {image_path}')
            if image_path.suffix.lower() in {'.tif', '.tiff'}:
                import tifffile

                image = tifffile.imread(str(image_path))
            else:
                import cv2

                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f'Unable to read image {image_path}')
        if isinstance(image, torch.Tensor):
            tensor = image.float()
        else:
            image = np.asarray(image)
            if image.ndim == 2:
                image = image[..., None]
            if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
                raise ValueError(f'Unsupported image channel count: {image.shape[2]}')
            tensor = torch.from_numpy(image).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3:
            raise ValueError(f'Unsupported image tensor shape {tensor.shape}. Expected HxW or HxWxC.')
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor = tensor / 255.0
        if imgsz is not None:
            imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
            tensor = torch.nn.functional.interpolate(tensor, size=imgsz, mode='bilinear', align_corners=False)
        return tensor

    def _get_export_input(self, imgsz: Union[int, Tuple[int, int]], batch: int, device: torch.device) -> torch.Tensor:
        size = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        channels = 3 if self.config.get('model_name', 'maskrcnn-resnet50_fpn') == 'maskrcnn-resnet50_fpn' else int(self.config.get('dim_in', 1) or 1)
        if channels == 0:
            channels = 3
        return torch.randn((batch, channels, size[0], size[1]), device=device)

    def _infer_tensor(self, tensor: torch.Tensor) -> Any:
        if self.exported_model_format == 'onnx':
            assert self.onnx_input_name is not None
            inputs = {self.onnx_input_name: tensor.cpu().numpy()}
            outputs = self.model.run(self.onnx_output_names, inputs)
            return outputs[0] if len(outputs) == 1 else outputs
        if self.exported_model_format == 'openvino':
            result = self.model([tensor.cpu().numpy()])
            return result[0] if len(result) == 1 else result
        if self.model_name == 'maskrcnn-resnet50_fpn':
            imgs = [tensor[i].to(self.device) for i in range(tensor.shape[0])]
            return self.model(imgs)
        return self.model(tensor)

    def _infer_single(self, image: Union[str, Path, np.ndarray, torch.Tensor], imgsz: Optional[Union[int, Tuple[int, int]]] = None) -> Any:
        tensor = self._prepare_image_tensor(image, imgsz=imgsz or self.config.get('imgsz', 512))
        result = self._infer_tensor(tensor)
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        return result

    def export(
        self,
        format: str = 'onnx',
        imgsz: Union[int, Tuple[int, int]] = 512,
        optimize: bool = False,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: Optional[int] = None,
        batch: int = 1,
        device: Optional[Union[str, int]] = None,
        data: Optional[str] = None,
        fraction: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Path:
        if self.is_exported_model:
            raise RuntimeError('Cannot export a model that was loaded from an exported file.')
        export_format = str(format).lower()
        if export_format not in {'onnx', 'torchscript', 'openvino'}:
            raise ValueError('Unsupported export format: ' + format)
        if half and int8:
            raise ValueError('half and int8 export options are mutually exclusive.')
        if int8:
            raise NotImplementedError(
                'INT8 quantization export is not implemented in this wrapper. '
                'Install onnxruntime and add calibration logic to support this feature.'
            )
        device_target = torch.device(_choose_device(device, verbose=False)) if device is not None else self.device
        if half and device_target.type == 'cpu':
            raise ValueError('half precision export is not supported on CPU-only exports.')

        self.model.eval()
        model_to_export = self.model
        if half:
            model_to_export = model_to_export.half()

        if output_path is None:
            output_dir = Path.cwd()
            output_path = output_dir / f'instanseg_export.{export_format if export_format != "openvino" else "xml"}'
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if export_format == 'torchscript':
            example_input = self._get_export_input(imgsz, batch, device_target)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='The given buffer is not writable, and PyTorch does not support non-writable tensors.*',
                    category=UserWarning,
                )
                exported_program = torch_export.export(model_to_export, (example_input,))
                with open(str(output_path), 'wb') as output_file:
                    torch_export.save(exported_program, output_file)
        elif export_format == 'onnx':
            if self.model_name == 'maskrcnn-resnet50_fpn':
                raise RuntimeError('ONNX export for Mask R-CNN models is not supported by the current export helper.')
            example_input = self._get_export_input(imgsz, batch, device_target)
            dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic else None
            try:
                torch.onnx.export(
                    model_to_export,
                    example_input,
                    str(output_path),
                    opset_version=opset,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=optimize,
                )
            except ModuleNotFoundError as exc:
                if 'onnxscript' in str(exc) or 'onnx' in str(exc):
                    raise ImportError(
                        'ONNX export requires onnxscript and onnx to be installed. '
                        'Install onnxscript and onnx to enable ONNX export.'
                    ) from exc
                raise
            if simplify:
                try:
                    import onnx
                    from onnxsim import simplify as simplify_onnx

                    onnx_model = onnx.load(str(output_path))
                    simplified_model, check = simplify_onnx(onnx_model)
                    if check:
                        onnx.save(simplified_model, str(output_path))
                except ImportError:
                    warnings.warn('onnxsim is not installed; ONNX simplification was skipped.')
        else:
            if self.model_name == 'maskrcnn-resnet50_fpn':
                raise RuntimeError('OpenVINO export for Mask R-CNN models is not supported by the current export helper.')
            onnx_path = output_path.with_suffix('.onnx')
            example_input = self._get_export_input(imgsz, batch, device_target)
            try:
                torch.onnx.export(
                    model_to_export,
                    example_input,
                    str(onnx_path),
                    opset_version=opset,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic else None,
                    do_constant_folding=optimize,
                )
            except ModuleNotFoundError as exc:
                if 'onnxscript' in str(exc) or 'onnx' in str(exc):
                    raise ImportError(
                        'ONNX export requires onnxscript and onnx to be installed. '
                        'Install onnxscript and onnx to enable OpenVINO export.'
                    ) from exc
                raise
            try:
                import openvino.runtime as ov
            except ImportError as exc:
                raise ImportError('OpenVINO runtime is required to export to OpenVINO. Install openvino.') from exc
            core = ov.Core()
            ov_model = core.read_model(str(onnx_path))
            compiled = core.compile_model(ov_model, 'CPU')
            output_path = output_path.with_suffix('.xml')
            ov.serialize(ov_model, str(output_path), str(output_path.with_suffix('.bin')))

        metadata = {
            'model_name': self.model_name,
            'imgsz': imgsz,
            'export_format': export_format,
            'batch': batch,
            'half': half,
            'dynamic': dynamic,
        }
        self._save_export_metadata(output_path, metadata)
        return output_path

    def __call__(self, image: Union[str, Path, np.ndarray, torch.Tensor], imgsz: Optional[Union[int, Tuple[int, int]]] = None) -> Any:
        return self._infer_single(image, imgsz=imgsz)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.checkpoint_path = checkpoint_path

    def train(self, epochs: Optional[int] = None, imgsz: Optional[int] = None, resume: bool = False, **kwargs: Any) -> Any:
        if self.is_exported_model:
            raise RuntimeError('Cannot train a model that was loaded from an exported file.')
        if epochs is not None:
            kwargs['epochs'] = epochs
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        kwargs['resume'] = resume

        if self.checkpoint_path is not None:
            kwargs['checkpoint_path'] = str(self.checkpoint_path)

        self.config.update(kwargs)
        validate_embedding_mode(self.config.get('embedding_mode', 'center-seed'))
        seed_everything(self.config.get('seed', 42))
        self.experiment_name = self.config.get('experiment_name') or self.experiment_name
        self.experiment_dir = Path('runs') / self.experiment_name
        model, metrics_history = train_model(config_path=self.config_path, overrides=self.config, checkpoint_path=self.config.get('checkpoint_path'))
        self.model = model
        if self._loss_method is not None:
            self._loss_method.initialize_pixel_classifier(self.model, MLP_width=5)
        if self.experiment_dir.exists():
            best_model = self.experiment_dir / 'checkpoints' / 'best.pt'
            if best_model.exists():
                self._load_checkpoint(best_model)
        return metrics_history

    def eval(
        self,
        data_dir: Optional[str] = None,
        subset: str = 'test',
        imgsz: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        use_best: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        config = dict(self.config)
        if data_dir is not None:
            config['data_dir'] = data_dir
        if imgsz is not None:
            config['imgsz'] = imgsz
        if batch_size is not None:
            config['batch_size'] = batch_size
        if num_workers is not None:
            config['num_workers'] = num_workers
        if kwargs:
            config.update(kwargs)

        if self.is_exported_model:
            checkpoint_path = None
        elif checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        if checkpoint_path is None and use_best and not self.is_exported_model:
            checkpoint_path = str(Path('runs') / self.experiment_name / 'checkpoints' / 'best.pt')
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self._load_checkpoint(Path(checkpoint_path))

        seed_everything(config.get('seed', 42))
        data_dir = Path(config['data_dir'])

        if (data_dir / 'train').exists() and (data_dir / 'test').exists():
            if subset == 'test':
                image_paths, mask_paths = collect_image_mask_pairs(data_dir / 'test')
                selected_pairs = list(zip(image_paths, mask_paths))
            else:
                train_images, train_masks = collect_image_mask_pairs(data_dir / 'train')
                if not train_images:
                    raise FileNotFoundError(f"No image/mask pairs found in data_dir/train: {data_dir / 'train'}")
                pairs = list(zip(train_images, train_masks))
                pairs.sort(key=lambda x: x[0].name)
                num_val = int(round(len(pairs) * config['val_ratio']))
                val_pairs = pairs[:num_val]
                train_pairs = pairs[num_val:]
                subset_map = {
                    'train': train_pairs,
                    'val': val_pairs,
                    'validation': val_pairs,
                    'test': [],
                }
                if subset not in subset_map:
                    raise ValueError(f"Unknown subset {subset}. Choose from train, val, validation, test.")
                selected_pairs = subset_map[subset]
        else:
            image_paths, mask_paths = collect_image_mask_pairs(data_dir)
            if not image_paths:
                raise FileNotFoundError(f"No image/mask pairs found in data_dir: {data_dir}")
            pairs = list(zip(image_paths, mask_paths))
            pairs.sort(key=lambda x: x[0].name)
            N = len(pairs)
            val_ratio = config['val_ratio']
            test_ratio = config['test_ratio']
            num_val = int(round(N * val_ratio))
            num_test = int(round(N * test_ratio))
            val_pairs = pairs[:num_val]
            test_pairs = pairs[num_val:num_val + num_test]
            train_pairs = pairs[num_val + num_test:]
            subset_map = {
                'train': train_pairs,
                'val': val_pairs,
                'validation': val_pairs,
                'test': test_pairs,
            }
            if subset not in subset_map:
                raise ValueError(f"Unknown subset {subset}. Choose from train, val, validation, test.")
            selected_pairs = subset_map[subset]

        if len(selected_pairs) == 0:
            raise ValueError(f"No samples in subset '{subset}' for evaluation.")

        imgsz = config['imgsz']
        dataset = Segmentation_Dataset(tuple(zip(*selected_pairs)), common_transforms=False, imgsz=imgsz)
        bs = batch_size or config['batch_size']
        workers = num_workers if num_workers is not None else config.get('num_workers', 4)
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=bs,
            shuffle=False,
            pin_memory=self.device.type == 'cuda',
            num_workers=workers,
            persistent_workers=workers > 0,
            worker_init_fn=seed_worker,
        )

        self.model.eval()
        test_loss_sum = 0.0
        test_count = 0
        all_pred = []
        all_gt = []

        with torch.no_grad():
            for images, labels, _ in loader:
                if self.model_name == 'maskrcnn-resnet50_fpn':
                    images = [img.to(self.device, non_blocking=True) / 255.0 for img in images]
                    targets = get_maskrcnn_targets(labels, self.device)
                    loss_dict = self.model(images, targets)
                    loss = sum(l for l in loss_dict.values())
                    test_loss_sum += loss.item() * len(images)
                    predictions = self.model(images)
                    for i, prediction in enumerate(predictions):
                        H, W = images[i].shape[1:]
                        pred_lbl = maskrcnn_to_labeled_mask(prediction, (H, W))
                        all_pred.append(pred_lbl)
                        all_gt.append(labels[i].cpu().numpy().squeeze())
                else:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    output = self.model(images)
                    loss_fn = self._loss_fn()
                    loss = loss_fn(output, labels).mean()
                    test_loss_sum += loss.item() * len(images)
                    predicted = torch.stack([self.postprocessing_fn(out) for out in output])
                    for i in range(len(images)):
                        all_pred.append(predicted[i].cpu().numpy())
                        all_gt.append(labels[i].cpu().numpy().squeeze())
                test_count += images[0].shape[0] if isinstance(images, list) else images.shape[0]

        metrics = compute_yolo_style_metrics(all_gt, all_pred)
        metrics['loss'] = safe_average(test_loss_sum, test_count)
        metrics_path = Path('runs') / self.experiment_name / f'{subset}_metrics.csv'
        from pandas import DataFrame

        DataFrame([metrics]).to_csv(metrics_path, index=False)
        return metrics

    def _loss_fn(self) -> Any:
        if self._loss_method is None:
            raise RuntimeError('Loss function is not available for this model type.')
        self._loss_method.initialize_pixel_classifier(self.model, MLP_width=5)
        return self._loss_method
