import os

os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from instanseg.utils.embedding_modes import (
    embedding_vector_channels,
    validate_embedding_mode,
)


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            torch.set_deterministic(True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from instanseg.utils.AI_utils import Segmentation_Dataset, collate_fn
from instanseg.utils.metrics import compute_yolo_style_metrics
from instanseg.utils.kaggle_checkpointing import (
    KaggleCheckpointManager,
    validate_kaggle_checkpoint_config,
)


DEFAULT_CONFIG = {
    'seed': 42,
    'data_dir': '../data',
    'model_name': 'maskrcnn-resnet50_fpn',
    'experiment_name': None,
    'train_data_ratio': 1.0,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'epochs': 10,
    'batch_size': 4,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'num_classes': 2,
    'val_interval': 1,
    'patience': 5,
    'resume': False,
    'freeze_backbone_epochs': 0,
    'backbone_lr_mult': 1.0,
    'imgsz': 512,
    'warmup_epochs': 2,
    'dropout': 0.0,
    'num_workers': 4,
    'embedding_mode': 'center-seed',
    'save_kgl_ckp': False,
    'kgl_best_ckp_path': None,
    'kgl_last_ckp_path': None,
    'kgl_ckp_freq': 1,
    'kgl_creds_path': None,
}


class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
    def log(self, message):
        print(message)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def load_config(config_path=None):
    config_path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[2] / 'config.yaml'
    if not config_path.exists():
        fallback_path = Path('config.yaml')
        if fallback_path.exists():
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Could not find config file: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
        
    flat_cfg = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat_cfg.update(section)
            
    for k, v in DEFAULT_CONFIG.items():
        if k not in flat_cfg:
            flat_cfg[k] = v
            
    # Normalize resume parameter if it is a string representation of boolean
    resume_val = flat_cfg.get('resume', False)
    if isinstance(resume_val, str):
        if resume_val.lower() in ('true', 'yes', '1'):
            flat_cfg['resume'] = True
        elif resume_val.lower() in ('false', 'no', '0'):
            flat_cfg['resume'] = False

    tdr = flat_cfg['train_data_ratio']
    if not isinstance(tdr, (int, float)) or tdr <= 0.0 or tdr > 1.0:
        raise ValueError(f"train_data_ratio must be greater than 0.0 and at most 1.0, but got {tdr}")

    val_ratio = flat_cfg['val_ratio']
    test_ratio = flat_cfg['test_ratio']
    if not isinstance(val_ratio, (int, float)) or not isinstance(test_ratio, (int, float)):
        raise ValueError("val_ratio and test_ratio must be numeric")
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio and test_ratio must be non-negative and sum to less than 1.0, "
            f"but got val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    positive_int_keys = ('epochs', 'batch_size', 'num_classes', 'val_interval', 'patience', 'imgsz')
    for key in positive_int_keys:
        if not isinstance(flat_cfg[key], int) or flat_cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer, but got {flat_cfg[key]}")
    if not isinstance(flat_cfg['num_workers'], int) or flat_cfg['num_workers'] < 0:
        raise ValueError(f"num_workers must be a non-negative integer, but got {flat_cfg['num_workers']}")
    if not isinstance(flat_cfg['warmup_epochs'], int) or flat_cfg['warmup_epochs'] < 0:
        raise ValueError(f"warmup_epochs must be a non-negative integer, but got {flat_cfg['warmup_epochs']}")
    validate_embedding_mode(flat_cfg['embedding_mode'])

    save_kgl_ckp = flat_cfg.get('save_kgl_ckp', False)
    if isinstance(save_kgl_ckp, str):
        save_kgl_ckp = save_kgl_ckp.lower() in ('true', 'yes', '1')
    flat_cfg['save_kgl_ckp'] = bool(save_kgl_ckp)

    kgl_ckp_freq = flat_cfg.get('kgl_ckp_freq', 1)
    if not isinstance(kgl_ckp_freq, int) or kgl_ckp_freq <= 0:
        raise ValueError(f"kgl_ckp_freq must be a positive integer, but got {kgl_ckp_freq}")
    flat_cfg['kgl_ckp_freq'] = kgl_ckp_freq

    if flat_cfg['save_kgl_ckp']:
        validate_kaggle_checkpoint_config(flat_cfg)
        
    return cfg, flat_cfg, config_path


def _collect_image_mask_pairs_from_dir(data_dir):
    data_dir = Path(data_dir)
    image_paths = []
    mask_paths = []

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    def find_mask(img_p, m_dir):
        mask_p = m_dir / img_p.name
        if mask_p.exists():
            return mask_p
        for old_suffix, new_suffix in [("_img", "_masks"), ("_img", "_mask"), ("_img", "_label"), ("_img", "_labels"), ("_image", "_mask"), ("_image", "_masks")]:
            if old_suffix in img_p.name:
                candidate = m_dir / img_p.name.replace(old_suffix, new_suffix)
                if candidate.exists():
                    return candidate
        stem = img_p.stem
        candidates = []
        for suffix in ["_img", "_image"]:
            if stem.endswith(suffix):
                base_stem = stem[:-len(suffix)]
                candidates.extend(m_dir.glob(f"{base_stem}*"))
        candidates.extend(m_dir.glob(f"{stem}*"))
        if candidates:
            return sorted(candidates)[0]
        return None

    if images_dir.exists() and masks_dir.exists():
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.tif'):
                mask_p = find_mask(f, masks_dir)
                if mask_p is not None:
                    image_paths.append(f)
                    mask_paths.append(mask_p)
        return image_paths, mask_paths

    for root, dirs, files in os.walk(data_dir):
        if Path(root).name == "images":
            mask_dir = Path(root).parent / "masks"
            if mask_dir.exists():
                for f in sorted(files):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                        img_p = Path(root) / f
                        mask_p = find_mask(img_p, mask_dir)
                        if mask_p is not None:
                            image_paths.append(img_p)
                            mask_paths.append(mask_p)

    return image_paths, mask_paths


def collect_image_mask_pairs(data_dir, subset=None):
    data_dir = Path(data_dir)
    if subset is not None and subset in {'train', 'val', 'validation', 'test'}:
        explicit_dir = data_dir / subset
        if explicit_dir.exists():
            return _collect_image_mask_pairs_from_dir(explicit_dir)

    if (data_dir / 'train').exists() and (data_dir / 'test').exists():
        if subset == 'test':
            return _collect_image_mask_pairs_from_dir(data_dir / 'test')
        if subset in {'train', 'val', 'validation'}:
            return _collect_image_mask_pairs_from_dir(data_dir / 'train')
        train_images, train_masks = _collect_image_mask_pairs_from_dir(data_dir / 'train')
        test_images, test_masks = _collect_image_mask_pairs_from_dir(data_dir / 'test')
        return train_images + test_images, train_masks + test_masks

    return _collect_image_mask_pairs_from_dir(data_dir)


def set_backbone_requires_grad(model, requires_grad):
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = requires_grad
    elif hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = requires_grad


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def get_optimizer_params(model, base_lr, backbone_lr_mult):
    if backbone_lr_mult == 1.0:
        return model.parameters()
        
    backbone_params = []
    other_params = []
    
    backbone_modules = []
    if hasattr(model, 'encoder'):
        backbone_modules = [model.encoder]
    elif hasattr(model, 'backbone'):
        backbone_modules = [model.backbone]
        
    backbone_param_ids = set()
    for m in backbone_modules:
        for p in m.parameters():
            backbone_param_ids.add(id(p))
            
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in backbone_param_ids:
            backbone_params.append(p)
        else:
            other_params.append(p)
            
    return [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": other_params, "lr": base_lr}
    ]


def create_optimizer(model, lr, weight_decay, momentum, backbone_lr_mult):
    params = get_optimizer_params(model, lr, backbone_lr_mult)
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, backbone_lr_mult):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
        lr = base_lr * factor
    else:
        lr = base_lr

    for i, param_group in enumerate(optimizer.param_groups):
        if len(optimizer.param_groups) == 2 and i == 0:
            param_group['lr'] = lr * backbone_lr_mult
        else:
            param_group['lr'] = lr


def get_maskrcnn_targets(labels_batch, device):
    targets = []
    for i in range(len(labels_batch)):
        mask = labels_batch[i]
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.to(device)
            
        unique_ids = torch.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        boxes = []
        labels = []
        masks = []
        
        for uid in unique_ids:
            binary_mask = (mask == uid)
            pos = torch.where(binary_mask)
            if len(pos[0]) == 0:
                continue
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                labels.append(1)
                masks.append(binary_mask.to(dtype=torch.uint8))
                
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            labels_t = torch.as_tensor(labels, dtype=torch.int64, device=device)
            masks_t = torch.stack(masks).to(device=device)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_t = torch.zeros((0,), dtype=torch.int64, device=device)
            masks_t = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8, device=device)
            
        targets.append({
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t
        })
    return targets


def maskrcnn_to_labeled_mask(pred_dict, shape, threshold=0.5, score_threshold=0.5):
    H, W = shape
    labeled_mask = np.zeros((H, W), dtype=np.int16)
    
    masks = pred_dict['masks']
    scores = pred_dict['scores']
    
    if len(masks) == 0:
        return labeled_mask
        
    masks = masks.cpu().numpy()
    scores = scores.cpu().numpy()
    
    sorted_idxs = np.argsort(scores)
    
    inst_id = 1
    for idx in sorted_idxs:
        score = scores[idx]
        if score < score_threshold:
            continue
        mask = masks[idx, 0]
        binary_mask = mask > threshold
        labeled_mask[binary_mask] = inst_id
        inst_id += 1
        
    return labeled_mask


def safe_average(total, count):
    return total / count if count > 0 else 0.0


def build_maskrcnn(num_classes, imgsz=512):
    return maskrcnn_resnet50_fpn(
        num_classes=num_classes,
        weights=None,
        weights_backbone=None,
        min_size=imgsz,
        max_size=imgsz,
    )


def train_model(config_path=None, overrides=None, checkpoint_path=None, embedding_mode=None):
    _, flat, config_path = load_config(config_path)
    if overrides is not None:
        flat.update(overrides)
    if embedding_mode is not None:
        flat['embedding_mode'] = embedding_mode
    validate_embedding_mode(flat['embedding_mode'])
    
    # Setup directories
    experiment_name = flat['experiment_name'] or 'default_experiment'
    experiment_dir = Path("runs") / experiment_name
    checkpoints_dir = experiment_dir / 'checkpoints'
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(experiment_dir / 'train.log')
    logger.log("="*50)
    logger.log(f"Starting experiment: {experiment_name}")
    logger.log("="*50)
    
    # Log effective config dict after overrides
    logger.log("Configuration settings:")
    logger.log(yaml.dump(flat, default_flow_style=False))
    
    # Set seed
    seed = flat['seed']
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_dir = Path(flat['data_dir'])
    val_ratio = flat['val_ratio']
    test_ratio = flat['test_ratio']
    train_data_ratio = flat['train_data_ratio']

    if (data_dir / 'train').exists() and (data_dir / 'test').exists():
        train_images, train_masks = collect_image_mask_pairs(data_dir / 'train')
        test_images, test_masks = collect_image_mask_pairs(data_dir / 'test')

        if not train_images:
            raise FileNotFoundError(f"No image/mask pairs found in data_dir/train: {data_dir / 'train'}")
        if not test_images:
            raise FileNotFoundError(f"No image/mask pairs found in data_dir/test: {data_dir / 'test'}")

        logger.log("Explicit train/test folder layout detected; configured test_ratio will be ignored.")
        train_pairs = list(zip(train_images, train_masks))
        train_pairs.sort(key=lambda x: x[0].name)
        random.Random(seed).shuffle(train_pairs)

        num_val = int(round(len(train_pairs) * val_ratio))
        val_pairs = train_pairs[:num_val]
        train_pairs = train_pairs[num_val:]

        if train_data_ratio < 1.0:
            num_train_subset = max(1, int(round(len(train_pairs) * train_data_ratio)))
            train_pairs = train_pairs[:num_train_subset]

        val_images, val_masks = zip(*val_pairs) if len(val_pairs) > 0 else ([], [])
        train_images, train_masks = zip(*train_pairs) if len(train_pairs) > 0 else ([], [])
        test_images = list(test_images)
        test_masks = list(test_masks)
        N = len(train_images) + len(val_images) + len(test_images)

        logger.log("Dataset split summary:")
        logger.log(f"  Total samples: {N} (train/val from data_dir/train, test from data_dir/test)")
        logger.log(f"  Train samples: {len(train_images)}")
        logger.log(f"  Val samples:   {len(val_images)}")
        logger.log(f"  Test samples:  {len(test_images)}")
    else:
        image_paths, mask_paths = collect_image_mask_pairs(data_dir)
        if not image_paths:
            raise FileNotFoundError(f"No image/mask pairs found in data_dir: {data_dir}")

        pairs = list(zip(image_paths, mask_paths))
        pairs.sort(key=lambda x: x[0].name)
        random.Random(seed).shuffle(pairs)

        N = len(pairs)
        num_val = int(round(N * val_ratio))
        num_test = int(round(N * test_ratio))
        val_pairs = pairs[:num_val]
        test_pairs = pairs[num_val:num_val + num_test]
        train_pairs = pairs[num_val + num_test:]

        if train_data_ratio < 1.0:
            num_train_subset = max(1, int(round(len(train_pairs) * train_data_ratio)))
            train_pairs = train_pairs[:num_train_subset]

        train_images, train_masks = zip(*train_pairs) if len(train_pairs) > 0 else ([], [])
        val_images, val_masks = zip(*val_pairs) if len(val_pairs) > 0 else ([], [])
        test_images, test_masks = zip(*test_pairs) if len(test_pairs) > 0 else ([], [])

        logger.log("Dataset split summary:")
        logger.log(f"  Total samples: {N}")
        logger.log(f"  Train samples: {len(train_images)}")
        logger.log(f"  Val samples:   {len(val_images)}")
        logger.log(f"  Test samples:  {len(test_images)}")

    if len(train_images) == 0:
        raise ValueError("Training split is empty. Reduce val_ratio/test_ratio or provide more data.")

    imgsz = flat['imgsz']
    train_dataset = Segmentation_Dataset(
        (train_images, train_masks),
        common_transforms=True,
        imgsz=imgsz,
        augmentation_seed=seed,
    )
    val_dataset = Segmentation_Dataset((val_images, val_masks), common_transforms=False, imgsz=imgsz)
    test_dataset = Segmentation_Dataset((test_images, test_masks), common_transforms=False, imgsz=imgsz)

    batch_size = flat['batch_size']
    num_workers = flat['num_workers']
    pin_memory = device.type == 'cuda'
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(seed + 2)

    train_workers = max(0, num_workers)
    val_workers = max(0, num_workers // 2)
    test_workers = max(0, num_workers // 2)

    # Restore the train generator state from checkpoint when resuming.
    resume = flat['resume']
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    else:
        checkpoint_path = None
    if resume and checkpoint_path is None and checkpoints_dir.exists():
        last_pt = checkpoints_dir / 'last.pt'
        if last_pt.exists():
            checkpoint_path = last_pt
        else:
            pt_files = list(checkpoints_dir.glob("*.pt"))
            if pt_files:
                pt_files.sort(key=lambda x: x.stat().st_mtime)
                checkpoint_path = pt_files[-1]

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'train_generator_state' in checkpoint:
                train_generator.set_state(checkpoint['train_generator_state'])
        except Exception:
            logger.log(f"Warning: could not restore train generator state from checkpoint {checkpoint_path}.")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=train_workers,
        persistent_workers=train_workers > 0,
        worker_init_fn=seed_worker,
        prefetch_factor=2 if train_workers > 0 else None,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=val_workers,
        persistent_workers=val_workers > 0,
        worker_init_fn=seed_worker,
        prefetch_factor=2 if val_workers > 0 else None,
        generator=val_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=test_workers,
        persistent_workers=test_workers > 0,
        worker_init_fn=seed_worker,
        prefetch_factor=2 if test_workers > 0 else None,
        generator=test_generator,
    )

    with open(experiment_dir / 'config.yaml', 'w', encoding='utf-8') as config_out:
        config_out.write(yaml.safe_dump(flat, sort_keys=False))

    logger.log("Data loaded successfully")
    logger.log("Augmentation pipeline initialized")
    
    # Initialize Model
    model_name = flat['model_name']
    num_classes = flat['num_classes']
    dropout = flat['dropout']
    embedding_mode = flat['embedding_mode']
    
    resume = flat['resume']
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Check model architecture
    if model_name == 'maskrcnn-resnet50_fpn':
        model = build_maskrcnn(num_classes, imgsz=flat['imgsz'])
        postprocessing_fn = None
        loss_fn = None
    else:
        # InstanSeg model loader
        from instanseg.utils.model_loader import build_model_from_dict
        from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss
        
        embedding_vector_dim = embedding_vector_channels(embedding_mode, dim_coords=2)
        args_dict = {
            'model_str': model_name,
            'dropprob': dropout,
            'dim_in': 1,
            'dim_coords': 2,
            'n_sigma': 4,
            'norm': 'BATCH',
            'layers': [32, 64, 128, 256],
            'multihead': True,
            'dim_out': embedding_vector_dim + 4 + 1
        }
        model = build_model_from_dict(args_dict)
        
        # Check pixel classifier initialization
        from instanseg.utils.loss.instanseg_loss import has_pixel_classifier_model
        method = InstanSegLoss(
            binary_loss_fn_str="lovasz_hinge",
            seed_loss_fn="l1_distance",
            device=device,
            n_sigma=4,
            cells_and_nuclei=False,
            to_centre=False,
            window_size=256,
            tile_size=512,
            dim_coords=2,
            multi_centre=True,
            embedding_mode=embedding_mode,
        )
        if not has_pixel_classifier_model(model):
            model = method.initialize_pixel_classifier(model, MLP_width=5)
            
        def loss_fn(*args, **kwargs):
            return method.forward(*args, **kwargs)
        postprocessing_fn = method.postprocessing

    model.to(device)
    
    # Setup optimizer and lr scheduler parameters
    learning_rate = flat['learning_rate']
    weight_decay = flat['weight_decay']
    momentum = flat['momentum']
    backbone_lr_mult = flat['backbone_lr_mult']
    epochs = flat['epochs']
    patience = flat['patience']
    val_interval = flat['val_interval']
    warmup_epochs = flat['warmup_epochs']
    freeze_backbone_epochs = flat['freeze_backbone_epochs']
    
    no_improve_epochs = 0
    
    # Resume functionality
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    else:
        checkpoint_path = None
    if resume and checkpoint_path is None and checkpoints_dir.exists():
        last_pt = checkpoints_dir / 'last.pt'
        if last_pt.exists():
            checkpoint_path = last_pt
        else:
            pt_files = list(checkpoints_dir.glob("*.pt"))
            if pt_files:
                pt_files.sort(key=lambda x: x.stat().st_mtime)
                checkpoint_path = pt_files[-1]

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer = create_optimizer(model, learning_rate, weight_decay, momentum, backbone_lr_mult)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                optimizer = create_optimizer(model, learning_rate, weight_decay, momentum, backbone_lr_mult)
                logger.log("Resume checkpoint did not contain optimizer state; optimizer reinitialized.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', -1)
            no_improve_epochs = checkpoint.get('no_improve_epochs', 0)

            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if 'numpy_rng_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if 'random_rng_state' in checkpoint:
                random.setstate(checkpoint['random_rng_state'])
            if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        else:
            model.load_state_dict(checkpoint)
            optimizer = create_optimizer(model, learning_rate, weight_decay, momentum, backbone_lr_mult)
            start_epoch = 0
            logger.log("Loaded plain state_dict checkpoint; optimizer initialized from scratch.")

        # Set backbone freeze state to match the resumed epoch before creating optimizer
        resumed_backbone_frozen = (start_epoch < freeze_backbone_epochs)
        set_backbone_requires_grad(model, not resumed_backbone_frozen)
        
        logger.log(f"Model loaded: resumed training from checkpoint {checkpoint_path} at epoch {start_epoch+1}")
    else:
        optimizer = create_optimizer(model, learning_rate, weight_decay, momentum, backbone_lr_mult)
        if resume:
            logger.log(f"Warning: resume is True but no checkpoint was found in {checkpoints_dir}. Training from scratch.")
        logger.log("Model initialized: training from scratch")
        
    on_cluster = False # Enable tqdm progress bars by default
    metrics_history = []
    kaggle_checkpoint_manager = KaggleCheckpointManager(flat, experiment_dir, logger=logger)
    
    # Load existing metrics history if resuming
    metrics_csv_path = experiment_dir / 'metrics.csv'
    if resume and metrics_csv_path.exists():
        try:
            metrics_history = pd.read_csv(metrics_csv_path).to_dict('records')
        except Exception:
            metrics_history = []
    
    # Main training loop
    for epoch in range(start_epoch, epochs):
        logger.log(f"\nEPOCH {epoch+1}/{epochs}")
        
        # 1. Warmup
        adjust_learning_rate(optimizer, epoch, warmup_epochs, learning_rate, backbone_lr_mult)
        
        # 2. Backbone freezing
        is_backbone_frozen = (epoch < freeze_backbone_epochs)
        set_backbone_requires_grad(model, not is_backbone_frozen)
        
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            set_backbone_requires_grad(model, True)
            optimizer = create_optimizer(model, learning_rate, weight_decay, momentum, backbone_lr_mult)
            
        model.train()
        set_batchnorm_eval(model)
        train_loss_sum = 0
        train_count = 0
        
        # Train progress bar
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", disable=on_cluster)
        for batch_idx, (image_batch, labels_batch, _) in enumerate(train_bar):
            optimizer.zero_grad()
            
            if model_name == 'maskrcnn-resnet50_fpn':
                images_list = [img.to(device, non_blocking=pin_memory) / 255.0 for img in image_batch]
                targets = get_maskrcnn_targets(labels_batch, device)
                loss_dict = model(images_list, targets)
                loss = sum(l for l in loss_dict.values())
            else:
                image_batch = image_batch.to(device)
                labels_batch = labels_batch.to(device)
                output = model(image_batch)
                loss = loss_fn(output, labels_batch).mean()
                
            loss.backward()
            if model_name != 'maskrcnn-resnet50_fpn':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            
            train_loss_sum += loss.item() * len(image_batch)
            train_count += len(image_batch)
            train_bar.set_postfix(loss=loss.item())
            
        train_loss = safe_average(train_loss_sum, train_count)
        logger.log(f"Average train loss: {train_loss:.4f}")
        
        # 3. Validation step
        if (epoch + 1) % val_interval == 0:
            logger.log("Starting validation...")
            model.eval()
            val_loss_sum = 0
            val_count = 0
            all_gt_masks = []
            all_pred_masks = []
            
            val_bar = tqdm(val_loader, desc="Validation", disable=on_cluster)
            with torch.no_grad():
                for image_batch, labels_batch, _ in val_bar:
                    if model_name == 'maskrcnn-resnet50_fpn':
                        model.train()
                        set_batchnorm_eval(model)
                        images_list = [img.to(device, non_blocking=pin_memory) / 255.0 for img in image_batch]
                        targets = get_maskrcnn_targets(labels_batch, device)
                        loss_dict = model(images_list, targets)
                        loss = sum(l for l in loss_dict.values())
                        val_loss_sum += loss.item() * len(image_batch)
                        
                        model.eval()
                        predictions = model(images_list)
                        for i in range(len(image_batch)):
                            H, W = image_batch[i].shape[1:]
                            pred_lbl = maskrcnn_to_labeled_mask(predictions[i], (H, W))
                            gt_lbl = labels_batch[i].cpu().numpy().squeeze()
                            all_pred_masks.append(pred_lbl)
                            all_gt_masks.append(gt_lbl)
                    else:
                        image_batch = image_batch.to(device)
                        labels_batch = labels_batch.to(device)
                        output = model(image_batch)
                        loss = loss_fn(output, labels_batch).mean()
                        val_loss_sum += loss.item() * len(image_batch)
                        
                        # Extract postprocessing parameters from config with defaults
                        postproc_params = {
                            'mask_threshold': flat.get('mask_threshold', 0.53),
                            'seed_threshold': flat.get('seed_threshold', 0.5),
                            'peak_distance': int(flat.get('peak_distance', 5)),
                            'overlap_threshold': flat.get('overlap_threshold', 0.3),
                            'mean_threshold': flat.get('mean_threshold', 0.1),
                            'min_size': int(flat.get('min_size', 10)),
                        }
                        
                        # Apply parameters during validation postprocessing
                        predicted_labels = torch.stack([
                            postprocessing_fn(out, **postproc_params) 
                            for out in output
                        ])
                        for i in range(len(image_batch)):
                            pred_lbl = predicted_labels[i].cpu().numpy()
                            gt_lbl = labels_batch[i].cpu().numpy().squeeze()
                            all_pred_masks.append(pred_lbl)
                            all_gt_masks.append(gt_lbl)
                            
                    val_count += len(image_batch)
                    
            val_loss = safe_average(val_loss_sum, val_count)
            val_metrics = compute_yolo_style_metrics(all_gt_masks, all_pred_masks)
            
            logger.log(f"Validation loss: {val_loss:.4f}")
            logger.log(f"Validation Quality Metrics:")
            logger.log(f"  Precision: {val_metrics['precision']:.4f}")
            logger.log(f"  Recall:    {val_metrics['recall']:.4f}")
            logger.log(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
            logger.log(f"  AP@50:     {val_metrics['ap50']:.4f}")
            logger.log(f"  AP@75:     {val_metrics['ap75']:.4f}")
            logger.log(f"  AP@50-95:  {val_metrics['ap50_95']:.4f}")
            
            # Save metrics to metrics.csv
            metrics_row = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'accuracy': val_metrics['accuracy'],
                'ap50': val_metrics['ap50'],
                'ap75': val_metrics['ap75'],
                'ap50_95': val_metrics['ap50_95']
            }
            metrics_history.append(metrics_row)
            pd.DataFrame(metrics_history).to_csv(experiment_dir / 'metrics.csv', index=False)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_epochs = 0
                is_best = True
            else:
                no_improve_epochs += 1
                is_best = False
                
            # Checkpoint saving
            checkpoint = {
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'no_improve_epochs': no_improve_epochs,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'train_generator_state': train_generator.get_state(),
            }
            if torch.cuda.is_available():
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                
            torch.save(checkpoint, checkpoints_dir / 'last.pt')
            torch.save(checkpoint, experiment_dir / 'last.pt')
            
            if is_best:
                torch.save(checkpoint, checkpoints_dir / 'best.pt')
                torch.save(checkpoint, experiment_dir / 'best.pt')
                logger.log(f"New best model saved to {checkpoints_dir / 'best.pt'} with val loss: {best_val_loss:.4f}")
            else:
                logger.log(f"Val loss did not improve. Early stopping: {no_improve_epochs}/{patience}")

            kaggle_checkpoint_manager.save_checkpoints(checkpoint, epoch, is_best)
                
            if no_improve_epochs >= patience:
                logger.log(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # 4. Testing evaluation
    logger.log("\n" + "="*50)
    logger.log("Training complete. Starting testing evaluation...")
    logger.log("="*50)
    
    best_checkpoint_path = checkpoints_dir / 'best.pt'
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.log(f"Loaded best model from {best_checkpoint_path} for testing.")
        
    model.eval()
    test_loss_sum = 0
    test_count = 0
    all_gt_test = []
    all_pred_test = []
    
    test_bar = tqdm(test_loader, desc="Testing Evaluation", disable=on_cluster)
    with torch.no_grad():
        for image_batch, labels_batch, _ in test_bar:
            if model_name == 'maskrcnn-resnet50_fpn':
                model.train()
                set_batchnorm_eval(model)
                images_list = [img.to(device, non_blocking=pin_memory) / 255.0 for img in image_batch]
                targets = get_maskrcnn_targets(labels_batch, device)
                loss_dict = model(images_list, targets)
                loss = sum(l for l in loss_dict.values())
                test_loss_sum += loss.item() * len(image_batch)
                
                model.eval()
                predictions = model(images_list)
                for i in range(len(image_batch)):
                    H, W = image_batch[i].shape[1:]
                    pred_lbl = maskrcnn_to_labeled_mask(predictions[i], (H, W))
                    gt_lbl = labels_batch[i].cpu().numpy().squeeze()
                    all_pred_test.append(pred_lbl)
                    all_gt_test.append(gt_lbl)
            else:
                image_batch = image_batch.to(device)
                labels_batch = labels_batch.to(device)
                output = model(image_batch)
                loss = loss_fn(output, labels_batch).mean()
                test_loss_sum += loss.item() * len(image_batch)
                
                # Extract postprocessing parameters from config with defaults
                postproc_params = {
                    'mask_threshold': flat.get('mask_threshold', 0.53),
                    'seed_threshold': flat.get('seed_threshold', 0.5),
                    'peak_distance': int(flat.get('peak_distance', 5)),
                    'overlap_threshold': flat.get('overlap_threshold', 0.3),
                    'mean_threshold': flat.get('mean_threshold', 0.1),
                    'min_size': int(flat.get('min_size', 10)),
                }
                
                predicted_labels = torch.stack([
                    postprocessing_fn(out, **postproc_params)
                    for out in output
                ])
                for i in range(len(image_batch)):
                    pred_lbl = predicted_labels[i].cpu().numpy()
                    gt_lbl = labels_batch[i].cpu().numpy().squeeze()
                    all_pred_test.append(pred_lbl)
                    all_gt_test.append(gt_lbl)
                    
            test_count += len(image_batch)
            
    test_loss = safe_average(test_loss_sum, test_count)
    test_metrics = compute_yolo_style_metrics(all_gt_test, all_pred_test)
    
    logger.log(f"Test Loss: {test_loss:.4f}")
    logger.log(f"Test Quality Metrics:")
    logger.log(f"  Precision: {test_metrics['precision']:.4f}")
    logger.log(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.log(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.log(f"  AP@50:     {test_metrics['ap50']:.4f}")
    logger.log(f"  AP@75:     {test_metrics['ap75']:.4f}")
    logger.log(f"  AP@50-95:  {test_metrics['ap50_95']:.4f}")
    
    # Save test metrics
    pd.DataFrame([test_metrics]).to_csv(experiment_dir / 'test_metrics.csv', index=False)
    logger.log("Testing evaluation complete.")
    return model, metrics_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config. Defaults to root config.yaml.")
    args = parser.parse_args()
    train_model(config_path=args.config)
