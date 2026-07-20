import os
import random
from pathlib import Path
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm.auto import tqdm
from instanseg.utils.metrics import _robust_average_precision, _robust_f1_mean_calculator

from instanseg.utils.augmentations import Augmentations
import time

from instanseg.utils.utils import show_images
import warnings


global_step = 0
def train_epoch(train_model, 
                train_device, 
                train_dataloader, 
                train_loss_fn, 
                train_optimizer, 
                args,
                ):

    global global_step
    start = time.time()
    train_model.train()
    train_loss = []
    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):
        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)
        output = train_model(image_batch)
        loss = train_loss_fn(output, labels.clone()).mean()
        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)
        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    end = time.time()
    return sum(train_loss) / 3188, end - start


global_step_test = 0
def test_epoch(test_model, 
               test_device, 
               test_dataloader, 
               test_loss_fn, 
               args,
               postprocessing_fn,
               method,
               iou_threshold,
               debug=False, 
               save_str=None, 
               save_bool=False,
               best_f1=None):
    global global_step_test
    start = time.time()

    test_model.eval()
    test_loss = []

    current_f1_list = []
    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device) 
            output = test_model(image_batch)  
            loss = test_loss_fn(output, labels.clone()).mean()
            test_loss.append(loss.detach().cpu().numpy())
            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                predicted_labels = torch.stack([postprocessing_fn(out) for out in output])
                f1i = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                               threshold=iou_threshold)
                current_f1_list.append((f1i))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
            global_step_test += 1
    f1_array = np.array(current_f1_list)  # either N,2 or N,
    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    #  mean_f1 = current_f1_list

    if mean_f1 > best_f1 or save_bool:
        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]
        labels_dst = labels[0]
        lab = postprocessing_fn(output[0])

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)
    end = time.time()
    return sum(test_loss) / 569, mean1_f1, end - start


def resize_keeping_aspect_ratio(image, imgsz, is_mask=False):
    H, W = image.shape[:2]
    max_side = max(H, W)
    scale = imgsz / max_side
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    if is_mask:
        interpolation = getattr(cv2, 'INTER_NEAREST_EXACT', cv2.INTER_NEAREST)
    else:
        interpolation = getattr(cv2, 'INTER_LINEAR_EXACT', cv2.INTER_LINEAR)
    return cv2.resize(image, (new_W, new_H), interpolation=interpolation)


def collate_fn(data):
    imgs, labels = zip(*data)
    
    # Find max H and max W in this batch
    max_H = max(img.shape[-2] for img in imgs)
    max_W = max(img.shape[-1] for img in imgs)
    
    padded_imgs = []
    padded_labels = []
    
    for img, label in zip(imgs, labels):
        C, H, W = img.shape
        padded_img = torch.zeros((C, max_H, max_W), dtype=img.dtype)
        padded_img[:, :H, :W] = img
        padded_imgs.append(padded_img)
        
        l_shape = label.shape
        if len(l_shape) == 3:
            padded_label = torch.full((l_shape[0], max_H, max_W), -1, dtype=label.dtype)
            padded_label[:, :H, :W] = label
        else:
            padded_label = torch.full((max_H, max_W), -1, dtype=label.dtype)
            padded_label[:H, :W] = label
        padded_labels.append(padded_label)
        
    images = torch.stack(padded_imgs)
    labels = torch.stack(padded_labels)
    lengths = torch.tensor([img.shape[0] for img in imgs])
    return images, labels, lengths.int()


class Segmentation_Dataset(Dataset):
    def __init__(self, input_data_dir, common_transforms=True, metadata=None, size=(512, 512),
                 augmentation_dict=None, dim_in=3, debug=False, cells_and_nuclei=False,
                 target_segmentation="C", channel_invariant=True, imgsz=512, augmentation_seed=None):
        import warnings
        if cells_and_nuclei:
            warnings.warn(
                "cells_and_nuclei is deprecated and will be ignored. Only cell segmentation is supported.",
                DeprecationWarning,
                stacklevel=2,
            )
        if metadata is not None:
            warnings.warn(
                "The metadata parameter is deprecated and will be ignored. "
                "Cell segmentation uses only the raw input image.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.common_transforms = common_transforms
        self.augmentation_seed = augmentation_seed
        self.imgsz = imgsz
        self.size = size
        self.dim_in = dim_in
        self.channel_invariant = channel_invariant
        self.debug = debug

        self.config = {
            "bright_limit": 0.1,
            "contrast_limit": 0.1,
            "bright_prob": 0.5,
            "flip_prob": 0.5,
            "crop_scale": (0.3, 1.0),
            "crop_ratio": (0.75, 1.3333),
            "crop_prob": 0.3,
            "scale_limit": [-0.2, 0.2],
            "rotate_prob": 0.4,
            "size": 256
        }
        if isinstance(input_data_dir, tuple):
            self.img_paths, self.mask_paths = input_data_dir
        else:
            self.input_data_dir = Path(input_data_dir)
            img_dir = self.input_data_dir / "images"
            mask_dir = self.input_data_dir / "masks"
            assert img_dir.exists() and mask_dir.exists(), f"Expected folder structure <input_data_dir>/images and <input_data_dir>/masks, got {input_data_dir}"
            self.img_paths = sorted([img_dir / f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
            self.mask_paths = sorted([mask_dir / f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
            assert len(self.img_paths) == len(self.mask_paths), "The number of images and labels must be the same"

        from instanseg.utils.preprocessing import build_augmentation_pipeline
        self.transform = build_augmentation_pipeline(
            bright_limit=self.config["bright_limit"],
            contrast_limit=self.config["contrast_limit"],
            bright_prob=self.config["bright_prob"],
            flip_prob=self.config["flip_prob"],
            scale_limit=tuple(self.config["scale_limit"]),
            rotate_prob=self.config["rotate_prob"],
            seed=int(augmentation_seed) if augmentation_seed is not None else None,
        )

    def __len__(self):
        return len(self.img_paths)

    def _read_pair(self, index):
        image_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        suffix = image_path.suffix.lower()
        if suffix in ('.tif', '.tiff'):
            import tifffile

            image = tifffile.imread(image_path)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Unable to read image {image_path}")
            if image.ndim == 2:
                image = image[:, :, None]
        mask_suffix = mask_path.suffix.lower()
        if mask_suffix in ('.tif', '.tiff'):
            import tifffile

            mask = tifffile.imread(mask_path)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Unable to read mask {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int16)
        image = image.astype(np.uint8) if image.dtype != np.uint8 else image
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        image = resize_keeping_aspect_ratio(image, self.imgsz, is_mask=False)
        mask = resize_keeping_aspect_ratio(mask, self.imgsz, is_mask=True)
        return image, mask

    def __getitem__(self, i):
        data, label = self._read_pair(i)

        if self.common_transforms:
            # albumentations expects HWC for image, HW for mask
            # Do NOT transpose to CHW before passing to albumentations
            if self.augmentation_seed is not None:
                self.transform.set_random_seed(int(self.augmentation_seed) + int(i))
            transformed = self.transform(image=data, mask=label)
            data, label = transformed['image'], transformed['mask']
        else:
            if isinstance(data, np.ndarray):
                if data.ndim == 3:
                    data = np.ascontiguousarray(np.transpose(data, (2, 0, 1)))
                else:
                    data = np.ascontiguousarray(data[np.newaxis, ...])
                data = torch.as_tensor(data, dtype=torch.float32)
            elif isinstance(data, torch.Tensor):
                data = data.float()
                if data.ndim == 2:
                    data = data.unsqueeze(0)

            if isinstance(label, np.ndarray):
                if label.ndim == 3:
                    label = label[0]
                label = np.ascontiguousarray(label)
                label = torch.as_tensor(label, dtype=torch.int16)
            elif isinstance(label, torch.Tensor):
                if label.ndim == 3:
                    label = label[0]
                label = label.to(torch.int16)

        if isinstance(data, torch.Tensor) and data.ndim == 2:
            data = data.unsqueeze(0)
        if isinstance(label, torch.Tensor) and label.ndim == 2:
            label = label.unsqueeze(0)

        data = data.float()
        label = label.to(torch.int16)
        assert not data.isnan().any(), "Transformed images contains NaN"
        return data, label


def plot_loss(_model):
    loss_fig = plt.figure()
    timer = loss_fig.canvas.new_timer(interval=300000)
    timer.add_callback(plt.close)

    losses = [param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None]
    names = [name for name, param in _model.named_parameters() if param.grad is not None]

    plt.plot(losses)
    plt.xticks(np.arange(len(names))[::1], names[::1])
    plt.xticks(fontsize=8, rotation=90)
    spacing = 0.5
    loss_fig.subplots_adjust(bottom=spacing)
    timer.start()
    plt.show()


def check_max_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.max()


def check_min_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.min()


def check_mean_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.mean()


def optimize_hyperparameters(model,postprocessing_fn, data_loader = None, val_images = None, val_labels = None,max_evals = 50, verbose = False, threshold = [0.5, 0.7, 0.9], show_progressbar = True, device = None):


    from instanseg.utils.metrics import _robust_average_precision
    from instanseg.utils.utils import _choose_device

    from hyperopt import fmin
    from hyperopt import hp
    from hyperopt import Trials
    from hyperopt import tpe
    import copy

    if device is None:
        device = _choose_device()

    bayes_trials = Trials()

    space = {  # instanseg
        'mask_threshold': hp.uniform('mask_threshold', 0.3, 0.7),
        'seed_threshold': hp.uniform('seed_threshold', 0.5, 1),
        'overlap_threshold': hp.uniform('overlap_threshold', 0.1, 0.9),
        #'min_size': hp.uniform('min_size', 0, 30),
        'peak_distance': hp.uniform('peak_distance', 3, 10),
        'mean_threshold': hp.uniform('mean_threshold', 0.0, 0.3)} #the max could be increased, but may cuase the method not to converge for some reason.
    
    _model = model # copy.deepcopy(model)
    _model.eval()
    predictions = []

    with torch.no_grad():
        if data_loader is not None:
            for image_batch, labels_batch, _ in data_loader:
                    image_batch = image_batch.to(device)
                    output = _model(image_batch).cpu()
                    predictions.extend([pred,masks] for pred,masks in zip(output,labels_batch))


            def objective(params={}):
                pred_masks = []
                gt_masks = []
                for pred, masks in predictions:
                    lab = postprocessing_fn(pred.to(device), **params).cpu()
                    pred_masks.append(lab)
                    gt_masks.append(masks)

                mean_f1 = _robust_average_precision(torch.stack(gt_masks),torch.stack(pred_masks),threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        
        elif val_images is not None and val_labels is not None:
            from instanseg.utils.tiling import _instanseg_padding, _recover_padding
            def objective(params={}):
                pred_masks = []
                gt_masks = []
                #randomly shuffle val_images and val_labels

                np.random.seed(0)
                indexes = np.random.permutation(len(val_images))[:300]
                indexes.sort()

                for i in indexes:
                    imgs = val_images[i]
                    gt_mask = val_labels[i]
                    with torch.no_grad():
                        imgs = imgs.to(device)
                        imgs, pad = _instanseg_padding(imgs, min_dim = 32)
                        output = _model(imgs[None,])
                        output = _recover_padding(output, pad).squeeze(0)
                        lab = postprocessing_fn(output.to(device), **params).cpu()
                        pred_masks.append(lab)
                        gt_masks.append(gt_mask)

                mean_f1 = _robust_average_precision(gt_masks,pred_masks,threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        else:
            raise ValueError("Either data_loader or val_images and val_labels must be provided")

        print("Optimizing hyperparameters")
        # Optimize
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=bayes_trials, show_progressbar = show_progressbar)
    
    if verbose:
        print(best)
    return best



