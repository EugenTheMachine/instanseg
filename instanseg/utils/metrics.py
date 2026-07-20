import numpy as np
from stardist import matching
import fastremap

from typing import Union
def _robust_f1_mean_calculator(nan_list: Union[list, np.ndarray]):
    nan_list = np.array(nan_list)
    if len(nan_list) == 0:
        return np.nan
    elif np.isnan(nan_list).all():
        return np.nan
    else:
        return np.nanmean(nan_list)


def _robust_average_precision(labels, predicted, threshold):

    for i in range(len(labels)):
        if labels[i].min() < 0 and not (labels[i] < 0).all():
            labels[i][labels[i] < 0] = 0 #sparse labels
            predicted[i][labels[i] < 0] = 0 

    if labels[0].shape[0] != 2: #cells or nuclei
        labels = [labels[i].detach().cpu().numpy().astype(np.int32) for i, l in enumerate(labels) if labels[i].min() >= 0 and labels[i].max() > 0]
        predicted = [predicted[i].detach().cpu().numpy().astype(np.int32) for i, l in enumerate(labels) if labels[i].min() >= 0 and labels[i].max() > 0]

        if len(labels)==0:
            return np.nan
        

        stats = matching.matching_dataset([l for l in labels], [p for p in predicted], thresh=threshold, show_progress = False)
        f1i = [stat.f1 for stat in stats]
        return _robust_f1_mean_calculator(f1i)
    else:
        f1is = [] 
        for i, _ in enumerate(["nuclei", "cells"]):
            labels_tmp = [fastremap.renumber(labels[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]
            predicted_tmp = [fastremap.renumber(predicted[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]

            if len(labels_tmp)==0:
                f1is.append(np.nan)
                continue

            stats = matching.matching_dataset([l for l in labels_tmp], [p for p in predicted_tmp],thresh=threshold, show_progress = False)
            f1i = [stat.f1 for stat in stats]


            f1is.append(_robust_f1_mean_calculator(f1i))

        return f1is
    



import pandas as pd
from tqdm import tqdm
import torch

try:
    from ultralytics.utils.metrics import ap_per_class
except Exception:  # pragma: no cover - only used when the optional package is unavailable
    ap_per_class = None

def get_binary_masks_and_scores(labeled_mask):
    if isinstance(labeled_mask, torch.Tensor):
        labeled_mask = labeled_mask.cpu().numpy()
    if labeled_mask.ndim == 3:
        labeled_mask = labeled_mask[0]
        
    unique_ids = np.unique(labeled_mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    masks = []
    scores = []
    
    for uid in unique_ids:
        masks.append(labeled_mask == uid)
        scores.append(1.0)
        
    return masks, scores

def match_masks_for_image(pred_masks, gt_masks, pred_scores, iou_thresholds):
    num_preds = len(pred_masks)
    num_gts = len(gt_masks)
    num_thresholds = len(iou_thresholds)
    
    correct = np.zeros((num_preds, num_thresholds), dtype=bool)
    if num_preds == 0 or num_gts == 0:
        return correct
        
    pred_flat = np.array([m.flatten() for m in pred_masks], dtype=float)
    gt_flat = np.array([m.flatten() for m in gt_masks], dtype=float)
    
    intersection = np.dot(pred_flat, gt_flat.T)
    pred_area = pred_flat.sum(axis=1, keepdims=True)
    gt_area = gt_flat.sum(axis=1, keepdims=True).T
    union = pred_area + gt_area - intersection
    
    iou = intersection / (union + 1e-7)
    sorted_idxs = np.argsort(-np.array(pred_scores))
    
    for t_idx, threshold in enumerate(iou_thresholds):
        matched_gts = set()
        for p_idx in sorted_idxs:
            best_iou = 0
            best_gt_idx = -1
            for g_idx in range(num_gts):
                if g_idx in matched_gts:
                    continue
                if iou[p_idx, g_idx] > best_iou:
                    best_iou = iou[p_idx, g_idx]
                    best_gt_idx = g_idx
                    
            if best_iou >= threshold and best_gt_idx != -1:
                correct[p_idx, t_idx] = True
                matched_gts.add(best_gt_idx)
                
    return correct

def compute_yolo_style_metrics(gt_masks_list, pred_masks_list):
    if ap_per_class is None:
        raise ImportError("ultralytics is required for AP metric calculation. Install it with `pip install ultralytics`.")

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []
    
    for gt_mask, pred_mask in zip(gt_masks_list, pred_masks_list):
        p_masks, p_scores = get_binary_masks_and_scores(pred_mask)
        g_masks, _ = get_binary_masks_and_scores(gt_mask)
        
        num_preds = len(p_masks)
        num_gts = len(g_masks)
        
        tp = match_masks_for_image(p_masks, g_masks, p_scores, iou_thresholds)
        
        all_tp.append(tp)
        all_conf.append(np.array(p_scores))
        all_pred_cls.append(np.ones(num_preds, dtype=int))
        all_target_cls.append(np.ones(num_gts, dtype=int))
        
    if len(all_tp) == 0 or sum(len(x) for x in all_tp) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
            "ap50_95": 0.0
        }
        
    tp = np.concatenate(all_tp, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    pred_cls = np.concatenate(all_pred_cls, axis=0)
    target_cls = np.concatenate(all_target_cls, axis=0)
    
    res = ap_per_class(tp, conf, pred_cls, target_cls, names={1: "object"})
    
    tp_50 = tp[:, 0].sum()
    fp_50 = len(tp) - tp_50
    fn_50 = len(target_cls) - tp_50
    
    precision_50 = float(tp_50 / (tp_50 + fp_50)) if (tp_50 + fp_50) > 0 else 0.0
    recall_50 = float(tp_50 / (tp_50 + fn_50)) if (tp_50 + fn_50) > 0 else 0.0
    accuracy_50 = float(tp_50 / (tp_50 + fp_50 + fn_50)) if (tp_50 + fp_50 + fn_50) > 0 else 0.0
    
    ap = res[5]
    ap50 = float(ap[0, 0]) if ap.shape[0] > 0 else 0.0
    ap75 = float(ap[0, 5]) if ap.shape[0] > 0 and ap.shape[1] > 5 else 0.0
    ap50_95 = float(ap[0].mean()) if ap.shape[0] > 0 else 0.0
    
    return {
        "precision": precision_50,
        "recall": recall_50,
        "accuracy": accuracy_50,
        "ap50": ap50,
        "ap75": ap75,
        "ap50_95": ap50_95
    }

def compute_and_export_metrics(gt_masks, pred_masks, output_path, target, return_metrics = False, show_progress = False, verbose = True, logger = None):
    taus = [ 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching.matching_dataset(gt_masks, pred_masks, thresh=t, show_progress=False, by_image = False) for t in tqdm(taus, disable=not show_progress)]
    df_list = []

    for stat in stats:
        df_list.append(pd.DataFrame([stat]))
    df = pd.concat(df_list, ignore_index=True)

    mean_f1 = df[["thresh", "f1"]].iloc[:].mean()["f1"]
    mean_panoptic_quality = df[["thresh", "panoptic_quality"]].iloc[:].mean()["panoptic_quality"]
    panoptic_quality_05 = df[["thresh", "panoptic_quality"]].iloc[0]["panoptic_quality"]
    f1_05 = df[["thresh", "f1"]].iloc[0]["f1"]

    df["mean_f1"] = mean_f1
    df["f1_05"] = f1_05
    df["mean_PQ"] = mean_panoptic_quality
    df["SQ"] = panoptic_quality_05 / f1_05

    yolo_metrics = compute_yolo_style_metrics(gt_masks, pred_masks)

    def emit(message):
        if logger is not None:
            logger.log(message)
        else:
            print(message)

    if verbose:
        emit(f"Target: {target}")
        emit(f"Mean f1 score: {mean_f1}")
        emit(f"f1 score at 0.5: {f1_05}")
        emit(f"SQ: {panoptic_quality_05 / f1_05}")
        emit(f"Precision: {yolo_metrics['precision']:.4f}")
        emit(f"Recall: {yolo_metrics['recall']:.4f}")
        emit(f"Accuracy: {yolo_metrics['accuracy']:.4f}")
        emit(f"AP@50: {yolo_metrics['ap50']:.4f}")
        emit(f"AP@75: {yolo_metrics['ap75']:.4f}")
        emit(f"AP@50-95: {yolo_metrics['ap50_95']:.4f}")

    if return_metrics:
        return mean_f1, f1_05, panoptic_quality_05 / f1_05

    if output_path is not None:
        # Save both standard and YOLO style metrics
        df.to_csv(output_path / str(target + "_matching_metrics.csv"))
        pd.DataFrame([yolo_metrics]).to_csv(output_path / str(target + "_yolo_metrics.csv"), index=False)
