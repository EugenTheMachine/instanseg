"""
postprocessing.py
-----------------
Post-processing utilities for InstanSeg cell segmentation predictions.

This module centralises all post-processing operations that convert raw
model outputs (embedding fields + seed maps) into final integer label maps.

Functions extracted / consolidated from:
  - instanseg/utils/loss/instanseg_loss.py   (merge_sparse_predictions, convert,
                                               torch_peak_local_max, centre_crop,
                                               compute_crops, find_connected_components)
  - instanseg/utils/pytorch_utils.py         (referenced helpers)
"""

from typing import Optional, Tuple, Union, Dict, List
import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F

from instanseg.utils.pytorch_utils import (
    torch_fastremap, torch_onehot, remap_values,
    fast_iou, fast_sparse_iou, eccentricity_batch, connected_components
)
from instanseg.utils.tiling import _instanseg_padding, _recover_padding


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def find_all_local_maxima(image: torch.Tensor, neighbourhood_size: int, minimum_value: float) -> torch.Tensor:
    """Return a mask whose non-zero pixels are local maxima within *neighbourhood_size*."""
    kernel_size = 2 * neighbourhood_size + 1
    pooled = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mask = (pooled == image) * (image >= minimum_value)
    return image * mask


def torch_peak_local_max(
    image: torch.Tensor,
    neighbourhood_size: int,
    minimum_value: float,
    return_map: bool = False,
    dtype: torch.dtype = torch.int,
) -> torch.Tensor:
    """Efficient peak local maximum detection using max-pool with indices.

    Parameters
    ----------
    image:             2-D tensor (H, W).
    neighbourhood_size: radius of the suppression neighbourhood.
    minimum_value:     minimum pixel value for a peak to be considered.
    return_map:        if True, return the full boolean map instead of coordinates.
    dtype:             output dtype for the coordinate tensor.

    Returns
    -------
    Coordinate tensor of shape (N, 2) or a boolean map of the same shape as *image*.
    """
    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    kernel_size = 2 * neighbourhood_size + 1
    pooled, max_inds = F.max_pool2d(
        image, kernel_size=kernel_size, stride=1, padding=neighbourhood_size, return_indices=True
    )
    inds = torch.arange(0, image.numel(), device=device, dtype=dtype).reshape(image.shape)
    peak_local_max = (max_inds == inds) * (pooled > minimum_value)

    if return_map:
        return peak_local_max

    return torch.nonzero(peak_local_max.squeeze()).to(dtype)


def centre_crop(centroids: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Return flat mesh-grid indices for centred crops around each centroid.

    Parameters
    ----------
    centroids:   (C, 2) integer tensor of (row, col) centroid positions.
    window_size: half-width of the crop (final crop is 2*window_size × 2*window_size).
    h, w:        spatial dimensions of the full image.

    Returns
    -------
    mesh_grid_flat: (2, C * (2*window_size)^2) long tensor of row/col indices.
    """
    window_size = min(window_size, h, w)
    window_size = window_size - (window_size % 2)

    C = centroids.shape[0]
    centroids = centroids.clone()
    centroids[:, 0] = centroids[:, 0].clamp(min=window_size // 2, max=h - window_size // 2)
    centroids[:, 1] = centroids[:, 1].clamp(min=window_size // 2, max=w - window_size // 2)
    window_slices = centroids[:, None] + torch.tensor([[-1, -1], [1, 1]], device=centroids.device) * (window_size // 2)

    grid_x, grid_y = torch.meshgrid(
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        indexing="ij",
    )
    mesh = torch.stack((grid_x, grid_y))
    mesh_grid = mesh.expand(C, 2, window_size, window_size)
    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_grid_flat = mesh_grid_flat + idx
    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)
    mesh_grid_flat = mesh_grid_flat.round().long()
    mesh_grid_flat[0].clamp_(0, h - 1)
    mesh_grid_flat[1].clamp_(0, w - 1)

    return mesh_grid_flat


def border_distance_peaks_to_crop_anchors(
    border_distance_map: torch.Tensor,
    neighbourhood_size: int = 5,
    seed_threshold: float = 0.7,
    max_seeds: int = 2000,
) -> torch.Tensor:
    """Convert border-distance map peaks into crop-anchor coordinates.

    This utility extracts local maxima from a (normalised) border-distance map
    — corresponding to medial-axis positions — and returns them as (N, 2)
    integer (row, col) coordinates that can be fed directly to
    ``centre_crop`` or ``compute_crops``.

    Parameters
    ----------
    border_distance_map : (H, W) or (1, H, W) float tensor.
        Predicted border-distance map (higher → further from boundary).
    neighbourhood_size : int
        Half-window for peak detection.
    seed_threshold : float
        Minimum value a peak must exceed to be accepted.
    max_seeds : int
        Cap on the number of returned seeds (largest values kept).

    Returns
    -------
    anchors : (N, 2) long tensor of (row, col) coordinates.
    """
    if border_distance_map.dim() == 3:
        border_distance_map = border_distance_map.squeeze(0)

    anchors = torch_peak_local_max(
        border_distance_map,
        neighbourhood_size=neighbourhood_size,
        minimum_value=seed_threshold,
    )

    if anchors.shape[0] > max_seeds:
        # Keep the seeds with the highest border-distance values.
        vals = border_distance_map[anchors[:, 0], anchors[:, 1]]
        top_idx = vals.argsort(descending=True)[:max_seeds]
        anchors = anchors[top_idx]

    return anchors.long()


def convert(
    prob_input: torch.Tensor,
    coords_input: torch.Tensor,
    size: Tuple[int, int],
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a batch of per-object probability maps to a single integer label image.

    Parameters
    ----------
    prob_input:     (C, H, W) probability maps, one per detected object.
    coords_input:   (2, H, W) coordinate maps (row, col).
    size:           output spatial dimensions (H, W).
    mask_threshold: probability threshold above which a pixel is considered foreground.

    Returns
    -------
    (H, W) float32 label tensor (background = 0, objects = 1 … C).
    """
    all_labels = torch.arange(1, 1 + prob_input.shape[0], dtype=torch.float32, device=prob_input.device)
    labels = torch.ones_like(prob_input) * torch.reshape(all_labels, (-1, 1, 1, 1))

    labels = labels.flatten()
    prob = prob_input.flatten()
    x = coords_input[0, ...].flatten()
    y = coords_input[1, ...].flatten()

    if size is None:
        size = (int(y.max() + 1), int(x.max() + 1))

    inds_prob = prob >= mask_threshold
    n_thresholded = torch.count_nonzero(inds_prob)
    if n_thresholded == 0:
        return torch.zeros(size, dtype=torch.float32, device=labels.device)

    arr = torch.zeros((int(n_thresholded), 5), dtype=coords_input.dtype, device=labels.device)
    arr[:, 1] = y[inds_prob]
    arr[:, 2] = x[inds_prob]
    arr[:, 0] = arr[:, 2] * size[1] + arr[:, 1]
    arr[:, 3] = labels[inds_prob]

    inds_sorted = prob[inds_prob].argsort(descending=True, stable=True)
    arr = arr[inds_sorted, :]
    inds_sorted = arr[:, 0].argsort(descending=False, stable=True)
    arr = arr[inds_sorted, :]

    inds_unique = torch.ones_like(arr[:, 0], dtype=torch.bool)
    inds_unique[1:] = arr[1:, 0] != arr[:-1, 0]

    output = torch.zeros(size, dtype=torch.float32, device=labels.device)
    output[arr[inds_unique, 2], arr[inds_unique, 1]] = arr[inds_unique, 3].float()

    return output


def find_connected_components(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Compute connected components of a graph via matrix exponentiation.

    Parameters
    ----------
    adjacency_matrix: (N, N) float32 boolean matrix.

    Returns
    -------
    remapping: (2, N+2) tensor mapping each node to its component representative.
    """
    M = adjacency_matrix + torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device)
    num_iterations = 10
    out = torch.matrix_power(M, num_iterations)
    col = torch.arange(0, out.shape[0], device=out.device).view(-1, 1).expand(out.shape[0], out.shape[0])
    out_col_idx = ((out > 1).int() - torch.eye(out.shape[0], device=out.device)) * col
    maxes = out_col_idx.argmax(0) * (out_col_idx.max(0)[0] > 0).int()
    maxes = torch.maximum(
        maxes + 1,
        torch.arange(0, out.shape[0], device=out.device) + 1,
    )
    tentative_remapping = torch.stack(((torch.arange(0, out.shape[0], device=out.device) + 1), maxes))
    remapping = torch.cat(
        (torch.zeros(2, 1, device=tentative_remapping.device), tentative_remapping), dim=1
    )
    return remapping


def merge_sparse_predictions(
    x: torch.Tensor,
    coords: torch.Tensor,
    mask_map: torch.Tensor,
    size: list,
    mask_threshold: float = 0.5,
    window_size: int = 128,
    min_size: int = 10,
    overlap_threshold: float = 0.5,
    mean_threshold: float = 0.5,
) -> torch.Tensor:
    """Merge per-object crop predictions into a single label image with NMS.

    Parameters
    ----------
    x:                (C, 1, H_crop, W_crop) per-object sigmoid maps.
    coords:           (2, C * H_crop * W_crop) pixel coordinates.
    mask_map:         (H, W) seed probability map for mean filtering.
    size:             [C, H, W] spatial info for reconstruction.
    mask_threshold:   pixel-level threshold.
    window_size:      crop half-width (used for shape assertions).
    min_size:         minimum object area in pixels.
    overlap_threshold: IoU above which two detections are merged.
    mean_threshold:   minimum mean seed-map value inside an object.

    Returns
    -------
    (1, H, W) integer label tensor.
    """
    from instanseg.utils.pytorch_utils import fast_sparse_iou, remap_values

    labels = convert(x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold)[None]

    idx = torch.arange(1, size[0] + 1, device=x.device, dtype=coords.dtype)
    stack_ID = torch.ones((size[0], window_size, window_size), device=x.device, dtype=coords.dtype)
    stack_ID = stack_ID * (idx[:, None, None] - 1)

    coords = torch.stack((stack_ID.flatten(), coords[0] * size[2] + coords[1])).to(coords.dtype)

    fg = x.flatten() > mask_threshold
    x = x.flatten()[fg]
    coords = coords[:, fg]

    using_mps = False
    if x.is_mps:
        using_mps = True
        device = "cpu"
        x = x.to(device)
        mask_map = mask_map.to(device)

    sparse_onehot = torch.sparse_coo_tensor(
        coords,
        x.flatten() > mask_threshold,
        size=(size[0], size[1] * size[2]),
        dtype=x.dtype,
        device=x.device,
        requires_grad=False,
    )

    object_areas = torch.sparse.sum(sparse_onehot, dim=1).values()
    sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map.flatten()[None]), dim=1).values()
    mean_mask_value = sum_mask_value / object_areas
    objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

    if window_size ** 2 * sparse_onehot.shape[0] == sparse_onehot.sum():
        return labels

    iou = fast_sparse_iou(sparse_onehot)
    remapping = find_connected_components((iou > overlap_threshold).float())

    if using_mps:
        remapping = remapping.to("mps")
        labels = labels.to("mps")

    labels = remap_values(remapping, labels)

    labels_to_remove = (
        torch.arange(0, len(objects_to_remove), device=objects_to_remove.device, dtype=coords.dtype) + 1
    )[objects_to_remove]
    labels[torch.isin(labels, labels_to_remove)] = 0

    return labels


def filter_small_blobs(
    label: Union[np.ndarray, torch.Tensor],
    min_size: int = 10,
) -> Union[np.ndarray, torch.Tensor]:
    """Remove labelled objects smaller than *min_size* pixels.

    Parameters
    ----------
    label:    2-D integer label array / tensor.
    min_size: minimum object area (inclusive) to keep.

    Returns
    -------
    Filtered label array of the same type as the input.
    """
    is_tensor = isinstance(label, torch.Tensor)
    if is_tensor:
        device = label.device
        label_np = label.cpu().numpy().astype(np.int32)
    else:
        label_np = label.astype(np.int32)

    import fastremap

    ids, counts = np.unique(label_np, return_counts=True)
    small = ids[(counts < min_size) & (ids != 0)]
    label_np[np.isin(label_np, small)] = 0
    label_np = fastremap.renumber(label_np)[0]

    if is_tensor:
        return torch.from_numpy(label_np).to(device=device, dtype=label.dtype)
    return label_np.astype(label.dtype)


def compute_boundary_evidence(
    embeddings: torch.Tensor,
    mask_map: torch.Tensor,
) -> torch.Tensor:
    """Compute a learned boundary map B from absolute embeddings and border distance.

    embeddings: (D, H, W)
    mask_map: (H, W) in [0, 1] (predicted border distance map)
    """
    b_dist = torch.clamp(1.0 - mask_map, 0.0, 1.0)

    dy = torch.zeros_like(embeddings)
    dx = torch.zeros_like(embeddings)
    dy[:, 1:, :] = torch.abs(embeddings[:, 1:, :] - embeddings[:, :-1, :])
    dx[:, :, 1:] = torch.abs(embeddings[:, :, 1:] - embeddings[:, :, :-1])

    grad_mag = torch.sqrt(dy**2 + dx**2 + 1e-6).sum(dim=0)
    b_grad = 1.0 - torch.exp(-grad_mag / 2.0)

    boundary = torch.maximum(b_dist, b_grad)
    return boundary


def refine_clusters(
    label_map: torch.Tensor,
    embeddings: torch.Tensor,
    mask_map: torch.Tensor,
    min_size: int = 10,
    affinity_threshold: float = 0.65,
    sigma_e: float = 1.0,
    sigma_g: float = 2.0,
) -> torch.Tensor:
    device = label_map.device
    h, w = label_map.shape
    refined = label_map.clone()

    unique_labels = torch.unique(refined)
    unique_labels = unique_labels[unique_labels > 0]

    next_new_label = int(refined.max().item()) + 1

    for label_val in unique_labels:
        mask = (refined == label_val)
        if not mask.any():
            continue

        cluster_mask_map = mask_map * mask.float()
        max_val = cluster_mask_map.max().item()
        if max_val <= 0:
            continue

        high_conf_thresh = max(0.5, 0.7 * max_val)
        high_conf_mask = (cluster_mask_map >= high_conf_thresh)

        if not high_conf_mask.any():
            continue

        high_conf_labels = connected_components(
            high_conf_mask.unsqueeze(0).unsqueeze(0).float(),
            num_iterations=16
        ).squeeze()

        # connected_components may return a scalar if there is only one pixel
        if high_conf_labels.dim() < 2:
            continue

        high_conf_unique = torch.unique(high_conf_labels)
        high_conf_unique = high_conf_unique[high_conf_unique > 0]

        valid_components = []
        for hc_id in high_conf_unique:
            hc_mask = (high_conf_labels == hc_id)
            if hc_mask.sum().item() >= max(2, min_size // 3):
                valid_components.append(hc_mask)

        if len(valid_components) >= 2:
            comp_embeddings = []
            for hc_mask in valid_components:
                comp_emb = embeddings[:, hc_mask].mean(dim=1)
                comp_embeddings.append(comp_emb)
            comp_embeddings = torch.stack(comp_embeddings)

            cluster_indices = torch.nonzero(mask, as_tuple=False)
            pixel_embs = embeddings[:, mask].T

            dists = torch.cdist(pixel_embs.unsqueeze(0), comp_embeddings.unsqueeze(0)).squeeze(0)
            closest_comp = dists.argmin(dim=1)

            for idx, hc_mask in enumerate(valid_components):
                new_lbl = label_val if idx == 0 else next_new_label
                if idx > 0:
                    next_new_label += 1

                pixel_mask = (closest_comp == idx)
                if pixel_mask.any():
                    refined[cluster_indices[pixel_mask][:, 0], cluster_indices[pixel_mask][:, 1]] = int(new_lbl)

    unique_labels = torch.unique(refined)
    unique_labels = unique_labels[unique_labels > 0]

    for label_val in unique_labels:
        mask = (refined == label_val)
        size = mask.sum().item()
        if size < min_size:
            dilated = F.max_pool2d(
                mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze() > 0

            boundary_mask = dilated & ~mask
            neighbor_labels = refined[boundary_mask]
            neighbor_labels = neighbor_labels[neighbor_labels > 0]

            merged = False
            if len(neighbor_labels) > 0:
                neigh_unique = torch.unique(neighbor_labels)
                best_neighbor = None
                best_affinity = -1.0

                frag_emb = embeddings[:, mask].mean(dim=1, keepdim=True)

                for neigh_val in neigh_unique:
                    neigh_mask = (refined == neigh_val)
                    if not neigh_mask.any():
                        continue
                    neigh_emb = embeddings[:, neigh_mask].mean(dim=1, keepdim=True)

                    emb_diff = ((frag_emb - neigh_emb) ** 2).sum()
                    affinity = torch.exp(-emb_diff / (2 * sigma_e ** 2)).item()
                    if affinity > best_affinity:
                        best_affinity = affinity
                        best_neighbor = int(neigh_val.item())

                if best_neighbor is not None and best_affinity >= affinity_threshold * 0.8:
                    refined[mask] = best_neighbor
                    merged = True

            if not merged:
                refined[mask] = 0

    fg_mask = (refined > 0)
    if fg_mask.any():
        # Use replicate padding on 4D tensor (supported), then squeeze back
        padded = F.pad(
            refined.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='replicate'
        ).squeeze(0).squeeze(0)
        patches = padded.unfold(0, 3, 1).unfold(1, 3, 1)
        flat_patches = patches.reshape(h, w, 9).long()
        center_labels = flat_patches[:, :, 4]
        matches = (flat_patches == center_labels.unsqueeze(-1)).sum(dim=-1)
        outliers = (matches <= 2) & fg_mask

        if outliers.any():
            outlier_indices = torch.nonzero(outliers, as_tuple=False)
            for yx in outlier_indices:
                y_idx, x_idx = int(yx[0].item()), int(yx[1].item())
                patch = flat_patches[y_idx, x_idx]
                nonzero_patch = patch[patch > 0]
                if len(nonzero_patch) > 0:
                    vals, counts = torch.unique(nonzero_patch, return_counts=True)
                    majority_label = int(vals[counts.argmax()].item())
                    refined[y_idx, x_idx] = majority_label
                else:
                    refined[y_idx, x_idx] = 0

    positive = torch.unique(refined)
    positive = positive[positive > 0]
    remapped = torch.zeros_like(refined)
    for new_id, old_id in enumerate(positive, start=1):
        remapped[refined == old_id] = new_id

    return remapped


def cluster_embedding_pixels(
    embeddings: torch.Tensor,
    foreground: torch.Tensor,
    boundary: Optional[torch.Tensor] = None,
    affinity_threshold: float = 0.65,
    sigma_e: float = 1.0,
    sigma_g: float = 2.0,
    min_size: int = 10,
) -> torch.Tensor:
    """Group foreground pixels with a local embedding-affinity graph.

    This deterministic seedless grouping connects 8-neighbour foreground
    pixels when their embedding and spatial affinity exceeds the threshold.
    The resulting connected components are returned as a dense label map.
    """
    if embeddings.dim() != 3:
        raise ValueError("embeddings must have shape (D, H, W)")
    if foreground.shape != embeddings.shape[-2:]:
        raise ValueError("foreground shape must match embedding spatial shape")
    if boundary is not None and boundary.shape != foreground.shape:
        raise ValueError("boundary shape must match foreground shape")
    if sigma_e <= 0 or sigma_g <= 0:
        raise ValueError("sigma_e and sigma_g must be positive")

    device = embeddings.device
    h, w = foreground.shape
    foreground = foreground.bool()
    if not foreground.any():
        return torch.zeros((h, w), dtype=torch.int64, device=device)

    parent = torch.arange(h * w, device=device, dtype=torch.int64)
    rank = torch.zeros(h * w, device=device, dtype=torch.int8)
    flat_fg = foreground.flatten()
    offsets = ((0, 1), (1, -1), (1, 0), (1, 1))

    def find(idx: int) -> int:
        root = idx
        while int(parent[root].item()) != root:
            root = int(parent[root].item())
        while int(parent[idx].item()) != idx:
            next_idx = int(parent[idx].item())
            parent[idx] = root
            idx = next_idx
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if int(rank[root_a].item()) < int(rank[root_b].item()):
            parent[root_a] = root_b
        elif int(rank[root_a].item()) > int(rank[root_b].item()):
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for dy, dx in offsets:
        y0_start = max(0, -dy)
        y0_end = h - max(0, dy)
        x0_start = max(0, -dx)
        x0_end = w - max(0, dx)
        y1_start = y0_start + dy
        y1_end = y0_end + dy
        x1_start = x0_start + dx
        x1_end = x0_end + dx

        src_fg = foreground[y0_start:y0_end, x0_start:x0_end]
        dst_fg = foreground[y1_start:y1_end, x1_start:x1_end]
        pair_fg = src_fg & dst_fg
        if not pair_fg.any():
            continue

        src_e = embeddings[:, y0_start:y0_end, x0_start:x0_end]
        dst_e = embeddings[:, y1_start:y1_end, x1_start:x1_end]

        spatial_sq = float(dy * dy + dx * dx)
        spatial_affinity = torch.exp(torch.tensor(-spatial_sq / (2 * sigma_g ** 2), device=device))

        if embeddings.shape[0] == 4:
            src_center = src_e[:2]
            dst_center = dst_e[:2]
            src_border = src_e[2:]
            dst_border = dst_e[2:]

            emb_sq_center = ((src_center - dst_center) ** 2).sum(0)
            emb_sq_border = ((src_border - dst_border) ** 2).sum(0)

            affinity = (
                torch.exp(-emb_sq_center / (2 * sigma_e ** 2))
                * torch.exp(-emb_sq_border / (2 * sigma_e ** 2))
                * spatial_affinity
            )
        else:
            emb_sq = ((src_e - dst_e) ** 2).sum(0)
            affinity = torch.exp(-emb_sq / (2 * sigma_e ** 2)) * spatial_affinity

        if boundary is not None:
            boundary_pair = torch.maximum(
                boundary[y0_start:y0_end, x0_start:x0_end],
                boundary[y1_start:y1_end, x1_start:x1_end],
            )
            affinity = affinity * (1 - boundary_pair.clamp(0, 1))

        edges = torch.nonzero(pair_fg & (affinity >= affinity_threshold), as_tuple=False)
        for yx in edges:
            y = int(yx[0].item())
            x = int(yx[1].item())
            a = (y0_start + y) * w + (x0_start + x)
            b = (y1_start + y) * w + (x1_start + x)
            union(a, b)

    labels = torch.zeros(h * w, dtype=torch.int64, device=device)
    root_to_label = {}
    next_label = 1
    fg_indices = torch.nonzero(flat_fg, as_tuple=False).flatten()
    for flat_idx_t in fg_indices:
        flat_idx = int(flat_idx_t.item())
        root = find(flat_idx)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[flat_idx] = root_to_label[root]

    label_map = labels.reshape(h, w)
    
    # Run the cluster refinement loop
    mask_map_refinement = 1.0 - boundary if boundary is not None else foreground.float()
    label_map = refine_clusters(
        label_map=label_map,
        embeddings=embeddings,
        mask_map=mask_map_refinement,
        min_size=min_size,
        affinity_threshold=affinity_threshold,
        sigma_e=sigma_e,
        sigma_g=sigma_g,
    )

    return label_map


def generate_coordinate_map(
    mode: str = "linear",
    spatial_dim: int = 2,
    height: int = 256,
    width: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a (spatial_dim, H, W) coordinate map.

    Parameters
    ----------
    mode:        currently only "linear" is supported.
    spatial_dim: number of spatial dimensions (≥ 2).
    height, width: spatial resolution of the output map.
    device:      target device.

    Returns
    -------
    (spatial_dim, H, W) float32 coordinate tensor.
    """
    if mode == "linear":
        xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height, width)
        yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
        if spatial_dim == 2:
            xxyy = torch.cat((xx, yy), 0)
        elif spatial_dim >= 3:
            zz = torch.zeros_like(xx).expand(spatial_dim - 2, -1, -1)
            xxyy = torch.cat((xx, yy, zz), 0)
        else:
            xxyy = torch.zeros((spatial_dim, height, width), device=device)
    else:
        xxyy = torch.zeros((spatial_dim, height, width), device=device)

    return xxyy


# ---------------------------------------------------------------------------
# Moved from instanseg/utils/loss/instanseg_loss.py
# ---------------------------------------------------------------------------

def has_pixel_classifier_model(model):
    """Return True if *model* already contains a ProbabilityNet sub-module."""
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'ProbabilityNet' in module_class:
                return True
    return False


def guide_function(params: torch.Tensor, device=None, width: int = 256):
    # params must be depth,3
    depth = params.shape[0]
    if device is None:
        device = params.device
    xx = torch.linspace(0, 1, width, device=device).view(1, 1, -1).expand(1, width, width)
    yy = torch.linspace(0, 1, width, device=device).view(1, -1, 1).expand(1, width, width)
    xxyy = torch.cat((xx, yy), 0).expand(depth, 2, width, width)

    xx = xxyy[:, 0] * params[:, 0][:, None, None]
    yy = xxyy[:, 1] * params[:, 1][:, None, None]

    return torch.sin(xx + yy + params[:, 2, None, None])[None]


def compute_crops(
    x: torch.Tensor,
    c: torch.Tensor,
    sigma: torch.Tensor,
    centroids_idx: torch.Tensor,
    feature_engineering,
    pixel_classifier,
    window_size: int = 128,
):
    """Compute per-object probability crops using feature engineering + pixel classifier."""
    h, w = x.shape[-2:]
    C = c.shape[0]

    window_size = min(window_size, h, w)
    window_size = window_size - (window_size % 2)

    mesh_grid_flat = centre_crop(centroids_idx, window_size, h, w)

    x = feature_engineering(x, c, sigma, window_size // 2, mesh_grid_flat)
    x = pixel_classifier(x)  # C*H*W, 1
    x = x.view(C, 1, window_size, window_size)

    idx = torch.arange(1, C + 1, device=x.device, dtype=mesh_grid_flat.dtype)
    rep = torch.ones((C, window_size, window_size), device=x.device, dtype=mesh_grid_flat.dtype)
    rep = rep * (idx[:, None, None] - 1)
    iidd = torch.cat((rep.flatten()[None], mesh_grid_flat)).to(mesh_grid_flat.dtype)

    return x, iidd


class ProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width=5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        x = self._relu_non_empty(self.fc1(x))
        x = self._relu_non_empty(self.fc2(x))
        x = self.fc3(x)
        return x

    def _relu_non_empty(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        else:
            return torch.relu_(x)


class MyBlock(nn.Sequential):
    def __init__(self, embedding_dim, width):
        super(MyBlock, self).__init__()
        self.fc1 = nn.Conv2d(embedding_dim, width, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(width, width, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Conv2d(width, 1, 1)


class ConvProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width=5, depth=5):
        super().__init__()
        self.layer1 = MyBlock(embedding_dim + depth, width)
        self.layer2 = MyBlock(embedding_dim, width)
        self.layer3 = MyBlock(embedding_dim + 2, width)
        self.positional_embedding_params = nn.Parameter(torch.rand(depth, 3) * 10)

    def forward(self, x):
        positional_embedding = guide_function(self.positional_embedding_params, device=x.device, width=100)
        one = self.layer1(torch.cat((x, positional_embedding.expand(x.shape[0], -1, -1, -1)), dim=1))
        two = self.layer2(x)
        output = self.layer3(torch.cat((x, one, two), dim=1))
        return output


class MedianFilter(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int]):
        from kornia.filters import MedianBlur
        super(MedianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.MedianBlur = MedianBlur(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MedianBlur(x)


from einops import rearrange


def feature_engineering(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x = torch.cat([x, sigma])[:, mesh_grid_flat[0], mesh_grid_flat[1]]
    x = rearrange(x, '(E) (C H W) -> C (E) H W', E=E+S, C=C, H=2*window_size, W=2*window_size)
    c_shaped = c.view(-1, E, 1, 1)
    x[:, :E] -= c_shaped
    x = rearrange(x, 'C (E) H W-> (C H W) (E)', E=E+S)
    return x


def feature_engineering_border(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)

    device = x.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(2 * window_size, device=device, dtype=x.dtype) - window_size,
        torch.arange(2 * window_size, device=device, dtype=x.dtype) - window_size,
        indexing="ij"
    )
    rel_coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(C, -1, -1, -1)  # C, 2, H, W
    dist_to_seed = torch.sqrt(torch.sum(rel_coords ** 2, dim=1, keepdim=True) + 1e-6)  # C, 1, H, W

    feat = torch.cat([x_slices, sigma_slices, rel_coords, dist_to_seed], dim=1)
    feat = feat.flatten(2).permute(0, -1, 1)
    feat = feat.reshape((feat.shape[0] * feat.shape[1]), feat.shape[2])
    return feat


def feature_engineering_border_slow(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    return feature_engineering_border(x, c, sigma, window_size, mesh_grid_flat)


def feature_engineering_border_and_dist(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                                      mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)

    device = x.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(2 * window_size, device=device, dtype=x.dtype) - window_size,
        torch.arange(2 * window_size, device=device, dtype=x.dtype) - window_size,
        indexing="ij"
    )
    rel_coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(C, -1, -1, -1)  # C, 2, H, W
    dist_to_seed = torch.sqrt(torch.sum(rel_coords ** 2, dim=1, keepdim=True) + 1e-6)  # C, 1, H, W
    
    border_rel_to_seed = x_slices + rel_coords  # C, 2, H, W

    feat = torch.cat([x_slices, sigma_slices, rel_coords, dist_to_seed, border_rel_to_seed], dim=1)
    feat = feat.flatten(2).permute(0, -1, 1)
    feat = feat.reshape((feat.shape[0] * feat.shape[1]), feat.shape[2])
    return feat


def feature_engineering_combined(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                               mesh_grid_flat: torch.Tensor):
    """Feature engineering for combined-center / combined-cluster modes.

    The embedding tensor *x* contains 4 channels:
      - channels 0:2  — center-relative vectors (pointing toward instance centroid)
      - channels 2:4  — border-relative vectors (pointing toward nearest border)

    The centroid seed vector *c* contains only the first 2 (center) channels.

    Features produced per pixel per candidate:
      [center_diff (2), border_vecs (2), sigma (S), rel_coords (2), dist_to_seed (1)]
    Output width = 2 + 2 + S + 2 + 1 = 7 + S
    """
    # Split center and border channels
    x_center = x[:2]   # (2, H, W)
    x_border = x[2:]   # (2, H, W)
    E_c = x_center.shape[0]  # 2
    E_b = x_border.shape[0]  # 2
    C = c.shape[0]
    S = sigma.shape[0]

    hw = 2 * window_size

    # Slice windows for each seed
    center_slices = x_center[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(
        E_c, C, hw, hw).permute(1, 0, 2, 3)  # C, 2, H, W
    border_slices = x_border[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(
        E_b, C, hw, hw).permute(1, 0, 2, 3)  # C, 2, H, W
    sigma_slices  = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(
        S, C, hw, hw).permute(1, 0, 2, 3)    # C, S, H, W

    # Center-relative difference (seed centroid subtracted from prediction)
    c_shaped = c[:, :E_c].reshape(C, E_c, 1, 1)  # only the center half of c
    center_diff = center_slices - c_shaped         # C, 2, H, W

    # Relative spatial coords and distance to seed
    device = x.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(hw, device=device, dtype=x.dtype) - window_size,
        torch.arange(hw, device=device, dtype=x.dtype) - window_size,
        indexing="ij",
    )
    rel_coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(C, -1, -1, -1)  # C, 2, H, W
    dist_to_seed = torch.sqrt(torch.sum(rel_coords ** 2, dim=1, keepdim=True) + 1e-6)     # C, 1, H, W

    feat = torch.cat([center_diff, border_slices, sigma_slices, rel_coords, dist_to_seed], dim=1)
    feat = feat.flatten(2).permute(0, -1, 1)                 # C, H*W, features
    feat = feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2])
    return feat


def feature_engineering_slow(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    return feature_engineering(x, c, sigma, window_size, mesh_grid_flat)


def feature_engineering_2(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped
    norm = torch.sqrt(torch.sum(torch.pow(x_slices - c_shaped, 2) + 1e-6, dim=1, keepdim=True))

    x = torch.cat([diff, sigma_slices, norm], dim=1)
    x = x.flatten(2).permute(0, -1, 1)
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])
    return x


def feature_engineering_3(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped
    x = torch.cat([diff, sigma_slices * 0], dim=1)
    x = x.flatten(2).permute(0, -1, 1)
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])
    return x


def feature_engineering_10(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    c_shaped = c.reshape(-1, E, 1, 1)
    diff = x_slices - c_shaped
    x = torch.cat([diff, sigma_slices], dim=1)
    return x


def feature_engineering_generator(feature_engineering_function):
    """Return (fn, output_width_per_channel) for the given FE key.

    Width values are the number of *extra* feature dimensions added beyond the
    raw embedding (i.e. sigma channels + any spatial features).  They are used
    by the caller to size the pixel-classifier MLP correctly.

    Supported keys
    --------------
    ``"0"`` / ``"7"``        Standard centroid-based FE (width = 2).
    ``"2"``                  Centroid diff + norm (width = 3).
    ``"3"``                  Centroid diff only (width = 2).
    ``"10"``                 Centroid diff + sigma (width = 2).
    ``"border"``             Border-based FE — rel coords + dist (width = 5).
    ``"border_slow"``        Alias for ``"border"``.
    ``"border_and_dist"``    Border FE + border-rel-to-seed channel (width = 7).
    ``"combined"``           Combined center+border FE for 4-channel embeddings.
                             Produces center_diff[2]+border[2]+sigma[S]+rel[2]+dist[1]
                             per pixel; reported width = 5 (satisfies the shared
                             mlp_input_dim formula: 5 + n_sigma - 2 + 4 = 7+n_sigma).
    """
    if feature_engineering_function == "0" or feature_engineering_function == "7":
        return feature_engineering, 2
    elif feature_engineering_function == "2":
        return feature_engineering_2, 3
    elif feature_engineering_function == "3":
        return feature_engineering_3, 2
    elif feature_engineering_function == "10":
        return feature_engineering_10, 2
    elif feature_engineering_function == "border":
        return feature_engineering_border, 5
    elif feature_engineering_function == "border_slow":
        return feature_engineering_border_slow, 5
    elif feature_engineering_function == "border_and_dist":
        return feature_engineering_border_and_dist, 7
    elif feature_engineering_function == "combined":
        # Actual output: center_diff(2) + border(2) + sigma(S) + rel_coords(2) + dist(1) = 7+S
        # The shared formula adds embedding_vector_dim (4) and n_sigma (S) and subtracts 2,
        # so width must satisfy: width + n_sigma - 2 + 4 = 7 + n_sigma  →  width = 5
        return feature_engineering_combined, 5
    else:
        raise NotImplementedError("Feature engineering function " + str(feature_engineering_function) + " is not implemented")


class IdentityTransform:
    def augment_image(self, img):
        return img
    def deaugment_mask(self, mask):
        return mask


class InstanSeg_Torchscript(nn.Module):
    def __init__(self, model, 
                 cells_and_nuclei: bool = False,
                 pixel_size: float = 0, 
                 n_sigma: int = 2, 
                 dim_coords: int = 2, 
                 to_centre: bool = True,
                 backbone_dim_in: int = 3,  
                 feature_engineering_function: str = "0",
                 params = None):
        super(InstanSeg_Torchscript, self).__init__()

        if cells_and_nuclei:
            import warnings
            warnings.warn(
                "cells_and_nuclei=True is deprecated in InstanSeg_Torchscript. "
                "InstanSeg now performs cell segmentation only. "
                "This flag is preserved only for loading pre-trained joint models; "
                "the forward pass will return the cells channel only.",
                DeprecationWarning,
                stacklevel=2,
            )

        model.eval()
        use_mixed_precision = True

        with torch.amp.autocast("cuda", enabled=use_mixed_precision):
            with torch.no_grad():
                self.fcn = torch.jit.trace(model, torch.rand(1, backbone_dim_in, 256, 256))

        try:
            self.pixel_classifier = model.pixel_classifier
        except:
            self.pixel_classifier = model.model.pixel_classifier
        self.cells_and_nuclei = cells_and_nuclei
        self.pixel_size = pixel_size
        self.dim_coords = dim_coords
        self.n_sigma = n_sigma
        self.to_centre = to_centre
        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        if feature_engineering_function == "0" or feature_engineering_function == "7":
            self.fe_type = 0
        elif feature_engineering_function == "2":
            self.fe_type = 2
        elif feature_engineering_function == "3":
            self.fe_type = 3
        elif feature_engineering_function == "10":
            self.fe_type = 10
        elif feature_engineering_function in ["border", "border_slow"]:
            self.fe_type = 11
        elif feature_engineering_function == "border_and_dist":
            self.fe_type = 12
        else:
            self.fe_type = 0
        self.params = params or {}
        self.index_dtype = torch.long

        self.default_target_segmentation = self.params.get('target_segmentation', torch.tensor([1, 1]))
        self.default_min_size = self.params.get('min_size', 10)
        self.default_mask_threshold = self.params.get('mask_threshold', 0.53)
        self.default_peak_distance = int(self.params.get('peak_distance', 5))
        self.default_seed_threshold = self.params.get('seed_threshold', 0.7)
        self.default_overlap_threshold = self.params.get('overlap_threshold', 0.3)
        self.default_mean_threshold = self.params.get('mean_threshold', 0.0)
        self.default_window_size = self.params.get('window_size', 32)
        self.default_cleanup_fragments = self.params.get('cleanup_fragments', False)

    def forward(self, x: torch.Tensor,
                args: Optional[Dict[str, torch.Tensor]] = None,
                target_segmentation: torch.Tensor = torch.tensor([1, 1]),
                min_size: Optional[int] = None,
                mask_threshold: Optional[float] = None,
                peak_distance: Optional[int] = None,
                seed_threshold: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                mean_threshold: Optional[float] = None,
                window_size: Optional[int] = None,
                cleanup_fragments: Optional[bool] = None,
                precomputed_seeds: torch.Tensor = torch.tensor([]),
                ) -> torch.Tensor:
        
        min_size = int(min_size) if min_size is not None else self.default_min_size
        mask_threshold = float(mask_threshold) if mask_threshold is not None else self.default_mask_threshold
        peak_distance = int(peak_distance) if peak_distance is not None else self.default_peak_distance
        seed_threshold = float(seed_threshold) if seed_threshold is not None else self.default_seed_threshold
        overlap_threshold = float(overlap_threshold) if overlap_threshold is not None else self.default_overlap_threshold
        mean_threshold = float(mean_threshold) if mean_threshold is not None else self.default_mean_threshold
        window_size = int(window_size) if window_size is not None else self.default_window_size
        cleanup_fragments = bool(cleanup_fragments) if cleanup_fragments is not None else self.default_cleanup_fragments

        if args is None:
            args = {"None": torch.tensor([0])}

        target_segmentation = args.get('target_segmentation', target_segmentation)
        min_size = int(args.get('min_size', torch.tensor(float(min_size))).item())
        mask_threshold = args.get('mask_threshold', torch.tensor(mask_threshold)).item()
        peak_distance = args.get('peak_distance', torch.tensor(peak_distance)).item()
        seed_threshold = args.get('seed_threshold', torch.tensor(seed_threshold)).item()
        overlap_threshold = args.get('overlap_threshold', torch.tensor(overlap_threshold)).item()
        mean_threshold = args.get('mean_threshold', torch.tensor(mean_threshold)).item()
        window_size = int(args.get('window_size', torch.tensor(float(window_size))).item())
        cleanup_fragments = args.get('cleanup_fragments', torch.tensor(cleanup_fragments)).item()
        precomputed_seeds = args.get('precomputed_seeds', precomputed_seeds)

        torch.clamp_max_(x, 3)
        torch.clamp_min_(x, -2)

        x, pad = _instanseg_padding(x, extra_pad=0)

        with torch.no_grad():
            x_full = self.fcn(x)
            dim_out = x_full.shape[1]

            if self.cells_and_nuclei:
                dim_out = int(dim_out / 2)

            output_labels_list = []

            for image_index in range(x_full.shape[0]):
                if self.cells_and_nuclei:
                    # Cell segmentation channel is the second half
                    x = x_full[image_index, dim_out:, :, :]
                else:
                    x = x_full[image_index, 0:dim_out, :, :]

                x = _recover_padding(x, pad)
                height, width = x.size(1), x.size(2)

                xxyy = generate_coordinate_map(mode="linear", spatial_dim=self.dim_coords, height=height, width=width, device=x.device)

                vectors = x[0:self.dim_coords]
                sigma = x[self.dim_coords:self.dim_coords + self.n_sigma]
                border_distance_map = ((x[self.dim_coords + self.n_sigma]) / 15) + 0.5

                if precomputed_seeds is None or precomputed_seeds.shape[0] == 0:
                    centroids_idx = torch_peak_local_max(border_distance_map, neighbourhood_size=peak_distance,
                                                        minimum_value=seed_threshold, dtype=self.index_dtype)
                else:
                    centroids_idx = precomputed_seeds.to(border_distance_map.device).long()

                if self.to_centre or not self.to_centre:
                    vectors_at_centroids = vectors[:, centroids_idx[:, 0], centroids_idx[:, 1]]

                x = vectors
                c = vectors_at_centroids.T
                E = x.shape[0]
                h, w = x.shape[-2:]
                C = c.shape[0]
                S = sigma.shape[0]

                if C == 0:
                    label = torch.zeros(border_distance_map.shape, dtype=torch.float32, device=border_distance_map.device).squeeze()
                    output_labels_list.append(label)
                    continue

                centroids = centroids_idx.clone().cpu()
                centroids[:, 0].clamp_(min=window_size, max=h - window_size)
                centroids[:, 1].clamp_(min=window_size, max=w - window_size)
                window_slices = centroids[:, None].to(x.device) + torch.tensor([[-1, -1], [1, 1]], device=x.device, dtype=centroids.dtype) * window_size

                slice_size = window_size * 2

                grid_x, grid_y = torch.meshgrid(
                    torch.arange(slice_size, device=x.device, dtype=self.index_dtype),
                    torch.arange(slice_size, device=x.device, dtype=self.index_dtype), indexing="ij")
                mesh = torch.stack((grid_x, grid_y))

                mesh_grid = mesh.expand(C, 2, slice_size, slice_size)
                mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)
                idx = window_slices[:, 0].permute(1, 0)[:, :, None]
                mesh_grid_flat = mesh_grid_flat + idx
                mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)

                if self.fe_type == 11 or self.fe_type == 12:
                    dist_map = border_distance_map.unsqueeze(0)
                else:
                    dist_map = sigma

                if self.fe_type == 11:
                    x = feature_engineering_border(x, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)
                elif self.fe_type == 12:
                    x = feature_engineering_border_and_dist(x, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)
                elif self.fe_type == 2:
                    x = feature_engineering_2(x, xxyy, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)
                elif self.fe_type == 3:
                    x = feature_engineering_3(x, xxyy, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)
                elif self.fe_type == 10:
                    x = feature_engineering_10(x, xxyy, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)
                else:
                    x = feature_engineering_slow(x, c, dist_map, torch.tensor(window_size).int(), mesh_grid_flat)

                x = torch.sigmoid(self.pixel_classifier(x))
                x = x.reshape(C, 1, slice_size, slice_size)

                C = x.shape[0]
                if C == 0:
                    label = torch.zeros(border_distance_map.shape, dtype=torch.float32, device=border_distance_map.device).squeeze()
                    output_labels_list.append(label)
                    continue

                original_device = x.device
                if x.is_mps:
                    device = 'cpu'
                    mesh_grid_flat = mesh_grid_flat.to(device)
                    x = x.to(device)
                    mask_map = mask_map.to(device)

                coords = mesh_grid_flat.reshape(2, C, slice_size, slice_size)

                if cleanup_fragments:
                    top_left = window_slices[:, 0, :]
                    shifted_centroid = centroids_idx - top_left
                    cc = connected_components((x > mask_threshold).float(), num_iterations=64)
                    labels_to_keep = cc[torch.arange(cc.shape[0]), 0, shifted_centroid[:, 0], shifted_centroid[:, 1]]
                    in_mask = cc == labels_to_keep[:, None, None, None]
                    x *= in_mask

                labels = convert(x, coords, size=(h, w), mask_threshold=mask_threshold)[None]

                idx = torch.arange(1, C + 1, device=x.device, dtype=self.index_dtype)
                stack_ID = torch.ones((C, slice_size, slice_size), device=x.device, dtype=self.index_dtype)
                stack_ID = stack_ID * (idx[:, None, None] - 1)

                iidd = torch.stack((stack_ID.flatten(), mesh_grid_flat[0] * w + mesh_grid_flat[1]))

                fg = x.flatten() > mask_threshold
                x = x.flatten()[fg]
                sparse_onehot = torch.sparse_coo_tensor(
                    iidd[:, fg],
                    (x.flatten() > mask_threshold).float(),
                    size=(C, h * w),
                    dtype=x.dtype,
                    device=x.device
                )

                object_areas = torch.sparse.sum(sparse_onehot.to(torch.bool).float(), dim=(1,)).values()
                sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map.flatten()[None]), dim=(1,)).values()
                mean_mask_value = sum_mask_value / object_areas
                objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

                iou = fast_sparse_iou(sparse_onehot)

                remapping = find_connected_components((iou > overlap_threshold).to(self.index_dtype))
                labels = remap_values(remapping, labels)

                labels_to_remove = (torch.arange(0, len(objects_to_remove), device=objects_to_remove.device) + 1)[
                    objects_to_remove]
                labels[torch.isin(labels, labels_to_remove)] = 0

                output_labels_list.append(labels.squeeze().to(original_device))

            lab = torch.stack(output_labels_list)
            lab = lab.unsqueeze(1)
            return lab.to(torch.float32)
