import torch
import numpy as np

from typing import Tuple, List, Union

from instanseg.utils.loss.lovasz_losses import binary_xloss
from instanseg.utils.pytorch_utils import (
    torch_fastremap, torch_onehot, remap_values,
    fast_iou, fast_sparse_iou, eccentricity_batch, connected_components,
    instance_wise_border_vectors_and_distances,
)
from instanseg.utils.embedding_modes import (
    embedding_uses_border,
    embedding_uses_center,
    embedding_uses_clustering,
    embedding_vector_channels,
    validate_embedding_mode,
)
from instanseg.utils.tiling import _instanseg_padding, _recover_padding

import torch.nn.functional as F
import torch.nn as nn

binary_xloss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

from instanseg.utils.utils import show_images, timer

# Post-processing helpers and classes are defined in postprocessing.py and imported here
from instanseg.utils.postprocessing import (
    convert,
    find_all_local_maxima,
    torch_peak_local_max,
    centre_crop,
    find_connected_components,
    merge_sparse_predictions,
    generate_coordinate_map,
    filter_small_blobs,
    has_pixel_classifier_model,
    guide_function,
    compute_crops,
    ProbabilityNet,
    MyBlock,
    ConvProbabilityNet,
    MedianFilter,
    feature_engineering,
    feature_engineering_slow,
    feature_engineering_border,
    feature_engineering_border_slow,
    feature_engineering_border_and_dist,
    feature_engineering_2,
    feature_engineering_3,
    feature_engineering_10,
    feature_engineering_combined,
    feature_engineering_generator,
    IdentityTransform,
    InstanSeg_Torchscript,
    cluster_embedding_pixels,
    compute_boundary_evidence,
)

integer_dtype = torch.int64




class InstanSeg(nn.Module):

    def __init__(self,
                 n_sigma: int = 1, 
                 instance_weight: float = 1.5, 
                 device: str = 'cuda', 
                 binary_loss_fn_str: str = "lovasz_hinge", 
                 seed_loss_fn = "binary_xloss", 
                 cells_and_nuclei: bool = False, 
                 to_centre: bool = True, 
                 multi_centre: bool = False,
                 window_size = 256, 
                 tile_size = 256,
                 feature_engineering_function = "0",
                 dim_coords = 2,
                 border_vector_weight: float = 1.0,
                 embedding_mode: str = "center-seed"):
        
        super().__init__()
        self.n_sigma = n_sigma
        self.instance_weight = instance_weight
        self.device = device
        self.dim_coords = dim_coords
        self.embedding_mode = validate_embedding_mode(embedding_mode)
        self.embedding_vector_dim = embedding_vector_channels(self.embedding_mode, self.dim_coords)

        self.dim_out = self.embedding_vector_dim + self.n_sigma + 1
        self.parameters_have_been_updated = False

        if cells_and_nuclei:
            import warnings
            warnings.warn(
                "cells_and_nuclei=True is deprecated in InstanSeg loss. "
                "InstanSeg now performs cell segmentation only. "
                "The dual-channel loss path is preserved only for loading "
                "pre-trained joint models; do not use for new training.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.dim_out = self.dim_out * 2
        self.cells_and_nuclei = cells_and_nuclei

        self.to_centre = to_centre
        self.multi_centre = multi_centre
        self.window_size = window_size

        self.num_instance_cap = 50
        self.sort_by_eccentricity = False

        xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = tile_size, width = tile_size, device = device)

        # For combined modes, auto-select the dedicated combined FE unless the
        # caller explicitly requested a different one.
        if (
            self.embedding_mode in ("combined-center", "combined-cluster")
            and feature_engineering_function == "0"
        ):
            feature_engineering_function = "combined"

        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        self.feature_engineering_function = feature_engineering_function
        self._is_border_fe = feature_engineering_function in ("border", "border_slow", "border_and_dist")


        self.register_buffer("xxyy", xxyy)

        self.update_binary_loss(binary_loss_fn_str)
        self.vector_loss = torch.nn.L1Loss(reduction='none')
        self.border_vector_weight = border_vector_weight
        self.update_seed_loss(seed_loss_fn)

    def update_binary_loss(self,binary_loss_fn_str):

        if binary_loss_fn_str == "lovasz_hinge":
            from instanseg.utils.loss.lovasz_losses import lovasz_hinge
            def binary_loss_fn(pred, gt, **kwargs):
               # pred = torch.sigmoid_(pred)
                return lovasz_hinge((pred.squeeze(1)), gt,per_image = True)

        elif binary_loss_fn_str == "binary_xloss":
            from instanseg.utils.loss.lovasz_losses import binary_xloss
            self.binary_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif binary_loss_fn_str == "dicefocal_loss":
            from monai.losses import DiceFocalLoss
            
            binary_loss_fn_ = DiceFocalLoss(sigmoid=True)
            def binary_loss_fn(pred, gt, **kwargs):
                l = binary_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l
        elif binary_loss_fn_str == "dice_loss":
            from monai.losses import DiceLoss
            
            binary_loss_fn_ = DiceLoss(sigmoid=True)
            def binary_loss_fn(pred, gt, **kwargs):
                l = binary_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l


        elif binary_loss_fn_str == "general_dice_loss":
            from monai.losses import GeneralizedDiceLoss
            def binary_loss_fn(pred, gt):
                return GeneralizedDiceLoss(sigmoid=True)(pred, gt.unsqueeze(1))
            

        elif binary_loss_fn_str == "cross_entropy":
            from torch.nn import NLLLoss
            assert self.window_size == 256, "Cross entropy loss only works with window size 256"
            assert self.num_instance_cap is None, "Cross entropy loss only works with num_instance_cap = None"
            
            self.l_fn = NLLLoss()
            self.m = nn.LogSoftmax(dim=1)
            
            def binary_loss_fn(pred, gt, sigma):
                pred = torch.cat([sigma[None,None],pred])
                

                gt = torch.cat(((gt.sum(0)==0)[None],gt))
                target = gt.argmax(0)[None]

                pred = pred.squeeze(1).unsqueeze(0)
              
                pred = self.m(pred)

                return self.l_fn(pred,target.long()) * 7
            
        else:
            raise NotImplementedError("Binary loss function",binary_loss_fn,"is not implemented")
        self.binary_loss_fn = binary_loss_fn

    def update_seed_loss(self,seed_loss_fn):
        if seed_loss_fn in ["binary_xloss"]:
            binary_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

            def seed_loss(x,y, mask = None):
                if mask is not None:
                    mask = mask.float()  # Ensure the mask is float for multiplication
                    loss = binary_loss(x, (y > 0).float())  # Calculate the element-wise binary loss
                    masked_loss = loss * mask  # Apply the mask to the loss
                    return masked_loss.sum() / mask.sum()
                else:
                    return binary_loss(x, (y > 0).float()).mean()
                
            self.seed_loss = seed_loss

        elif seed_loss_fn in ["l1_distance"]:
            distance_loss = torch.nn.L1Loss(reduction='none')
            def seed_loss(x, y, mask = None):
                _, border_distance = instance_wise_border_vectors_and_distances(y.float(), normalize=True)
                target = (border_distance - 0.5) * 15
                loss = distance_loss(x, target)

                if mask is not None:
                    mask = mask.float()
                    masked_loss = loss * mask
                    return masked_loss.sum() / mask.sum()
                else:
                    return loss.mean()

            self.seed_loss = seed_loss
        else:
            raise NotImplementedError("Seedloss function",seed_loss_fn,"is not implemented")

    def _center_vectors_and_centroids(self, instance: torch.Tensor, xxyy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        instance_ids = torch.unique(instance)
        instance_ids = instance_ids[instance_ids > 0]
        h, w = instance.shape[-2:]
        center_vectors = torch.zeros((self.dim_coords, h, w), dtype=torch.float32, device=instance.device)
        centroids = []

        for instance_id in instance_ids:
            mask = instance[0] == instance_id
            if not mask.any():
                continue
            coords = xxyy[:, mask]
            center = coords.mean(dim=1)
            center_vectors[:, mask] = center[:, None] - xxyy[:, mask]
            rows_cols = torch.nonzero(mask, as_tuple=False).float()
            centroids.append(rows_cols.mean(dim=0))

        if centroids:
            centroid_idx = torch.stack(centroids).to(device=instance.device)
        else:
            centroid_idx = torch.zeros((0, 2), dtype=torch.float32, device=instance.device)
        return center_vectors, centroid_idx

    def generate_embedding_targets(self, instance: torch.Tensor, xxyy: torch.Tensor = None) -> torch.Tensor:
        """Generate center, border, or combined vector targets for one label mask."""
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        if instance.ndim != 3 or instance.shape[0] != 1:
            raise ValueError("instance must have shape (H, W) or (1, H, W)")
        if instance.numel() == 0:
            raise ValueError("instance mask must not be empty")
        if torch.isnan(instance.float()).any():
            raise ValueError("instance mask contains NaN values")

        h, w = instance.shape[-2:]
        if xxyy is None:
            xxyy = generate_coordinate_map(
                mode="linear",
                spatial_dim=self.dim_coords,
                height=h,
                width=w,
                device=instance.device,
            )
        if xxyy.shape[-2:] != (h, w):
            raise ValueError("xxyy spatial shape must match instance shape")

        targets = []
        if embedding_uses_center(self.embedding_mode):
            center_vectors, _ = self._center_vectors_and_centroids(instance, xxyy[: self.dim_coords])
            targets.append(center_vectors)
        if embedding_uses_border(self.embedding_mode):
            border_vectors, _ = instance_wise_border_vectors_and_distances(instance.float(), normalize=True)
            targets.append(border_vectors[: self.dim_coords])
        if not targets:
            raise ValueError("No embedding targets were generated")
        return torch.cat(targets, dim=0)

    def initialize_pixel_classifier(self, model, MLP_width = 10, MLP_input_dim = None):

        if has_pixel_classifier_model(model):
            try:
                self.pixel_classifier = model.pixel_classifier
            except:
                self.pixel_classifier = model.model.pixel_classifier  # This happens when there is an adaptornet
            return model
        else:
            if MLP_input_dim is None:
                MLP_input_dim = self.feature_engineering_width + self.n_sigma - 2 + self.embedding_vector_dim
            model.pixel_classifier = ProbabilityNet( MLP_input_dim, width = MLP_width)
            if self.feature_engineering_function != "10":
                model.pixel_classifier = ProbabilityNet( MLP_input_dim, width = MLP_width)
            else:
                model.pixel_classifier = ConvProbabilityNet( MLP_input_dim, width = MLP_width)
            self.pixel_classifier = model.pixel_classifier.to(self.device)
            return model

    def forward(self, prediction: torch.Tensor, instances: torch.Tensor, w_inst: float = 1.5, w_seed: float = 1.0):

        w_inst = self.instance_weight

        if instances.dim() == 3:
            instances = instances.unsqueeze(1)
        while instances.dim() > 4 and instances.shape[1] == 1:
            instances = instances.squeeze(1)
        if instances.dim() != 4:
            raise ValueError("instances must have shape (B, C, H, W) or (B, H, W)")

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xxyy = self.xxyy[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
  
        if self.cells_and_nuclei:
            dim_out = int(self.dim_out / 2)
        else:
            dim_out = self.dim_out

        for mask_channel in range(0, instances.shape[1]):

            if mask_channel == 0:
                prediction_b = prediction[:, 0: dim_out, :, :]
            else:
                prediction_b = prediction[:, dim_out:, :, :]

            instances_batch = instances

            if not self.to_centre:
                spatial_emb_batch = (torch.sigmoid((prediction_b[:, 0: self.embedding_vector_dim]))-0.5) * 8
            else:
                spatial_emb_batch = prediction_b[:, 0: self.embedding_vector_dim]
            sigma_batch = prediction_b[:, self.embedding_vector_dim: self.embedding_vector_dim + self.n_sigma]  # n_sigma x h x w
            border_distance_batch = prediction_b[:, self.embedding_vector_dim + self.n_sigma: self.embedding_vector_dim + self.n_sigma + 1]  # 1 x h x w

            for b in range(0, batch_size):

                spatial_emb = spatial_emb_batch[b]
                sigma = sigma_batch[b]
                border_distance_pred = border_distance_batch[b]

                instance_loss = 0
                seed_loss = 0
                vector_loss_val = 0

                instance = instances_batch[b, mask_channel].unsqueeze(0)  # 1 x h x w

                if (instance < 0).all(): #-1 means not annotated
                    continue
       

                elif instance.min() < 0: #label is sparse
                    mask = instance >=0
                    instance[instance < 0] = 0
                else:
                    mask = None

                _, border_distance = instance_wise_border_vectors_and_distances(instance.float(), normalize=True)
                embedding_target = self.generate_embedding_targets(instance, xxyy=xxyy)
                valid_mask = (instance > 0)
                if mask is not None:
                    valid_mask = valid_mask & mask

                seed_loss_tmp = self.seed_loss(border_distance_pred, instance, mask = mask)
                seed_loss += seed_loss_tmp

                if valid_mask.any():
                    if spatial_emb.shape[0] != embedding_target.shape[0]:
                        raise ValueError(
                            f"prediction has {spatial_emb.shape[0]} embedding channels, "
                            f"but {self.embedding_mode} targets require {embedding_target.shape[0]}"
                        )
                    vector_loss_map = self.vector_loss(spatial_emb, embedding_target)
                    vector_loss_map = vector_loss_map * valid_mask.float()
                    vector_loss_tmp = vector_loss_map.sum() / valid_mask.float().sum()
                    vector_loss_val = self.border_vector_weight * vector_loss_tmp

                if w_inst == 0:
                    loss += w_seed * seed_loss + vector_loss_val
                    continue

                instance_ids = instance.unique()
                instance_ids = instance_ids[instance_ids != 0]

                if len(instance_ids) > 0:

                    instance = torch_fastremap(instance)

                    onehot_labels = torch_onehot(instance).squeeze(0)  # C x h x w

                    if self.num_instance_cap is not None: #This is to cap the number of objects to avoid OOM errors.
                         if self.num_instance_cap < onehot_labels.shape[0]:
                            if self.sort_by_eccentricity:
                                eccentricities = eccentricity_batch(onehot_labels.float())
                                idx = eccentricities.argsort(descending = True)[:self.num_instance_cap]
                            else:
                                idx = torch.arange(onehot_labels.shape[0], device=onehot_labels.device)[:self.num_instance_cap]
                            onehot_labels = onehot_labels[idx]


                    if self.multi_centre:
                        border_distance_tmp = torch.sigmoid(border_distance_pred)
                        
                        centroids = torch_peak_local_max(border_distance_tmp.squeeze() * onehot_labels.sum(0), neighbourhood_size = 3, minimum_value = 0.5).T

                        if self.to_centre and self.embedding_mode in ("center-seed", "center-cluster"):
                            centres = xxyy[:,centroids[0],centroids[1]].detach().T
                        else:
                            centres = spatial_emb[:,centroids[0],centroids[1]].detach().T

                        idx = torch.arange(centroids.shape[1], device=centroids.device)[:self.num_instance_cap]
                  
                        centres = centres[idx]
                        centroids = centroids[:,idx]

                        instance_labels = onehot_labels[:,centroids[0],centroids[1]].float().argmax(0)
                        onehot_labels = onehot_labels[instance_labels]

                        centroids = centroids.T
             
                    else:
                        if self.to_centre and self.embedding_mode in ("center-seed", "center-cluster"):

                            border_distance_min = border_distance_pred.min()
           
                            border_distance_tmp = (border_distance_pred - border_distance_min).detach()
                            centres = xxyy.flatten(1).T[((border_distance_tmp * onehot_labels).flatten(1)).argmax(1)]  # location at max border distance (medial axis)
                            border_distance_pred = border_distance_tmp + border_distance_min
                        else:
                            border_distance_tmp = border_distance_pred - border_distance_pred.min()
                            centres = spatial_emb.flatten(1).T[((border_distance_tmp * onehot_labels).flatten(1)).argmax(1)].detach()  # embedding at max border distance (medial axis)

                        if self._is_border_fe:
                            # For border-based FE, anchor crops at the medial-axis
                            # peak (border distance maximum) rather than at the
                            # centre of mass, so the crop is centered on the seed.
                            medial_axis_idx = ((border_distance_tmp * onehot_labels).flatten(1)).argmax(1)
                            h_tmp, w_tmp = border_distance_pred.shape[-2:]
                            centroids_row = medial_axis_idx // w_tmp
                            centroids_col = medial_axis_idx % w_tmp
                            centroids = torch.stack((centroids_row, centroids_col), dim=1).float()
                        else:
                            centroids = (torch.sum(((xxyy[:2] * onehot_labels.unsqueeze(1))).flatten(2),dim=2)/onehot_labels.flatten(1).sum(1)[:,None] )* (256 / 64) #coordinates of centre of mass
                            centroids = torch.stack((centroids[:, 1], centroids[:, 0])).T

                    if len(centroids) == 0:
                        loss += w_seed * seed_loss
                        continue

                    effective_window_size = min(self.window_size, *spatial_emb.shape[-2:])
                    effective_window_size = effective_window_size - (effective_window_size % 2)

                    # For border-based FE, use the border distance map
                    # instead of learned sigma as the auxiliary channel.
                    if self._is_border_fe:
                        crop_sigma = ((border_distance_pred) / 15 + 0.5)
                    else:
                        crop_sigma = sigma

                    dist, coords = compute_crops(spatial_emb, 
                                                 centres, 
                                                 crop_sigma, 
                                                 centroids, 
                                                 feature_engineering = self.feature_engineering,
                                                 pixel_classifier=self.pixel_classifier,
                                                 window_size = effective_window_size)
                    
                    crop = onehot_labels.squeeze(1)[coords[0], coords[1], coords[2]].reshape(-1, effective_window_size, effective_window_size)

                    instance_loss = instance_loss + self.binary_loss_fn(dist,crop.float(), sigma = sigma[0])

                loss += w_inst * instance_loss + w_seed * seed_loss + vector_loss_val


        loss = loss / (b + 1)

        if self.cells_and_nuclei:
            loss = loss / 2

        return loss
    
    

    def update_hyperparameters(self,params):
        self.parameters_have_been_updated = True
        self.params = params



    #@timer
    def postprocessing(self, prediction: Union[torch.Tensor, np.ndarray],
                        mask_threshold: float = 0.53,
                        peak_distance: int = 5,
                        seed_threshold: float = 0.8,
                        overlap_threshold: float = 0.3,
                        mean_threshold: float = 0.1,
                        window_size: int = 128,
                        min_size = 10,
                       device=None,
                       classifier=None,
                       cleanup_fragments: bool = False,
                       max_seeds: int = 2000,
                       return_intermediate_objects: bool = False,
                       precomputed_crops: torch.Tensor = None,
                       precomputed_seeds: torch.Tensor = None,
                       img=None):

        if device is None:
            device = self.device
        if classifier is None and not embedding_uses_clustering(self.embedding_mode):
            classifier = self.pixel_classifier

        if self.parameters_have_been_updated:
            mask_threshold = self.params['mask_threshold']
            peak_distance = self.params['peak_distance']
            seed_threshold = self.params['seed_threshold']
            overlap_threshold = self.params['overlap_threshold']
            if "min_size" in self.params:
                min_size = self.params['min_size']
            if "mean_threshold" in self.params:
                mean_threshold = self.params['mean_threshold']
            

        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction, device=device)

        if self.cells_and_nuclei:
            iterations = 2
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        labels = []

        for i in range(iterations):

            if precomputed_crops is None:

                if i == 0:
                    prediction_i = prediction[0: dim_out, :, :]
                else:
                    prediction_i = prediction[dim_out:, :, :]

                height, width = prediction_i.size(1), prediction_i.size(2)

                ##torch.cuda.synchronize()

                xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = height, width = width, device = device)

                #torch.cuda.synchronize()

                if not self.to_centre:
                    fields = (torch.sigmoid(prediction_i[0:self.embedding_vector_dim])-0.5) * 8
                else:
                    fields = prediction_i[0:self.embedding_vector_dim]

                sigma = prediction_i[self.embedding_vector_dim:self.embedding_vector_dim + self.n_sigma]
            #    mask_map = torch.sigmoid(prediction_i[self.dim_coords + self.n_sigma])

                mask_map = ((prediction_i[self.embedding_vector_dim + self.n_sigma]) / 15) + 0.5

                if (mask_map > mask_threshold).max() == 0:  # no foreground pixels
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                if embedding_uses_clustering(self.embedding_mode):
                    cluster_fields = fields
                    if self.embedding_mode in ("center-cluster", "combined-cluster"):
                        cluster_fields = cluster_fields.clone()
                        cluster_fields[: self.dim_coords] = cluster_fields[: self.dim_coords] + xxyy
                    label = cluster_embedding_pixels(
                        cluster_fields,
                        mask_map > mask_threshold,
                        affinity_threshold=0.65,
                        sigma_e=1.0,
                        sigma_g=2.0,
                        min_size=min_size,
                    )
                    labels.append(label.squeeze())
                    continue

            # local_centroids_idx = #torch.tensor([[20,21,32,32,34],[30,35,36,364,346]],device = device).long().T

                #torch.cuda.synchronize()

                if precomputed_seeds is None:
                    local_centroids_idx = torch_peak_local_max(mask_map, neighbourhood_size=int(peak_distance), minimum_value=seed_threshold)
                else:
                    local_centroids_idx = precomputed_seeds


                #torch.cuda.synchronize()

                if self._is_border_fe:
                    # Border-based FE: pass raw border vectors (no xxyy addition)
                    # and use border distance map as the auxiliary channel.
                    crop_fields = fields
                    crop_sigma = mask_map.unsqueeze(0)
                    fields_at_centroids = fields[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]
                else:
                    # Centroid-based FE: add coordinate map and compute centroid embeddings
                    if self.embedding_vector_dim == self.dim_coords:
                        fields = fields + xxyy
                    else:
                        fields = fields.clone()
                        fields[: self.dim_coords] = fields[: self.dim_coords] + xxyy
                    if self.to_centre and self.embedding_mode in ("center-seed", "center-cluster"):
                        fields_at_centroids = xxyy[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]
                    else:
                        fields_at_centroids = fields[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]
                    crop_fields = fields
                    crop_sigma = sigma

                if local_centroids_idx.shape[0] > max_seeds:
                    print("Too many seeds, skipping", local_centroids_idx.shape[0])
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                
                C = fields_at_centroids.shape[0]

                h, w = mask_map.shape[-2:]
                window_size = min(window_size, h, w)
                window_size = window_size - window_size % 2

                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                #torch.cuda.synchronize()
                crops, coords = compute_crops(crop_fields, 
                                                fields_at_centroids.T, 
                                                crop_sigma, 
                                                local_centroids_idx.int(), 
                                                feature_engineering = self.feature_engineering,
                                                pixel_classifier=self.pixel_classifier,
                                                window_size=window_size) # about 65% of the time
                #torch.cuda.synchronize()
                coords = coords[1:] # The first channel are just channel indices, not required here.

                if return_intermediate_objects:
                    return crops, coords, mask_map

                C = crops.shape[0]
                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                

            else:
                crops,coords,mask_map = precomputed_crops
                C = crops.shape[0]



            h, w = mask_map.shape[-2:]

            label = merge_sparse_predictions(crops, coords, mask_map, size=(C,h, w), mask_threshold=mask_threshold, window_size=window_size, min_size=min_size, overlap_threshold=overlap_threshold, mean_threshold=mean_threshold).int() #about 30% of the time


            # from utils.pytorch_utils import centroids_from_lab
            # centroids, ids = centroids_from_lab(label)
            # coords = centre_crop(centroids=centroids, window_size=window_size, h=h, w=w)
            # crops = label[...,coords[0],coords[1]].view(centroids.shape[0],1,window_size,window_size)
            # clean = (crops == ids[1:][:,None,None,None]) * ids[1:][:,None,None,None]

            # clean = connected_components((clean > 0).float(),num_iterations= 128)

            # for ii, cc_map in enumerate(clean):
            #     l= torch.unique(cc_map[cc_map>0], sorted=True)
            #     clean[ii] = cc_map == l[-1]
            # clean = clean.int()
            # clean = clean * torch.arange(clean.shape[0],device = clean.device,dtype = label.dtype)[:,None,None,None]

            # label = convert(clean,coords,size = [h,w]).int()



            labels.append(label.squeeze())


        if len(labels) == 1:
            return labels[0][None]  # 1,H,W
        else:
            return torch.stack(labels)  # 2,H,W
        


    def TTA_postprocessing(self, img, model, transforms,
                        mask_threshold: float = 0.53,
                        peak_distance: int = 5,
                        seed_threshold: float = 0.8,
                        overlap_threshold: float = 0.3,
                        mean_threshold: float = 0.1,
                        window_size: int =64,
                        min_size = 10,
                       device=None,
                       classifier=None,
                       cleanup_fragments: bool = False,
                       reduction = "mean",
                       max_seeds: int = 2000,):

        

        cells_and_nuclei = self.cells_and_nuclei
        if self.cells_and_nuclei:
            iterations = 2
            assert self.dim_out % 2 == 0,  print("The model should an even number of output channels for cells and nuclei.")
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        out_labels = []

        transforms = [t for t in transforms] + [IdentityTransform()]
        
        for i in range(iterations):

            all_masks_list = []
            all_predictions = []

            self.cells_and_nuclei = False

            
            for t in transforms:
                with torch.cuda.amp.autocast():
                    augmented_image = t.augment_image(img)
                    augmented_image, pad = _instanseg_padding(augmented_image, extra_pad= 0, min_dim = 32)
                    prediction = model(augmented_image)[:,i * dim_out:(i+1) * dim_out]
                    prediction = _recover_padding(prediction, pad)
                    mask_map = prediction[:,-1][None] 
                    mask_map = t.deaugment_mask(mask_map)
                #  show_images(mask_map)
                    all_masks_list.append(mask_map.cpu())
                    all_predictions.append(prediction.cpu())

          #  pdb.set_trace()

            if reduction == "local_max":

                local_maxima_maps = [torch_peak_local_max(mask.squeeze().float().to(device), int(peak_distance),seed_threshold, return_map = True) for mask in all_masks_list]
                local_maxima_map = torch_peak_local_max(torch.stack(local_maxima_maps).max(0)[0].squeeze(),int(peak_distance),seed_threshold, return_map = True)
                all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)

            elif reduction in ["mean", "median"]:
                if reduction == "mean":
                    all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)
                elif reduction == "median":
                    all_masks = torch.median(torch.stack(all_masks_list),dim=0)[0].to(device)

                local_maxima_map = torch_peak_local_max(all_masks.squeeze(), neighbourhood_size=int(peak_distance), minimum_value=seed_threshold, return_map = True)
            
            local_maxima_map = (local_maxima_map > 0).float()
            centroids = torch.stack(torch.where(local_maxima_map.squeeze())).T


            if len(centroids) == 0:
                out_labels.append(torch.zeros((1,*all_masks.shape[-2:]), dtype=int, device=device))
                continue

            local_maxima_map[...,centroids[:,0],centroids[:,1]] = torch.arange(1,centroids.shape[0]+1,device = local_maxima_map.device).float()


            all_crops = []

            for (t, prediction) in (zip(transforms, all_predictions)):
                prediction_tmp = prediction.clone().float().to(device)
                prediction_tmp[:,-1] = t.augment_image(all_masks)
                prediction_tmp = prediction_tmp.squeeze(0)

                local_maxima_map_tmp = t.augment_image( local_maxima_map )
                centroids = torch.stack(torch.where(local_maxima_map_tmp.squeeze())).T
                values = local_maxima_map_tmp[...,centroids[:,0],centroids[:,1]]
                centroids = centroids[values.sort()[1]][0,0]

                out = self.postprocessing(prediction_tmp, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier, 
                                                            cleanup_fragments, max_seeds, return_intermediate_objects = True, precomputed_seeds = centroids)
                
                if len(out)==3:
                    crops, coords, mask_map = out
                else:
                    pdb.set_trace()

                crops = t.deaugment_mask(crops)
                all_crops.append(crops.cpu())


         #   show_images(torch.cat([*torch.cat(all_crops,dim = 3)[:50]],dim = 1),colorbar= False)


            all_crops = torch.median(torch.stack(all_crops).float(),dim=0)[0].to(device)
         #   all_crops = torch.mean(torch.stack(all_crops).float(),dim=0).to(device)
         #   all_crops = torch.max(torch.stack(all_crops).float(),dim=0)[0].to(device)


            labels = self.postprocessing(prediction, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier,
                                        cleanup_fragments, max_seeds, precomputed_crops = (all_crops, coords, mask_map))

            
            out_labels.append(labels)
        self.cells_and_nuclei = cells_and_nuclei

        labels = torch.stack(out_labels, dim = 1).squeeze(0)
        #show_images(labels)
        return labels


        # IdentityTransform and InstanSeg_Torchscript have been relocated to instanseg/utils/postprocessing.py






