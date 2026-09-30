from instanseg.utils.utils import show_images, _move_channel_axis
import torchvision
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
from torchvision.transforms import RandomCrop, Resize, RandomPerspective
from monai.transforms import RandGaussianNoise, AdjustContrast
from instanseg.utils.utils import percentile_normalize
import warnings

import time
import fastremap

time_dict = {}


def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        if f.__name__ in time_dict.keys():
            time_dict[f.__name__] += te - ts
        else:
            time_dict[f.__name__] = te - ts
        # print((f.__name__, ))
        return result

    return timed


def measure_average_instance_area(lab):
    if lab.shape[0] == 2:
        median_area = max(measure_average_instance_area(lab[1]),measure_average_instance_area(lab[0]))
        return median_area
    if torch.is_tensor(lab):
        if lab.max() > 0:
            fg = lab[lab > 0]
            median_area = torch.median(torch.unique(fg, return_counts=True)[1])
            return median_area.item()
        else:
            return 0
    else:
        if lab.max() > 0:
            fg = lab[lab > 0]
            median_area = np.median(np.unique(fg, return_counts=True)[1])
            return median_area
        else:
            return 0

def resize_with_log_scale(lab, mean_diameter=30, min_scale=0.25, max_scale=4):
    # Get the original cell diameter
    original_diameter = 2 * np.sqrt(measure_average_instance_area(lab) / np.pi)  # Assuming circular cells
    
    # Generate a logarithmically distributed scale factor
    log_min = np.log2(min_scale)
    log_max = np.log2(max_scale)
    scale_factor = 2 ** (random.uniform(log_min, log_max))
    
    # Calculate the target diameter and corresponding resize scale
    target_diameter = mean_diameter * scale_factor
    scale = target_diameter / original_diameter

    return scale
    



class Augmentations(object):
    def __init__(self, augmentation_dict={},
                 shape=(256, 256), 
                 dim_in=3,
                 nuclei_channel=None,
                 debug=False, 
                 modality=None, 
                 cells_and_nuclei=None,
                 target_segmentation=None,
                 channel_invariant = False,
                 random_seed=None):
        
        self.debug = debug
        self.shape = shape
        self.augmentation_dict = augmentation_dict
        self.modality = modality
        if cells_and_nuclei is not None:
            warnings.warn(
                "cells_and_nuclei is deprecated and ignored; only cell segmentation is supported.",
                DeprecationWarning,
                stacklevel=2,
            )
        if target_segmentation is not None:
            warnings.warn(
                "target_segmentation is deprecated and ignored; only cell segmentation is supported.",
                DeprecationWarning,
                stacklevel=2,
            )
        if nuclei_channel is not None:
            warnings.warn(
                "nuclei_channel is deprecated and ignored; only brightfield and phase-contrast cell segmentation is supported.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.dim_in = dim_in  # Note, this is the number of input channels to the model, not the number of channels in the raw image. (Can be 'None' for channel invariant models)
        self.channel_invariant = channel_invariant

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)


    def to_tensor(self, image, labels=None, normalize=False, amount=None, metadata=None):


        if isinstance(image, np.ndarray):
            if self.debug:
                orig = image.copy()

            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float32)
            if normalize:
                image, _ = self.normalize(image)

            out = torch.tensor(_move_channel_axis(image), dtype=torch.float32)

        elif isinstance(image, torch.Tensor):
            if self.debug:
                orig = torch.clone(image)
            out = _move_channel_axis(image).float()
        
            if normalize:
                out, _ = self.normalize(out)

        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = np.atleast_3d(labels)
                labels = _move_channel_axis(labels)
                for n, lab in enumerate(labels):

                    if not (lab == -1).any():  # convention is to skip labels with a value of -1
                        labels[n] = fastremap.renumber(lab)[0]
                    else:
                        labels[n] = lab

                labels = torch.tensor(labels.astype(np.int32), dtype=torch.int32)
            labels = torch.atleast_3d(labels)
            labels = _move_channel_axis(labels)

        out = out.squeeze()
        out = _move_channel_axis(torch.atleast_3d(out))

    
        if self.debug:
            print("Tensor")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def normalize(self, image: torch.Tensor, labels=None, amount: float = 0., subsampling_factor: int = 1,
                  percentile=0.1, metadata=None):
        out = percentile_normalize(image, subsampling_factor=subsampling_factor, percentile=percentile)

        return out, labels
    

    def extract_hematoxylin_stain(self, image: torch.Tensor, labels=None, amount=0, metadata=None):
        # image should be 3 channel RGB between 0 and 255 (float32)
        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels

        import torchstain
        if self.debug:
            orig = torch.clone(image)

        tensor = (image / (image.max() + 0.001)) * 255

        tensor = torch.clamp(tensor, 0., 255.)

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        try:

            normalizer.HERef += (torch.rand_like(normalizer.HERef) - 0.5) * normalizer.HERef * amount
            normalizer.maxCRef += (torch.rand_like(normalizer.maxCRef) - 0.5) * normalizer.maxCRef * amount
            norm, H, E = normalizer.normalize(I=tensor, stains=True, Io=240)
        except:
            return image,labels

        out = _move_channel_axis(H) / 255.

        if self.debug:
            print("Stain separation")
            show_images([orig, out], titles=["Original", "Transformed"])

        return out, labels

    def normalize_HE_stains(self, image: torch.Tensor, labels=None, amount=0, metadata=None):
        # image should be 3 channel RGB between 0 and 255 (float32)
        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels
        
        assert image.shape[0] == 3
        
        import torchstain
        if self.debug:
            orig = torch.clone(image)

        tensor = (image / (image.max() + 0.001)) * 255

        tensor = torch.clamp(tensor, 0., 255.)

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        try:
            normalizer.HERef += (torch.rand_like(normalizer.HERef) - 0.5) * normalizer.HERef * amount
            normalizer.maxCRef += (torch.rand_like(normalizer.maxCRef) - 0.5) * normalizer.maxCRef * amount
            norm, _, _ = normalizer.normalize(I=tensor, stains=False, Io=240, beta=0.15)
        except:
            return image,labels


        out = _move_channel_axis(norm) / 255.

        if self.debug:
            print("Stain normalization")
            show_images([orig, out], titles=["Original", "Transformed"])

        return out, labels

    def randomJPEGcompression(self, image, labels=None, amount=0, metadata=None):
        """ This function applies random JPEG compression to an image. It is used to simulate the effect of JPEG compression on the image.

        :param image: A tensor of shape C,H,W
        :param labels: any labels associated with the image
        """

        if image.shape[0] != 3:  # Not implemented for >3 channels
            return image, labels

        import io
        from PIL import Image
        from torchvision import transforms
        import torchvision.transforms.functional as F
        import random

        if self.debug:
            orig = torch.clone(image)  # .copy()
        # Convert tensor to PIL image

        image = image.clamp(0, 1) * 255
        image = image.byte()


        image = F.to_pil_image(image)

        # # Apply random JPEG compression
        qf = random.randrange(int(100 * (1 - amount) - 1), 100)
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimize=True)
        outputIoStream.seek(0)

        # # Convert compressed image back to Torch tensor
        out = transforms.ToTensor()(Image.open(outputIoStream))

        if self.debug:
            print("JPEG")
            show_images([orig, out, orig - out], titles=["Original", "Transformed"])

        return out, labels

    def brightness_augment(self, image, labels=None, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)  # .copy()
        rand = 1 + ((torch.rand(len(image)) - 0.5) * amount)
        out = (image.permute(1, 2, 0) * rand).permute(2, 0, 1)
        if self.debug:
            print("Brightness")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def RandGaussianNoise(self, image, labels=None, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()

        out = RandGaussianNoise(prob=1, mean=0.0, std=amount / 5)(image)

        if self.debug:
            print("RandGaussianNoise")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def HistogramNormalize(self, image, labels=None, amount=0, metadata=None):
        from monai.transforms import RandStdShiftIntensity

        if self.debug:
            orig = torch.clone(image)  # .copy()

        normalizer = RandStdShiftIntensity(10)

        out = torch.stack([normalizer(c) for c in image])

        if self.debug:
            print("RandStdShiftIntensity")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels
    

    def AdjustContrast(self, image, labels=None, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()

        out = AdjustContrast(gamma=amount)(image)

        if self.debug:
            print("AdjustContrast")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels
    
    def kornia_base_augmentations(self, image, labels=None, amount=0, metadata=None):
        
        import kornia

       # orig = image.clone()
        #get stats
        min = torch.min(image)
        max = torch.max(image)

        image = image - min
        image = image / (max - min + 0.001)

        image  = torch.nn.Sequential(
                kornia.augmentation.RandomLinearIllumination(gain = (0.1,0.4),p = 1),
                kornia.augmentation.RandomPlasmaContrast(p=0.5),
                kornia.augmentation.RandomSaltAndPepperNoise(),
                kornia.augmentation.RandomMedianBlur(p = 0.2),
                kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
                kornia.augmentation.RandomBoxBlur((3, 3), p=0.2),
                kornia.augmentation.RandomSharpness(sharpness=1, p=0.5,),
              #  kornia.augmentation.RandomInvert(max_val=1.0, p=0.1),
                kornia.augmentation.RandomContrast(contrast=(1, 1.5),p = 0.5),
                kornia.augmentation.RandomGamma(gamma=(0.5, .5), gain=(0.5, 1.5), p=0.5), 
            )(image).squeeze(0)
        
        image = image * (max - min + 0.001) + min

        return image, labels
    def flips(self, image, labels, amount=0, metadata=None):

        amount = 0.5

        if self.debug:
            orig = torch.clone(image)  # .copy()
        if random.random() > (1 - amount):
            image = TF.hflip(image)
            labels = TF.hflip(labels)
        if random.random() > (1 - amount):
            image = TF.vflip(image)
            labels = TF.vflip(labels)
        out = image
        if self.debug:
            print("Flips")
            show_images([orig, out])
        return out, labels

    def rotate(self, image, labels, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)

        angle = int(np.random.choice([180, 90, 270, 0]))
        out = TF.rotate(image, angle)
        labels = TF.rotate(labels, angle)

        if self.debug:
            print("Rotate")
            show_images([orig, out])
        return out, labels

    def adjust_hue(self, image, labels, amount=0, metadata=None):

        assert image.shape[0] == 3

        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels

        if self.debug:
            orig = torch.clone(image)

        out = TF.adjust_hue(image, hue_factor=(torch.rand(1) - 0.5) * amount * 0.5)

        out = torch.nan_to_num(out, nan=0)  # This occasionally produces nan values, which we replace with 0

        if self.debug:
            print("Adjust Hue")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def perspective(self, image, labels, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()
        perspective_transformer = RandomPerspective(distortion_scale=amount / 2, p=1.0,
                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                    fill=float(image.max() if metadata[
                                                                                  "image_modality"] == "Brightfield" else image.min()))
        state = torch.get_rng_state()

        out = perspective_transformer(image)

        perspective_transformer = RandomPerspective(distortion_scale=amount / 2, p=1.0,
                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                    fill=0)

        torch.set_rng_state(state)
        labels = perspective_transformer(labels)

        if self.debug:
            print("perspective")
            show_images([orig, out, labels], )
        return out, labels

    def invert(self, image, labels, amount=0, metadata=None):

        out = 1 - image

        if self.debug:
            print("Inversion")
            show_images([image, out])
        return out, labels


    # @measure_time
    def add_gradient(self, image, labels=None, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)  # .copy()
        _, h, w = image.shape
        xs = torch.linspace(0, 0.5, steps=w)
        ys = torch.linspace(0, 0.5, steps=h)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        if random.random() > 0.5:
            x = TF.hflip(x)
        if random.random() > 0.5:
            y = TF.hflip(y)
        if random.random() > 0.5:
            x = TF.vflip(x)
        if random.random() > 0.5:
            y = TF.vflip(y)
        out = (amount * (np.random.random() * x + np.random.random() * y)) * image.max() + image
        if self.debug:
            print("Gradient")
            show_images([orig, out], titles=["Original", "Transformed"])

        return torch.Tensor(out), labels

    
    def torch_rescale(self, image, 
                      labels=None, 
                      amount=0, 
                      current_pixel_size=None, 
                      requested_pixel_size=None, 
                      crop=True,
                      random_seed=None, 
                      metadata=None,
                      diameter_range = None, 
                      modality = None):
        

        if random_seed is not None:
            torch.manual_seed(random_seed)

        if labels is not None:
            assert image.shape[-2:] == labels.shape[-2:]

        shape = (torch.tensor(image.shape[-2:]) * (1 + ((torch.rand(1) - 0.5) * amount))).int().tolist()

        if metadata is not None:
            if current_pixel_size is None and "pixel_size" in metadata.keys():
                current_pixel_size = metadata["pixel_size"]
            if modality is None and "image_modality" in metadata.keys():
                modality = metadata["image_modality"]
            else:
                modality = None

        if modality is None:
            modality = "Brightfield"
        #    print("Modality not specified in metadata or in function call, assuming Brightfield")

        if current_pixel_size is not None and requested_pixel_size is not None:
            scale = (current_pixel_size / requested_pixel_size)
            shape = (torch.tensor(image.shape[-2:]) * scale).int().tolist()

        if diameter_range is not None and labels is not None:
            scale = resize_with_log_scale(labels, mean_diameter=diameter_range[0], min_scale=diameter_range[1], max_scale=diameter_range[2])
            if scale == 0:
                scale = 1
            shape = (torch.tensor(image.shape[-2:]) * scale).int().tolist()

        resized_data = Resize(size=shape, antialias=True)(image)

        if labels is not None:
            resized_labels = Resize(size=shape, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(labels)

        while self.shape is not None and np.any(np.array(resized_data[0].shape) < self.shape[0]) and crop:
            pad = int((self.shape[0] - min(resized_data[0].shape)) / 2) + 3
            pad = torch.Tensor([pad, resized_data.shape[1], resized_data.shape[2]]).min().int() - 1

            
            resized_data = torch.nn.functional.pad(resized_data, (pad, pad, pad, pad), mode='constant',
                                                   value=image.max() if modality == "Brightfield" else resized_data.min())

            if labels is not None:
                resized_labels = torch.nn.functional.pad(resized_labels, (pad, pad, pad, pad), mode='constant', value = min(abs(resized_labels.min()),0)).to(
                    labels.dtype)
                
              
        if not crop:
            if labels is None:
                return resized_data, None
            else:
                return resized_data, resized_labels

        cropper = RandomCrop(size=self.shape)
        (i, j, h, w) = cropper.get_params(resized_data, output_size=self.shape)
        out_image = resized_data[:, i:i + h, j:j + w]

        if labels is not None:
            out_labels = resized_labels[:, i:i + h, j:j + w]

            assert out_image.shape[-2:] == self.shape and out_labels.shape[-2:] == self.shape

        if self.debug:
            print("Rescaling")
            show_images([image[0], labels, out_image[0], out_labels],
                        titles=["Source", "Target", "Resize source", "Resized target"], n_cols=2, axes=True)
            
        

        if labels is None:
            return out_image.float(), None
        else:
            return out_image.float(), out_labels #torch.Tensor(out_labels)  # .short()
        
        
    def duplicate_grayscale_channels(self, image, labels, metadata=None):
        if self.debug:
            orig = torch.clone(image)
        if image.shape[0] == 1 and self.dim_in != 1 and not self.channel_invariant:
            image = image.repeat(self.dim_in, 1, 1)
        elif image.shape[0] != self.dim_in and not self.channel_invariant:
            raise ValueError(
                "Image has {} channels, but model expects {}, check the augmentations pipeline!".format(image.shape[0],
                                                                                                        self.dim_in))

        if self.debug:
            print("Duplicate channels")
            show_images([orig, image], titles=["Original", "Transformed"])
        return image, labels
    def __call__(self, image, labels, meta=None):

        from instanseg.utils.utils import _estimate_image_modality

        if self.modality is None:
            if meta is not None and ("image_modality" in meta or "modality" in meta):
                image_modality = meta.get("image_modality", meta.get("modality"))
                normalized_modality = str(image_modality).strip().lower().replace("_", "-")
                if normalized_modality in {"brightfield", "chromogenic"}:
                    observed_modality = "Brightfield"
                    if min(image.squeeze().shape) != 3:
                        observed_modality = "phase-contrast"
                elif normalized_modality in {"phase-contrast", "phase contrast", "phasecontrast"}:
                    observed_modality = "phase-contrast"
                else:
                    observed_modality = image_modality
            else:
                observed_modality = _estimate_image_modality(image, labels)
            modality = observed_modality
        else:
            modality = self.modality

        if modality not in {"Brightfield", "phase-contrast"}:
            raise ValueError(
                f"Unsupported image modality {modality!r}; only brightfield and phase-contrast cell segmentation is supported."
            )

        if self.debug:
            print("Observed modality:", modality)


        augmentation_dict = self.augmentation_dict[modality]

        if self.debug:
            print(augmentation_dict.keys())

        pixel_size = meta.get("pixel_size") if meta is not None else None
        if not isinstance(pixel_size, float):
            pixel_size = None

        metadata = {"image_modality": modality, "pixel_size": pixel_size}

        has_been_normalized = False

        for augmentation, values in augmentation_dict.items():

            if np.random.random() < values[0]:

                if augmentation in ["normalize_HE_stains", "extract_hematoxylin_stain", "normalize"]:
                    if not has_been_normalized:
                        amount = values[1] if len(values) > 1 else None
                        image, labels = getattr(self, augmentation)(image, labels, amount=amount, metadata=metadata)
                        has_been_normalized = True
                elif augmentation == "torch_rescale":
                    _, requested_pixel_size, diameter_range = values
                    if not (isinstance(diameter_range, tuple) and len(diameter_range) == 3):
                        diameter_range = None
                        
                    image, labels = self.torch_rescale(image, labels, current_pixel_size=None,
                                                       requested_pixel_size=requested_pixel_size, 
                                                       diameter_range=diameter_range,
                                                       metadata=metadata)

                else:
                    p, *rest = values
                    amount = rest[0] if len(rest) == 1 else None

                    image, labels = getattr(self, augmentation)(image, labels, amount=amount, metadata=metadata)

                assert not image.isnan().any()
        

        image, labels = self.duplicate_grayscale_channels(image,
                                                          labels,
                                                          metadata=metadata)  # This will only catch single channel images fed to a multi channel network and duplicate the channel if required.

        if image.var() > 1e2:
            import warnings
            image = torch.clip(image, min=-1, max=5)
            warnings.warn("Warning, variance of image is very high, check augmentations")

      

        return image, labels





