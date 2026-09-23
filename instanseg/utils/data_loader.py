
import numpy as np
import warnings

def _keep_images(item, args):

    #args.source_dataset = str(args.source_dataset).lower().replace("[","").replace("]","").replace("'","").split(",")

    if args.source_dataset != ["all"] and item[
        'parent_dataset'].lower() not in args.source_dataset:  # remove items that are not of the desired dataset
        return False
    elif 'duplicate' in item.keys() and item['duplicate']:  # remove items that are duplicates
        return False
    elif args.target_segmentation == "N" and "nucleus_masks" not in item.keys():
        return False
    elif args.target_segmentation == "C" and "cell_masks" not in item.keys():
        return False
    else:
        return True
    
 
def _format_labels(item,target_segmentation):
            
    if "cell_masks" in item.keys():
        item["cell_masks"] = get_image(item["cell_masks"])
    
    if "nucleus_masks" in item.keys():
        item["nucleus_masks"] = get_image(item["nucleus_masks"])
 
    elif "masks" in item.keys():
        item["nucleus_masks"] = get_image(item["masks"])
    
    if target_segmentation == "N":
        if "nucleus_masks" not in item.keys():
            c,h,w = item['image'].shape
            labels = np.zeros((h,w)) -1
        else:
            labels = item["nucleus_masks"]
    elif target_segmentation == "C":
        if "cell_masks" not in item.keys():
            c,h,w = item['image'].shape
            labels = np.zeros((h,w)) -1
        else:
            labels = item["cell_masks"]
    elif "N" in target_segmentation and "C" in target_segmentation:
            if "nucleus_masks" in item.keys() and "cell_masks" in item.keys():
                labels = np.stack((item["nucleus_masks"], item["cell_masks"]))
            elif "nucleus_masks" in item.keys() and "cell_masks" not in item.keys():
                labels = item['nucleus_masks']
                if isinstance(labels, np.ndarray):
                    labels = labels.astype(np.int32)
                else:
                    labels = np.array(labels).astype(np.int32)
                labels = np.stack((labels, np.zeros_like(labels) - 1))
            elif "nucleus_masks" not in item.keys() and "cell_masks" in item.keys():
                labels = item['cell_masks']
                if isinstance(labels, np.ndarray):
                    labels = labels.astype(np.int32)
                else:
                    labels = np.array(labels).astype(np.int32)
                labels = np.stack((np.zeros_like(labels) - 1, labels))
            else:
                raise NotImplementedError("No labels found")
    else:
        raise NotImplementedError("Target segmentation not recognized", target_segmentation)
 
    return labels
 


def export_dataset_dict_as_folder(dataset,destination = "benchmarking_data"):
    from collections import defaultdict
    from tqdm import tqdm
    import pandas as pd
    import torch
    import tifffile
    import os
    from instanseg.utils.utils import _move_channel_axis
    from pathlib import Path

    expanded_datasets = Path("../datasets/") / destination


    dataset_dfs = defaultdict(lambda: defaultdict(lambda: list()))

    for fold_name, fold in dataset.items():
        for i, data_info in tqdm(enumerate(fold), ncols=100, total=len(fold)):
            img = data_info['image']
            dataset_name = data_info['parent_dataset']
            if dataset_name == "Aleynik" or dataset_name == "TissueNet":
                continue
            if "nucleus_masks" not in data_info.keys():
                mask = data_info['masks']
            else:
                mask = data_info['nucleus_masks']
            storage_folder = os.path.join(expanded_datasets, dataset_name, fold_name)
            if not os.path.exists(storage_folder):
                os.makedirs(storage_folder)
            tifffile.imwrite(os.path.join(storage_folder, f"img{i}.tiff"), _move_channel_axis(img))
            tifffile.imwrite(os.path.join(storage_folder, f"img{i}_mask.tiff"), mask)
            for key in data_info.keys():
                if key not in ['masks','image', 'nucleus_masks', 'cell_masks']:
                    dataset_dfs[dataset_name][key].append(data_info[key])
            dataset_dfs[dataset_name]["ID"].append(i)


    for df_name, df in dataset_dfs.items():
        print(df_name)
        pd.DataFrame(df).to_csv(os.path.join(expanded_datasets, df_name, f"{df_name}_metadata.csv"))


def get_image(img_object):
    import tifffile
    import os
    from pathlib import Path

    if type(img_object) == str:

        data_path = os.environ["INSTANSEG_DATASET_PATH"]

        img_path = Path(os.path.join(data_path,img_object))

        if Path(img_path).exists():
            img = tifffile.imread(img_path)
            return img
        else:
           
            if Path(str(Path(img_path).parents[1]) + ".zip").exists():
                import shutil
                import os
                print("Inflating zip file")

                print((str(Path(img_path).parents[1]) + ".zip"))

                shutil.unpack_archive(str(Path(img_path).parents[1]) + ".zip", Path(img_path).parents[2])
            
            #breakpoint()
            img = tifffile.imread(img_path)
            return img
    else:
        return img_object



def _read_images_from_path(data_path= "../datasets", 
                          dataset = "segmentation", 
                          data_slice = None, 
                          dummy = False, 
                          args = None, 
                          sets = ["Train","Validation"], 
                          ):
    
    from pathlib import Path
    import os
    import skimage.io

    datasets_available = sorted(os.listdir(data_path))
    print("Datasets available ", datasets_available)

    source_dataset = args.source_dataset

    assert len(sets) !=2, "Only one set can be loaded at a time"
    data_dicts = {}

    source_dataset = str(args.source_dataset).lower().replace("[","").replace("]","").replace("'","").split(",")


    for folder in datasets_available:
        if folder.lower() in source_dataset:
            dataset_path = Path(data_path) / folder
            for _set in sets:

                if _set not in data_dicts.keys():
                    data_dicts[_set] = [[],[],[]]

                for image_str in sorted(os.listdir(dataset_path / f"{_set}")):
                    if "mask" in image_str:
                        continue
                    image = skimage.io.imread(dataset_path / f"{_set}/{image_str}")
                    

                    mask_str = image_str.replace(".tiff","_mask.tiff")

                    mask = skimage.io.imread(dataset_path / f"{_set}/{mask_str}")


                    meta = {"parent_dataset": folder, "modality": "Brightfield", "pixel_size": None, "name": image_str}

                    data_dicts[_set][0].append(image)
                    data_dicts[_set][1].append(mask)
                    data_dicts[_set][2].append(meta)
                  #  breakpoint()


    return_list = []
    for _set in sets:
        return_list.extend(data_dicts[_set])

        assert len(data_dicts[_set][0]) > 0, "No images in the dataset meet the requirements. (Hint: Check that the source argument is correct)"
    
    return return_list
   # breakpoint()


def _read_images_from_pth(data_path= "../datasets", dataset = "segmentation", data_slice = None, dummy = False, args = None, sets = ["Train","Validation"], complete_dataset = None):
    from pathlib import Path
    import torch
    import os 

    if complete_dataset is None:
        if not os.environ.get("INSTANSEG_DATASET_PATH"):
            os.environ["INSTANSEG_DATASET_PATH"] = Path(os.path.join(os.path.dirname(__file__),data_path))
        data_path = os.environ["INSTANSEG_DATASET_PATH"]
        if ".pth" in dataset:
            path_of_pth = os.path.join(data_path,dataset)
        else:
            path_of_pth = os.path.join(data_path,str(dataset + "_dataset.pth"))

        print("Loading dataset from ", os.path.abspath(path_of_pth))

        try:
            complete_dataset = torch.load(path_of_pth,weights_only = False)
        except:
            complete_dataset = torch.load(path_of_pth)
    

    data_dicts = {}

    for _set in sets:
        print("Datasets available in ", _set)
        unique_values, counts = np.unique([item['parent_dataset'] for item in complete_dataset[_set]], return_counts=True)
        print(set((k.item(), v.item()) for k, v in zip(unique_values, counts)))

        data_dicts[_set] = []
        images_local = [get_image(item['image']) for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]
 
        labels_local = [_format_labels(item,target_segmentation = args.target_segmentation) for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]
        metadata = [{k: v for k, v in item.items() if k not in ('image', 'cell_masks','nucleus_masks', 'class_masks')} for item in complete_dataset[_set] if _keep_images(item, args)][:data_slice]

        data_dicts[_set].extend([images_local,labels_local,metadata])

        print("After filtering using:")
        unique_values, counts = np.unique([item['parent_dataset'] for item in data_dicts[_set][2]], return_counts=True)
        print(set((k.item(), v.item()) for k, v in zip(unique_values, counts)))

    if dummy:
        warnings.warn("Using same train and validation sets !")
        data_dicts["Validation"] = data_dicts["Train"]

    return_list = []
    for _set in sets:
        return_list.extend(data_dicts[_set])

        assert len(data_dicts[_set][0]) > 0, "No images in the dataset meet the requirements. (Hint: Check that the source argument is correct)"

    return return_list


def seed_worker(worker_id):
    import torch
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_directory_dataset(
    data_dir,
    train_data_ratio: float = 1.0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Reads dataset structured as:
    data_dir/
      train/images/, train/masks/
      val/images/, val/masks/ (optional)
      test/images/, test/masks/ (optional)
    """
    from pathlib import Path
    import skimage.io
    import tifffile
    import cv2
    import numpy as np

    if not (0.0 < float(train_data_ratio) <= 1.0):
        raise ValueError(f"train_data_ratio must be between 0.0 and 1.0, got {train_data_ratio}")

    data_dir = Path(data_dir)

    def _read_img(p):
        p_str = str(p)
        if p_str.lower().endswith((".tif", ".tiff")):
            try:
                return tifffile.imread(p_str)
            except Exception:
                pass
        return skimage.io.imread(p_str)

    def _read_msk(p):
        p_str = str(p)
        if p_str.lower().endswith(".png"):
            img = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.astype(np.int32)
        return skimage.io.imread(p_str).astype(np.int32)

    def _load_set(folder_path):
        images_dir = folder_path / "images"
        masks_dir = folder_path / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            return [], [], []

        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        img_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in valid_exts])
        mask_files = sorted([f for f in masks_dir.iterdir() if f.suffix.lower() in valid_exts])

        if len(img_files) == 0:
            return [], [], []

        paired_images = []
        paired_masks = []
        paired_meta = []

        mask_map = {f.name: f for f in mask_files}
        mask_stem_map = {f.stem: f for f in mask_files}

        for i, img_f in enumerate(img_files):
            m_f = None
            if img_f.name in mask_map:
                m_f = mask_map[img_f.name]
            elif img_f.stem in mask_stem_map:
                m_f = mask_stem_map[img_f.stem]
            elif f"mask_{img_f.stem.replace('image_', '')}.png" in mask_map:
                m_f = mask_map[f"mask_{img_f.stem.replace('image_', '')}.png"]
            elif i < len(mask_files) and len(mask_files) == len(img_files):
                m_f = mask_files[i]

            if m_f is not None and m_f.exists():
                img = _read_img(img_f)
                msk = _read_msk(m_f)
                meta = {
                    "parent_dataset": data_dir.name,
                    "modality": "Brightfield",
                    "pixel_size": None,
                    "name": img_f.name,
                }
                paired_images.append(img)
                paired_masks.append(msk)
                paired_meta.append(meta)

        return paired_images, paired_masks, paired_meta

    train_imgs, train_msks, train_meta = _load_set(data_dir / "train")
    if len(train_imgs) == 0:
        train_imgs, train_msks, train_meta = _load_set(data_dir)

    if len(train_imgs) == 0:
        raise FileNotFoundError(f"No valid image/mask pairs found in dataset path {data_dir}")

    val_imgs, val_msks, val_meta = _load_set(data_dir / "val")
    test_imgs, test_msks, test_meta = _load_set(data_dir / "test")

    if len(val_imgs) == 0:
        n = len(train_imgs)
        num_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
        if num_val > 0:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            val_idx = perm[:num_val]
            train_idx = perm[num_val:]

            val_imgs = [train_imgs[i] for i in val_idx]
            val_msks = [train_msks[i] for i in val_idx]
            val_meta = [train_meta[i] for i in val_idx]

            train_imgs = [train_imgs[i] for i in train_idx]
            train_msks = [train_msks[i] for i in train_idx]
            train_meta = [train_meta[i] for i in train_idx]

    if train_data_ratio < 1.0 and len(train_imgs) > 0:
        n_train = len(train_imgs)
        num_keep = max(1, int(round(n_train * train_data_ratio)))
        train_imgs = train_imgs[:num_keep]
        train_msks = train_msks[:num_keep]
        train_meta = train_meta[:num_keep]

    if len(test_imgs) == 0:
        test_imgs, test_msks, test_meta = val_imgs, val_msks, val_meta

    return {
        "train": (train_imgs, train_msks, train_meta),
        "val": (val_imgs, val_msks, val_meta),
        "test": (test_imgs, test_msks, test_meta),
    }


def get_loaders(train_images_local, train_labels_local, val_images_local, val_labels_local, train_meta, val_meta, args):
    from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
    from instanseg.utils.augmentation_config import get_augmentation_dict
    from instanseg.utils.AI_utils import Segmentation_Dataset, collate_fn
    from torch.utils.data import DataLoader
    from instanseg.utils.utils import count_instances
    import torch
    import random

    seed_val = getattr(args, "seed", None) or getattr(args, "rng_seed", None) or 42
    torch.manual_seed(seed_val)
    generator = torch.Generator()
    generator.manual_seed(seed_val)

    augmentation_dict = get_augmentation_dict(args.dim_in, 
                                              nuclei_channel=None, 
                                              amount=args.transform_intensity,
                                              pixel_size=args.requested_pixel_size,
                                              mean_diameter=args.mean_object_diameter, 
                                              augmentation_type=args.augmentation_type)

    train_data = Segmentation_Dataset(train_images_local, 
                                      train_labels_local, 
                                      metadata=train_meta,
                                      size=(args.tile_size, args.tile_size), 
                                      augmentation_dict=augmentation_dict['train'],
                                      debug=False,
                                      dim_in=args.dim_in,
                                      cells_and_nuclei=args.cells_and_nuclei,
                                      random_seed=seed_val,
                                      target_segmentation=args.target_segmentation, 
                                      channel_invariant = args.channel_invariant)

    test_data = Segmentation_Dataset(val_images_local, val_labels_local, 
                                     size=(args.tile_size, args.tile_size), 
                                     metadata=val_meta,
                                     dim_in=args.dim_in,
                                     augmentation_dict=augmentation_dict['test'],
                                     random_seed = seed_val,
                                     cells_and_nuclei=args.cells_and_nuclei,
                                     target_segmentation=args.target_segmentation,
                                     channel_invariant = args.channel_invariant)

    if getattr(args, "length_of_epoch", None) is not None:
        train_sampler = RandomSampler(train_data, num_samples=args.length_of_epoch)
        test_sampler = RandomSampler(test_data, num_samples=max(1, int(args.length_of_epoch * 0.2)))
    else:
        train_sampler = RandomSampler(train_data)
        test_sampler = RandomSampler(test_data)

    num_workers = getattr(args, "num_workers", 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = (num_workers > 0)
    prefetch_factor = 2 if (num_workers > 0) else None

    train_loader = DataLoader(
        train_data,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_data,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
    )


    return train_loader, test_loader

