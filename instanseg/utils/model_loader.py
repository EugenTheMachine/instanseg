import numpy as np


def build_monai_model(model_str: str, build_model_dictionary: dict):


    if model_str == "AttentionUNet":
        from monai.networks.nets import AttentionUnet

        model = AttentionUnet(spatial_dims=2, in_channels=int(build_model_dictionary["dim_in"]),
                              out_channels=build_model_dictionary["dim_out"], \
                              dropout=build_model_dictionary["dropprob"], channels=build_model_dictionary["layers"], \
                              strides=tuple([2 for _ in build_model_dictionary["layers"][:-1]])
                              )
    elif model_str == "FlexibleUNet":
        from monai.networks.nets import FlexibleUNet
        model = FlexibleUNet(in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"],
                             backbone="efficientnet-b0")
        

    elif model_str == "BasicUNetPlusPlus":
        from monai.networks.nets import BasicUNetPlusPlus
        model = BasicUNetPlusPlus(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                                  out_channels=build_model_dictionary["dim_out"],
                                  dropout=build_model_dictionary["dropprob"])

        class ModelWrapper(BasicUNetPlusPlus):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, inputs):
                output = super().forward(inputs)
                modified_output = output[0]  # Modify the output here as needed
                return modified_output

        model = ModelWrapper(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"])

    elif model_str == "UNETR":
        from monai.networks.nets import UNETR
        model = UNETR(in_channels=build_model_dictionary["dim_in"], out_channels=build_model_dictionary["dim_out"],
                      img_size=256, feature_size=32, norm_name='batch', spatial_dims=2)


    else:
        raise NotImplementedError("Model not implemented: " + model_str)

    return model


def read_model_args_from_csv(path=r"../results/", folder=""):
    import pandas as pd
    from pathlib import Path
    import yaml
    model_path = Path(path) / folder
    
    # Check if config.yaml exists in the folder
    config_yaml_path = model_path / "config.yaml"
    if config_yaml_path.exists():
        with open(config_yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        flat_cfg = {}
        for section in cfg.values():
            if isinstance(section, dict):
                flat_cfg.update(section)
                
        build_model_dictionary = {}
        build_model_dictionary["model_str"] = flat_cfg.get("model_name", "maskrcnn-resnet50_fpn")
        build_model_dictionary["dim_in"] = flat_cfg.get("dim_in", 1)
        build_model_dictionary["dim_out"] = flat_cfg.get("dim_out", 7)
        build_model_dictionary["n_sigma"] = flat_cfg.get("n_sigma", 4)
        build_model_dictionary["dim_coords"] = flat_cfg.get("dim_coords", 2)
        build_model_dictionary["dropprob"] = flat_cfg.get("dropout", 0.0)
        build_model_dictionary["layers"] = tuple(flat_cfg.get("layers", [32, 64, 128, 256]))
        build_model_dictionary["pixel_size"] = flat_cfg.get("pixel_size", 0.5)
        build_model_dictionary["cells_and_nuclei"] = flat_cfg.get("cells_and_nuclei", False)
        build_model_dictionary["norm"] = flat_cfg.get("norm", "BATCH")
        build_model_dictionary["feature_engineering"] = flat_cfg.get("feature_engineering", "0")
        build_model_dictionary["adaptor_net_str"] = flat_cfg.get("adaptor_net_str", "1")
        build_model_dictionary["multihead"] = flat_cfg.get("multihead", True)
        build_model_dictionary["channel_invariant"] = flat_cfg.get("channel_invariant", False)
        build_model_dictionary["to_centre"] = flat_cfg.get("to_centre", False)
        build_model_dictionary["mlp_width"] = flat_cfg.get("mlp_width", 5)
        build_model_dictionary["loss_function"] = flat_cfg.get("loss_function", "instanseg_loss")
        build_model_dictionary["binary_loss_fn"] = flat_cfg.get("binary_loss_fn", "lovasz_hinge")
        build_model_dictionary["seed_loss_fn"] = flat_cfg.get("seed_loss_fn", "l1_distance")
        build_model_dictionary["target_segmentation"] = flat_cfg.get("target_segmentation", "C")
        build_model_dictionary["source_dataset"] = flat_cfg.get("source_dataset", "all")
        build_model_dictionary["num_classes"] = flat_cfg.get("num_classes", 2)
        build_model_dictionary["imgsz"] = flat_cfg.get("imgsz", 512)
        
        return build_model_dictionary

    df = pd.read_csv(model_path / "experiment_log.csv", header=None)
    build_model_dictionary = dict(zip(list(df[0]), list(df[1])))

    if "model_shape" in build_model_dictionary.keys():
        build_model_dictionary["model_shape"] = eval(build_model_dictionary["model_shape"])
    for key in ["dim_in", "n_sigma", "dim_out", "dim_coords"]:
        build_model_dictionary[key] = eval(str(build_model_dictionary[key])) if str(
            build_model_dictionary[key]) != "nan" else None
    if "to_centre" in build_model_dictionary.keys():
        build_model_dictionary["to_centre"] = eval(build_model_dictionary["to_centre"])
    if "dropprob" in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = float(build_model_dictionary["dropprob"])
    if "layers" in build_model_dictionary.keys():
        build_model_dictionary["layers"] = tuple(eval(build_model_dictionary["layers"]))
    if "requested_pixel_size" in build_model_dictionary.keys():
        build_model_dictionary["pixel_size"] = float(build_model_dictionary["requested_pixel_size"])
    if "cells_and_nuclei" in build_model_dictionary.keys():
        build_model_dictionary["cells_and_nuclei"] = bool(eval(build_model_dictionary["cells_and_nuclei"]))
    if "norm" in build_model_dictionary.keys():
        if build_model_dictionary["norm"] == "None" or str(build_model_dictionary["norm"]).lower() == "nan":
            build_model_dictionary["norm"] = None
        else:
            build_model_dictionary["norm"] = str(build_model_dictionary["norm"])
    else:
        print("Norm not specified in model dictionary")
        build_model_dictionary["norm"] = None
    if "feature_engineering" in build_model_dictionary.keys():
        build_model_dictionary["feature_engineering"] = str(build_model_dictionary["feature_engineering"])
    else:
        print("Feature engineering not specified in model dictionary")
        build_model_dictionary["feature_engineering"] = "0"
    if "adaptor_net_str" in build_model_dictionary.keys():
        build_model_dictionary["adaptor_net_str"] = str(build_model_dictionary["adaptor_net_str"])
    if "multihead" in build_model_dictionary.keys():
        build_model_dictionary["multihead"] = bool(eval(build_model_dictionary["multihead"]))
    else:
        build_model_dictionary["multihead"] = False
    if "channel_invariant" in build_model_dictionary.keys():
        build_model_dictionary["channel_invariant"] = bool(eval(build_model_dictionary["channel_invariant"]))

    return build_model_dictionary


def build_model_from_dict(build_model_dictionary):
    model_str = build_model_dictionary.get("model_str", build_model_dictionary.get("model_name", "InstanSeg_UNet"))
    
    if model_str == 'maskrcnn-resnet50_fpn':
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        num_classes = build_model_dictionary.get("num_classes", 2)
        imgsz = build_model_dictionary.get("imgsz", 512)
        model = maskrcnn_resnet50_fpn(
            num_classes=num_classes,
            weights=None,
            weights_backbone=None,
            min_size=imgsz,
            max_size=imgsz,
        )
        return model

    dim_in_value = build_model_dictionary.get("dim_in", 3)
    if dim_in_value == 0 or dim_in_value is None:
        dim_in = 3  # Channel invariance currently outputs a 3 channel image
    else:
        dim_in = dim_in_value

    if "dropprob" not in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = 0.0

    supported_unets = {
        "instanseg_unet",
        "efficientunetb0",
        "efficientunetb1",
        "efficientunetb2",
        "efficientunetb3",
        "efficientunetv2s",
        "mobileunetv2",
        "mobileunetv3s",
        "mobileunetv3l",
        "regnetunety400mf",
        "regnetunety800mf",
        "resnetunet18",
    }

    if model_str == "InstanSeg_UNet" or model_str.lower() in supported_unets:
            from instanseg.utils.models.InstanSeg_UNet import UNet
            model_type = "instanseg_unet" if model_str == "InstanSeg_UNet" else model_str.lower()
            print(f"Generating UNet: {model_type}")

            multihead = bool(build_model_dictionary.get("multihead", False))
            cells_and_nuclei = bool(build_model_dictionary.get("cells_and_nuclei", False))
            dim_coords = int(build_model_dictionary.get("dim_coords", 2))
            n_sigma = int(build_model_dictionary.get("n_sigma", 4))
            layers = [int(x) for x in list(np.array(build_model_dictionary.get("layers", [32, 64, 128, 256]))[::-1])]
            norm = build_model_dictionary.get("norm", "BATCH")
            dropprob = float(build_model_dictionary.get("dropprob", 0.0))

            if cells_and_nuclei:
                if not multihead:
                    from itertools import chain
                    out_channels = [[dim_coords, n_sigma, 1] for _ in range(2)]
                    out_channels = list(chain(*out_channels))
                else:
                    out_channels = [[dim_coords, n_sigma, 1] for _ in range(2)]
            else:
                if not multihead:
                    out_channels = [[dim_coords, n_sigma, 1]]
                else:
                    out_channels = [[dim_coords], [n_sigma], [1]]

            model = UNet(
                model_type=model_type,
                in_channels=dim_in,
                layers=layers,
                out_channels=out_channels,
                norm=norm,
                dropout=dropprob,
                peft=build_model_dictionary.get("peft"),
                r=build_model_dictionary.get("r", 4),
                lora_alpha=build_model_dictionary.get("lora_alpha"),
                lora_dropout=build_model_dictionary.get("lora_dropout", 0.0),
                bias=build_model_dictionary.get("bias", "lora-only"),
            )
    else:
        model = build_monai_model(model_str, build_model_dictionary)

    return model


def remove_module_prefix_from_dict(dictionary):
    modified_dict = {}
    for key, value in dictionary.items():
        if key.startswith('module.'):
            modified_dict[key[7:]] = value
        else:
            modified_dict[key] = value
    return modified_dict


def has_pixel_classifier_state_dict(state_dict):
    return bool(sum(['pixel_classifier' in key for key in state_dict.keys()]))


def has_adaptor_net_state_dict(state_dict):
    return bool(sum(['AdaptorNet' in key for key in state_dict.keys()]))

def has_pixel_classifier_model(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'pixel_classifier' in module_class or 'ProbabilityNet' in module_class:
                return True
    return False


def load_model_weights(model, device, folder, path=r"../models/", dict = None):
    import torch
    from pathlib import Path
    model_path = Path(path) / folder
    
    # Try multiple possible file names for model weights
    weight_files = ["model_weights.pth", "checkpoints/best.pt", "checkpoints/last.pt"]
    model_dict = None
    
    for f_name in weight_files:
        p = model_path / f_name
        if p.exists():
            if torch.cuda.is_available():
                model_dict = torch.load(p, weights_only=False)
            else:
                if device is None:
                    if torch.backends.mps.is_available():
                        device = 'mps'
                    else:
                        device = 'cpu'
                model_dict = torch.load(p, map_location=device, weights_only=False)
            break
            
    if model_dict is None:
        raise FileNotFoundError(f"Could not find model weights in {model_path}")

    model_dict['model_state_dict'] = remove_module_prefix_from_dict(model_dict['model_state_dict'])

    if has_pixel_classifier_state_dict(model_dict['model_state_dict']) and not has_pixel_classifier_model(model):
        from instanseg.utils.loss.instanseg_loss import InstanSeg

        method = InstanSeg(n_sigma=int(dict["n_sigma"]), feature_engineering_function= dict["feature_engineering"],dim_coords = dict["dim_coords"],device =device)
        model = method.initialize_pixel_classifier(model, MLP_width=int(dict["mlp_width"]))

    from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
    if has_adaptor_net_state_dict(model_dict['model_state_dict']) and not has_AdaptorNet(model):
        from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
        model = AdaptorNetWrapper(model, norm = dict["norm"],adaptor_net_str = dict["adaptor_net_str"])

    model.load_state_dict(model_dict['model_state_dict'], strict=True)
    model.to(device)

    return model, model_dict

def load_model(folder,path=r"../models/", device='cpu'):
    build_model_dictionary = read_model_args_from_csv(path=path, folder=folder)

    empty_model = build_model_from_dict(build_model_dictionary)

    model, _ = load_model_weights(empty_model, path=path, folder=folder, device=device, dict = build_model_dictionary)

    return model, build_model_dictionary
