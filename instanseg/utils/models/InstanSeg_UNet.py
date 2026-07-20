from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import torchvision.models as torchvision_models
except ImportError:  # pragma: no cover - optional dependency guard
    torchvision_models = None


class _SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        if self.training and (input.numel() // input.shape[1]) == 1:
            self.eval()
            out = super().forward(input)
            self.train()
            return out
        return super().forward(input)


def create_gaussian_grid(N, sigma, device="cuda", channels=1):
    x = torch.linspace(-N // 2, N // 2, N, device=device)
    y = torch.linspace(-N // 2, N // 2, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    gaussian_values = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).repeat(channels, 1, 1)
    return gaussian_values


class LocalInstanceNorm(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.kernel_size = int(64 / (in_channels // 32))
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.sigma = int(64 / (in_channels // 32))
        self.gaussian = (
            create_gaussian_grid(
                self.kernel_size,
                sigma=self.sigma,
                device="cuda",
                channels=in_channels,
            )
            .flatten()
            .view(1, -1, 1)
            + 1e-5
        )

    def forward(self, x):
        assert x.dim() == 4, print("Only implemented for batch 4d tensor", x.dim())

        b, c, h, w = x.shape
        x = F.unfold(x, kernel_size=(self.kernel_size, self.kernel_size), stride=self.kernel_size // 2)
        x = rearrange(
            x,
            "b (c k1 k2) n -> (b n) c (k1 k2)",
            c=c,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-5)

        x = rearrange(
            x,
            "(b n) c (k1 k2) -> b (c k1 k2) n",
            b=b,
            c=c,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        x = x * self.gaussian
        counter = F.fold(
            torch.ones_like(x) * self.gaussian,
            (h, w),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.kernel_size // 2,
        )
        x = F.fold(x, (h, w), kernel_size=(self.kernel_size, self.kernel_size), stride=self.kernel_size // 2)

        result = x / counter
        return result


def _normalize_peft_mode(peft):
    if peft is None or peft == "" or str(peft).lower() in {"none", "null", "false"}:
        return None

    normalized = str(peft).strip().lower().replace("-", "_")
    if normalized in {"lorac", "convlora", "cplora", "dorac", "convdora", "cpdora"}:
        return normalized
    raise ValueError(
        "Unsupported peft mode. Expected one of: None, lorac, convlora, cplora, dorac, convdora, cpdora."
    )


def _normalize_bias_mode(bias):
    if bias is None:
        return "none"
    normalized = str(bias).strip().lower().replace("-", "_")
    if normalized in {"lora_only", "loraonly"}:
        return "lora_only"
    if normalized in {"none", "null", "false"}:
        return "none"
    if normalized in {"all", "true"}:
        return "all"
    raise ValueError("Unsupported bias mode. Expected one of: none, lora_only, all.")


class _PEFTConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        peft=None,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    ):
        super().__init__()
        self.peft = _normalize_peft_mode(peft)
        self.r = max(1, int(r))
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else 4 * self.r)
        self.lora_dropout = float(lora_dropout if lora_dropout is not None else 0.0)
        self.bias_mode = _normalize_bias_mode(bias)
        self.dropout = nn.Dropout(self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.lora_bias = nn.Parameter(torch.zeros(out_channels))
        self.lora_bias.requires_grad_(self.bias_mode in {"lora_only", "all"})
        self.conv.bias.requires_grad_(self.bias_mode == "all")

        if self.peft is None:
            return

        flattened_input_dim = in_channels * kernel_size[0] * kernel_size[1]
        if self.peft in {"lorac", "dorac"}:
            self.lora_a = nn.Parameter(torch.empty(self.r, flattened_input_dim))
            self.lora_b = nn.Parameter(torch.empty(out_channels, self.r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
            if self.peft == "dorac":
                self.magnitude = nn.Parameter(torch.ones(out_channels))
        elif self.peft in {"convlora", "convdora"}:
            self.lora_x = nn.Parameter(torch.empty(self.r, in_channels, *kernel_size))
            self.lora_y = nn.Parameter(torch.empty(out_channels, self.r, 1, 1))
            nn.init.kaiming_uniform_(self.lora_x, a=math.sqrt(5))
            nn.init.zeros_(self.lora_y)
            if self.peft == "convdora":
                self.magnitude = nn.Parameter(torch.ones(out_channels))
        elif self.peft in {"cplora", "cpdora"}:
            self.lora_a = nn.Parameter(torch.empty(out_channels, self.r))
            self.lora_b = nn.Parameter(torch.empty(in_channels, self.r))
            self.lora_c = nn.Parameter(torch.empty(kernel_size[0], self.r))
            self.lora_d = nn.Parameter(torch.empty(kernel_size[1], self.r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_b, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_c, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_d, a=math.sqrt(5))
            if self.peft == "cpdora":
                self.magnitude = nn.Parameter(torch.ones(out_channels))
        else:
            raise ValueError(f"Unsupported PEFT mode: {self.peft}")

    def _get_delta_weight(self):
        if self.peft is None:
            return None

        scale = self.lora_alpha / max(self.r, 1)
        if self.peft in {"lorac", "dorac"}:
            delta = self.lora_b @ self.lora_a
            delta = delta.view(self.conv.weight.shape)
        elif self.peft in {"convlora", "convdora"}:
            delta = torch.einsum("orhw,rihw->oihw", self.lora_y, self.lora_x)
        else:
            delta = torch.einsum("or,ir,hr,wr->oihw", self.lora_a, self.lora_b, self.lora_c, self.lora_d)

        delta = self.dropout(delta) * scale
        if self.peft in {"dorac", "convdora", "cpdora"}:
            weight = self.conv.weight + delta
            weight_flat = weight.reshape(weight.size(0), -1)
            weight_norm = weight_flat.norm(dim=1, keepdim=True) + 1e-6
            return self.magnitude.view(-1, 1, 1, 1) * weight / weight_norm.view(-1, 1, 1, 1)
        return self.conv.weight + delta

    def forward(self, x):
        weight = self._get_delta_weight()
        if weight is None:
            return self.conv(x)
        bias = self.conv.bias + self.lora_bias
        return self.conv._conv_forward(x, weight, bias)


def conv_norm_act(
    in_channels,
    out_channels,
    sz,
    norm,
    act="ReLU",
    peft=None,
    r=4,
    lora_alpha=None,
    lora_dropout=0.0,
    bias="none",
):
    if norm == "None" or norm is None:
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = _SafeBatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels, eps=1e-5, track_running_stats=False, affine=True)
    elif norm.lower() == "local":
        norm_layer = LocalInstanceNorm(in_channels=out_channels)
    else:
        raise ValueError("Norm must be None, batch or instance")

    if act == "None" or act is None:
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    else:
        raise ValueError("Act must be None, ReLU or Mish")

    conv_layer = _PEFTConv2d(
        in_channels,
        out_channels,
        sz,
        padding=sz // 2,
        peft=peft,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    return nn.Sequential(conv_layer, norm_layer, act_layer)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm="BATCH",
        act="ReLU",
        shallow=False,
        peft=None,
        r=4,
        lora_alpha=None,
        lora_dropout=0.0,
        bias="none",
    ):
        super().__init__()

        self.conv0 = conv_norm_act(
            in_channels,
            out_channels,
            1,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.conv_skip = conv_norm_act(
            skip_channels,
            out_channels,
            1,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.conv1 = conv_norm_act(
            in_channels,
            out_channels,
            3,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.conv2 = conv_norm_act(
            out_channels,
            out_channels,
            3,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.conv3 = conv_norm_act(
            out_channels,
            out_channels,
            3,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        self.conv4 = conv_norm_act(
            out_channels,
            out_channels,
            3,
            norm,
            act,
            peft=peft,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x + self.conv_skip(skip))
        x = x + self.conv4(self.conv3(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True,
        norm="BATCH",
        act="ReLU",
        shallow=False,
    ):
        super().__init__()

        if pool:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = nn.Identity()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers, out_channels, norm, act, peft=None, r=4, lora_alpha=None, lora_dropout=0.0, bias="none"):
        super().__init__()

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    layers[i],
                    layers[i + 1],
                    layers[i + 1],
                    norm=norm,
                    act=act,
                    peft=peft,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=bias,
                )
                for i in range(len(layers) - 1)
            ]
        )
        self.final_block = nn.ModuleList(
            [
                conv_norm_act(
                    layers[-1],
                    out_channel,
                    1,
                    norm=norm if (norm is not None) and norm.lower() != "instance" else None,
                    act=None,
                )
                for out_channel in out_channels
            ]
        )

    def forward(self, x, skips):
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)

        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
        return x


class UNetMeta(type):
    def __call__(cls, *args, **kwargs):
        if cls is not UNet:
            return super().__call__(*args, **kwargs)
        
        if args:
            model_type = args[0]
            remaining_args = args[1:]
        else:
            model_type = kwargs.pop("model_type", "instanseg_unet")
            remaining_args = args
            
        model_type = (model_type or "instanseg_unet").lower()
        model_map = {
            "instanseg_unet": InstanSeg_UNet,
            "efficientunetb0": EfficientUNetB0,
            "efficientunetb1": EfficientUNetB1,
            "efficientunetb2": EfficientUNetB2,
            "efficientunetb3": EfficientUNetB3,
            "efficientunetv2s": EfficientUNetV2S,
            "mobileunetv2": MobileUNetV2,
            "mobileunetv3s": MobileUNetV3S,
            "mobileunetv3l": MobileUNetV3L,
            "regnetunety400mf": RegNetUNetY400mf,
            "regnetunety800mf": RegNetUNetY800mf,
            "resnetunet18": ResNetUNet18,
        }
        if model_type not in model_map:
            raise ValueError(f"Unsupported model_type: {model_type}")
        target_cls = model_map[model_type]
        return target_cls(*remaining_args, **kwargs)


class UNet(nn.Module, metaclass=UNetMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def _normalize_out_channels(out_channels):
        if isinstance(out_channels, int):
            return [[out_channels]]
        if not out_channels:
            return [[1]]
        if isinstance(out_channels[0], int):
            return [out_channels]
        return list(out_channels)


class InstanSeg_UNet(UNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        layers=(256, 128, 64, 32),
        norm="BATCH",
        dropout=0,
        act="ReLu",
        peft=None,
        r=4,
        lora_alpha=None,
        lora_dropout=0.0,
        bias="lora-only",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.peft = _normalize_peft_mode(peft)
        self.r = max(1, int(r))
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else 4 * self.r)
        self.lora_dropout = float(lora_dropout if lora_dropout is not None else 0.0)
        self.bias_mode = _normalize_bias_mode(bias)

        layers = tuple(layers[::-1])
        self.encoder = nn.ModuleList(
            [EncoderBlock(in_channels, layers[0], pool=False, norm=norm, act=act)]
            + [EncoderBlock(layers[i], layers[i + 1], norm=norm, act=act) for i in range(len(layers) - 1)]
        )
        layers = tuple(layers[::-1])
        normalized_out_channels = self._normalize_out_channels(out_channels)
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    layers,
                    out_channel,
                    norm,
                    act,
                    peft=self.peft,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    bias=self.bias_mode,
                )
                for out_channel in normalized_out_channels
            ]
        )
        self.model_type = "instanseg_unet"

    def forward(self, x):
        skips = []
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder) - 1:
                skips.append(x)

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)


def _get_encoder_info(encoder, in_channels):
    was_training = encoder.training
    encoder.eval()
    try:
        device = next(encoder.parameters()).device if list(encoder.parameters()) else torch.device("cpu")
        dummy = torch.randn(1, in_channels, 64, 64, device=device)
        with torch.no_grad():
            bottleneck, skips = encoder(dummy)
        bottleneck_channels = bottleneck.shape[1]
        skip_channels = [skip.shape[1] for skip in skips]
        return bottleneck_channels, skip_channels
    finally:
        if was_training:
            encoder.train()


class _BackboneUNet(UNet):
    def __init__(self, encoder, decoder_class, in_channels, out_channels, model_type, *args, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.encoder = encoder
        
        bottleneck_channels, skip_channels = _get_encoder_info(self.encoder, in_channels)
        
        decoder_channels = [bottleneck_channels] + list(reversed(skip_channels))
        skip_channels_list = list(reversed(skip_channels))
        
        normalized_out_channels = self._normalize_out_channels(out_channels)
        self.decoder = decoder_class(decoder_channels, skip_channels_list, normalized_out_channels)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        bottleneck, skips = self.encoder(x)
        features = self.decoder(bottleneck, skips)
        
        outputs = [head(features) for head in self.decoder.final_heads]
        out = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.resnet18(weights=None)
        self.conv1 = backbone.conv1
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
    def forward(self, x):
        x0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3, [x0, x1, x2]


class ResNet18DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _SafeBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _SafeBatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(out_channels)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.upsample(x)
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet18Decoder(nn.Module):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            self.blocks.append(
                ResNet18DecoderBlock(decoder_channels[i], skip_channels_list[i], decoder_channels[i+1])
            )
        self.final_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[-1], heads[0], kernel_size=1)
            for heads in out_channels
        ])
        
    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        return x


class ResNetUNet18(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = ResNet18Encoder(in_channels)
        super().__init__(encoder, ResNet18Decoder, in_channels, out_channels, "resnetunet18", *args, **kwargs)


class EfficientNetB0Encoder(nn.Module):
    def __init__(self, in_channels=3, backbone_fn=None):
        super().__init__()
        if backbone_fn is None:
            backbone_fn = torchvision_models.efficientnet_b0
        backbone = backbone_fn(weights=None)
        self.features = backbone.features
        if in_channels != 3:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = nn.Sequential(*self.features[0:2])
        self.stage1 = nn.Sequential(*self.features[2:4])
        self.stage2 = nn.Sequential(*self.features[4:7])
        self.stage3 = nn.Sequential(*self.features[7:])
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class EfficientNetB1Encoder(EfficientNetB0Encoder):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, torchvision_models.efficientnet_b1)


class EfficientNetB2Encoder(EfficientNetB0Encoder):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, torchvision_models.efficientnet_b2)


class EfficientNetB3Encoder(EfficientNetB0Encoder):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, torchvision_models.efficientnet_b3)


class EfficientNetV2SEncoder(EfficientNetB0Encoder):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, torchvision_models.efficientnet_v2_s)


class EfficientNetB0DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _SafeBatchNorm2d(mid_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.bn2 = _SafeBatchNorm2d(mid_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _SafeBatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(out_channels)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.upsample(x)
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return F.silu(out)


class EfficientNetB1DecoderBlock(EfficientNetB0DecoderBlock): pass
class EfficientNetB2DecoderBlock(EfficientNetB0DecoderBlock): pass
class EfficientNetB3DecoderBlock(EfficientNetB0DecoderBlock): pass
class EfficientNetV2SDecoderBlock(EfficientNetB0DecoderBlock): pass


class EfficientNetB0Decoder(nn.Module):
    def __init__(self, decoder_channels, skip_channels_list, out_channels, block_cls=EfficientNetB0DecoderBlock):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            self.blocks.append(
                block_cls(decoder_channels[i], skip_channels_list[i], decoder_channels[i+1])
            )
        self.final_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[-1], heads[0], kernel_size=1)
            for heads in out_channels
        ])
        
    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        return x


class EfficientNetB1Decoder(EfficientNetB0Decoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, EfficientNetB1DecoderBlock)


class EfficientNetB2Decoder(EfficientNetB0Decoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, EfficientNetB2DecoderBlock)


class EfficientNetB3Decoder(EfficientNetB0Decoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, EfficientNetB3DecoderBlock)


class EfficientNetV2SDecoder(EfficientNetB0Decoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, EfficientNetV2SDecoderBlock)


class EfficientUNetB0(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = EfficientNetB0Encoder(in_channels)
        super().__init__(encoder, EfficientNetB0Decoder, in_channels, out_channels, "efficientunetb0", *args, **kwargs)


class EfficientUNetB1(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = EfficientNetB1Encoder(in_channels)
        super().__init__(encoder, EfficientNetB1Decoder, in_channels, out_channels, "efficientunetb1", *args, **kwargs)


class EfficientUNetB2(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = EfficientNetB2Encoder(in_channels)
        super().__init__(encoder, EfficientNetB2Decoder, in_channels, out_channels, "efficientunetb2", *args, **kwargs)


class EfficientUNetB3(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = EfficientNetB3Encoder(in_channels)
        super().__init__(encoder, EfficientNetB3Decoder, in_channels, out_channels, "efficientunetb3", *args, **kwargs)


class EfficientUNetV2S(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = EfficientNetV2SEncoder(in_channels)
        super().__init__(encoder, EfficientNetV2SDecoder, in_channels, out_channels, "efficientunetv2s", *args, **kwargs)


class MobileNetV2Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.mobilenet_v2(weights=None)
        self.features = backbone.features
        if in_channels != 3:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = nn.Sequential(*self.features[0:2])
        self.stage1 = nn.Sequential(*self.features[2:4])
        self.stage2 = nn.Sequential(*self.features[4:7])
        self.stage3 = nn.Sequential(*self.features[7:])
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class MobileNetV2DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _SafeBatchNorm2d(mid_channels)
        self.act1 = nn.ReLU6(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.bn2 = _SafeBatchNorm2d(mid_channels)
        self.act2 = nn.ReLU6(inplace=True)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _SafeBatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(out_channels)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.upsample(x)
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return out


class MobileNetV2Decoder(nn.Module):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            self.blocks.append(
                MobileNetV2DecoderBlock(decoder_channels[i], skip_channels_list[i], decoder_channels[i+1])
            )
        self.final_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[-1], heads[0], kernel_size=1)
            for heads in out_channels
        ])
        
    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        return x


class MobileUNetV2(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = MobileNetV2Encoder(in_channels)
        super().__init__(encoder, MobileNetV2Decoder, in_channels, out_channels, "mobileunetv2", *args, **kwargs)


class MobileNetV3SEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        if in_channels != 3:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = nn.Sequential(*self.features[0:2])
        self.stage1 = nn.Sequential(*self.features[2:4])
        self.stage2 = nn.Sequential(*self.features[4:7])
        self.stage3 = nn.Sequential(*self.features[7:])
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class MobileNetV3LEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        if in_channels != 3:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = nn.Sequential(*self.features[0:2])
        self.stage1 = nn.Sequential(*self.features[2:4])
        self.stage2 = nn.Sequential(*self.features[4:7])
        self.stage3 = nn.Sequential(*self.features[7:])
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class MobileNetV3SDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = in_channels + skip_channels
        
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _SafeBatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.bn2 = _SafeBatchNorm2d(mid_channels)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4 if mid_channels >= 4 else 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4 if mid_channels >= 4 else 1, mid_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True)
        )
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _SafeBatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(out_channels)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.upsample(x)
        residual = self.shortcut(x)
        out = F.hardswish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        out = F.hardswish(out)
        out = self.bn3(self.conv3(out))
        out += residual
        return out


class MobileNetV3LDecoderBlock(MobileNetV3SDecoderBlock): pass


class MobileNetV3SDecoder(nn.Module):
    def __init__(self, decoder_channels, skip_channels_list, out_channels, block_cls=MobileNetV3SDecoderBlock):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            self.blocks.append(
                block_cls(decoder_channels[i], skip_channels_list[i], decoder_channels[i+1])
            )
        self.final_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[-1], heads[0], kernel_size=1)
            for heads in out_channels
        ])
        
    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        return x


class MobileNetV3LDecoder(MobileNetV3SDecoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, MobileNetV3LDecoderBlock)


class MobileUNetV3S(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = MobileNetV3SEncoder(in_channels)
        super().__init__(encoder, MobileNetV3SDecoder, in_channels, out_channels, "mobileunetv3s", *args, **kwargs)


class MobileUNetV3L(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = MobileNetV3LEncoder(in_channels)
        super().__init__(encoder, MobileNetV3LDecoder, in_channels, out_channels, "mobileunetv3l", *args, **kwargs)


class RegNetY400mfEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.regnet_y_400mf(weights=None)
        self.stem = backbone.stem
        if in_channels != 3:
            first_conv = self.stem[0]
            self.stem[0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = self.stem
        self.stage1 = backbone.trunk_output.block1
        self.stage2 = backbone.trunk_output.block2
        self.stage3 = backbone.trunk_output.block3
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class RegNetY800mfEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = torchvision_models.regnet_y_800mf(weights=None)
        self.stem = backbone.stem
        if in_channels != 3:
            first_conv = self.stem[0]
            self.stem[0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        self.stage0 = self.stem
        self.stage1 = backbone.trunk_output.block1
        self.stage2 = backbone.trunk_output.block2
        self.stage3 = backbone.trunk_output.block3
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3, [x0, x1, x2]


class RegNetY400mfDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _SafeBatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        groups = 8 if mid_channels % 8 == 0 else 1
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = _SafeBatchNorm2d(mid_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4 if mid_channels >= 4 else 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4 if mid_channels >= 4 else 1, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _SafeBatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(out_channels)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.upsample(x)
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = out * self.se(out)
        out = self.bn3(self.conv3(out))
        out += residual
        return self.act3(out)


class RegNetY800mfDecoderBlock(RegNetY400mfDecoderBlock): pass


class RegNetY400mfDecoder(nn.Module):
    def __init__(self, decoder_channels, skip_channels_list, out_channels, block_cls=RegNetY400mfDecoderBlock):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(skip_channels_list)):
            self.blocks.append(
                block_cls(decoder_channels[i], skip_channels_list[i], decoder_channels[i+1])
            )
        self.final_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[-1], heads[0], kernel_size=1)
            for heads in out_channels
        ])
        
    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        return x


class RegNetY800mfDecoder(RegNetY400mfDecoder):
    def __init__(self, decoder_channels, skip_channels_list, out_channels):
        super().__init__(decoder_channels, skip_channels_list, out_channels, RegNetY800mfDecoderBlock)


class RegNetUNetY400mf(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = RegNetY400mfEncoder(in_channels)
        super().__init__(encoder, RegNetY400mfDecoder, in_channels, out_channels, "regnetunety400mf", *args, **kwargs)


class RegNetUNetY800mf(_BackboneUNet):
    def __init__(self, in_channels=3, out_channels=1, *args, **kwargs):
        if torchvision_models is None:
            raise ImportError("torchvision is required for backbone-based U-Net models")
        encoder = RegNetY800mfEncoder(in_channels)
        super().__init__(encoder, RegNetY800mfDecoder, in_channels, out_channels, "regnetunety800mf", *args, **kwargs)


if __name__ == "__main__":
    net = UNet(model_type="instanseg_unet", in_channels=3, out_channels=[5, 3])
    print(net(torch.randn(1, 3, 256, 256)).shape)
