"""Model factory: ViT and CNN backbones with N-channel (multispectral) input.

Relies on timm's `in_chans` to adapt the stem to 10/13-band Sentinel-2 input
(pretrained RGB stem weights are repeated/averaged across the extra channels).
"""
import timm
import torch.nn as nn

# name -> timm model id.  All run at 64x64 (pos-embed interpolated for ViT).
MODEL_IDS = {
    "vit_s":   "vit_small_patch16_224",
    "vit_ti":  "vit_tiny_patch16_224",
    "resnet50":"resnet50",
    "effb0":   "efficientnet_b0",
    "swin_t":  "swin_tiny_patch4_window7_224",
}
NEEDS_224 = {"swin_t"}  # models that require the default 224 input


def create_model(name, in_chans, num_classes, pretrained=True, img_size=64):
    """Build a classifier. Returns (model, input_size)."""
    mid = MODEL_IDS[name]
    kw = dict(pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)
    if name.startswith("vit"):
        kw["img_size"] = img_size            # interpolate positional embedding
        inp = img_size
    elif name in NEEDS_224:
        inp = 224                            # swin: resize inputs to 224
    else:
        inp = img_size                       # CNNs: fully convolutional, any size
    model = timm.create_model(mid, **kw)
    return model, inp


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # millions
