"""Backbone feature extractors, stripped of their classification head."""

import torch.nn as nn
from torchvision import models

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
}


def build_backbone(name: str, pretrained: bool = True):
    """Returns (backbone_module, feature_dim). backbone_module(x) -> [B, feature_dim]."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Choose from {sorted(_BACKBONES)}")

    constructor, weights, feature_dim = _BACKBONES[name]
    backbone = constructor(weights=weights if pretrained else None)

    if name.startswith("resnet"):
        backbone.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        backbone.classifier = nn.Identity()

    return backbone, feature_dim
