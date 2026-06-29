"""Binary defective/non_defective classifier: backbone + linear head."""

import torch.nn as nn

from .backbone import build_backbone


class PillDefectClassifier(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=2, freeze_backbone=False):
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
