"""Feature-based one-class anomaly detector: frozen backbone embeddings + Mahalanobis distance.

A simplified, image-level analogue of approaches like PaDiM / deep feature
modeling: extract pretrained backbone features for non-defective ("good")
training images only, fit a Gaussian over those features, and score new
images by their Mahalanobis distance from that Gaussian. Higher distance
means more anomalous, i.e. more likely defective.
"""

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance

from .backbone import build_backbone


class FeatureGaussianAnomalyDetector:
    def __init__(self, backbone_name="resnet18", device="cpu"):
        self.device = device
        self.backbone, self.feature_dim = build_backbone(backbone_name, pretrained=True)
        self.backbone.to(device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.cov_estimator = EmpiricalCovariance()

    @torch.no_grad()
    def extract_features(self, loader):
        all_features, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            features = self.backbone(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.tolist())
        return np.concatenate(all_features, axis=0), all_labels

    def fit(self, loader):
        features, _ = self.extract_features(loader)
        self.cov_estimator.fit(features)

    def score(self, loader):
        """Returns (mahalanobis_distances, labels) -- higher distance = more anomalous."""
        features, labels = self.extract_features(loader)
        return self.cov_estimator.mahalanobis(features), labels
