import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import FeatureGaussianAnomalyDetector


def test_anomaly_detector_assigns_higher_score_to_shifted_inputs():
    torch.manual_seed(0)
    normal = torch.randn(16, 3, 64, 64) * 0.1 + 0.5
    shifted = torch.randn(4, 3, 64, 64) * 0.1 + 5.0  # clearly different distribution

    normal_loader = DataLoader(TensorDataset(normal, torch.zeros(16, dtype=torch.long)), batch_size=8)
    shifted_loader = DataLoader(TensorDataset(shifted, torch.ones(4, dtype=torch.long)), batch_size=4)

    detector = FeatureGaussianAnomalyDetector(backbone_name="resnet18")
    detector.fit(normal_loader)

    normal_scores, normal_labels = detector.score(normal_loader)
    shifted_scores, shifted_labels = detector.score(shifted_loader)

    assert shifted_scores.mean() > normal_scores.mean()
    assert normal_labels == [0] * 16
    assert shifted_labels == [1] * 4
