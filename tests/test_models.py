import pytest
import torch

from src.models import PillDefectClassifier, build_backbone


def test_build_backbone_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_backbone("not_a_real_backbone")


@pytest.mark.parametrize("backbone_name", ["resnet18", "efficientnet_b0"])
def test_classifier_forward_pass_shape(backbone_name):
    model = PillDefectClassifier(backbone_name=backbone_name, pretrained=False, num_classes=2)
    model.eval()

    batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (4, 2)


def test_freeze_backbone_disables_gradients():
    model = PillDefectClassifier(backbone_name="resnet18", pretrained=False, freeze_backbone=True)

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
