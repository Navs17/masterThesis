import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval import compute_bwt, compute_forgetting
from src.strategies import EWC


def _tiny_model():
    return nn.Linear(4, 2)


def _tiny_loader(n=16):
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_penalty_is_zero_before_consolidation():
    model = _tiny_model()
    ewc = EWC(lambda_ewc=1000.0)
    assert ewc.penalty(model).item() == 0.0


def test_penalty_positive_after_consolidation_and_param_change():
    model = _tiny_model()
    loader = _tiny_loader()
    ewc = EWC(lambda_ewc=1000.0)
    ewc.consolidate(model, loader, device="cpu")

    # Perturb weights and check penalty grows
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 10.0)

    assert ewc.penalty(model).item() > 0.0


def test_penalty_zero_when_lambda_is_zero():
    model = _tiny_model()
    loader = _tiny_loader()
    ewc = EWC(lambda_ewc=0.0)
    ewc.consolidate(model, loader, device="cpu")

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 10.0)

    assert ewc.penalty(model).item() == 0.0


def test_compute_bwt_negative_on_forgetting():
    # Task 0 perf=0.9 when first trained, drops to 0.6 after task 1
    result_matrix = [[0.9, None], [0.6, 0.8]]
    bwt = compute_bwt([[0.9, 0.0], [0.6, 0.8]])
    assert bwt < 0.0  # forgetting occurred


def test_compute_forgetting_detects_drop():
    result_matrix = [[0.9, 0.0], [0.6, 0.8]]
    forgetting = compute_forgetting(result_matrix)
    assert abs(forgetting - 0.3) < 1e-6  # 0.9 - 0.6 = 0.3
