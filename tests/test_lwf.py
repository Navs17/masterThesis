"""Tests for the LwF strategy (src/strategies/lwf.py)."""

import copy

import torch
from torch import nn

from src.strategies import LwF


def _tiny_model(in_features=8, num_classes=2):
    return nn.Linear(in_features, num_classes)


def _batch(n=16, in_features=8):
    return torch.randn(n, in_features)


# ── Construction ──────────────────────────────────────────────────────────────

def test_no_teacher_before_snapshot():
    lwf = LwF()
    assert lwf._teacher is None


# ── distillation_loss before snapshot ────────────────────────────────────────

def test_distillation_loss_is_zero_before_snapshot():
    model = _tiny_model()
    lwf = LwF(lambda_lwf=10.0)
    inputs = _batch()
    logits = model(inputs)
    loss = lwf.distillation_loss(logits, inputs)
    assert loss.item() == 0.0


# ── snapshot ──────────────────────────────────────────────────────────────────

def test_snapshot_stores_teacher():
    model = _tiny_model()
    lwf = LwF()
    lwf.snapshot(model, device="cpu")
    assert lwf._teacher is not None


def test_snapshot_freezes_teacher_params():
    model = _tiny_model()
    lwf = LwF()
    lwf.snapshot(model, device="cpu")
    for p in lwf._teacher.parameters():
        assert not p.requires_grad


def test_snapshot_is_independent_copy():
    """Mutating the original model after snapshot should not affect the teacher."""
    model = _tiny_model()
    lwf = LwF()
    lwf.snapshot(model, device="cpu")

    teacher_weight_before = lwf._teacher.weight.data.clone()
    with torch.no_grad():
        model.weight.fill_(99.0)

    assert torch.allclose(lwf._teacher.weight.data, teacher_weight_before)


def test_snapshot_replaces_previous_teacher():
    model = _tiny_model()
    lwf = LwF()
    lwf.snapshot(model, device="cpu")
    teacher_id_1 = id(lwf._teacher)

    with torch.no_grad():
        model.weight.fill_(5.0)
    lwf.snapshot(model, device="cpu")

    assert id(lwf._teacher) != teacher_id_1
    assert torch.allclose(lwf._teacher.weight.data, torch.full_like(lwf._teacher.weight, 5.0))


# ── distillation_loss after snapshot ─────────────────────────────────────────

def test_distillation_loss_positive_after_snapshot_and_param_change():
    model = _tiny_model()
    lwf = LwF(lambda_lwf=1.0, temperature=2.0)
    lwf.snapshot(model, device="cpu")

    # Add different amounts to each class row so the output distribution changes.
    # Uniform row-wise shifts cancel out in softmax; we need asymmetric shifts.
    with torch.no_grad():
        model.weight[0].add_(100.0)   # push class-0 logit up
        model.weight[1].add_(-100.0)  # push class-1 logit down

    inputs = _batch()
    student_logits = model(inputs)
    loss = lwf.distillation_loss(student_logits, inputs)
    assert loss.item() > 0.0


def test_distillation_loss_near_zero_when_student_equals_teacher():
    """If student == teacher the KL divergence should be (near) zero."""
    model = _tiny_model()
    lwf = LwF(lambda_lwf=1.0, temperature=2.0)
    lwf.snapshot(model, device="cpu")

    inputs = _batch()
    student_logits = model(inputs)
    loss = lwf.distillation_loss(student_logits, inputs)
    assert loss.item() < 1e-5


def test_lambda_zero_gives_zero_loss():
    model = _tiny_model()
    lwf = LwF(lambda_lwf=0.0, temperature=2.0)
    lwf.snapshot(model, device="cpu")

    with torch.no_grad():
        model.weight.add_(50.0)

    inputs = _batch()
    loss = lwf.distillation_loss(model(inputs), inputs)
    assert loss.item() == 0.0


def test_higher_lambda_scales_loss():
    model_a = _tiny_model()
    model_b = copy.deepcopy(model_a)
    lwf_low = LwF(lambda_lwf=1.0, temperature=2.0)
    lwf_high = LwF(lambda_lwf=5.0, temperature=2.0)
    lwf_low.snapshot(model_a, device="cpu")
    lwf_high.snapshot(model_b, device="cpu")

    # Asymmetric perturbation so the softmax distribution actually changes
    with torch.no_grad():
        model_a.weight[0].add_(10.0)
        model_a.weight[1].add_(-10.0)
        model_b.weight[0].add_(10.0)
        model_b.weight[1].add_(-10.0)

    inputs = _batch()
    loss_low = lwf_low.distillation_loss(model_a(inputs), inputs)
    loss_high = lwf_high.distillation_loss(model_b(inputs), inputs)
    assert loss_high.item() > loss_low.item()


def test_distillation_loss_does_not_update_teacher():
    """Backprop through the student should not change teacher weights."""
    model = _tiny_model()
    lwf = LwF(lambda_lwf=1.0)
    lwf.snapshot(model, device="cpu")

    teacher_weight_before = lwf._teacher.weight.data.clone()

    with torch.no_grad():
        model.weight.add_(5.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = _batch()
    loss = lwf.distillation_loss(model(inputs), inputs)
    loss.backward()
    optimizer.step()

    assert torch.allclose(lwf._teacher.weight.data, teacher_weight_before)


# ── integration: combined CE + KD loss ───────────────────────────────────────

def test_combined_loss_is_differentiable():
    """CE + KD loss should produce valid gradients for a training step."""
    model = _tiny_model()
    lwf = LwF(lambda_lwf=1.0, temperature=2.0)
    lwf.snapshot(model, device="cpu")

    with torch.no_grad():
        model.weight.add_(1.0)

    inputs = _batch(n=8)
    labels = torch.randint(0, 2, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(inputs)
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    kd_loss = lwf.distillation_loss(logits, inputs)
    total = ce_loss + kd_loss

    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    # If we got here without error, gradients are valid
    assert total.item() > 0.0
