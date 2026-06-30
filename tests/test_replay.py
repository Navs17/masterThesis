"""Tests for the Experience Replay buffer (src/strategies/replay.py)."""

import random

import torch
from torch.utils.data import Dataset

from src.strategies import ReplayBuffer


class _FakeDataset(Dataset):
    """Minimal dataset that mimics PillDefectDataset's .rows interface."""

    def __init__(self, n_per_class=10, img_channels=3, img_size=8):
        self.rows = []
        labels = ["non_defective"] * n_per_class + ["defective"] * n_per_class
        for label in labels:
            self.rows.append({"label": label})
        self._img_channels = img_channels
        self._img_size = img_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        label_str = self.rows[idx]["label"]
        label_int = 0 if label_str == "non_defective" else 1
        img = torch.zeros(self._img_channels, self._img_size, self._img_size) + label_int
        return img, label_int


# ── Construction ──────────────────────────────────────────────────────────────

def test_buffer_initially_empty():
    buf = ReplayBuffer(n_per_task=50)
    assert len(buf) == 0


# ── populate ──────────────────────────────────────────────────────────────────

def test_populate_adds_images():
    ds = _FakeDataset(n_per_class=20)
    buf = ReplayBuffer(n_per_task=10)
    buf.populate(ds, seed=0)
    assert len(buf) > 0


def test_populate_respects_n_per_task_cap():
    ds = _FakeDataset(n_per_class=50)  # 100 total images
    buf = ReplayBuffer(n_per_task=20)
    buf.populate(ds, seed=0)
    # At most n_per_task images should be stored (may be slightly less if class
    # count doesn't divide evenly, but never more than n_per_task)
    assert len(buf) <= 20


def test_populate_is_stratified_both_classes_represented():
    ds = _FakeDataset(n_per_class=30)
    buf = ReplayBuffer(n_per_task=20)
    buf.populate(ds, seed=42)
    labels_in_buffer = [label for _, label in buf._buffer]
    assert 0 in labels_in_buffer, "non_defective class missing from buffer"
    assert 1 in labels_in_buffer, "defective class missing from buffer"


def test_populate_accumulates_across_tasks():
    ds1 = _FakeDataset(n_per_class=20)
    ds2 = _FakeDataset(n_per_class=20)
    buf = ReplayBuffer(n_per_task=10)
    buf.populate(ds1, seed=0)
    size_after_task1 = len(buf)
    buf.populate(ds2, seed=1)
    assert len(buf) > size_after_task1


def test_populate_is_deterministic_with_same_seed():
    ds = _FakeDataset(n_per_class=30)
    buf_a = ReplayBuffer(n_per_task=10)
    buf_b = ReplayBuffer(n_per_task=10)
    buf_a.populate(ds, seed=7)
    buf_b.populate(ds, seed=7)
    labels_a = [lbl for _, lbl in buf_a._buffer]
    labels_b = [lbl for _, lbl in buf_b._buffer]
    assert labels_a == labels_b


def test_populate_with_imbalanced_dataset():
    """Buffer should not crash when one class has very few samples."""
    ds = _FakeDataset(n_per_class=2)  # only 4 images total (2 per class)
    buf = ReplayBuffer(n_per_task=100)  # ask for more than available
    buf.populate(ds, seed=0)
    assert len(buf) > 0


# ── sample_batch ──────────────────────────────────────────────────────────────

def test_sample_batch_returns_correct_shapes():
    ds = _FakeDataset(n_per_class=20)
    buf = ReplayBuffer(n_per_task=30)
    buf.populate(ds, seed=0)
    imgs, labels = buf.sample_batch(batch_size=8)
    assert imgs.ndim == 4                   # (B, C, H, W)
    assert labels.ndim == 1
    assert imgs.shape[0] == labels.shape[0]
    assert labels.dtype == torch.long


def test_sample_batch_capped_by_buffer_size():
    ds = _FakeDataset(n_per_class=3)  # 6 images total
    buf = ReplayBuffer(n_per_task=4)
    buf.populate(ds, seed=0)
    imgs, labels = buf.sample_batch(batch_size=100)  # ask for more than buffer holds
    assert imgs.shape[0] <= len(buf)


def test_sample_batch_labels_match_images():
    """Images for label 0 contain only zeros; for label 1 contain only ones."""
    ds = _FakeDataset(n_per_class=20)
    buf = ReplayBuffer(n_per_task=40)
    buf.populate(ds, seed=0)
    imgs, labels = buf.sample_batch(batch_size=20)
    for img, lbl in zip(imgs, labels):
        expected_val = float(lbl.item())
        assert img.unique().tolist() == [expected_val], (
            f"Image content {img.unique()} doesn't match label {lbl}"
        )


# ── integration: replay during training ──────────────────────────────────────

def test_replay_batch_can_be_concatenated_with_training_batch():
    """Verify that buffer tensors can be cat'd with an incoming batch (the main training use-case)."""
    ds = _FakeDataset(n_per_class=20)
    buf = ReplayBuffer(n_per_task=20)
    buf.populate(ds, seed=0)

    # Simulate an incoming training batch
    train_imgs = torch.zeros(8, 3, 8, 8)
    train_labels = torch.zeros(8, dtype=torch.long)

    r_imgs, r_labels = buf.sample_batch(batch_size=8)
    combined_imgs = torch.cat([train_imgs, r_imgs], dim=0)
    combined_labels = torch.cat([train_labels, r_labels], dim=0)

    assert combined_imgs.shape == (16, 3, 8, 8)
    assert combined_labels.shape == (16,)
