"""Experience Replay buffer for continual learning.

Stores a small, stratified sample of past-task images in memory and mixes
them into every training batch for subsequent tasks, preventing catastrophic
forgetting by regularly revisiting earlier examples.
"""

import random

import torch

from src.data.dataset import LABEL_TO_IDX


def _get_labels(dataset):
    """Extract integer labels without loading any images.

    Works for PillDefectDataset directly and for torch Subset wrappers over it.
    """
    if hasattr(dataset, "rows"):
        return [LABEL_TO_IDX[r["label"]] for r in dataset.rows]
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "rows"):
        return [LABEL_TO_IDX[dataset.dataset.rows[i]["label"]] for i in dataset.indices]
    raise TypeError(f"Cannot extract labels from {type(dataset)}")


class ReplayBuffer:
    def __init__(self, n_per_task: int = 100):
        self.n_per_task = n_per_task
        self._buffer: list = []  # list of (image_tensor, int_label)

    def __len__(self):
        return len(self._buffer)

    def populate(self, dataset, seed: int) -> None:
        """Stratified-sample up to n_per_task images from dataset and add to buffer."""
        labels = _get_labels(dataset)
        by_label: dict = {}
        for idx, label in enumerate(labels):
            by_label.setdefault(label, []).append(idx)

        rng = random.Random(seed)
        per_class = max(1, self.n_per_task // max(len(by_label), 1))
        sampled = []
        for indices in by_label.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            sampled.extend(shuffled[:per_class])

        for idx in sampled:
            img, label = dataset[idx]
            self._buffer.append((img.cpu(), label))

    def sample_batch(self, batch_size: int):
        """Return a random batch from the buffer as (images, labels) tensors."""
        indices = random.sample(range(len(self._buffer)), min(batch_size, len(self._buffer)))
        imgs, labels = zip(*[self._buffer[i] for i in indices])
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)
