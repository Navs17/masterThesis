"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

After finishing training on each task, compute the diagonal Fisher
information matrix and record the current model weights as theta*.
When training on subsequent tasks, add an EWC regularisation penalty
that resists important weights from drifting far from theta*.
"""

import torch
from torch import nn


class EWC:
    def __init__(self, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc
        self._tasks = []  # list of (params_star, fisher) per consolidated task

    def consolidate(self, model, loader, device, n_batches=None):
        """Call after finishing a task to record theta* and the Fisher diagonal."""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        n_seen = 0
        for batch_idx, (images, labels) in enumerate(loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * images.size(0)
            n_seen += images.size(0)

        for n in fisher:
            fisher[n] /= max(n_seen, 1)

        params_star = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self._tasks.append((params_star, fisher))
        model.zero_grad()

    def penalty(self, model):
        """EWC regularisation term to add to the task loss."""
        if not self._tasks:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for params_star, fisher in self._tasks:
            for n, p in model.named_parameters():
                if n in fisher and p.requires_grad:
                    penalty = penalty + (fisher[n] * (p - params_star[n].to(p.device)) ** 2).sum()
        return (self.lambda_ewc / 2.0) * penalty
