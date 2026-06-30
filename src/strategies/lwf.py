"""Learning without Forgetting (LwF) — Li & Hoiem, ECCV 2016.

Before moving to a new task, the current model is snapshotted as a frozen
teacher. During new-task training the teacher's softened output distribution
is used as a distillation target (knowledge distillation loss), pushing the
student model to stay close to the old task's decision surface without
storing any past-task images.

Temperature T controls how soft the teacher distribution is:
  - T=1 → hard teacher argmax (no smoothing)
  - T>1 → smoother, emphasises relative class similarities
  - T^2 rescaling preserves gradient magnitude (Hinton et al., 2015)
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn


class LwF:
    def __init__(self, temperature: float = 2.0, lambda_lwf: float = 1.0):
        self.temperature = temperature
        self.lambda_lwf = lambda_lwf
        self._teacher: nn.Module | None = None

    def snapshot(self, model: nn.Module, device: torch.device | str) -> None:
        """Deep-copy the current model and freeze it as the teacher for the next task."""
        self._teacher = copy.deepcopy(model).to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

    def distillation_loss(self, student_logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """KL-divergence loss between student and (frozen) teacher output distributions.

        Returns zero before the first snapshot has been taken.
        """
        if self._teacher is None:
            return torch.tensor(0.0, device=student_logits.device)

        with torch.no_grad():
            teacher_logits = self._teacher(inputs)

        T = self.temperature
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        log_probs = F.log_softmax(student_logits / T, dim=1)
        kd_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (T**2)
        return self.lambda_lwf * kd_loss
