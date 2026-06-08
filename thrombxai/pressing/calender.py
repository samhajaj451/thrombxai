from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from thrombxai.fabric import ThrombXAI
from thrombxai.order import EdgeConfig, OptimConfig
from thrombxai.spindle import run_backward
from thrombxai.yarn import Batch


def distillation_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    soft_teacher = torch.sigmoid(teacher_logit / temperature)
    soft_student = student_logit / temperature
    soft = F.binary_cross_entropy_with_logits(soft_student, soft_teacher) * (temperature**2)
    hard = F.binary_cross_entropy_with_logits(student_logit, target)
    return alpha * soft + (1.0 - alpha) * hard


def distill(
    teacher: ThrombXAI,
    student: ThrombXAI,
    loader: DataLoader[Batch],
    cfg: EdgeConfig,
    device: torch.device,
) -> ThrombXAI:
    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=OptimConfig().lr)
    for _ in range(cfg.distill_epochs):
        for batch in loader:
            signal = batch["signal"].to(device)
            label = batch["label"].to(device)
            with torch.no_grad():
                teacher_logit = teacher(signal).logit
            student_logit = student(signal).logit
            loss = distillation_loss(
                student_logit, teacher_logit, label, cfg.distill_temperature, cfg.distill_alpha
            )
            optimizer.zero_grad()
            run_backward(loss)
            optimizer.step()
    return student
