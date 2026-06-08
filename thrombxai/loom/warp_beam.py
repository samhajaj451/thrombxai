from __future__ import annotations

import torch
from torch import nn

from thrombxai.order import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, eta_min: float
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=eta_min
    )
