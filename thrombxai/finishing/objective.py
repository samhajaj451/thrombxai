from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from thrombxai.order import ObjectiveConfig


def bce_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target)


def focal_loss(logit: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    p_t = torch.where(target > 0.5, prob, 1.0 - prob).clamp_min(1e-6)
    per_sample = -((1.0 - p_t) ** gamma) * p_t.log()
    return per_sample.mean()


def prediction_loss(
    logit: torch.Tensor, target: torch.Tensor, cfg: ObjectiveConfig
) -> torch.Tensor:
    if cfg.use_focal:
        return focal_loss(logit, target, cfg.focal_gamma)
    return bce_loss(logit, target)


def l2_penalty(model: nn.Module, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=next(model.parameters()).device)
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            total = total + param.pow(2).sum()
    return weight * total
