from __future__ import annotations

import torch

from thrombxai.finishing import objective


def test_focal_reduces_easy_example_weight() -> None:
    logit = torch.tensor([4.0, 4.0, -4.0, -4.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    focal = float(objective.focal_loss(logit, target, gamma=2.0))
    bce = float(objective.bce_loss(logit, target))
    assert focal < bce


def test_focal_gamma_zero_matches_bce() -> None:
    torch.manual_seed(0)
    logit = torch.randn(32)
    target = (torch.rand(32) > 0.5).float()
    focal = float(objective.focal_loss(logit, target, gamma=0.0))
    bce = float(objective.bce_loss(logit, target))
    assert abs(focal - bce) < 1e-5


def test_l2_penalty_zero_when_disabled() -> None:
    model = torch.nn.Linear(4, 1)
    assert float(objective.l2_penalty(model, 0.0)) == 0.0
    assert float(objective.l2_penalty(model, 1e-3).detach()) > 0.0
