from __future__ import annotations

import torch
from torch.nn import functional as F

from thrombxai.fabric import build_model
from thrombxai.order import ModelConfig


def test_all_parameters_receive_gradient(tiny_cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    features = torch.randn(8, 30, tiny_cfg.num_features)
    labels = (torch.rand(8) > 0.5).float()
    output = model(features)
    loss = F.binary_cross_entropy_with_logits(output.logit, labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
