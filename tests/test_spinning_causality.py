from __future__ import annotations

import torch

from thrombxai.order import ModelConfig
from thrombxai.spinning.twist import TemporalConvEncoder


def test_encoder_is_causal(tiny_cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    encoder = TemporalConvEncoder(tiny_cfg).eval()
    x = torch.randn(2, 30, tiny_cfg.num_features)
    with torch.no_grad():
        base = encoder(x)
        perturbed = x.clone()
        perturbed[:, 20:, :] += 5.0
        altered = encoder(perturbed)
    assert torch.allclose(base[:, :19, :], altered[:, :19, :], atol=1e-5)
    assert not torch.allclose(base[:, 25, :], altered[:, 25, :], atol=1e-5)


def test_encoder_preserves_time_dimension(tiny_cfg: ModelConfig) -> None:
    encoder = TemporalConvEncoder(tiny_cfg)
    x = torch.randn(3, 36, tiny_cfg.num_features)
    out = encoder(x)
    assert out.shape == (3, 36, tiny_cfg.hidden_dim)
