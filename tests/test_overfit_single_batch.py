from __future__ import annotations

import torch
from torch.nn import functional as F

from thrombxai.fabric import build_model
from thrombxai.opening.sliver import synthesize
from thrombxai.order import ModelConfig


def test_model_overfits_a_single_batch() -> None:
    torch.manual_seed(0)
    features, labels, _ = synthesize(16, 32, 19, positive_rate=0.5, seed=1)
    cfg = ModelConfig(
        hidden_dim=32,
        tcn_layers=3,
        attn_heads=4,
        attn_dim=8,
        spatial_attn_dim=16,
        feature_embed_dim=32,
        mlp_hidden=32,
        dropout=0.0,
    )
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = None
    loss = torch.zeros(())
    output = model(features)
    for _ in range(200):
        output = model(features)
        loss = F.binary_cross_entropy_with_logits(output.logit, labels)
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert initial is not None
    assert float(loss.detach()) < initial * 0.5
    accuracy = ((output.risk > 0.5).float() == labels).float().mean()
    assert float(accuracy) > 0.9
