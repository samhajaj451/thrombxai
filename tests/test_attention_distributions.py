from __future__ import annotations

import dataclasses

import torch

from thrombxai.order import ModelConfig
from thrombxai.warping.heddle import TemporalSelfAttention, aggregate_temporal
from thrombxai.weaving.shed import SpatialFeatureAttention


def test_temporal_attention_rows_sum_to_one(tiny_cfg: ModelConfig) -> None:
    attn = TemporalSelfAttention(tiny_cfg).eval()
    z = torch.randn(4, 25, tiny_cfg.hidden_dim)
    with torch.no_grad():
        _, weights = attn(z)
    assert weights.shape == (4, tiny_cfg.attn_heads, 25, 25)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, tiny_cfg.attn_heads, 25), atol=1e-5)


def test_temporal_aggregation_is_a_distribution(tiny_cfg: ModelConfig) -> None:
    attn = TemporalSelfAttention(tiny_cfg).eval()
    z = torch.randn(4, 25, tiny_cfg.hidden_dim)
    with torch.no_grad():
        _, weights = attn(z)
    pooled = aggregate_temporal(weights)
    assert pooled.shape == (4, 25)
    assert torch.allclose(pooled.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_spatial_attention_sums_to_one(tiny_cfg: ModelConfig) -> None:
    spatial = SpatialFeatureAttention(tiny_cfg)
    pooled = torch.randn(5, tiny_cfg.hidden_dim)
    fused, weights = spatial(pooled)
    assert weights.shape == (5, tiny_cfg.num_features)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert fused.shape == (5, tiny_cfg.feature_embed_dim)


def test_spatial_attention_disabled_is_uniform(tiny_cfg: ModelConfig) -> None:
    disabled = dataclasses.replace(tiny_cfg, use_spatial_attention=False)
    spatial = SpatialFeatureAttention(disabled)
    pooled = torch.randn(5, disabled.hidden_dim)
    _, weights = spatial(pooled)
    expected = torch.full((5, disabled.num_features), 1.0 / disabled.num_features)
    assert torch.allclose(weights, expected, atol=1e-6)
