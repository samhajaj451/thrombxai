from __future__ import annotations

import torch
from torch import nn

from thrombxai.finishing.press import PredictionHead
from thrombxai.order import ModelConfig, edge_model_config
from thrombxai.spinning.twist import TemporalConvEncoder
from thrombxai.warping.heddle import TemporalSelfAttention, aggregate_temporal
from thrombxai.weaving.shed import SpatialFeatureAttention
from thrombxai.yarn import ForwardOutput


class ThrombXAI(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalConvEncoder(cfg)
        self.temporal = TemporalSelfAttention(cfg)
        self.spatial = SpatialFeatureAttention(cfg)
        self.head = PredictionHead(cfg)
        self.use_temporal = cfg.use_temporal_attention

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        encoded = self.encoder(x)
        if self.use_temporal:
            attended, attention = self.temporal(encoded)
            temporal_weights = aggregate_temporal(attention)
        else:
            attended = encoded
            steps = encoded.shape[1]
            temporal_weights = encoded.new_full((encoded.shape[0], steps), 1.0 / steps)
        pooled = attended.mean(dim=1)
        fused, spatial_weights = self.spatial(pooled)
        logit = self.head(pooled, fused)
        return ForwardOutput(
            logit=logit,
            risk=torch.sigmoid(logit),
            temporal_attention=temporal_weights,
            spatial_attention=spatial_weights,
            pooled=pooled,
        )


def build_model(cfg: ModelConfig) -> ThrombXAI:
    return ThrombXAI(cfg)


def build_edge_model() -> ThrombXAI:
    return ThrombXAI(edge_model_config())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
