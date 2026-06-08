from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from thrombxai.order import ModelConfig


class SpatialFeatureAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.features = cfg.num_features
        self.embed = nn.Parameter(torch.randn(cfg.num_features, cfg.feature_embed_dim) * 0.02)
        self.score_in = nn.Linear(cfg.hidden_dim + cfg.feature_embed_dim, cfg.spatial_attn_dim)
        self.score_out = nn.Linear(cfg.spatial_attn_dim, 1, bias=False)
        self.use_attention = cfg.use_spatial_attention

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = pooled.shape[0]
        embed = self.embed.unsqueeze(0).expand(batch, -1, -1)
        if self.use_attention:
            context = pooled.unsqueeze(1).expand(-1, self.features, -1)
            energy = self.score_out(torch.tanh(self.score_in(torch.cat([context, embed], dim=-1))))
            weights = F.softmax(energy.squeeze(-1), dim=-1)
        else:
            weights = pooled.new_full((batch, self.features), 1.0 / self.features)
        fused = torch.bmm(weights.unsqueeze(1), embed).squeeze(1)
        return fused, weights
