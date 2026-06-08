from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from thrombxai.order import ModelConfig


class PredictionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden = nn.Linear(cfg.hidden_dim + cfg.feature_embed_dim, cfg.mlp_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.mlp_hidden, 1)

    def forward(self, pooled: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([pooled, fused], dim=-1)
        hidden = self.dropout(F.relu(self.hidden(joined)))
        logit: torch.Tensor = self.out(hidden).squeeze(-1)
        return logit
