from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from thrombxai.order import ModelConfig


class TemporalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.heads = cfg.attn_heads
        self.head_dim = cfg.hidden_dim // cfg.attn_heads
        self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.key = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.value = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        return x.view(batch, steps, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._reshape(self.query(z))
        k = self._reshape(self.key(z))
        v = self._reshape(self.value(z))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attention), v)
        batch, _, steps, _ = context.shape
        merged = context.transpose(1, 2).reshape(batch, steps, self.heads * self.head_dim)
        return self.out(merged), attention


def aggregate_temporal(attention: torch.Tensor) -> torch.Tensor:
    return attention.mean(dim=1).mean(dim=1)
