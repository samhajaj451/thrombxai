from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from thrombxai.order import ModelConfig


def receptive_field(layers: int, kernel: int, dilated: bool) -> int:
    total = 1
    for layer in range(layers):
        dilation = (2**layer) if dilated else 1
        total += (kernel - 1) * dilation
    return total


class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad, 0))
        out: torch.Tensor = self.conv(padded)
        return out


class TwistBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = DilatedCausalConv1d(channels, kernel, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv(x))
        out = self.dropout(out)
        out = out + residual
        out = out.transpose(1, 2)
        out = self.norm(out)
        return out.transpose(1, 2)


class TemporalConvEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.project = nn.Conv1d(cfg.num_features, cfg.hidden_dim, 1)
        blocks: list[nn.Module] = []
        for layer in range(cfg.tcn_layers):
            dilation = (2**layer) if cfg.use_dilation else 1
            blocks.append(TwistBlock(cfg.hidden_dim, cfg.tcn_kernel, dilation, cfg.dropout))
        self.blocks = nn.ModuleList(blocks)
        self.receptive_field = receptive_field(cfg.tcn_layers, cfg.tcn_kernel, cfg.use_dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.project(x.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        encoded: torch.Tensor = hidden.transpose(1, 2)
        return encoded
