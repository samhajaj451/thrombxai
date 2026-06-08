from __future__ import annotations

import torch
from torch import nn


def fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    padding: str | int = conv.padding if isinstance(conv.padding, str) else int(conv.padding[0])
    fused = nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        int(conv.kernel_size[0]),
        stride=int(conv.stride[0]),
        padding=padding,
        dilation=int(conv.dilation[0]),
        bias=True,
    )
    weight = conv.weight.clone()
    running_var = bn.running_var if bn.running_var is not None else torch.ones(conv.out_channels)
    running_mean = (
        bn.running_mean if bn.running_mean is not None else torch.zeros(conv.out_channels)
    )
    scale = (
        (bn.weight / torch.sqrt(running_var + bn.eps))
        if bn.weight is not None
        else torch.ones(conv.out_channels)
    )
    assert fused.bias is not None
    with torch.no_grad():
        fused.weight.copy_(weight * scale.reshape(-1, 1, 1))
        bias = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
        beta = bn.bias if bn.bias is not None else torch.zeros(conv.out_channels)
        fused.bias.copy_((bias - running_mean) * scale + beta)
    return fused


def model_size_mb(model: nn.Module, bits: int) -> float:
    params = sum(p.numel() for p in model.parameters())
    return params * bits / 8.0 / 1_000_000.0


def estimate_latency_ms(model: nn.Module, time_steps: int, bits: int) -> float:
    params = sum(p.numel() for p in model.parameters())
    flops = 2.0 * params * time_steps
    throughput = 0.85e9 * (32.0 / bits)
    return flops / throughput * 1000.0
