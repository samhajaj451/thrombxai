from __future__ import annotations

import torch


def kl_attention(spatial: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    prior = relevance.unsqueeze(0).clamp_min(eps)
    posterior = spatial.clamp_min(eps)
    divergence = posterior * (posterior.log() - prior.log())
    return divergence.sum(dim=-1).mean()


def temporal_smoothness(temporal: torch.Tensor) -> torch.Tensor:
    diff = temporal[:, 1:] - temporal[:, :-1]
    return (diff * diff).sum(dim=-1).mean()


def crga_loss(
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    relevance: torch.Tensor,
    lambda_kl: float,
    lambda_smooth: float,
) -> torch.Tensor:
    return lambda_kl * kl_attention(spatial, relevance) + lambda_smooth * temporal_smoothness(
        temporal
    )


def kl_warmup(base: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return base
    return base * max(0.0, 1.0 - epoch / warmup_epochs)
