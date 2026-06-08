from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from thrombxai.fabric import ThrombXAI


def _mask_top(weights: torch.Tensor, fraction: float, keep: bool) -> torch.Tensor:
    features = weights.shape[-1]
    k = max(1, int(round(fraction * features)))
    order = torch.argsort(weights, dim=-1, descending=True)
    mask = torch.ones_like(weights)
    top = order[:, :k]
    selector = torch.zeros_like(weights)
    selector.scatter_(1, top, 1.0)
    return selector if keep else (mask - selector)


@torch.no_grad()
def comprehensiveness(model: ThrombXAI, signal: torch.Tensor, fraction: float) -> float:
    model.eval()
    base = model(signal)
    mask = _mask_top(base.spatial_attention, fraction, keep=False)
    perturbed = signal * mask.unsqueeze(1)
    altered = model(perturbed)
    return float((base.risk - altered.risk).abs().mean().item())


@torch.no_grad()
def sufficiency(model: ThrombXAI, signal: torch.Tensor, fraction: float) -> float:
    model.eval()
    base = model(signal)
    mask = _mask_top(base.spatial_attention, fraction, keep=True)
    perturbed = signal * mask.unsqueeze(1)
    altered = model(perturbed)
    retained = 1.0 - (base.risk - altered.risk).abs().mean().item()
    return float(max(0.0, retained))


def attention_overlap(
    spatial: NDArray[np.float64], clinical_rank: NDArray[np.int64], top_k: int
) -> float:
    model_top = set(np.argsort(-spatial)[:top_k].tolist())
    clinical_top = set(np.argsort(clinical_rank)[:top_k].tolist())
    return len(model_top & clinical_top) / float(top_k)


def ranking_correlation(
    mean_attention: NDArray[np.float64], clinical_rank: NDArray[np.int64]
) -> float:
    model_rank = stats.rankdata(-mean_attention)
    rho, _ = stats.spearmanr(model_rank, clinical_rank)
    return float(rho)
