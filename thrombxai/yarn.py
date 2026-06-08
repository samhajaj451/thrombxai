from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class Batch(TypedDict):
    signal: torch.Tensor
    label: torch.Tensor
    subject: torch.Tensor


@dataclass(frozen=True)
class ForwardOutput:
    logit: torch.Tensor
    risk: torch.Tensor
    temporal_attention: torch.Tensor
    spatial_attention: torch.Tensor
    pooled: torch.Tensor


@dataclass(frozen=True)
class Split:
    train: list[int]
    val: list[int]
    test: list[int]
