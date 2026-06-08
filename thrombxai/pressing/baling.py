from __future__ import annotations

import copy
from typing import TypeVar

import torch
from torch import nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def fake_quantize(weight: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    levels = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
    if levels <= 0:
        return weight
    if symmetric:
        scale = weight.abs().max().clamp_min(1e-8) / levels
        quantized = torch.clamp(torch.round(weight / scale), -levels - 1, levels)
        result: torch.Tensor = quantized * scale
        return result
    low = weight.min()
    scale = (weight.max() - low).clamp_min(1e-8) / levels
    quantized = torch.clamp(torch.round((weight - low) / scale), 0, levels)
    affine: torch.Tensor = quantized * scale + low
    return affine


def quantize_model(model: ModuleT, bits: int, symmetric: bool) -> ModuleT:
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for param in clone.parameters():
            param.copy_(fake_quantize(param.data, bits, symmetric))
    clone.eval()
    return clone
