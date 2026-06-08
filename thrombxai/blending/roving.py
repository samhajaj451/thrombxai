from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


def integrate(hrv: NDArray[np.float64], temp: NDArray[np.float64]) -> NDArray[np.float64]:
    steps = min(hrv.shape[0], temp.shape[0])
    return np.concatenate([hrv[:steps], temp[:steps]], axis=1)


def modality_mask(modalities: list[str], hrv_dim: int, temp_dim: int) -> NDArray[np.float64]:
    mask = np.zeros(hrv_dim + temp_dim, dtype=np.float64)
    if "hrv" in modalities:
        mask[:hrv_dim] = 1.0
    if "temp" in modalities:
        mask[hrv_dim:] = 1.0
    return mask


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> Standardizer:
        flat = features.reshape(-1, features.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std

    def state(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor]) -> Standardizer:
        return cls(mean=state["mean"], std=state["std"])


def apply_mask(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return features * mask
