from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

FEATURE_NAMES: tuple[str, ...] = (
    "sdnn",
    "rmssd",
    "pnn50",
    "hr_mean",
    "lf",
    "hf",
    "lf_hf",
    "vlf",
    "sd1",
    "sd2",
    "sampen",
    "dfa_alpha",
    "t_mean",
    "t_std",
    "t_slope",
    "t_range",
    "dt_5min",
    "dt_15min",
    "t_accel",
)

NODES: tuple[str, ...] = (
    "hypercoagulability",
    "autonomic_dysfunction",
    "vagal_withdrawal",
    "sympathetic_drive",
    "microcirculation",
    "inflammation",
    "thermoregulation",
)

EDGES: tuple[tuple[str, str], ...] = (
    ("lf_hf", "sympathetic_drive"),
    ("lf_hf", "autonomic_dysfunction"),
    ("rmssd", "vagal_withdrawal"),
    ("rmssd", "autonomic_dysfunction"),
    ("sdnn", "autonomic_dysfunction"),
    ("pnn50", "vagal_withdrawal"),
    ("hr_mean", "sympathetic_drive"),
    ("lf", "sympathetic_drive"),
    ("hf", "vagal_withdrawal"),
    ("vlf", "inflammation"),
    ("sd1", "vagal_withdrawal"),
    ("sd2", "autonomic_dysfunction"),
    ("sampen", "autonomic_dysfunction"),
    ("dfa_alpha", "hypercoagulability"),
    ("lf_hf", "hypercoagulability"),
    ("rmssd", "hypercoagulability"),
    ("t_slope", "microcirculation"),
    ("t_range", "thermoregulation"),
    ("t_accel", "microcirculation"),
    ("dt_5min", "thermoregulation"),
    ("dt_15min", "microcirculation"),
    ("t_mean", "thermoregulation"),
    ("t_std", "thermoregulation"),
    ("microcirculation", "hypercoagulability"),
    ("inflammation", "hypercoagulability"),
    ("autonomic_dysfunction", "hypercoagulability"),
)


def degree_vector() -> NDArray[np.float64]:
    degree = {name: 0 for name in FEATURE_NAMES}
    for source, target in EDGES:
        if source in degree:
            degree[source] += 1
        if target in degree:
            degree[target] += 1
    return np.array([float(degree[name]) for name in FEATURE_NAMES], dtype=np.float64)


def relevance_vector(gamma: float) -> torch.Tensor:
    degree = degree_vector()
    smoothed = degree + gamma
    normalized = smoothed / smoothed.sum()
    return torch.tensor(normalized, dtype=torch.float32)


def clinical_rank() -> NDArray[np.int64]:
    degree = degree_vector()
    order = np.argsort(-degree, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, order.size + 1)
    return rank.astype(np.int64)
