from __future__ import annotations

import numpy as np
import pytest
import torch

from thrombxai.order import ModelConfig, SignalConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        tcn_layers=4,
        tcn_kernel=3,
        attn_heads=4,
        attn_dim=8,
        spatial_attn_dim=16,
        feature_embed_dim=32,
        mlp_hidden=32,
        dropout=0.0,
        num_features=19,
    )


@pytest.fixture
def signal() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(6, 40, 19)


@pytest.fixture
def fast_signal_cfg() -> SignalConfig:
    return SignalConfig(sampling_hz=100.0, hrv_window_s=60.0, hrv_stride_s=30.0)


def synth_ecg(duration_s: float, fs: float, hr_hz: float) -> np.ndarray:
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    beats = np.zeros(n)
    period = 1.0 / hr_hz
    centers = np.arange(period, duration_s, period)
    for c in centers:
        beats += np.exp(-0.5 * ((t - c) / 0.02) ** 2)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n)
    return (beats + noise).astype(np.float64)
