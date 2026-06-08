from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sps

from thrombxai.order import SignalConfig

TEMP_FEATURE_NAMES: tuple[str, ...] = (
    "t_mean",
    "t_std",
    "t_slope",
    "t_range",
    "dt_5min",
    "dt_15min",
    "t_accel",
)


def despike(temp: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    kernel = window if window % 2 == 1 else window + 1
    if temp.size < kernel:
        return temp.astype(np.float64)
    filtered: NDArray[np.float64] = sps.medfilt(temp, kernel_size=kernel).astype(np.float64)
    return filtered


def statistical_features(segment: NDArray[np.float64]) -> NDArray[np.float64]:
    if segment.size < 2:
        return np.zeros(4, dtype=np.float64)
    index = np.arange(segment.size, dtype=np.float64)
    slope = float(np.polyfit(index, segment, 1)[0])
    return np.array(
        [
            float(np.mean(segment)),
            float(np.std(segment, ddof=1)),
            slope,
            float(np.max(segment) - np.min(segment)),
        ],
        dtype=np.float64,
    )


def dynamic_features(
    segment: NDArray[np.float64], samples_per_minute: float
) -> NDArray[np.float64]:
    if segment.size < 3:
        return np.zeros(3, dtype=np.float64)
    span_5 = min(segment.size - 1, int(round(5.0 * samples_per_minute)))
    span_15 = min(segment.size - 1, int(round(15.0 * samples_per_minute)))
    dt_5 = float(segment[-1] - segment[-1 - span_5]) if span_5 >= 1 else 0.0
    dt_15 = float(segment[-1] - segment[-1 - span_15]) if span_15 >= 1 else 0.0
    second = np.gradient(np.gradient(segment))
    accel = float(np.mean(second))
    return np.array([dt_5, dt_15, accel], dtype=np.float64)


def window_features(segment: NDArray[np.float64], samples_per_minute: float) -> NDArray[np.float64]:
    return np.concatenate(
        [statistical_features(segment), dynamic_features(segment, samples_per_minute)]
    )


def temperature_matrix(
    temp: NDArray[np.float64], cfg: SignalConfig, time_steps: int
) -> NDArray[np.float64]:
    clean = despike(temp, cfg.temp_median_window)
    if clean.size < time_steps:
        clean = np.pad(clean, (0, time_steps - clean.size), mode="edge")
    bounds = np.linspace(0, clean.size, time_steps + 1, dtype=int)
    samples_per_minute = clean.size / (cfg.hrv_window_s * time_steps / 60.0 + 1e-9)
    rows: list[NDArray[np.float64]] = []
    for i in range(time_steps):
        segment = clean[bounds[i] : bounds[i + 1]]
        if segment.size < 3:
            segment = clean[max(0, bounds[i + 1] - 3) : bounds[i + 1]]
        rows.append(window_features(segment, max(samples_per_minute, 1.0)))
    return np.vstack(rows)
