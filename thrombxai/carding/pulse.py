from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sps

from thrombxai.order import SignalConfig

HRV_FEATURE_NAMES: tuple[str, ...] = (
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
)


def bandpass(raw: NDArray[np.float64], fs: float, low: float, high: float) -> NDArray[np.float64]:
    nyquist = 0.5 * fs
    high = min(high, nyquist * 0.99)
    b, a = sps.butter(2, [low / nyquist, high / nyquist], btype="band")
    filtered: NDArray[np.float64] = sps.filtfilt(b, a, raw).astype(np.float64)
    return filtered


def detect_rpeaks(ecg: NDArray[np.float64], fs: float) -> NDArray[np.int64]:
    filtered = bandpass(ecg, fs, 5.0, min(15.0, 0.49 * fs))
    derivative = np.gradient(filtered)
    squared = derivative * derivative
    window = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    threshold = integrated.mean() + 0.5 * integrated.std()
    min_gap = int(0.3 * fs)
    peaks, _ = sps.find_peaks(integrated, height=threshold, distance=max(1, min_gap))
    result: NDArray[np.int64] = peaks.astype(np.int64)
    return result


def interbeat_intervals(
    rpeaks: NDArray[np.int64], fs: float, ibi_min_ms: float, ibi_max_ms: float
) -> NDArray[np.float64]:
    if rpeaks.size < 2:
        return np.empty(0, dtype=np.float64)
    intervals = np.diff(rpeaks) / fs * 1000.0
    valid = (intervals >= ibi_min_ms) & (intervals <= ibi_max_ms)
    return intervals[valid].astype(np.float64)


def time_domain(ibi_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    if ibi_ms.size < 2:
        return np.zeros(4, dtype=np.float64)
    diff = np.diff(ibi_ms)
    sdnn = float(np.std(ibi_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff * diff)))
    pnn50 = float(np.mean(np.abs(diff) > 50.0) * 100.0)
    hr_mean = float(60000.0 / np.mean(ibi_ms))
    return np.array([sdnn, rmssd, pnn50, hr_mean], dtype=np.float64)


def _tachogram(ibi_ms: NDArray[np.float64], resample_hz: float) -> NDArray[np.float64]:
    times = np.cumsum(ibi_ms) / 1000.0
    times = times - times[0]
    grid = np.arange(times[0], times[-1], 1.0 / resample_hz)
    if grid.size < 4:
        return np.zeros(4, dtype=np.float64)
    interpolated = np.interp(grid, times, ibi_ms)
    return interpolated.astype(np.float64) - float(np.mean(interpolated))


def _band_power(freqs: NDArray[np.float64], psd: NDArray[np.float64], band: list[float]) -> float:
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def frequency_domain(ibi_ms: NDArray[np.float64], cfg: SignalConfig) -> NDArray[np.float64]:
    if ibi_ms.size < 4:
        return np.zeros(4, dtype=np.float64)
    series = _tachogram(ibi_ms, cfg.resample_hz)
    if series.size < 8:
        return np.zeros(4, dtype=np.float64)
    nperseg = min(series.size, 256)
    freqs, psd = sps.welch(series, fs=cfg.resample_hz, nperseg=nperseg)
    lf = _band_power(freqs, psd, cfg.lf_band)
    hf = _band_power(freqs, psd, cfg.hf_band)
    vlf = _band_power(freqs, psd, cfg.vlf_band)
    lf_hf = lf / hf if hf > 0.0 else 0.0
    return np.array([lf, hf, lf_hf, vlf], dtype=np.float64)


def poincare(ibi_ms: NDArray[np.float64]) -> tuple[float, float]:
    if ibi_ms.size < 3:
        return 0.0, 0.0
    diff = np.diff(ibi_ms)
    sd1 = float(np.sqrt(0.5) * np.std(diff, ddof=1))
    sdnn_sq = float(np.var(ibi_ms, ddof=1))
    sd2 = float(np.sqrt(max(2.0 * sdnn_sq - 0.5 * np.var(diff, ddof=1), 0.0)))
    return sd1, sd2


def sample_entropy(ibi_ms: NDArray[np.float64], m: int = 2, r: float = 0.2) -> float:
    n = ibi_ms.size
    if n < m + 2:
        return 0.0
    tolerance = r * float(np.std(ibi_ms, ddof=1))
    if tolerance <= 0.0:
        return 0.0

    def _phi(length: int) -> float:
        templates = np.array([ibi_ms[i : i + length] for i in range(n - length + 1)])
        count = 0
        for i in range(templates.shape[0]):
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            count += int(np.sum(distances <= tolerance) - 1)
        return float(count)

    b = _phi(m)
    a = _phi(m + 1)
    if b <= 0.0 or a <= 0.0:
        return 0.0
    return float(-np.log(a / b))


def dfa_alpha(ibi_ms: NDArray[np.float64], scales: tuple[int, ...] = (4, 8, 16, 32)) -> float:
    n = ibi_ms.size
    if n < max(scales) + 1:
        return 0.0
    profile = np.cumsum(ibi_ms - float(np.mean(ibi_ms)))
    fluctuations: list[float] = []
    usable: list[int] = []
    for scale in scales:
        segments = n // scale
        if segments < 1:
            continue
        rms = np.zeros(segments, dtype=np.float64)
        for s in range(segments):
            window = profile[s * scale : (s + 1) * scale]
            x = np.arange(scale, dtype=np.float64)
            coef = np.polyfit(x, window, 1)
            trend = np.polyval(coef, x)
            rms[s] = np.sqrt(np.mean((window - trend) ** 2))
        fluctuations.append(float(np.sqrt(np.mean(rms * rms))))
        usable.append(scale)
    if len(usable) < 2:
        return 0.0
    log_scales = np.log(np.array(usable, dtype=np.float64))
    log_fluct = np.log(np.array(fluctuations, dtype=np.float64) + 1e-12)
    slope = float(np.polyfit(log_scales, log_fluct, 1)[0])
    return slope


def nonlinear(ibi_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    sd1, sd2 = poincare(ibi_ms)
    return np.array([sd1, sd2, sample_entropy(ibi_ms), dfa_alpha(ibi_ms)], dtype=np.float64)


def window_features(ibi_ms: NDArray[np.float64], cfg: SignalConfig) -> NDArray[np.float64]:
    return np.concatenate([time_domain(ibi_ms), frequency_domain(ibi_ms, cfg), nonlinear(ibi_ms)])


def hrv_matrix(ecg: NDArray[np.float64], cfg: SignalConfig) -> NDArray[np.float64]:
    fs = cfg.sampling_hz
    window = int(cfg.hrv_window_s * fs)
    stride = int(cfg.hrv_stride_s * fs)
    rows: list[NDArray[np.float64]] = []
    start = 0
    while start + window <= ecg.size:
        segment = bandpass(
            ecg[start : start + window], fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz
        )
        peaks = detect_rpeaks(segment, fs)
        ibi = interbeat_intervals(peaks, fs, cfg.ibi_min_ms, cfg.ibi_max_ms)
        rows.append(window_features(ibi, cfg))
        start += stride
    if not rows:
        return np.zeros((0, len(HRV_FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(rows)
