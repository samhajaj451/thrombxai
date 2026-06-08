from __future__ import annotations

import numpy as np

from tests.conftest import synth_ecg
from thrombxai.blending import roving
from thrombxai.carding import pulse, thermal
from thrombxai.order import SignalConfig


def test_hrv_matrix_has_twelve_columns(fast_signal_cfg: SignalConfig) -> None:
    ecg = synth_ecg(180.0, fast_signal_cfg.sampling_hz, 1.1)
    matrix = pulse.hrv_matrix(ecg, fast_signal_cfg)
    assert matrix.shape[0] > 0
    assert matrix.shape[1] == 12


def test_temperature_matrix_has_seven_columns(fast_signal_cfg: SignalConfig) -> None:
    temp = 36.5 + 0.2 * np.sin(np.linspace(0, 6, 400))
    matrix = thermal.temperature_matrix(temp, fast_signal_cfg, time_steps=24)
    assert matrix.shape == (24, 7)


def test_integration_yields_nineteen_features(fast_signal_cfg: SignalConfig) -> None:
    ecg = synth_ecg(180.0, fast_signal_cfg.sampling_hz, 1.0)
    hrv = pulse.hrv_matrix(ecg, fast_signal_cfg)
    temp = thermal.temperature_matrix(
        36.5 + np.zeros(400), fast_signal_cfg, time_steps=hrv.shape[0]
    )
    fused = roving.integrate(hrv, temp)
    assert fused.shape[1] == 19
