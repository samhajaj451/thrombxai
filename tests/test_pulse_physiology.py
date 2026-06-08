from __future__ import annotations

import numpy as np

from thrombxai.carding import pulse
from thrombxai.spinning.twist import receptive_field


def test_ibi_constraint_filters_out_of_range() -> None:
    fs = 100.0
    rpeaks = np.array([0, 20, 120, 400, 410, 700], dtype=np.int64)
    ibi = pulse.interbeat_intervals(rpeaks, fs, 300.0, 2000.0)
    assert np.all(ibi >= 300.0)
    assert np.all(ibi <= 2000.0)


def test_receptive_field_matches_paper() -> None:
    assert receptive_field(4, 3, dilated=True) == 31
    assert receptive_field(3, 3, dilated=True) == 15
    assert receptive_field(4, 3, dilated=False) == 9


def test_time_domain_on_constant_intervals() -> None:
    ibi = np.full(20, 800.0)
    features = pulse.time_domain(ibi)
    assert features[0] == 0.0
    assert features[1] == 0.0
    assert features[2] == 0.0
    assert abs(features[3] - 75.0) < 1e-6
