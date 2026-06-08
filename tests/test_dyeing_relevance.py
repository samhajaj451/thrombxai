from __future__ import annotations

import torch
from torch.nn import functional as F

from thrombxai.dyeing import vat
from thrombxai.dyeing.mordant import FEATURE_NAMES, relevance_vector


def test_relevance_is_a_distribution() -> None:
    r = relevance_vector(1.0)
    assert r.shape == (len(FEATURE_NAMES),)
    assert torch.all(r > 0.0)
    assert abs(float(r.sum()) - 1.0) < 1e-6


def test_hrv_features_carry_more_prior_than_temperature() -> None:
    r = relevance_vector(1.0)
    assert float(r[6]) > float(r[12])


def test_kl_is_non_negative() -> None:
    r = relevance_vector(1.0)
    spatial = F.softmax(torch.randn(8, len(FEATURE_NAMES)), dim=-1)
    assert float(vat.kl_attention(spatial, r)) >= -1e-6


def test_temporal_smoothness_zero_for_constant() -> None:
    constant = torch.full((4, 10), 0.1)
    assert float(vat.temporal_smoothness(constant)) == 0.0


def test_kl_warmup_decays_to_zero() -> None:
    assert vat.kl_warmup(0.1, 0, 20) == 0.1
    assert vat.kl_warmup(0.1, 20, 20) == 0.0
    assert vat.kl_warmup(0.1, 30, 20) == 0.0
