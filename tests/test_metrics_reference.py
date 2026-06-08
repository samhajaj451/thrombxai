from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from thrombxai.inspection import gauge


def _data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = (rng.random(300) < 0.3).astype(np.float64)
    score = np.clip(0.25 * target + 0.6 * rng.random(300), 0.0, 1.0)
    return score, target


def test_roc_auc_matches_sklearn() -> None:
    score, target = _data(1)
    assert abs(gauge.roc_auc(score, target) - skm.roc_auc_score(target, score)) < 1e-9


def test_confusion_metrics_match_sklearn() -> None:
    score, target = _data(2)
    pred = (score >= 0.5).astype(np.float64)
    assert abs(gauge.accuracy(score, target, 0.5) - skm.accuracy_score(target, pred)) < 1e-9
    assert (
        abs(gauge.sensitivity(score, target, 0.5) - skm.recall_score(target, pred, zero_division=0))
        < 1e-9
    )


def test_pr_auc_in_unit_interval() -> None:
    score, target = _data(3)
    value = gauge.pr_auc(score, target)
    assert 0.0 <= value <= 1.0


def test_delong_variance_non_negative() -> None:
    score, target = _data(4)
    assert gauge.delong_variance(score, target) >= 0.0
