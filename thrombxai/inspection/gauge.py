from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def _counts(
    score: NDArray[np.float64], target: NDArray[np.float64], threshold: float
) -> tuple[int, int, int, int]:
    pred = score >= threshold
    truth = target >= 0.5
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    return tp, tn, fp, fn


def accuracy(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    tp, tn, fp, fn = _counts(score, target, threshold)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def sensitivity(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    tp, _, _, fn = _counts(score, target, threshold)
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    _, tn, fp, _ = _counts(score, target, threshold)
    return tn / (tn + fp) if (tn + fp) else 0.0


def precision(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    tp, _, fp, _ = _counts(score, target, threshold)
    return tp / (tp + fp) if (tp + fp) else 0.0


def npv(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    _, tn, _, fn = _counts(score, target, threshold)
    return tn / (tn + fn) if (tn + fn) else 0.0


def f1(score: NDArray[np.float64], target: NDArray[np.float64], threshold: float) -> float:
    prec = precision(score, target, threshold)
    rec = sensitivity(score, target, threshold)
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def roc_auc(score: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    truth = target >= 0.5
    n_pos = int(np.sum(truth))
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(score)
    rank_sum = float(np.sum(ranks[truth]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(score: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    truth = target >= 0.5
    order = np.argsort(-score, kind="stable")
    sorted_truth = truth[order]
    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(~sorted_truth)
    total_pos = int(np.sum(truth))
    if total_pos == 0:
        return 0.0
    recall = tp / total_pos
    denom = tp + fp
    prec = np.where(denom > 0, tp / np.maximum(denom, 1), 1.0)
    recall = np.concatenate([[0.0], recall])
    prec = np.concatenate([[1.0], prec])
    return float(np.trapezoid(prec, recall))


def delong_variance(score: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    truth = target >= 0.5
    pos = score[truth]
    neg = score[~truth]
    m = pos.size
    n = neg.size
    if m == 0 or n == 0:
        return 0.0
    v10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    v01 = np.array([np.mean(pos > q) + 0.5 * np.mean(pos == q) for q in neg])
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    return s10 / m + s01 / n


def bootstrap_ci(
    score: NDArray[np.float64],
    target: NDArray[np.float64],
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    iterations: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = score.size
    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric(score[idx], target[idx])
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lower, upper


def report(
    score: NDArray[np.float64], target: NDArray[np.float64], threshold: float
) -> dict[str, float]:
    return {
        "accuracy": accuracy(score, target, threshold),
        "auroc": roc_auc(score, target),
        "auprc": pr_auc(score, target),
        "sensitivity": sensitivity(score, target, threshold),
        "specificity": specificity(score, target, threshold),
        "f1": f1(score, target, threshold),
        "ppv": precision(score, target, threshold),
        "npv": npv(score, target, threshold),
    }
