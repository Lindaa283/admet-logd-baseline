"""Regression metrics with bootstrap confidence intervals.

A single MAE on a 840-compound test set is not a precise number.  Quoting it
to three decimals without an interval invites over-reading small differences
between models, so every test metric here carries a bootstrap 95% interval and
the report compares models on whether those intervals overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Scores:
    """Point estimates for one set of predictions."""

    n: int
    mae: float
    rmse: float
    r2: float
    spearman: float
    pearson: float
    bias: float          # mean(pred - obs); a check for systematic offset
    max_abs_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Scores:
    """Compute all metrics for one prediction vector."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty set")

    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    # R2 against the variance of the *observed* test labels.  With a constant
    # predictor fitted on train this can go slightly negative, which is
    # informative and is reported as-is rather than clipped.
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator

    if y_true.size < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        spearman = float("nan")
        pearson = float("nan")
    else:
        spearman = float(stats.spearmanr(y_true, y_pred).statistic)
        pearson = float(stats.pearsonr(y_true, y_pred).statistic)

    return Scores(
        n=int(y_true.size),
        mae=mae,
        rmse=rmse,
        r2=r2,
        spearman=spearman,
        pearson=pearson,
        bias=float(np.mean(residual)),
        max_abs_error=float(np.max(np.abs(residual))),
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "mae",
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for one metric, resampling test compounds."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n = y_true.size
    values = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        values[draw] = getattr(score(y_true[pick], y_pred[pick]), metric)
    low = (1.0 - level) / 2.0 * 100.0
    return (
        float(np.nanpercentile(values, low)),
        float(np.nanpercentile(values, 100.0 - low)),
    )
