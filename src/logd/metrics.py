"""Regression and ranking metrics, with bootstrap confidence intervals.

Both error metrics (RMSE, MAE, R2) and rank-correlation metrics (Spearman,
Kendall) are reported. Rank metrics are included because a triage model is
used to *order* candidates, and because a model can have respectable RMSE
while ordering compounds badly, or vice versa.

Point estimates on a few hundred test molecules are noisy, so every headline
number carries a 95% bootstrap interval. A difference between two models that
is smaller than these intervals is not a result.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

METRIC_NAMES = ("rmse", "mae", "r2", "spearman", "kendall")
HIGHER_IS_BETTER = {"rmse": False, "mae": False, "r2": True, "spearman": True, "kendall": True}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) == 0.0:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) == 0.0:
        return float("nan")
    return float(stats.kendalltau(y_true, y_pred, variant="b").statistic)


_FUNCS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman, "kendall": kendall}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return {name: _FUNCS[name](y_true, y_pred) for name in METRIC_NAMES}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap over test molecules for every metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) == 0.0:
            continue
        for name in METRIC_NAMES:
            draws[name].append(_FUNCS[name](yt, yp))

    out: dict[str, dict[str, float]] = {}
    point = compute_all(y_true, y_pred)
    for name in METRIC_NAMES:
        vals = np.asarray([v for v in draws[name] if np.isfinite(v)], dtype=float)
        if vals.size == 0:
            out[name] = {"point": point[name], "lo": float("nan"), "hi": float("nan")}
            continue
        out[name] = {
            "point": point[name],
            "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
        }
    return out


def format_table(rows: dict[str, dict[str, dict[str, float]]]) -> str:
    """Markdown table of model -> metric -> {point, lo, hi}."""
    header = "| model | " + " | ".join(
        f"{m.upper()} (95% CI)" for m in METRIC_NAMES
    ) + " |"
    sep = "|---" * (len(METRIC_NAMES) + 1) + "|"
    lines = [header, sep]
    for model, metrics in rows.items():
        cells = []
        for m in METRIC_NAMES:
            d = metrics[m]
            cells.append(f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)
