"""Metrics, checked against values worked out by hand."""

import math

import numpy as np
import pytest

from admet.metrics import bootstrap_ci, score


def test_metrics_match_hand_computed_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    # residuals [0, 0, +1]; SS_res = 1; mean(y)=2 so SS_tot = 1+0+1 = 2
    result = score(y_true, y_pred)
    assert result.n == 3
    assert result.mae == pytest.approx(1 / 3)
    assert result.rmse == pytest.approx(math.sqrt(1 / 3))
    assert result.r2 == pytest.approx(0.5)
    assert result.bias == pytest.approx(1 / 3)
    assert result.max_abs_error == pytest.approx(1.0)


def test_perfect_prediction():
    y = np.array([0.5, 1.5, -2.0, 3.25])
    result = score(y, y.copy())
    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.r2 == pytest.approx(1.0)
    assert result.spearman == pytest.approx(1.0)


def test_constant_prediction_gives_zero_r2_at_the_mean():
    """A constant at the test mean explains none of the variance."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = score(y, np.full_like(y, y.mean()))
    assert result.r2 == pytest.approx(0.0)
    assert math.isnan(result.spearman)  # no variance in the prediction


def test_worse_than_the_mean_gives_negative_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = score(y, np.full_like(y, 100.0))
    assert result.r2 < 0


def test_rmse_is_at_least_mae():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.normal(size=50)
        y_pred = y_true + rng.normal(scale=0.5, size=50)
        result = score(y_true, y_pred)
        assert result.rmse >= result.mae - 1e-12


def test_spearman_is_rank_based_not_scale_based():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = score(y_true, y_true * 10 + 7)  # monotone, badly calibrated
    assert result.spearman == pytest.approx(1.0)
    assert result.mae > 1.0  # but the absolute error is large


def test_shape_and_emptiness_are_errors():
    with pytest.raises(ValueError):
        score(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        score(np.array([]), np.array([]))


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=300)
    y_pred = y_true + rng.normal(scale=0.4, size=300)
    point = score(y_true, y_pred).mae
    low, high = bootstrap_ci(y_true, y_pred, metric="mae", n_resamples=400, seed=0)
    assert low < point < high
    assert high - low < 0.3  # a sane width for n=300


def test_bootstrap_is_reproducible_under_a_seed():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.3, size=100)
    first = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=5)
    second = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=5)
    assert first == second


def test_bootstrap_interval_is_tighter_with_more_data():
    rng = np.random.default_rng(3)
    widths = []
    for n in (60, 600):
        y_true = rng.normal(size=n)
        y_pred = y_true + rng.normal(scale=0.5, size=n)
        low, high = bootstrap_ci(y_true, y_pred, n_resamples=300, seed=0)
        widths.append(high - low)
    assert widths[1] < widths[0]
