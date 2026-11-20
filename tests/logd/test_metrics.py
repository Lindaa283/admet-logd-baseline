import numpy as np
import pytest

from logd import metrics


def test_rmse_and_mae_against_hand_computed_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    # errors: 0, 0, 0, 2  -> mse = 4/4 = 1 -> rmse = 1 ; mae = 2/4 = 0.5
    assert metrics.rmse(y_true, y_pred) == pytest.approx(1.0)
    assert metrics.mae(y_true, y_pred) == pytest.approx(0.5)


def test_r2_is_zero_for_the_mean_predictor():
    y_true = np.array([1.0, 2.0, 3.0, 10.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert metrics.r2(y_true, y_pred) == pytest.approx(0.0)


def test_r2_is_one_for_perfect_prediction():
    y = np.array([0.5, 1.5, 2.5])
    assert metrics.r2(y, y.copy()) == pytest.approx(1.0)


def test_rank_metrics_are_one_for_monotone_and_minus_one_for_reversed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    monotone = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    reversed_ = y_true[::-1].copy()

    assert metrics.spearman(y_true, monotone) == pytest.approx(1.0)
    assert metrics.kendall(y_true, monotone) == pytest.approx(1.0)
    assert metrics.spearman(y_true, reversed_) == pytest.approx(-1.0)
    assert metrics.kendall(y_true, reversed_) == pytest.approx(-1.0)


def test_rank_metrics_ignore_monotone_rescaling():
    """A constant offset cannot change a ranking, so tau must not move."""
    y_true = np.array([0.1, 1.7, 2.2, 3.9, 4.4, 5.1])
    pred = np.array([1.0, 1.2, 3.0, 3.1, 4.9, 6.0])
    assert metrics.kendall(y_true, pred) == pytest.approx(
        metrics.kendall(y_true, pred * 3.0 + 7.0)
    )


def test_rank_metrics_are_nan_for_a_constant_predictor():
    y_true = np.array([1.0, 2.0, 3.0])
    const = np.array([2.0, 2.0, 2.0])
    assert np.isnan(metrics.kendall(y_true, const))
    assert np.isnan(metrics.spearman(y_true, const))


def test_compute_all_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.compute_all(np.array([1.0, 2.0]), np.array([1.0]))


def test_bootstrap_interval_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=250)
    y_pred = y_true + rng.normal(scale=0.5, size=250)

    a = metrics.bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)
    b = metrics.bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)
    assert a == b, "same seed must give the same interval"

    for name in metrics.METRIC_NAMES:
        d = a[name]
        assert d["lo"] <= d["point"] <= d["hi"], f"{name} point estimate outside CI"
