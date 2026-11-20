import numpy as np
import pytest

from logd import similarity


def test_identical_molecule_has_similarity_one():
    sims = similarity.nearest_train_similarity(["CCO"], ["CCO", "c1ccccc1"])
    assert sims[0] == pytest.approx(1.0)


def test_close_analogue_scores_higher_than_unrelated_molecule():
    train = ["c1ccc2ccccc2c1"]  # naphthalene
    sims = similarity.nearest_train_similarity(
        ["Cc1ccc2ccccc2c1", "CCCCCCCCCCCC"], train
    )
    analogue, unrelated = sims
    assert analogue > unrelated
    assert 0.0 <= unrelated < analogue <= 1.0


def test_similarity_is_the_maximum_over_the_training_set():
    train = ["CCCCCCCCCCCC", "c1ccccc1", "CCO"]
    one_at_a_time = [
        similarity.nearest_train_similarity(["CCO"], [t])[0] for t in train
    ]
    against_all = similarity.nearest_train_similarity(["CCO"], train)[0]
    assert against_all == pytest.approx(max(one_at_a_time))


def test_empty_training_set_is_rejected():
    with pytest.raises(ValueError):
        similarity.nearest_train_similarity(["CCO"], [])


def test_unparseable_smiles_is_rejected():
    with pytest.raises(ValueError):
        similarity.nearest_train_similarity(["not-a-molecule"], ["CCO"])


def test_summary_reports_bounds_and_fractions():
    sims = np.array([0.1, 0.2, 0.45, 0.6, 0.85, 0.9])
    s = similarity.similarity_summary(sims)
    assert s["n"] == 6
    assert s["min"] == pytest.approx(0.1)
    assert s["max"] == pytest.approx(0.9)
    # 0.85 and 0.9 are >= 0.7  -> 2/6
    assert s["fraction_above_0.7_close_analogue"] == pytest.approx(2 / 6, abs=1e-4)
    # 0.1 and 0.2 are < 0.3    -> 2/6
    assert s["fraction_below_0.3_novel_chemotype"] == pytest.approx(2 / 6, abs=1e-4)


def test_bands_partition_every_molecule_exactly_once():
    rng = np.random.default_rng(0)
    sims = rng.uniform(0.0, 1.0, size=500)
    y_true = rng.normal(size=500)
    rows = similarity.error_by_similarity_band(y_true, y_true + 0.1, sims)
    assert sum(r["n"] for r in rows) == 500


def test_small_bands_suppress_their_metrics():
    sims = np.array([0.05, 0.06] + [0.8] * 40)
    y_true = np.arange(42, dtype=float)
    rows = similarity.error_by_similarity_band(y_true, y_true, sims, min_band_n=20)
    low = next(r for r in rows if r["band"] == "[0.0, 0.3)")
    high = next(r for r in rows if r["band"] == "[0.7, 1.0)")
    assert low["n"] == 2 and "rmse" not in low and "note" in low
    assert high["n"] == 40 and high["rmse"] == pytest.approx(0.0)


def test_band_rmse_matches_a_hand_computed_subset():
    sims = np.array([0.1] * 25 + [0.9] * 25)
    y_true = np.zeros(50)
    y_pred = np.concatenate([np.full(25, 2.0), np.zeros(25)])
    rows = similarity.error_by_similarity_band(y_true, y_pred, sims)
    low = next(r for r in rows if r["band"] == "[0.0, 0.3)")
    high = next(r for r in rows if r["band"] == "[0.7, 1.0)")
    assert low["rmse"] == pytest.approx(2.0)
    assert high["rmse"] == pytest.approx(0.0)
