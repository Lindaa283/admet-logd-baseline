"""The model ladder and the train/validate/test protocol."""

import numpy as np
import pytest

from admet.models import (
    ConstantRegressor,
    FeatureBlocks,
    fit_with_validation,
    model_specs,
)


@pytest.fixture
def blocks_and_labels():
    """A synthetic problem where the label is a known function of features.

    Descriptor column 1 is MolLogP by construction (see DESCRIPTOR_NAMES
    ordering), so a cLogP-style model has real signal to find.
    """
    rng = np.random.default_rng(0)
    n_rows, n_descriptors, n_bits = 400, 24, 64
    descriptors = rng.normal(size=(n_rows, n_descriptors))
    morgan = rng.integers(0, 3, size=(n_rows, n_bits)).astype(np.float32)
    # label depends on the cLogP column plus some fingerprint structure
    y = (
        1.5 * descriptors[:, 1]
        + 0.4 * morgan[:, 0]
        - 0.3 * morgan[:, 1]
        + rng.normal(scale=0.2, size=n_rows)
    )
    return FeatureBlocks(descriptors, morgan), y


def test_feature_blocks_slice_the_requested_rows():
    descriptors = np.arange(30, dtype=float).reshape(10, 3)
    morgan = np.arange(50, dtype=np.float32).reshape(10, 5)
    blocks = FeatureBlocks(descriptors, morgan)
    rows = np.array([0, 2, 4])

    assert blocks.n_rows == 10
    assert blocks.get("descriptors", rows).shape == (3, 3)
    assert blocks.get("morgan", rows).shape == (3, 5)
    assert blocks.get("both", rows).shape == (3, 8)
    assert blocks.get("clogp", rows).shape == (3, 1)
    assert blocks.get("none", rows).shape == (3, 1)


def test_feature_blocks_reject_mismatched_inputs():
    with pytest.raises(ValueError):
        FeatureBlocks(np.zeros((5, 3)), np.zeros((4, 3)))


def test_unknown_feature_block_is_an_error():
    blocks = FeatureBlocks(np.zeros((5, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        blocks.get("nonsense", np.array([0]))


def test_constant_regressor_predicts_the_training_mean():
    model = ConstantRegressor().fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))
    predictions = model.predict(np.zeros((3, 1)))
    assert predictions.shape == (3,)
    assert np.allclose(predictions, 2.5)


def test_constant_regressor_ignores_the_features():
    y = np.array([5.0, 7.0])
    first = ConstantRegressor().fit(np.zeros((2, 1)), y).predict(np.zeros((1, 1)))
    second = ConstantRegressor().fit(np.ones((2, 9)), y).predict(np.ones((1, 9)))
    assert first[0] == second[0] == 6.0


def test_every_spec_fits_and_predicts(blocks_and_labels):
    blocks, y = blocks_and_labels
    train = np.arange(0, 280)
    val = np.arange(280, 320)
    test = np.arange(320, 400)

    for spec in model_specs():
        fitted = fit_with_validation(spec, blocks, y, train, val)
        predictions = fitted.predict(blocks, test)
        assert predictions.shape == (test.size,)
        assert np.isfinite(predictions).all()
        assert fitted.chosen_params in spec.grid
        assert np.isfinite(fitted.validation_mae)


def test_the_ladder_beats_the_constant_when_there_is_signal(blocks_and_labels):
    """If a real model cannot beat a constant, the comparison is broken."""
    blocks, y = blocks_and_labels
    train = np.arange(0, 280)
    val = np.arange(280, 320)
    test = np.arange(320, 400)

    results = {}
    for spec in model_specs():
        fitted = fit_with_validation(spec, blocks, y, train, val)
        predictions = fitted.predict(blocks, test)
        results[spec.name] = float(np.mean(np.abs(predictions - y[test])))

    assert results["descriptor_ridge"] < results["constant"]
    assert results["morgan_desc_hgb"] < results["constant"]
    # column 1 is the designed cLogP-like driver, so this must find signal too
    assert results["clogp_calibrated"] < results["constant"]


def test_hyperparameters_are_chosen_from_the_grid(blocks_and_labels):
    blocks, y = blocks_and_labels
    spec = next(s for s in model_specs() if s.name == "descriptor_ridge")
    fitted = fit_with_validation(spec, blocks, y, np.arange(0, 300),
                                 np.arange(300, 360))
    assert "alpha" in fitted.chosen_params
    assert fitted.chosen_params["alpha"] in [g["alpha"] for g in spec.grid]


def test_fitting_is_reproducible(blocks_and_labels):
    blocks, y = blocks_and_labels
    spec = next(s for s in model_specs() if s.name == "morgan_rf")
    train, val, test = np.arange(0, 280), np.arange(280, 320), np.arange(320, 400)
    first = fit_with_validation(spec, blocks, y, train, val).predict(blocks, test)
    second = fit_with_validation(spec, blocks, y, train, val).predict(blocks, test)
    assert np.allclose(first, second)


def test_model_names_are_unique_and_blocks_are_known():
    specs = model_specs()
    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert {spec.block for spec in specs} <= {
        "none", "clogp", "descriptors", "morgan", "both"
    }
    for spec in specs:
        assert spec.description.strip()
