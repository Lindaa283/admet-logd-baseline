import numpy as np
import pytest

from logd import config as cfg
from logd import featurize

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCCCCCCCCCC"]


def test_descriptor_list_excludes_the_overflowing_descriptor():
    names = featurize.descriptor_names()
    for excluded in cfg.EXCLUDED_DESCRIPTORS:
        assert excluded not in names
    assert len(names) > 100


def test_ecfp4_shape_and_binary_values():
    X = featurize.ecfp4(SMILES)
    assert X.shape == (len(SMILES), cfg.MORGAN_NBITS)
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_ecfp4_distinguishes_different_molecules():
    X = featurize.ecfp4(["CCO", "c1ccccc1"])
    assert not np.array_equal(X[0], X[1])


def test_descriptors_contain_no_infinities():
    """Inf would survive imputation and destroy any standardised model."""
    X = featurize.descriptors(SMILES)
    assert X.shape == (len(SMILES), len(featurize.descriptor_names()))
    assert not np.isinf(X).any()


def test_clogp_is_monotone_in_an_obvious_homologous_series():
    X = featurize.clogp(["CCO", "CCCCO", "CCCCCCCCO"])
    assert X.shape == (3, 1)
    assert X[0, 0] < X[1, 0] < X[2, 0]


def test_combined_block_is_the_concatenation_of_its_parts():
    desc = featurize.descriptors(SMILES)
    fp = featurize.ecfp4(SMILES)
    combined = featurize.featurize(SMILES, "desc_ecfp4")
    assert combined.shape == (len(SMILES), desc.shape[1] + fp.shape[1])
    np.testing.assert_allclose(combined[:, desc.shape[1] :], fp)


def test_feature_names_match_matrix_width_for_every_block():
    for block in featurize.FEATURE_BLOCKS:
        X = featurize.featurize(SMILES, block)
        assert X.shape[1] == len(featurize.feature_names(block)), block


def test_featurisation_is_deterministic():
    a = featurize.featurize(SMILES, "desc")
    b = featurize.featurize(SMILES, "desc")
    np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_featurisers_reject_unparseable_smiles():
    with pytest.raises(ValueError):
        featurize.featurize(["definitely not a molecule"], "ecfp4")


def test_unknown_block_is_rejected():
    with pytest.raises(ValueError):
        featurize.featurize(SMILES, "graph_neural_magic")
