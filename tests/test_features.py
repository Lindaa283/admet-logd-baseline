"""Fingerprints, descriptor matrices, and similarity."""

import numpy as np
import pytest

from admet.chem import mol_from_canonical
from admet.features import (
    MORGAN_BITS,
    descriptor_matrix,
    morgan_bitvects,
    morgan_counts,
    nearest_train_similarity,
)

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCCCCCC",
    "Clc1ccc(Cl)cc1",
]
MOLS = [mol_from_canonical(s) for s in SMILES]


def test_fingerprint_matrix_has_the_declared_shape():
    matrix = morgan_counts(MOLS)
    assert matrix.shape == (len(MOLS), MORGAN_BITS)
    assert matrix.dtype == np.float32


def test_fingerprints_are_non_negative_counts():
    matrix = morgan_counts(MOLS)
    assert (matrix >= 0).all()
    assert np.allclose(matrix, np.round(matrix))


def test_counts_distinguish_multiplicity():
    """Four chlorines must not look like one — the point of counts over bits."""
    one = morgan_counts([mol_from_canonical("Clc1ccccc1")])
    many = morgan_counts([mol_from_canonical("Clc1c(Cl)c(Cl)c(Cl)cc1")])
    assert many.sum() > one.sum()


def test_fingerprints_are_deterministic():
    assert np.array_equal(morgan_counts(MOLS), morgan_counts(MOLS))


def test_bigger_molecules_set_more_bits():
    small = morgan_counts([mol_from_canonical("CCO")])
    large = morgan_counts([mol_from_canonical("CC(=O)Oc1ccccc1C(=O)O")])
    assert (large > 0).sum() > (small > 0).sum()


def test_descriptor_matrix_shape_and_names():
    matrix, names = descriptor_matrix(MOLS)
    assert matrix.shape == (len(MOLS), len(names))
    assert np.isfinite(matrix).all()
    assert "MolLogP" in names


def test_descriptor_matrix_respects_fragment_counts():
    matrix, names = descriptor_matrix(MOLS, [1, 2, 1, 1, 1])
    column = names.index("n_fragments")
    assert matrix[1, column] == 2
    assert matrix[0, column] == 1


def test_identical_molecule_has_similarity_one():
    fps = morgan_bitvects(MOLS)
    similarity = nearest_train_similarity([fps[0]], fps)
    assert similarity[0] == pytest.approx(1.0)


def test_similarity_is_between_zero_and_one():
    fps = morgan_bitvects(MOLS)
    similarity = nearest_train_similarity(fps[:2], fps[2:])
    assert ((similarity >= 0) & (similarity <= 1)).all()


def test_close_analogue_scores_higher_than_an_unrelated_molecule():
    train = morgan_bitvects([mol_from_canonical("CCCCCCCCCC")])
    analogue = morgan_bitvects([mol_from_canonical("CCCCCCCCCCC")])
    unrelated = morgan_bitvects([mol_from_canonical("c1ccncc1C(=O)N")])
    assert (
        nearest_train_similarity(analogue, train)[0]
        > nearest_train_similarity(unrelated, train)[0]
    )


def test_similarity_blocking_does_not_change_the_answer():
    """The blocked loop is an implementation detail, not a different result."""
    fps = morgan_bitvects(MOLS)
    whole = nearest_train_similarity(fps, fps, block=1024)
    chunked = nearest_train_similarity(fps, fps, block=2)
    assert np.allclose(whole, chunked)


def test_empty_training_set_gives_zero_similarity():
    fps = morgan_bitvects(MOLS)
    assert nearest_train_similarity(fps[:1], [])[0] == 0.0
