"""Bemis-Murcko scaffolds and the splitting guarantee."""

import numpy as np
import pytest

from admet.chem import mol_from_canonical
from admet.scaffold import (
    murcko_scaffold,
    random_split,
    scaffold_groups,
    scaffold_split,
)


def test_side_chains_are_stripped():
    # Aspirin's framework is its benzene ring; the ester and acid go.
    assert murcko_scaffold(mol_from_canonical("CC(=O)Oc1ccccc1C(=O)O")) == "c1ccccc1"


def test_linkers_are_kept():
    """Two rings joined by a CH2 are one framework, not two."""
    scaffold = murcko_scaffold(mol_from_canonical("c1ccccc1Cc1ccccc1"))
    assert "C" in scaffold          # the linker survived
    assert scaffold.count("c1") >= 1


def test_acyclic_molecules_share_the_empty_scaffold():
    assert murcko_scaffold(mol_from_canonical("CCO")) == ""
    assert murcko_scaffold(mol_from_canonical("CCCCCC")) == ""


def test_stereochemistry_does_not_split_a_series():
    left = murcko_scaffold(mol_from_canonical("C[C@H]1CCCCN1c1ccccc1"))
    right = murcko_scaffold(mol_from_canonical("C[C@@H]1CCCCN1c1ccccc1"))
    assert left == right


def test_generic_scaffold_merges_heteroatom_variants():
    benzene = murcko_scaffold(mol_from_canonical("c1ccccc1C(=O)N"), generic=True)
    pyridine = murcko_scaffold(mol_from_canonical("c1ccncc1C(=O)N"), generic=True)
    assert benzene == pyridine


def test_murcko_scaffold_rejects_none():
    with pytest.raises(ValueError):
        murcko_scaffold(None)


def test_groups_cover_every_row_exactly_once():
    scaffolds = ["a", "b", "a", "c", "b", "a"]
    groups = scaffold_groups(scaffolds)
    assert groups == {"a": [0, 2, 5], "b": [1, 4], "c": [3]}
    assert sum(len(v) for v in groups.values()) == len(scaffolds)


# A deliberately lumpy population: a few big series plus many singletons,
# which is what a real medicinal-chemistry table looks like.
LUMPY = ["core_a"] * 40 + ["core_b"] * 25 + ["core_c"] * 15 + [
    f"singleton_{i}" for i in range(20)
]


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_split_is_a_partition(seed):
    train, val, test = scaffold_split(LUMPY, seed=seed)
    combined = np.concatenate([train, val, test])
    assert sorted(combined.tolist()) == list(range(len(LUMPY)))
    assert len(np.unique(combined)) == len(LUMPY)


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_no_scaffold_appears_in_two_splits(seed):
    """The whole reason for using this split."""
    train, val, test = scaffold_split(LUMPY, seed=seed)
    sets = [{LUMPY[i] for i in part} for part in (train, val, test)]
    assert sets[0] & sets[2] == set()
    assert sets[0] & sets[1] == set()
    assert sets[1] & sets[2] == set()


def test_deterministic_split_is_reproducible_without_a_seed():
    first = scaffold_split(LUMPY, seed=None)
    second = scaffold_split(LUMPY, seed=None)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_seeded_splits_are_reproducible():
    assert np.array_equal(scaffold_split(LUMPY, seed=3)[2],
                          scaffold_split(LUMPY, seed=3)[2])


def test_seeded_splits_depend_on_the_seed():
    """Different seeds must be able to produce different partitions.

    Checked on a population of many equal-sized groups.  With only a handful
    of very unequal groups (as in LUMPY) the greedy fill is so constrained
    that two seeds can legitimately land on the same partition, so that would
    not be a fair test of seed sensitivity.
    """
    many_groups = [f"core_{index // 5}" for index in range(200)]  # 40 groups of 5
    test_sets = {
        tuple(scaffold_split(many_groups, seed=seed)[2].tolist())
        for seed in range(6)
    }
    assert len(test_sets) > 1


def test_deterministic_split_puts_the_biggest_series_in_train():
    train, _, test = scaffold_split(LUMPY, seed=None)
    assert "core_a" in {LUMPY[i] for i in train}
    assert "core_a" not in {LUMPY[i] for i in test}


def test_split_sizes_are_in_the_right_region():
    # Whole groups move together, so the fractions cannot be exact.
    train, val, test = scaffold_split(LUMPY, fractions=(0.7, 0.1, 0.2))
    total = len(LUMPY)
    assert 0.55 <= len(train) / total <= 0.85
    assert len(test) > 0
    assert len(val) > 0


def test_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        scaffold_split(LUMPY, fractions=(0.8, 0.3, 0.2))
    with pytest.raises(ValueError):
        random_split(100, fractions=(0.5, 0.2, 0.2))


def test_random_split_is_a_partition_and_seed_dependent():
    train, val, test = random_split(200, seed=0)
    combined = np.concatenate([train, val, test])
    assert sorted(combined.tolist()) == list(range(200))
    assert not np.array_equal(random_split(200, seed=0)[2],
                              random_split(200, seed=1)[2])


def test_random_split_does_not_respect_scaffolds():
    """Stated as a test so the contrast with the scaffold split is explicit."""
    train, val, test = random_split(len(LUMPY), seed=0)
    train_scaffolds = {LUMPY[i] for i in train}
    test_scaffolds = {LUMPY[i] for i in test}
    assert train_scaffolds & test_scaffolds  # overlap is expected here
