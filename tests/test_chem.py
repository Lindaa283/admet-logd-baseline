"""Structure standardisation and identity keys."""

import math

import pytest

from admet.chem import (
    DESCRIPTOR_NAMES,
    clogp,
    descriptors,
    mol_from_canonical,
    standardise,
)


def test_parses_a_simple_molecule():
    result = standardise("CCO")
    assert result is not None
    assert result.canonical_smiles == "CCO"
    assert result.heavy_atoms == 3
    assert result.n_fragments == 1
    assert len(result.inchikey) == 27  # 14-10-1 InChIKey layout
    assert result.connectivity_key == result.inchikey.split("-")[0]


def test_rejects_unparseable_input_instead_of_raising():
    # A count of these is reported by QC; a crash would lose the whole run.
    for bad in ["not_a_smiles", "C(((", "", "   ", None, 42]:
        assert standardise(bad) is None


def test_salt_is_reduced_to_its_parent():
    """A counter-ion must not change the structure we model."""
    salt = standardise("CC(=O)[O-].[Na+]")
    free = standardise("CC(=O)[O-]")
    assert salt is not None and free is not None
    assert salt.n_fragments == 2
    assert free.n_fragments == 1
    assert salt.canonical_smiles == free.canonical_smiles
    assert salt.connectivity_key == free.connectivity_key


def test_largest_fragment_wins_regardless_of_input_order():
    first = standardise("[Na+].CC(=O)[O-]")
    second = standardise("CC(=O)[O-].[Na+]")
    assert first is not None and second is not None
    assert first.canonical_smiles == second.canonical_smiles


def test_enantiomers_share_a_connectivity_key_but_not_a_full_key():
    """logD does not distinguish enantiomers, so leakage checks must not either."""
    left = standardise("C[C@H](N)C(=O)O")
    right = standardise("C[C@@H](N)C(=O)O")
    assert left is not None and right is not None
    assert left.inchikey != right.inchikey
    assert left.connectivity_key == right.connectivity_key


def test_canonical_smiles_round_trips():
    result = standardise("OCC")
    assert result is not None
    assert mol_from_canonical(result.canonical_smiles) is not None


def test_descriptor_set_is_complete_and_finite():
    mol = mol_from_canonical("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    values = descriptors(mol)
    assert set(values) == set(DESCRIPTOR_NAMES)
    for name, value in values.items():
        assert math.isfinite(value), f"{name} is not finite"


def test_descriptors_pick_up_the_obvious_chemistry():
    aspirin = descriptors(mol_from_canonical("CC(=O)Oc1ccccc1C(=O)O"))
    assert aspirin["n_acidic_groups"] >= 1       # the carboxylic acid
    assert aspirin["NumAromaticRings"] == 1
    assert aspirin["RingCount"] == 1

    amine = descriptors(mol_from_canonical("CCN(CC)CC"))  # triethylamine
    assert amine["n_basic_groups"] >= 1
    assert amine["n_acidic_groups"] == 0

    halide = descriptors(mol_from_canonical("Clc1ccc(Cl)cc1"))
    assert halide["n_halogens"] == 2

    phenol = descriptors(mol_from_canonical("Oc1ccccc1"))
    assert phenol["n_phenol"] == 1


def test_amide_nitrogen_is_not_counted_as_basic():
    """The ionisation term is only useful if it tracks real basicity."""
    amide = descriptors(mol_from_canonical("CC(=O)NC"))
    assert amide["n_basic_groups"] == 0


def test_n_fragments_is_passed_through():
    mol = mol_from_canonical("CCO")
    assert descriptors(mol, n_fragments=3)["n_fragments"] == 3


def test_clogp_orders_molecules_sensibly():
    """Not a precision claim — just that the baseline feature is not inverted."""
    assert clogp(mol_from_canonical("CCCCCCCC")) > clogp(mol_from_canonical("CCO"))
    assert clogp(mol_from_canonical("c1ccccc1")) > clogp(mol_from_canonical("OCC(O)CO"))


def test_descriptors_are_deterministic():
    mol = mol_from_canonical("CC(=O)Oc1ccccc1C(=O)O")
    assert descriptors(mol) == descriptors(mol)


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                                    "CN1CCC[C@H]1c1cccnc1"])
def test_standardise_is_idempotent(smiles):
    once = standardise(smiles)
    twice = standardise(once.canonical_smiles)
    assert twice is not None
    assert once.canonical_smiles == twice.canonical_smiles
    assert once.inchikey == twice.inchikey
