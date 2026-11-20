"""The leakage audit, tested by planting leaks it must catch."""

import json

import numpy as np
import pytest

from admet.chem import mol_from_canonical
from admet.features import morgan_bitvects
from admet.leakage import assert_all_passed, audit_split, write_audits

SCAFFOLDS = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
IDENTITIES = [f"KEY{i:03d}" for i in range(30)]


def clean_partition():
    train = np.arange(0, 10)    # scaffold a
    val = np.arange(10, 20)     # scaffold b
    test = np.arange(20, 30)    # scaffold c
    return train, val, test


def test_a_clean_scaffold_split_passes():
    train, val, test = clean_partition()
    audit = audit_split("clean", train, val, test, scaffolds=SCAFFOLDS,
                        identity_keys=IDENTITIES, require_scaffold_disjoint=True)
    assert audit.passed
    assert audit.failures == []
    assert audit.n_shared_scaffolds_train_test == 0
    assert audit.n_train == 10 and audit.n_val == 10 and audit.n_test == 10


def test_a_row_in_two_splits_is_caught():
    train = np.arange(0, 11)     # row 10 also appears in val
    val = np.arange(10, 20)
    test = np.arange(20, 30)
    audit = audit_split("overlap", train, val, test, scaffolds=SCAFFOLDS,
                        identity_keys=IDENTITIES, require_scaffold_disjoint=False)
    assert not audit.passed
    assert audit.n_overlapping_rows == 1
    assert any("more than one split" in f for f in audit.failures)


def test_the_same_compound_on_both_sides_is_caught():
    """Different rows, same compound — the leak that quietly flatters a model."""
    identities = list(IDENTITIES)
    identities[25] = identities[3]  # a test row is really a training compound
    train, val, test = clean_partition()
    audit = audit_split("identity", train, val, test, scaffolds=SCAFFOLDS,
                        identity_keys=identities, require_scaffold_disjoint=True)
    assert not audit.passed
    assert audit.n_duplicate_identity_across_splits == 1
    assert any("identity collision" in f for f in audit.failures)


def test_shared_scaffold_fails_only_when_disjointness_is_required():
    scaffolds = ["a"] * 10 + ["b"] * 10 + ["a"] * 10  # test reuses train's core
    train, val, test = clean_partition()

    strict = audit_split("strict", train, val, test, scaffolds=scaffolds,
                         identity_keys=IDENTITIES, require_scaffold_disjoint=True)
    assert not strict.passed
    assert strict.n_shared_scaffolds_train_test == 1
    assert strict.n_test_rows_with_train_scaffold == 10

    # The same partition is fine when it is explicitly a random split, where
    # scaffold overlap is expected and is itself the thing being measured.
    lenient = audit_split("lenient", train, val, test, scaffolds=scaffolds,
                          identity_keys=IDENTITIES, require_scaffold_disjoint=False)
    assert lenient.passed
    assert lenient.n_shared_scaffolds_train_test == 1


def test_similarity_statistics_are_computed_when_fingerprints_are_given():
    smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    fingerprints = morgan_bitvects([mol_from_canonical(s) for s in smiles])
    train = np.array([0, 1, 2])
    val = np.array([3])
    test = np.array([4, 5])
    audit = audit_split("sim", train, val, test,
                        scaffolds=["x", "x", "x", "y", "z", "w"],
                        identity_keys=[f"K{i}" for i in range(6)],
                        fingerprints=fingerprints,
                        require_scaffold_disjoint=True)
    assert 0.0 <= audit.mean_nearest_similarity <= 1.0
    assert 0.0 <= audit.max_nearest_similarity <= 1.0
    assert audit.n_test_with_identical_fingerprint_in_train == 0


def test_an_identical_structure_in_train_is_reported():
    smiles = ["CCO", "c1ccccc1", "CCO"]  # row 2 duplicates row 0
    fingerprints = morgan_bitvects([mol_from_canonical(s) for s in smiles])
    audit = audit_split("dup_fp", np.array([0]), np.array([1]), np.array([2]),
                        scaffolds=["", "ring", ""],
                        identity_keys=["A", "B", "C"],
                        fingerprints=fingerprints,
                        require_scaffold_disjoint=False)
    assert audit.n_test_with_identical_fingerprint_in_train == 1
    assert audit.max_nearest_similarity == pytest.approx(1.0)


def test_assert_all_passed_raises_with_a_readable_message():
    train = np.arange(0, 11)
    val = np.arange(10, 20)
    test = np.arange(20, 30)
    bad = audit_split("bad", train, val, test, scaffolds=SCAFFOLDS,
                      identity_keys=IDENTITIES, require_scaffold_disjoint=False)
    with pytest.raises(RuntimeError, match="leakage audit failed"):
        assert_all_passed([bad])


def test_assert_all_passed_is_quiet_when_everything_is_fine():
    train, val, test = clean_partition()
    good = audit_split("good", train, val, test, scaffolds=SCAFFOLDS,
                       identity_keys=IDENTITIES, require_scaffold_disjoint=True)
    assert_all_passed([good])  # must not raise


def test_audits_serialise_to_json(tmp_path):
    train, val, test = clean_partition()
    audit = audit_split("clean", train, val, test, scaffolds=SCAFFOLDS,
                        identity_keys=IDENTITIES, require_scaffold_disjoint=True)
    path = tmp_path / "audit.json"
    write_audits([audit], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["split_name"] == "clean"
    assert payload[0]["passed"] is True
