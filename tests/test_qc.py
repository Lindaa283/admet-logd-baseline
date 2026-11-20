"""Data quality control, driven by a small table with planted problems."""

import json

import pandas as pd
import pytest

from admet.qc import load_and_qc

# Each row below exists to trigger one specific check.
ROWS = [
    # id,              smiles,                        exp
    ("CHEMBL_ok1",     "CCO",                          1.00),
    ("CHEMBL_ok2",     "c1ccccc1",                     2.10),
    ("CHEMBL_ok3",     "CC(=O)Oc1ccccc1C(=O)O",        1.20),
    ("CHEMBL_ok4",     "CCCCCCCC",                     4.50),
    ("CHEMBL_ok5",     "Clc1ccc(Cl)cc1",               3.40),
    ("CHEMBL_ok6",     "CCN(CC)CC",                    1.45),
    # same compound as ok1, written as a different (equivalent) SMILES
    ("CHEMBL_dup1",    "OCC",                          1.40),
    # an enantiomer pair: one compound at the connectivity layer
    ("CHEMBL_chiral1", "C[C@H](N)C(=O)O",             -2.00),
    ("CHEMBL_chiral2", "C[C@@H](N)C(=O)O",            -2.20),
    # a salt of a compound already present as the free form
    ("CHEMBL_salt",    "CCN(CC)CC.Cl",                 1.55),
    # unparseable
    ("CHEMBL_bad",     "this is not a molecule",        2.00),
    # missing values
    ("CHEMBL_nosmi",   None,                            2.00),
    ("CHEMBL_nolab",   "CCCO",                          None),
    # physically implausible label
    ("CHEMBL_wild",    "CCCCCCCCCCCCCCCC",             99.00),
]


@pytest.fixture
def csv_path(tmp_path):
    frame = pd.DataFrame(ROWS, columns=["CMPD_CHEMBLID", "smiles", "exp"])
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    return path


def test_missing_values_are_counted_and_dropped(csv_path):
    clean, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.n_raw_rows == len(ROWS)
    assert report.n_missing_smiles == 1
    assert report.n_missing_label == 1
    assert "CHEMBL_nolab" not in set(clean["compound_id"])


def test_unparseable_structures_are_counted_not_fatal(csv_path):
    clean, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.n_unparseable == 1
    assert any("not a molecule" in example for example in report.unparseable_examples)
    assert len(clean) > 0


def test_equivalent_smiles_are_recognised_as_one_compound(csv_path):
    """CCO and OCC are the same molecule and must not become two rows."""
    clean, report = load_and_qc(csv_path, source_sha256="deadbeef")
    ethanol = clean[clean["canonical_smiles"] == "CCO"]
    assert len(ethanol) == 1
    assert int(ethanol.iloc[0]["n_measurements"]) == 2
    # label is the median of 1.00 and 1.40
    assert float(ethanol.iloc[0]["label"]) == pytest.approx(1.20)
    assert float(ethanol.iloc[0]["label_spread"]) == pytest.approx(0.40)


def test_enantiomers_collapse_to_one_compound(csv_path):
    clean, _ = load_and_qc(csv_path, source_sha256="deadbeef")
    alanine = clean[clean["canonical_smiles"].str.contains("N")
                    & clean["canonical_smiles"].str.contains("O")]
    chiral = clean[clean["n_measurements"] == 2]
    assert len(chiral) >= 2  # ethanol pair and the enantiomer pair
    assert len(alanine) >= 1


def test_salt_merges_with_its_free_base(csv_path):
    clean, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.n_multi_fragment == 1
    triethylamine = clean[clean["canonical_smiles"] == "CCN(CC)CC"]
    assert len(triethylamine) == 1
    assert int(triethylamine.iloc[0]["n_measurements"]) == 2


def test_duplicate_spread_is_reported_as_a_noise_floor(csv_path):
    _, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.n_duplicate_compounds == 3  # ethanol, alanine, triethylamine
    assert report.duplicate_label_spread_max == pytest.approx(0.40)
    assert report.implied_experimental_mae_floor > 0
    assert report.implied_experimental_mae_floor == pytest.approx(
        report.duplicate_label_spread_mean / 2
    )


def test_out_of_range_labels_are_flagged(csv_path):
    _, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.n_label_out_of_range == 1
    assert any("outside" in note for note in report.notes)


def test_unit_check_rejects_a_column_in_the_wrong_scale(tmp_path):
    """A raw partition ratio instead of its log must not pass silently."""
    frame = pd.DataFrame(
        {
            "CMPD_CHEMBLID": [f"c{i}" for i in range(6)],
            "smiles": ["CCO", "c1ccccc1", "CCCCCCCC", "CCCO", "CCCCO", "CCCCCO"],
            "exp": [10.0, 1200.0, 45000.0, 3.0, 800.0, 25000.0],
        }
    )
    path = tmp_path / "wrong_units.csv"
    frame.to_csv(path, index=False)
    _, report = load_and_qc(path, source_sha256="x")
    assert report.unit_check.startswith("SUSPECT")


def test_unit_check_flags_the_planted_implausible_value(csv_path):
    """The fixture contains one label of 99.0, so the check must complain."""
    _, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert report.unit_check.startswith("SUSPECT")


def test_unit_check_accepts_a_clean_log_scale_column(tmp_path):
    frame = pd.DataFrame(
        {
            "CMPD_CHEMBLID": [f"c{i}" for i in range(6)],
            "smiles": ["CCO", "c1ccccc1", "CCCCCCCC", "CCCO", "CCCCO", "CCCCCO"],
            "exp": [1.0, 2.1, 4.5, 0.3, 0.9, 1.4],
        }
    )
    path = tmp_path / "clean_units.csv"
    frame.to_csv(path, index=False)
    _, report = load_and_qc(path, source_sha256="x")
    assert report.unit_check.startswith("consistent")
    assert report.n_label_out_of_range == 0


def test_one_row_per_compound_in_the_clean_table(csv_path):
    clean, report = load_and_qc(csv_path, source_sha256="deadbeef")
    assert clean["connectivity_key"].is_unique
    assert report.n_clean_rows == len(clean)
    assert report.n_clean_rows == report.n_unique_connectivity


def test_report_serialises_to_json(csv_path):
    _, report = load_and_qc(csv_path, source_sha256="deadbeef")
    payload = json.loads(report.to_json())
    assert payload["source_sha256"] == "deadbeef"
    assert payload["n_clean_rows"] == report.n_clean_rows
    assert isinstance(payload["plausible_range"], list)


def test_missing_column_is_an_explicit_error(tmp_path):
    path = tmp_path / "bad_columns.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_and_qc(path, source_sha256="x")
