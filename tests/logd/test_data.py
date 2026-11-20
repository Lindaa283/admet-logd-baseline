import numpy as np
import pandas as pd
import pytest

from logd import config as cfg
from logd import data


def _raw(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {cfg.RAW_ID_COL: i, cfg.RAW_LABEL_COL: lab, cfg.RAW_SMILES_COL: smi}
            for i, lab, smi in rows
        ]
    )


def test_standardise_canonicalises_equivalent_smiles():
    a, _ = data.standardise("C1=CC=CC=C1")
    b, _ = data.standardise("c1ccccc1")
    assert a == b


def test_standardise_is_idempotent():
    once, _ = data.standardise("CC(=O)Oc1ccccc1C(=O)O")
    twice, _ = data.standardise(once)
    assert once == twice


def test_standardise_strips_counterions_and_reports_it():
    smiles, stripped = data.standardise("CCO.[Na+]")
    assert stripped is True
    assert smiles == "CCO"
    assert "Na" not in smiles


def test_standardise_keeps_charges_because_logd_is_ph_dependent():
    """Neutralising would change what the label means; it must not happen."""
    smiles, _ = data.standardise("CC(=O)[O-]")
    assert "[O-]" in smiles


def test_standardise_returns_none_for_garbage():
    smiles, _ = data.standardise("not a molecule at all")
    assert smiles is None


def test_curate_drops_unparseable_and_balances_the_ledger():
    raw = _raw(
        [
            ("A", 1.0, "CCO"),
            ("B", 2.0, "c1ccccc1"),
            ("C", 3.0, "%%%broken%%%"),
        ]
    )
    df, report = data.curate(raw)
    assert report.raw_rows == 3
    assert report.unparseable_smiles == 1
    assert report.final_rows == 2
    report.check_accounting()  # raises if the ledger does not balance


def test_curate_merges_duplicates_that_agree():
    raw = _raw([("A", 1.0, "CCO"), ("B", 1.2, "OCC"), ("C", 5.0, "c1ccccc1")])
    df, report = data.curate(raw)

    assert report.duplicate_structures == 1
    assert report.duplicates_merged == 1
    assert report.duplicates_dropped_conflicting == 0
    assert report.final_rows == 2

    value = df.loc[df[cfg.SMILES_COL] == "CCO", cfg.LABEL_COL].iloc[0]
    assert value == pytest.approx(1.1)
    report.check_accounting()


def test_curate_drops_duplicates_that_contradict_each_other():
    """Averaging contradictory measurements would invent an unmeasured label."""
    raw = _raw([("A", 1.0, "CCO"), ("B", 4.0, "OCC"), ("C", 5.0, "c1ccccc1")])
    df, report = data.curate(raw)

    assert report.duplicates_dropped_conflicting == 2
    assert report.duplicates_merged == 0
    assert "CCO" not in set(df[cfg.SMILES_COL])
    assert report.final_rows == 1
    report.check_accounting()


def test_curate_respects_the_tolerance_boundary():
    spread = cfg.DUPLICATE_LABEL_TOLERANCE
    raw = _raw([("A", 1.0, "CCO"), ("B", 1.0 + spread, "OCC")])
    _, report = data.curate(raw)
    assert report.duplicates_merged == 1, "spread exactly at tolerance must merge"


def test_curate_drops_missing_labels():
    raw = _raw([("A", np.nan, "CCO"), ("B", 2.0, "c1ccccc1")])
    df, report = data.curate(raw)
    assert report.missing_label == 1
    assert report.final_rows == 1
    report.check_accounting()


def test_curate_output_has_unique_structures_and_no_nan_labels():
    raw = _raw(
        [
            ("A", 1.0, "CCO"),
            ("B", 1.1, "OCC"),
            ("C", 2.0, "c1ccccc1"),
            ("D", 3.0, "CCOCC.[Cl-]"),
            ("E", 4.0, "garbage"),
        ]
    )
    df, report = data.curate(raw)
    assert df[cfg.SMILES_COL].is_unique
    assert df[cfg.LABEL_COL].notna().all()
    assert list(df.columns) == [cfg.ID_COL, cfg.SMILES_COL, cfg.LABEL_COL]
    report.check_accounting()


def test_check_accounting_fires_when_the_ledger_is_wrong():
    report = data.CurationReport(raw_rows=10, final_rows=3)
    with pytest.raises(AssertionError, match="ledger does not balance"):
        report.check_accounting()
