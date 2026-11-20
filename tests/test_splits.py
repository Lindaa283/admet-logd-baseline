import pandas as pd
import pytest

from logd import config as cfg
from logd import splits

# Four distinct ring scaffolds, several substituted analogues each, plus two
# acyclic molecules (which share the empty Murcko scaffold).
_FAMILIES = {
    "benzene": ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1", "COc1ccccc1"],
    "pyridine": ["c1ccncc1", "Cc1ccncc1", "CCc1ccncc1", "Clc1ccncc1"],
    "naphthalene": ["c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "CCc1ccc2ccccc2c1"],
    "indole": ["c1ccc2[nH]ccc2c1", "Cc1ccc2[nH]ccc2c1", "CCc1ccc2[nH]ccc2c1"],
    "acyclic": ["CCCCO", "CCCCCC"],
}


def _frame() -> pd.DataFrame:
    rows = []
    for family, smiles_list in _FAMILIES.items():
        for i, smi in enumerate(smiles_list):
            rows.append(
                {
                    cfg.ID_COL: f"{family}-{i}",
                    cfg.SMILES_COL: smi,
                    cfg.LABEL_COL: float(i) + len(family) / 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_murcko_scaffold_groups_analogues_and_empties_acyclics():
    assert splits.murcko_scaffold("Cc1ccccc1") == splits.murcko_scaffold("CCc1ccccc1")
    assert splits.murcko_scaffold("Cc1ccccc1") != splits.murcko_scaffold("Cc1ccncc1")
    assert splits.murcko_scaffold("CCCCO") == ""


def test_murcko_scaffold_rejects_unparseable_smiles():
    with pytest.raises(ValueError):
        splits.murcko_scaffold("this-is-not-a-molecule")


def test_scaffold_split_never_splits_a_scaffold_family():
    df = _frame()
    split = splits.scaffold_split(df, train_frac=0.6, valid_frac=0.2)

    # Every molecule appears exactly once across the three partitions.
    total = len(split.train) + len(split.valid) + len(split.test)
    assert total == len(df)

    where: dict[str, set[str]] = {}
    for part, frame in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        for smi in frame[cfg.SMILES_COL]:
            where.setdefault(splits.murcko_scaffold(smi), set()).add(part)
    for scaffold, parts in where.items():
        assert len(parts) == 1, f"scaffold {scaffold!r} spread across {parts}"


def test_scaffold_split_is_deterministic_under_row_shuffling():
    df = _frame()
    a = splits.scaffold_split(df, 0.6, 0.2)
    shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    b = splits.scaffold_split(shuffled, 0.6, 0.2)

    for part in ("train", "valid", "test"):
        assert set(getattr(a, part)[cfg.SMILES_COL]) == set(
            getattr(b, part)[cfg.SMILES_COL]
        ), f"{part} partition changed when input row order changed"


def test_assert_no_leakage_passes_on_a_clean_scaffold_split():
    split = splits.scaffold_split(_frame(), 0.6, 0.2)
    splits.assert_no_leakage(split)


def test_assert_no_leakage_catches_a_duplicated_molecule():
    """The leakage guard must actually fire, not just exist."""
    split = splits.scaffold_split(_frame(), 0.6, 0.2)
    poisoned = splits.Split(
        name="scaffold",
        train=split.train,
        valid=split.valid,
        # copy one training molecule into the test set
        test=pd.concat([split.test, split.train.iloc[[0]]], ignore_index=True),
    )
    with pytest.raises(AssertionError, match="shared between"):
        splits.assert_no_leakage(poisoned)


def test_assert_no_leakage_catches_a_shared_scaffold():
    """Distinct molecules sharing a scaffold across partitions is still leakage.

    Built by moving one analogue out of train into test, so no SMILES is
    duplicated: only the scaffold check can catch this.
    """
    split = splits.scaffold_split(_frame(), 0.6, 0.2)
    train = split.train.copy()
    train["_scaffold"] = [splits.murcko_scaffold(s) for s in train[cfg.SMILES_COL]]

    sizes = train["_scaffold"].value_counts()
    shared_scaffold = next(s for s, n in sizes.items() if n >= 2)
    members = train.index[train["_scaffold"] == shared_scaffold].tolist()
    moved, kept = members[0], members[1:]
    assert kept, "need at least one analogue left behind in train"

    poisoned = splits.Split(
        name="scaffold",
        train=train.drop(index=[moved]).drop(columns="_scaffold"),
        valid=split.valid,
        test=pd.concat(
            [split.test, train.loc[[moved]].drop(columns="_scaffold")],
            ignore_index=True,
        ),
    )
    # No molecule is duplicated...
    assert not set(poisoned.train[cfg.SMILES_COL]) & set(poisoned.test[cfg.SMILES_COL])
    # ...but the scaffold is shared, and that must still fail.
    with pytest.raises(AssertionError, match="scaffold"):
        splits.assert_no_leakage(poisoned)


def test_assert_no_leakage_rejects_an_empty_partition():
    df = _frame()
    split = splits.scaffold_split(df, 0.6, 0.2)
    empty = splits.Split(
        name="scaffold",
        train=split.train,
        valid=split.valid.iloc[0:0],
        test=split.test,
    )
    with pytest.raises(AssertionError, match="empty valid"):
        splits.assert_no_leakage(empty)


def test_random_split_is_seeded_and_has_no_molecule_overlap():
    df = _frame()
    a = splits.random_split(df, 0.6, 0.2, seed=5)
    b = splits.random_split(df, 0.6, 0.2, seed=5)
    assert set(a.test[cfg.SMILES_COL]) == set(b.test[cfg.SMILES_COL])
    splits.assert_no_leakage(a, check_scaffolds=False)


def test_make_split_rejects_unknown_kind():
    with pytest.raises(ValueError):
        splits.make_split("kfold", _frame())
