"""Train / validation / test splitting, with leakage checks that actually run.

Two split strategies are provided and both are reported:

* `scaffold` -- deterministic Bemis-Murcko scaffold split. Molecules sharing a
  ring scaffold are kept together, so the test set contains chemotypes the
  model never saw. This is the number that matters for prospective use.
* `random` -- uniform random split at a fixed seed. Reported only as a
  contrast, to quantify how much of the apparent performance comes from
  near-neighbour memorisation.

A split is never trusted without `assert_no_leakage`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config as cfg

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class Split:
    name: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
        }


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES. Acyclic molecules map to ''."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES reached splitting: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )


def scaffold_groups(smiles: pd.Series) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for pos, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(pos)
    return dict(groups)


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = cfg.TRAIN_FRAC,
    valid_frac: float = cfg.VALID_FRAC,
) -> Split:
    """Deterministic scaffold split: biggest scaffold families go to train.

    No RNG is involved, so the split is a pure function of the curated data.
    Scaffolds are ordered by (group size desc, scaffold SMILES) to break ties
    reproducibly regardless of row order.
    """
    groups = scaffold_groups(df[cfg.SMILES_COL])
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n = len(df)
    n_train = int(np.floor(train_frac * n))
    n_valid = int(np.floor(valid_frac * n))

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []
    for _scaffold, members in ordered:
        if len(train_idx) + len(members) <= n_train:
            train_idx.extend(members)
        elif len(valid_idx) + len(members) <= n_valid:
            valid_idx.extend(members)
        else:
            test_idx.extend(members)

    return Split(
        name="scaffold",
        train=df.iloc[sorted(train_idx)].reset_index(drop=True),
        valid=df.iloc[sorted(valid_idx)].reset_index(drop=True),
        test=df.iloc[sorted(test_idx)].reset_index(drop=True),
    )


def random_split(
    df: pd.DataFrame,
    train_frac: float = cfg.TRAIN_FRAC,
    valid_frac: float = cfg.VALID_FRAC,
    seed: int = cfg.RANDOM_SEED,
) -> Split:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n = len(df)
    n_train = int(np.floor(train_frac * n))
    n_valid = int(np.floor(valid_frac * n))
    return Split(
        name="random",
        train=df.iloc[sorted(perm[:n_train])].reset_index(drop=True),
        valid=df.iloc[sorted(perm[n_train : n_train + n_valid])].reset_index(drop=True),
        test=df.iloc[sorted(perm[n_train + n_valid :])].reset_index(drop=True),
    )


def make_split(kind: str, df: pd.DataFrame) -> Split:
    if kind == "scaffold":
        return scaffold_split(df)
    if kind == "random":
        return random_split(df)
    raise ValueError(f"unknown split kind: {kind!r}")


def assert_no_leakage(split: Split, check_scaffolds: bool | None = None) -> None:
    """Fail loudly if any molecule or (for scaffold splits) scaffold is shared."""
    parts = {"train": split.train, "valid": split.valid, "test": split.test}

    for part, frame in parts.items():
        if frame.empty:
            raise AssertionError(f"{split.name} split produced an empty {part} set")

    smiles = {k: set(v[cfg.SMILES_COL]) for k, v in parts.items()}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = smiles[a] & smiles[b]
        if shared:
            raise AssertionError(
                f"{len(shared)} molecule(s) shared between {a} and {b}, e.g. "
                f"{sorted(shared)[:3]}"
            )

    total = sum(len(v) for v in parts.values())
    if total != len(split.train) + len(split.valid) + len(split.test):
        raise AssertionError("split sizes inconsistent")

    if check_scaffolds is None:
        check_scaffolds = split.name == "scaffold"
    if check_scaffolds:
        scaff = {
            k: {murcko_scaffold(s) for s in v[cfg.SMILES_COL]} for k, v in parts.items()
        }
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = scaff[a] & scaff[b]
            if shared:
                raise AssertionError(
                    f"{len(shared)} scaffold(s) shared between {a} and {b} in a "
                    f"scaffold split, e.g. {sorted(shared)[:3]}"
                )


def split_summary(split: Split) -> dict:
    out: dict = {"kind": split.name, "sizes": split.sizes}
    for part, frame in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        y = frame[cfg.LABEL_COL].to_numpy()
        out[part] = {
            "n": int(len(y)),
            "n_scaffolds": len({murcko_scaffold(s) for s in frame[cfg.SMILES_COL]}),
            "label_mean": round(float(y.mean()), 4),
            "label_std": round(float(y.std(ddof=1)), 4),
            "label_min": round(float(y.min()), 4),
            "label_max": round(float(y.max()), 4),
        }
    return out
