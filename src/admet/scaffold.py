"""Bemis-Murcko scaffolds and scaffold-based splitting.

Why not a random split
----------------------
Medicinal-chemistry datasets are not samples of independent molecules.  They
are built from *series*: a project makes dozens of close analogues around one
core, measures them all, and deposits them together.  A random split therefore
puts near-identical analogues on both sides of the split, and the model is
scored on compounds whose answer it has effectively already seen.  The number
that comes out is a measure of interpolation within known series.

The question a project actually asks is different: *given a new chemotype, is
the prediction useful?*  A scaffold split answers something much closer to
that, by making sure that no Bemis-Murcko framework appears in both train and
test.  It is harder, the numbers are worse, and it is the honest one to quote.

We report both, and the gap between them is itself a result (see RESULTS.md).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(mol: Chem.Mol, *, generic: bool = False) -> str:
    """The Bemis-Murcko scaffold of a molecule, as canonical SMILES.

    The scaffold is the ring systems plus the linkers that connect them, with
    all terminal side chains removed.  Chirality is ignored: two enantiomers
    belong to the same series and must not be split apart.

    With ``generic=True`` every atom becomes a carbon and every bond an
    unspecified bond, so that e.g. a pyridine and a benzene core collapse
    together.  That is a *stricter* grouping; we use the normal form by
    default and expose the generic one for a sensitivity check.

    Molecules with no ring at all have an empty scaffold.  They are a real
    group in this dataset (short acyclic compounds), and keeping them together
    under the empty-string key is correct: they share "no framework".
    """
    if mol is None:
        raise ValueError("mol is None")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def scaffold_groups(scaffolds: list[str]) -> dict[str, list[int]]:
    """Map each scaffold to the row indices that carry it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(index)
    return dict(groups)


def scaffold_split(
    scaffolds: list[str],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices into train/val/test with disjoint scaffolds.

    Whole scaffold groups are assigned to one split, never divided, so the
    guarantee "no framework is in two splits" holds by construction.

    ``seed=None`` gives the deterministic, largest-group-first assignment used
    throughout the MoleculeNet literature: the common cores end up in train and
    the singletons in test, which is the most demanding arrangement.  Passing
    an integer seed shuffles the group order instead, which keeps the
    scaffold-disjointness but samples a different (generally easier) partition,
    letting us put error bars on the estimate instead of quoting one number
    from one arbitrary partition.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if any(fraction < 0 for fraction in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions}")

    groups = scaffold_groups(scaffolds)
    n_total = len(scaffolds)

    keys = list(groups)
    if seed is None:
        # Largest first; ties broken by the scaffold string so the result does
        # not depend on dict ordering.
        keys.sort(key=lambda key: (-len(groups[key]), key))
    else:
        keys.sort()  # deterministic starting order before shuffling
        np.random.default_rng(seed).shuffle(keys)

    n_train_target = int(round(fractions[0] * n_total))
    n_val_target = int(round(fractions[1] * n_total))

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for key in keys:
        members = groups[key]
        if len(train) + len(members) <= n_train_target:
            train.extend(members)
        elif len(val) + len(members) <= n_val_target:
            val.extend(members)
        else:
            test.extend(members)

    return (
        np.array(sorted(train), dtype=int),
        np.array(sorted(val), dtype=int),
        np.array(sorted(test), dtype=int),
    )


def random_split(
    n_rows: int,
    *,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plain random split, for the optimism comparison only.

    This is *not* the headline evaluation.  It exists so the report can state
    how much performance a random split appears to add over a scaffold split
    on the same data and the same models.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    indices = np.arange(n_rows)
    np.random.default_rng(seed).shuffle(indices)
    n_train = int(round(fractions[0] * n_rows))
    n_val = int(round(fractions[1] * n_rows))
    return (
        np.sort(indices[:n_train]),
        np.sort(indices[n_train : n_train + n_val]),
        np.sort(indices[n_train + n_val :]),
    )
