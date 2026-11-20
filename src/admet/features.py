"""Feature matrices: Morgan (ECFP-like) fingerprints and descriptors.

All featurisation is a pure function of the molecule.  Nothing here looks at
the label or at which split a row belongs to, so featurising the whole table
once cannot leak information between splits.  The only fitted quantities in
the pipeline are the model parameters and the feature scaler, and those are fit
on training rows only (see :mod:`admet.models`).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .chem import DESCRIPTOR_NAMES, descriptors

# Radius 2 over a 2048-bit space is the long-standing default for this kind of
# property model (ECFP4 in the Pipeline Pilot naming).  Counts rather than bits
# because how *many* times a substructure occurs matters for a bulk property
# like lipophilicity: four chlorines are not one chlorine.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_COUNT_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, countSimulation=False
)
_BIT_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def morgan_counts(mols: list[Chem.Mol]) -> np.ndarray:
    """Folded Morgan count fingerprints, shape ``(n_mols, MORGAN_BITS)``."""
    matrix = np.zeros((len(mols), MORGAN_BITS), dtype=np.float32)
    for row, mol in enumerate(mols):
        matrix[row] = _COUNT_GEN.GetCountFingerprintAsNumPy(mol)
    return matrix


def morgan_bitvects(mols: list[Chem.Mol]) -> list:
    """Morgan bit vectors, for Tanimoto similarity in the domain analysis."""
    return [_BIT_GEN.GetFingerprint(mol) for mol in mols]


def descriptor_matrix(
    mols: list[Chem.Mol], n_fragments: list[int] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The interpretable descriptor block, shape ``(n_mols, n_descriptors)``."""
    fragments = n_fragments or [1] * len(mols)
    matrix = np.zeros((len(mols), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    for row, (mol, n_frag) in enumerate(zip(mols, fragments)):
        values = descriptors(mol, n_fragments=n_frag)
        matrix[row] = [values[name] for name in DESCRIPTOR_NAMES]
    if not np.isfinite(matrix).all():
        # Descriptors such as BertzCT can overflow on pathological inputs.
        # Replace rather than crash, and say so loudly.
        n_bad = int((~np.isfinite(matrix)).sum())
        print(f"[features] WARNING: {n_bad} non-finite descriptor value(s) set to 0")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, DESCRIPTOR_NAMES


def nearest_train_similarity(
    test_fps: list, train_fps: list, *, block: int = 256
) -> np.ndarray:
    """For each test molecule, its highest Tanimoto similarity to any train molecule.

    Used for two things: an explicit leakage check (a value of 1.0 means an
    identical fingerprint is present in train) and the applicability-domain
    figure, which shows how prediction error grows as test compounds get
    further from the training set.
    """
    best = np.zeros(len(test_fps), dtype=np.float64)
    for start in range(0, len(test_fps), block):
        for offset, fingerprint in enumerate(test_fps[start : start + block]):
            sims = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
            best[start + offset] = max(sims) if sims else 0.0
    return best
