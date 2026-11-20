"""Molecular featurisation: ECFP4 bits, RDKit 2D descriptors, Crippen clogP.

All featurisers are pure functions of a SMILES list, so they cannot leak
information between splits. Anything that must be *fitted* (imputation,
variance filtering, scaling) lives inside the sklearn pipelines in `models.py`
and is therefore fitted on training folds only.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

from . import config as cfg

RDLogger.DisableLog("rdApp.*")

FEATURE_BLOCKS = ("clogp", "ecfp4", "desc", "desc_ecfp4")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=cfg.MORGAN_RADIUS, fpSize=cfg.MORGAN_NBITS
)


def _mols(smiles: list[str]):
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached featurisation: {smi!r}")
        mols.append(mol)
    return mols


def descriptor_names() -> list[str]:
    """Deterministic, version-pinned RDKit 2D descriptor list."""
    names = [n for n, _ in Descriptors.descList]
    return [n for n in names if n not in cfg.EXCLUDED_DESCRIPTORS]


def ecfp4(smiles: list[str]) -> np.ndarray:
    mols = _mols(smiles)
    out = np.zeros((len(mols), cfg.MORGAN_NBITS), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = _MORGAN.GetFingerprintAsNumPy(mol)
        out[i] = fp
    return out


def descriptors(smiles: list[str]) -> np.ndarray:
    """RDKit 2D descriptors. Failures become NaN for the pipeline's imputer."""
    mols = _mols(smiles)
    names = descriptor_names()
    out = np.full((len(mols), len(names)), np.nan, dtype=np.float64)
    for i, mol in enumerate(mols):
        values = Descriptors.CalcMolDescriptors(mol, missingVal=float("nan"))
        for j, name in enumerate(names):
            v = values.get(name, float("nan"))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            out[i, j] = fv
    # Non-finite values (inf from unstable descriptors) are NaN so the
    # imputer handles them rather than a scaler exploding on them.
    out[~np.isfinite(out)] = np.nan
    return out


def clogp(smiles: list[str]) -> np.ndarray:
    """Crippen calculated logP -- the knowledge-based domain baseline."""
    mols = _mols(smiles)
    return np.asarray([[Crippen.MolLogP(m)] for m in mols], dtype=np.float64)


def featurize(smiles: list[str], block: str) -> np.ndarray:
    if block == "clogp":
        return clogp(smiles)
    if block == "ecfp4":
        return ecfp4(smiles)
    if block == "desc":
        return descriptors(smiles)
    if block == "desc_ecfp4":
        return np.hstack([descriptors(smiles), ecfp4(smiles)])
    raise ValueError(f"unknown feature block: {block!r}")


def feature_names(block: str) -> list[str]:
    if block == "clogp":
        return ["Crippen_MolLogP"]
    if block == "ecfp4":
        return [f"ecfp4_{i}" for i in range(cfg.MORGAN_NBITS)]
    if block == "desc":
        return descriptor_names()
    if block == "desc_ecfp4":
        return descriptor_names() + [f"ecfp4_{i}" for i in range(cfg.MORGAN_NBITS)]
    raise ValueError(f"unknown feature block: {block!r}")
