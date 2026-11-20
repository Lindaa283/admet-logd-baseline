"""How far is the test set from the training set, really?

A scaffold split guarantees no shared Bemis-Murcko framework, but that is a
structural guarantee, not a distance guarantee: two different frameworks can
still be very similar molecules. So the split's difficulty is measured rather
than assumed.

For every test molecule we compute the Tanimoto similarity (ECFP4, 2048 bits)
to its nearest training molecule. Two things come out of it:

* the distribution of nearest-neighbour similarity, which says how much
  extrapolation the test set actually demands;
* error stratified by that similarity, which shows where a model stops
  working. A model whose RMSE is flat across similarity bands generalises;
  one whose RMSE collapses in the low-similarity band was interpolating.

This is the number that tells you whether a reported metric would survive
contact with a genuinely new chemotype.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from . import config as cfg
from . import metrics as metrics_mod

RDLogger.DisableLog("rdApp.*")

_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=cfg.MORGAN_RADIUS, fpSize=cfg.MORGAN_NBITS
)

# Bands chosen before looking at any result. 0.7 is a conventional "close
# analogue" threshold for ECFP4 Tanimoto; below 0.3 is genuinely novel chemistry.
SIMILARITY_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 1.0001),
)


def _fingerprints(smiles: list[str]) -> list:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES in similarity audit: {smi!r}")
        fps.append(_GEN.GetFingerprint(mol))
    return fps


def nearest_train_similarity(
    query_smiles: list[str], train_smiles: list[str]
) -> np.ndarray:
    """Max ECFP4 Tanimoto from each query molecule to any training molecule."""
    if not train_smiles:
        raise ValueError("training set is empty")
    train_fps = _fingerprints(train_smiles)
    out = np.empty(len(query_smiles), dtype=float)
    for i, fp in enumerate(_fingerprints(query_smiles)):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        out[i] = max(sims)
    return out


def similarity_summary(sims: np.ndarray) -> dict:
    return {
        "n": int(sims.size),
        "mean": round(float(sims.mean()), 4),
        "median": round(float(np.median(sims)), 4),
        "min": round(float(sims.min()), 4),
        "max": round(float(sims.max()), 4),
        "quantiles": {
            q: round(float(np.quantile(sims, float(q))), 4)
            for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "fraction_above_0.7_close_analogue": round(float((sims >= 0.7).mean()), 4),
        "fraction_below_0.3_novel_chemotype": round(float((sims < 0.3).mean()), 4),
    }


def error_by_similarity_band(
    y_true: np.ndarray, y_pred: np.ndarray, sims: np.ndarray, min_band_n: int = 20
) -> list[dict]:
    """RMSE and rank correlation within each nearest-neighbour similarity band.

    Bands with fewer than `min_band_n` molecules report `n` but suppress the
    metrics, because a correlation on a handful of points is noise.
    """
    rows: list[dict] = []
    for lo, hi in SIMILARITY_BANDS:
        mask = (sims >= lo) & (sims < hi)
        n = int(mask.sum())
        row: dict = {
            "band": f"[{lo:.1f}, {min(hi, 1.0):.1f})",
            "n": n,
        }
        if n >= min_band_n:
            row["rmse"] = round(metrics_mod.rmse(y_true[mask], y_pred[mask]), 4)
            row["mae"] = round(metrics_mod.mae(y_true[mask], y_pred[mask]), 4)
            k = metrics_mod.kendall(y_true[mask], y_pred[mask])
            row["kendall"] = None if not np.isfinite(k) else round(k, 4)
        else:
            row["note"] = f"fewer than {min_band_n} molecules; metrics suppressed"
        rows.append(row)
    return rows
