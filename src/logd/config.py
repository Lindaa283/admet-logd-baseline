"""Central configuration: paths, seeds, dataset provenance.

Every number that affects a result lives here or in a CLI flag, never inline
in a script. This is what makes the runs reproducible.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = RESULTS / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 20261120
N_CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Dataset provenance
# ---------------------------------------------------------------------------
# MoleculeNet "Lipophilicity": experimental octanol/water distribution
# coefficient at pH 7.4 (logD7.4) for 4200 compounds, curated by AstraZeneca
# and deposited in ChEMBL (ChEMBL3301363).
#   Wu et al., "MoleculeNet: A Benchmark for Molecular Machine Learning",
#   Chem. Sci. 2018, 9, 513-530.
DATASET_NAME = "MoleculeNet Lipophilicity (AstraZeneca / ChEMBL3301363)"
DATASET_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
)
# sha256 of the raw download, verified on every load. Recorded on first fetch
# by `python -m logd.data --download`; a mismatch means the upstream file moved
# and every result in this repo must be regenerated.
DATASET_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"

RAW_CSV = DATA_RAW / "Lipophilicity.csv"
CURATED_PARQUET = DATA_PROCESSED / "logd74_curated.csv"
QC_REPORT_JSON = RESULTS / "data_qc.json"

RAW_SMILES_COL = "smiles"
RAW_LABEL_COL = "exp"
RAW_ID_COL = "CMPD_CHEMBLID"

SMILES_COL = "smiles"
LABEL_COL = "logd74"
ID_COL = "chembl_id"

TARGET_DESCRIPTION = (
    "logD7.4: base-10 logarithm of the octanol/water distribution coefficient "
    "measured at pH 7.4. Higher means more lipophilic. Experimental, "
    "single assay family, single endpoint."
)
LABEL_UNITS = "log10 ratio (dimensionless)"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.8
VALID_FRAC = 0.1
TEST_FRAC = 0.1

# Two measurements of the same canonical structure are merged if they agree
# within this tolerance; beyond it the structure is dropped as irreconcilable.
DUPLICATE_LABEL_TOLERANCE = 0.5

# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Ipc is excluded: it is defined as an exponential of the information content
# of the adjacency matrix and overflows to +inf for medium-sized molecules,
# silently poisoning any model that standardises features.
EXCLUDED_DESCRIPTORS = ("Ipc",)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, RESULTS, ARTIFACTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
