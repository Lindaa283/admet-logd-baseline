"""A reproducible baseline for predicting logD7.4 from structure.

Public, permissively licensed data; scaffold-based evaluation; a model ladder
that starts at a constant and has to beat published cLogP to count as a
result.  See README.md for what the numbers mean and LIMITATIONS.md for what
they do not.
"""

import os

# --- thread caps ------------------------------------------------------------
# These must be set *before* NumPy (and therefore OpenBLAS) is imported, which
# is why they live here rather than in admet.models: the BLAS backend reads
# them once, at load time.  On a host with many logical cores OpenBLAS
# allocates per-thread scratch buffers up front and aborts with
# "Memory allocation still failed after 10 retries" when the system commit
# charge is already high.  Four threads is ample for a few thousand compounds.
# Override with ADMET_N_THREADS if you are running somewhere roomier.
N_THREADS = os.environ.get("ADMET_N_THREADS", "4")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, N_THREADS)

__version__ = "0.1.0"

__all__ = [
    "N_THREADS",
    "chem",
    "download",
    "features",
    "leakage",
    "metrics",
    "models",
    "pipeline",
    "plots",
    "qc",
    "scaffold",
]
