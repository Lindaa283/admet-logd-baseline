# Linux execution environment for the logD7.4 baseline.
#
# Why a container rather than a bare virtualenv:
# the author's Windows 11 machine runs Smart App Control in Enforced mode,
# which blocks RDKit's unsigned native DLLs (CodeIntegrity events 3033/3077).
# Disabling Smart App Control is irreversible without reinstalling Windows, so
# the project runs in a Linux container instead. This also makes the pinned
# environment reproducible for anyone else. See README "Environment".
FROM python:3.13-slim

# libgomp1  -> OpenMP runtime used by scikit-learn
# libxrender1, libxext6 -> shared objects RDKit links against
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/work/src \
    MPLBACKEND=Agg

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
CMD ["python", "-c", "import rdkit, sklearn; print('rdkit', rdkit.__version__, '| sklearn', sklearn.__version__)"]
