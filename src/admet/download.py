"""Fetch the public lipophilicity (logD7.4) dataset and record its provenance.

The dataset is the MoleculeNet "Lipophilicity" file: 4,200 experimental
octanol/water distribution coefficients at pH 7.4 (logD7.4) curated by
AstraZeneca and deposited in ChEMBL.  See DATA_SOURCES.md for licence terms.

Nothing here is run at training time: the raw file is downloaded once into
``data/raw/`` and every later step reads from disk, so the pipeline is
reproducible offline and the recorded SHA-256 pins exactly which bytes were
used.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# A plain, stable, publicly documented HTTP source.  No account, no API key,
# no cost.  Kept as data rather than hard-coded in the fetch call so that
# DATA_SOURCES.md and the code cannot drift apart.
LIPOPHILICITY = {
    "name": "lipophilicity_az_logd74",
    "url": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "filename": "Lipophilicity.csv",
    "smiles_column": "smiles",
    "label_column": "exp",
    "id_column": "CMPD_CHEMBLID",
    "label_name": "logD7.4",
    "label_units": "log10 ratio (dimensionless)",
    "plausible_label_range": (-3.0, 7.0),
    "expected_rows": 4200,
    # Measured on download 2026-11-20 (282,606 bytes); see
    # data/raw/provenance.json for the record written at fetch time.
    "expected_sha256": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
    "licence": "CC BY-SA 3.0 Unported (ChEMBL); redistributed via MoleculeNet/DeepChem",
    "attribution": (
        "Experimental logD7.4 curated from ChEMBL (AstraZeneca deposited data "
        "set), distributed as the MoleculeNet 'Lipophilicity' benchmark. "
        "ChEMBL data is released under CC BY-SA 3.0 Unported "
        "(https://www.ebi.ac.uk/chembl/). See DATA_SOURCES.md for the exact "
        "licence wording and for which identifiers were verified."
    ),
}


@dataclass(frozen=True)
class Provenance:
    """What was downloaded, from where, when, and with which bytes."""

    name: str
    url: str
    filename: str
    sha256: str
    n_bytes: int
    downloaded_utc: str
    licence: str
    attribution: str
    sha256_matches_expected: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks so large files stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    raw_dir: Path,
    spec: dict | None = None,
    *,
    force: bool = False,
    timeout: int = 120,
) -> tuple[Path, Provenance]:
    """Download the dataset into ``raw_dir`` unless it is already there.

    Returns the path to the CSV and its provenance record.  Re-running is
    cheap and does not hit the network: an existing file with the expected
    SHA-256 is accepted as-is.
    """
    spec = spec or LIPOPHILICITY
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / spec["filename"]

    if target.exists() and not force:
        existing = sha256_of(target)
        if existing == spec["expected_sha256"]:
            return target, _provenance(spec, target, existing, downloaded=False)
        # A file is there but the bytes differ from what we pinned.  Do not
        # silently train on unknown data; say so and re-download.
        print(
            f"[download] {target.name} present but SHA-256 {existing[:12]}… "
            f"!= expected {spec['expected_sha256'][:12]}…; re-downloading."
        )

    print(f"[download] GET {spec['url']}")
    request = urllib.request.Request(
        spec["url"],
        headers={"User-Agent": "admet-logd-baseline/0.1 (research; python-urllib)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    target.write_bytes(payload)

    digest = sha256_of(target)
    record = _provenance(spec, target, digest, downloaded=True)
    (raw_dir / "provenance.json").write_text(record.to_json(), encoding="utf-8")
    if not record.sha256_matches_expected:
        print(
            "[download] WARNING: SHA-256 differs from the pinned value. The "
            "upstream file may have been revised. Results are still "
            "reproducible from the recorded hash, but the number in "
            "download.py should be reviewed before the run is published."
        )
    return target, record


def _provenance(spec: dict, path: Path, digest: str, *, downloaded: bool) -> Provenance:
    return Provenance(
        name=spec["name"],
        url=spec["url"],
        filename=spec["filename"],
        sha256=digest,
        n_bytes=path.stat().st_size,
        downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        licence=spec["licence"],
        attribution=spec["attribution"],
        sha256_matches_expected=digest == spec["expected_sha256"],
    )


if __name__ == "__main__":  # pragma: no cover - manual use
    here = Path(__file__).resolve().parents[2]
    csv_path, prov = fetch(here / "data" / "raw")
    print(csv_path)
    print(prov.to_json())
