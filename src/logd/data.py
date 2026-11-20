"""Download, curate and quality-check the logD7.4 dataset.

Curation decisions are deliberate and recorded, because they change the label
semantics:

* Counterions / solvent fragments are stripped (largest organic fragment
  kept). A measured logD belongs to the parent compound, not to its chloride.
* Charges are NOT neutralised. logD7.4 is explicitly a pH-dependent quantity;
  rewriting the ionisation state of the deposited structure would silently
  change what the label means.
* Structures appearing more than once are merged only if their measurements
  agree within `DUPLICATE_LABEL_TOLERANCE` log units. Irreconcilable
  duplicates are dropped rather than averaged, since averaging two
  contradictory measurements manufactures a label that was never measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from . import config as cfg

RDLogger.DisableLog("rdApp.*")


@dataclass
class CurationReport:
    """Every compound that enters or leaves the dataset is accounted for."""

    raw_rows: int = 0
    unparseable_smiles: int = 0
    multi_fragment_stripped: int = 0
    missing_label: int = 0
    duplicate_structures: int = 0
    duplicates_merged: int = 0
    duplicates_dropped_conflicting: int = 0
    final_rows: int = 0
    label_min: float = 0.0
    label_max: float = 0.0
    label_mean: float = 0.0
    label_std: float = 0.0
    dropped_examples: list[dict] = field(default_factory=list)

    def check_accounting(self) -> None:
        """Assert the compound ledger balances."""
        accounted = (
            self.final_rows
            + self.unparseable_smiles
            + self.missing_label
            + self.duplicates_merged
            + self.duplicates_dropped_conflicting
        )
        if accounted != self.raw_rows:
            raise AssertionError(
                f"curation ledger does not balance: {accounted} accounted for "
                f"but {self.raw_rows} raw rows"
            )


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False) -> str:
    """Fetch the raw CSV and return its sha256."""
    cfg.ensure_dirs()
    if cfg.RAW_CSV.exists() and not force:
        return sha256_file(cfg.RAW_CSV)
    req = urllib.request.Request(
        cfg.DATASET_URL, headers={"User-Agent": "logd-baseline/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    cfg.RAW_CSV.write_bytes(payload)
    return sha256_file(cfg.RAW_CSV)


def verify_raw() -> str:
    """Verify the raw file against the recorded digest."""
    digest = sha256_file(cfg.RAW_CSV)
    if digest != cfg.DATASET_SHA256:
        raise RuntimeError(
            "Raw dataset digest mismatch.\n"
            f"  expected {cfg.DATASET_SHA256}\n"
            f"  actual   {digest}\n"
            "The upstream file changed. Do not trust committed results until "
            "they are regenerated against this new file."
        )
    return digest


_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


def standardise(smiles: str) -> tuple[str | None, bool]:
    """Return (canonical SMILES, whether fragments were stripped)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False
    stripped = False
    if len(Chem.GetMolFrags(mol)) > 1:
        mol = _LARGEST_FRAGMENT.choose(mol)
        stripped = True
        if mol is None:
            return None, True
    try:
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:
        return None, stripped
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None, stripped
    return Chem.MolToSmiles(mol), stripped


def curate(raw: pd.DataFrame) -> tuple[pd.DataFrame, CurationReport]:
    report = CurationReport(raw_rows=len(raw))

    records: list[dict] = []
    for row in raw.itertuples(index=False):
        smiles_in = getattr(row, cfg.RAW_SMILES_COL)
        label = getattr(row, cfg.RAW_LABEL_COL)
        ident = getattr(row, cfg.RAW_ID_COL)

        if label is None or (isinstance(label, float) and not np.isfinite(label)):
            report.missing_label += 1
            continue

        canonical, stripped = standardise(smiles_in)
        if canonical is None:
            report.unparseable_smiles += 1
            if len(report.dropped_examples) < 10:
                report.dropped_examples.append(
                    {"reason": "unparseable", "id": str(ident), "smiles": smiles_in}
                )
            continue
        if stripped:
            report.multi_fragment_stripped += 1

        records.append(
            {
                cfg.ID_COL: str(ident),
                cfg.SMILES_COL: canonical,
                cfg.LABEL_COL: float(label),
            }
        )

    df = pd.DataFrame.from_records(records)

    # --- duplicate structures -------------------------------------------------
    grouped = df.groupby(cfg.SMILES_COL)[cfg.LABEL_COL]
    counts = grouped.size()
    spreads = grouped.max() - grouped.min()
    dup_smiles = counts[counts > 1].index
    report.duplicate_structures = int(len(dup_smiles))

    conflicting = [s for s in dup_smiles if spreads[s] > cfg.DUPLICATE_LABEL_TOLERANCE]
    consistent = [s for s in dup_smiles if spreads[s] <= cfg.DUPLICATE_LABEL_TOLERANCE]

    # rows removed by merging consistent duplicates into one row each
    report.duplicates_merged = int(counts[consistent].sum() - len(consistent))
    report.duplicates_dropped_conflicting = int(counts[conflicting].sum())
    for s in conflicting[:10]:
        report.dropped_examples.append(
            {
                "reason": "conflicting_duplicate",
                "smiles": s,
                "spread_log_units": round(float(spreads[s]), 3),
            }
        )

    df = df[~df[cfg.SMILES_COL].isin(conflicting)]
    df = (
        df.groupby(cfg.SMILES_COL, as_index=False)
        .agg({cfg.LABEL_COL: "mean", cfg.ID_COL: "first"})
        .sort_values(cfg.SMILES_COL, kind="stable")
        .reset_index(drop=True)
    )
    df = df[[cfg.ID_COL, cfg.SMILES_COL, cfg.LABEL_COL]]

    report.final_rows = len(df)
    labels = df[cfg.LABEL_COL].to_numpy()
    report.label_min = float(labels.min())
    report.label_max = float(labels.max())
    report.label_mean = float(labels.mean())
    report.label_std = float(labels.std(ddof=1)) if len(labels) > 1 else 0.0
    report.check_accounting()
    return df, report


def build(force_download: bool = False, verify: bool = True) -> pd.DataFrame:
    cfg.ensure_dirs()
    digest = download_raw(force=force_download)
    if verify:
        verify_raw()
    raw = pd.read_csv(cfg.RAW_CSV)
    missing = {cfg.RAW_SMILES_COL, cfg.RAW_LABEL_COL, cfg.RAW_ID_COL} - set(raw.columns)
    if missing:
        raise RuntimeError(f"raw CSV missing expected columns: {sorted(missing)}")

    df, report = curate(raw)
    df.to_csv(cfg.CURATED_PARQUET, index=False)

    payload = {
        "dataset": cfg.DATASET_NAME,
        "url": cfg.DATASET_URL,
        "raw_sha256": digest,
        "curated_sha256": sha256_file(cfg.CURATED_PARQUET),
        "target": cfg.TARGET_DESCRIPTION,
        "units": cfg.LABEL_UNITS,
        "duplicate_label_tolerance": cfg.DUPLICATE_LABEL_TOLERANCE,
        "curation": asdict(report),
    }
    cfg.QC_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return df


def load_curated() -> pd.DataFrame:
    if not cfg.CURATED_PARQUET.exists():
        raise FileNotFoundError(
            "Curated dataset missing. Run: python -m logd.data --build"
        )
    return pd.read_csv(cfg.CURATED_PARQUET)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the curated logD7.4 dataset.")
    p.add_argument("--download", action="store_true", help="fetch raw CSV and print its sha256")
    p.add_argument("--build", action="store_true", help="curate and write the dataset")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip digest verification (only for recording a new digest)",
    )
    args = p.parse_args(argv)

    if args.download:
        digest = download_raw(force=args.force)
        print(f"raw sha256: {digest}")
        print(f"bytes:      {cfg.RAW_CSV.stat().st_size}")
        return 0

    if args.build:
        df = build(force_download=args.force, verify=not args.no_verify)
        report = json.loads(cfg.QC_REPORT_JSON.read_text(encoding="utf-8"))
        print(json.dumps(report["curation"], indent=2))
        print(f"\ncurated rows: {len(df)}  ->  {cfg.CURATED_PARQUET}")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
