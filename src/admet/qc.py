"""Data quality control: load the raw file and decide what is fit to model.

Everything this module finds is written to ``results/tables/qc_report.json``
and summarised in RESULTS.md, including the problems we chose to live with.
The checks are the ones that change a reported number if you skip them:

* missing structures or labels,
* structures RDKit cannot parse,
* labels outside a physically plausible range, or in the wrong unit,
* the same compound measured more than once (which both inflates the apparent
  dataset size and, if left in, can place one compound on both sides of a
  split),
* how far apart the repeat measurements of one compound are, which puts a
  floor under the error any model can achieve on this data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import standardise


@dataclass
class QCReport:
    """What the raw file contained and what survived cleaning."""

    source_file: str
    source_sha256: str
    n_raw_rows: int
    columns: list[str]
    label_name: str
    label_units: str

    n_missing_smiles: int = 0
    n_missing_label: int = 0
    n_unparseable: int = 0
    unparseable_examples: list[str] = field(default_factory=list)

    label_min: float = 0.0
    label_max: float = 0.0
    label_mean: float = 0.0
    label_median: float = 0.0
    label_std: float = 0.0
    n_label_out_of_range: int = 0
    plausible_range: tuple[float, float] = (-3.0, 7.0)
    unit_check: str = ""

    n_multi_fragment: int = 0
    n_unique_canonical_smiles: int = 0
    n_unique_inchikey: int = 0
    n_unique_connectivity: int = 0
    n_duplicate_compounds: int = 0
    n_rows_in_duplicate_groups: int = 0
    duplicate_label_spread_mean: float = 0.0
    duplicate_label_spread_max: float = 0.0
    duplicate_label_spread_median: float = 0.0
    implied_experimental_mae_floor: float = 0.0

    n_clean_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["plausible_range"] = list(self.plausible_range)
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_and_qc(
    csv_path: Path,
    *,
    source_sha256: str,
    smiles_column: str = "smiles",
    label_column: str = "exp",
    id_column: str | None = "CMPD_CHEMBLID",
    label_name: str = "logD7.4",
    label_units: str = "log10 ratio (dimensionless)",
    plausible_range: tuple[float, float] = (-3.0, 7.0),
) -> tuple[pd.DataFrame, QCReport]:
    """Return a cleaned, de-duplicated table plus the report of what happened.

    The returned frame has one row per distinct compound, with columns
    ``compound_id, canonical_smiles, inchikey, connectivity_key, label,
    n_measurements, label_spread, n_fragments``.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)

    for column in (smiles_column, label_column):
        if column not in raw.columns:
            raise KeyError(f"column {column!r} not in {list(raw.columns)}")

    report = QCReport(
        source_file=csv_path.name,
        source_sha256=source_sha256,
        n_raw_rows=len(raw),
        columns=list(raw.columns),
        label_name=label_name,
        label_units=label_units,
        plausible_range=plausible_range,
    )

    # --- missing values -----------------------------------------------------
    report.n_missing_smiles = int(raw[smiles_column].isna().sum())
    report.n_missing_label = int(raw[label_column].isna().sum())
    work = raw.dropna(subset=[smiles_column, label_column]).copy()
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    n_non_numeric = int(work[label_column].isna().sum())
    if n_non_numeric:
        report.notes.append(
            f"{n_non_numeric} label value(s) were not numeric and were dropped"
        )
        work = work.dropna(subset=[label_column])

    # --- label range and unit sanity ---------------------------------------
    labels = work[label_column].to_numpy(dtype=float)
    report.label_min = float(labels.min())
    report.label_max = float(labels.max())
    report.label_mean = float(labels.mean())
    report.label_median = float(np.median(labels))
    report.label_std = float(labels.std(ddof=1))
    low, high = plausible_range
    out_of_range = (labels < low) | (labels > high)
    report.n_label_out_of_range = int(out_of_range.sum())
    report.unit_check = _unit_verdict(labels, plausible_range)
    if report.n_label_out_of_range:
        report.notes.append(
            f"{report.n_label_out_of_range} label(s) outside {plausible_range}; "
            "kept but flagged — inspect before trusting them"
        )

    # --- structures ---------------------------------------------------------
    records = []
    for position, row in enumerate(work.itertuples(index=False)):
        smiles = getattr(row, smiles_column)
        std = standardise(smiles)
        if std is None:
            report.n_unparseable += 1
            if len(report.unparseable_examples) < 10:
                report.unparseable_examples.append(str(smiles)[:120])
            continue
        compound_id = (
            getattr(row, id_column) if id_column and hasattr(row, id_column)
            else f"row{position}"
        )
        records.append(
            {
                "compound_id": compound_id,
                "input_smiles": std.input_smiles,
                "canonical_smiles": std.canonical_smiles,
                "inchikey": std.inchikey,
                "connectivity_key": std.connectivity_key,
                "n_fragments": std.n_fragments,
                "heavy_atoms": std.heavy_atoms,
                "label": float(getattr(row, label_column)),
            }
        )

    parsed = pd.DataFrame.from_records(records)
    if parsed.empty:
        raise RuntimeError("no rows survived parsing; check the input file")

    report.n_multi_fragment = int((parsed["n_fragments"] > 1).sum())
    report.n_unique_canonical_smiles = int(parsed["canonical_smiles"].nunique())
    report.n_unique_inchikey = int(parsed["inchikey"].nunique())
    report.n_unique_connectivity = int(parsed["connectivity_key"].nunique())

    # --- duplicate compounds and measurement spread ------------------------
    # Group at the connectivity layer: stereoisomers and different protonation
    # states of one compound share a logD, so treating them as distinct would
    # let the "same" molecule sit in train and test.
    grouped = parsed.groupby("connectivity_key", sort=False)
    sizes = grouped.size()
    duplicate_keys = sizes[sizes > 1].index
    report.n_duplicate_compounds = int(len(duplicate_keys))
    report.n_rows_in_duplicate_groups = int(sizes[sizes > 1].sum())

    if len(duplicate_keys) > 0:
        spreads = (
            parsed[parsed["connectivity_key"].isin(duplicate_keys)]
            .groupby("connectivity_key")["label"]
            .agg(lambda series: float(series.max() - series.min()))
        )
        report.duplicate_label_spread_mean = float(spreads.mean())
        report.duplicate_label_spread_max = float(spreads.max())
        report.duplicate_label_spread_median = float(spreads.median())
        # If one compound measured twice differs by d, then a perfect model
        # scored against either single measurement still carries ~d/2 of
        # irreducible error.  Half the mean spread is a rough, honest floor.
        report.implied_experimental_mae_floor = float(spreads.mean() / 2.0)

    clean = (
        parsed.groupby("connectivity_key", sort=False)
        .agg(
            compound_id=("compound_id", "first"),
            canonical_smiles=("canonical_smiles", "first"),
            inchikey=("inchikey", "first"),
            n_fragments=("n_fragments", "first"),
            heavy_atoms=("heavy_atoms", "first"),
            label=("label", "median"),
            n_measurements=("label", "size"),
            label_spread=("label", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
    )
    report.n_clean_rows = int(len(clean))
    report.notes.append(
        "repeat measurements of one compound collapsed to their median; "
        "the spread is kept in the `label_spread` column"
    )

    return clean, report


def _unit_verdict(labels: np.ndarray, plausible: tuple[float, float]) -> str:
    """A blunt check that the column is a log-scale distribution coefficient.

    Catches the common mistake of modelling a raw partition ratio, or a column
    that is really in different units, without pretending to be clever.
    """
    low, high = plausible
    fraction_in_range = float(((labels >= low) & (labels <= high)).mean())
    spread = float(labels.max() - labels.min())
    if fraction_in_range < 0.95:
        return (
            f"SUSPECT: only {fraction_in_range:.1%} of values fall in {plausible}; "
            "the column may not be a log-scale value"
        )
    if spread > 20:
        return (
            f"SUSPECT: range spans {spread:.1f} units, which is too wide for "
            "a log-scale distribution coefficient"
        )
    return (
        f"consistent with a log10 distribution coefficient: "
        f"{fraction_in_range:.1%} of values within {plausible}, range {spread:.2f}"
    )
