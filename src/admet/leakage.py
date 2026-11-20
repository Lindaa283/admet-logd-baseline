"""Leakage audit: assert that a split is as clean as it claims to be.

Writing a scaffold split is easy; proving it worked is the part that keeps the
result honest.  Every split used in the report is passed through
:func:`audit_split`, the result is written to
``results/tables/leakage_audit.json``, and the pipeline refuses to continue if
a check that must hold is violated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .features import nearest_train_similarity


@dataclass
class LeakageAudit:
    """The outcome of auditing one train/val/test partition."""

    split_name: str
    n_train: int
    n_val: int
    n_test: int

    # Must be zero for any split we report.
    n_overlapping_rows: int = 0
    n_duplicate_identity_across_splits: int = 0

    # Must be zero for a scaffold split; expected to be large for a random one.
    n_shared_scaffolds_train_test: int = 0
    n_test_rows_with_train_scaffold: int = 0

    # Structural proximity, not an error in itself but the honest caveat.
    n_test_with_identical_fingerprint_in_train: int = 0
    max_nearest_similarity: float = 0.0
    mean_nearest_similarity: float = 0.0
    median_nearest_similarity: float = 0.0
    frac_nearest_similarity_above_0_9: float = 0.0
    frac_nearest_similarity_above_0_7: float = 0.0

    passed: bool = True
    failures: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def audit_split(
    split_name: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    *,
    scaffolds: list[str],
    identity_keys: list[str],
    fingerprints: list | None = None,
    require_scaffold_disjoint: bool,
) -> LeakageAudit:
    """Check one partition and record what was found."""
    audit = LeakageAudit(
        split_name=split_name,
        n_train=int(len(train)),
        n_val=int(len(val)),
        n_test=int(len(test)),
        failures=[],
    )

    # 1. The partition must actually be a partition.
    all_rows = np.concatenate([train, val, test])
    audit.n_overlapping_rows = int(len(all_rows) - len(np.unique(all_rows)))

    # 2. No compound identity may appear in more than one split.  After the QC
    #    step there is one row per compound, so this should be zero by
    #    construction; it is checked anyway because that is exactly the kind of
    #    assumption that quietly stops being true.
    train_identities = {identity_keys[row] for row in train}
    val_identities = {identity_keys[row] for row in val}
    test_identities = {identity_keys[row] for row in test}
    audit.n_duplicate_identity_across_splits = int(
        len(test_identities & train_identities)
        + len(test_identities & val_identities)
        + len(train_identities & val_identities)
    )

    # 3. Scaffold disjointness.
    train_scaffolds = {scaffolds[row] for row in train} | {
        scaffolds[row] for row in val
    }
    test_scaffolds = {scaffolds[row] for row in test}
    shared = train_scaffolds & test_scaffolds
    audit.n_shared_scaffolds_train_test = int(len(shared))
    audit.n_test_rows_with_train_scaffold = int(
        sum(1 for row in test if scaffolds[row] in train_scaffolds)
    )

    # 4. Structural proximity of test to train.
    if fingerprints is not None and len(test) and len(train):
        test_fps = [fingerprints[row] for row in test]
        train_fps = [fingerprints[row] for row in train]
        similarity = nearest_train_similarity(test_fps, train_fps)
        audit.n_test_with_identical_fingerprint_in_train = int(
            np.sum(similarity >= 0.999)
        )
        audit.max_nearest_similarity = float(similarity.max())
        audit.mean_nearest_similarity = float(similarity.mean())
        audit.median_nearest_similarity = float(np.median(similarity))
        audit.frac_nearest_similarity_above_0_9 = float(np.mean(similarity > 0.9))
        audit.frac_nearest_similarity_above_0_7 = float(np.mean(similarity > 0.7))

    # --- verdict ------------------------------------------------------------
    if audit.n_overlapping_rows:
        audit.failures.append(
            f"{audit.n_overlapping_rows} row(s) appear in more than one split"
        )
    if audit.n_duplicate_identity_across_splits:
        audit.failures.append(
            f"{audit.n_duplicate_identity_across_splits} compound identity "
            "collision(s) across splits"
        )
    if require_scaffold_disjoint and audit.n_shared_scaffolds_train_test:
        audit.failures.append(
            f"{audit.n_shared_scaffolds_train_test} scaffold(s) shared between "
            "train/val and test, but this split must be scaffold-disjoint"
        )
    audit.passed = not audit.failures
    return audit


def write_audits(audits: list[LeakageAudit], path) -> None:
    """Write all audits to one JSON file."""
    payload = [audit.to_dict() for audit in audits]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_all_passed(audits: list[LeakageAudit]) -> None:
    """Stop the run if any audit that must pass did not."""
    broken = [audit for audit in audits if not audit.passed]
    if broken:
        lines = [
            f"  {audit.split_name}: {'; '.join(audit.failures or [])}"
            for audit in broken
        ]
        raise RuntimeError("leakage audit failed:\n" + "\n".join(lines))
