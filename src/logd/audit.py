"""Standalone leakage audit. Reads structures only -- never labels, never models.

This exists because two leakage channels cannot be settled by reading code:

* **L3** -- how far the held-out molecules actually sit from training data.
  A scaffold split guarantees distinct Bemis-Murcko frameworks, not distant
  molecules, so the distance has to be measured.

* **L5** -- 2D descriptors and a Morgan fingerprint built without chirality
  are *bit-identical* for a pair of enantiomers. So two molecules with
  different canonical SMILES, which `assert_no_leakage` therefore accepts,
  can still be indistinguishable to every model in this project. This counts
  them directly.

Deliberately NOT a model evaluation. It loads no estimator, makes no
prediction and reads no label, so running it does not consume the project's
single permitted test-set evaluation (see `logd.evaluate`). It is safe to run
as often as you like.

Run:
    python -m logd.audit
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import config as cfg
from . import data as data_mod
from . import featurize, provenance, similarity, splits


def _identical_feature_twins(
    query_smiles: list[str], train_smiles: list[str]
) -> dict:
    """Count held-out molecules a model literally cannot tell from a training one.

    Two distinct structures that produce the same ECFP4 bit vector are, to
    every fingerprint model here, the same input. If their labels differ, the
    model is being asked an unanswerable question; if they are near-identical,
    the held-out molecule was effectively in the training set.
    """
    q = featurize.ecfp4(query_smiles)
    t = featurize.ecfp4(train_smiles)
    # Hash rows so the comparison is O(n) rather than O(n*m) on 2048 bits.
    train_hashes = {hash(row.tobytes()) for row in t.astype(np.uint8)}
    hits = [
        smi
        for smi, row in zip(query_smiles, q.astype(np.uint8))
        if hash(row.tobytes()) in train_hashes
    ]
    return {"n_with_identical_ecfp4_in_train": len(hits), "examples": hits[:5]}


def audit_split(kind: str, df) -> dict:
    split = splits.make_split(kind, df)
    splits.assert_no_leakage(split, check_scaffolds=(kind == "scaffold"))

    train_smiles = split.train[cfg.SMILES_COL].tolist()
    out: dict = {"sizes": split.sizes}

    for part_name, frame in (("valid", split.valid), ("test", split.test)):
        part_smiles = frame[cfg.SMILES_COL].tolist()
        sims = similarity.nearest_train_similarity(part_smiles, train_smiles)
        out[part_name] = {
            "nearest_train_tanimoto": similarity.similarity_summary(sims),
            "identical_feature_twins": _identical_feature_twins(
                part_smiles, train_smiles
            ),
        }
    return out


def run() -> dict:
    cfg.ensure_dirs()
    df = data_mod.load_curated()

    payload = {
        "what_this_is": (
            "Structure-only leakage audit. No labels read, no models loaded, "
            "no predictions made. Does not count as a test-set evaluation."
        ),
        "n_curated": int(len(df)),
        "splits": {},
        "provenance": provenance.snapshot(),
    }
    for kind in ("scaffold", "random"):
        print(f"[audit] {kind} split")
        payload["splits"][kind] = audit_split(kind, df)

    provenance.write_json(cfg.RESULTS / "leakage_audit_logd.json", payload)
    return payload


def _report(payload: dict) -> None:
    for kind, res in payload["splits"].items():
        print(f"\n=== {kind} split  {res['sizes']} ===")
        for part in ("valid", "test"):
            s = res[part]["nearest_train_tanimoto"]
            tw = res[part]["identical_feature_twins"]
            print(
                f"  {part:<6} nearest-train Tanimoto: "
                f"median {s['median']:.3f}, mean {s['mean']:.3f}, max {s['max']:.3f}"
            )
            print(
                f"         >=0.7 (close analogue) {s['fraction_above_0.7_close_analogue']:.1%}"
                f" | <0.3 (novel) {s['fraction_below_0.3_novel_chemotype']:.1%}"
                f" | identical ECFP4 twin in train: "
                f"{tw['n_with_identical_ecfp4_in_train']}"
            )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    payload = run()
    _report(payload)
    print(f"\n[audit] wrote results/leakage_audit_logd.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
