"""Fit every model. Hyperparameters chosen by cross-validation on TRAIN only.

Protocol, enforced by the code rather than promised in prose:

* Hyperparameter search runs `GridSearchCV` over the training set only. The
  validation set is not visible to the search.

* What this module does with the test partition, stated precisely because an
  earlier version of this docstring overclaimed: it READS the test rows and
  writes them to `partition_test.csv`, to freeze which molecules are held
  out. It never featurises them, never passes them to `fit`, and never
  computes a metric on them. So test labels do reach this process's memory
  and the disk; they do not reach any model or any selection decision.
  The honest guarantee is "not used", not "not loaded".
* The validation set is scored once per model, after refitting on train. It is
  used to choose *which model family* to put forward, and for nothing else.
* Fitted models are frozen to `artifacts/<split>/<model>.joblib`. The test
  evaluation loads those files and cannot refit.

Run:
    python -m logd.train --split scaffold
    python -m logd.train --split random
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV, KFold

from . import config as cfg
from . import data as data_mod
from . import featurize, metrics, models, provenance, splits


def _featurize_blocks(split: splits.Split, blocks: set[str]) -> dict[str, dict]:
    """Featurise train and valid once per block. Test is deliberately absent."""
    cache: dict[str, dict] = {}
    for block in sorted(blocks):
        t0 = time.perf_counter()
        cache[block] = {
            "train": featurize.featurize(
                split.train[cfg.SMILES_COL].tolist(), block
            ),
            "valid": featurize.featurize(
                split.valid[cfg.SMILES_COL].tolist(), block
            ),
        }
        cache[block]["seconds"] = round(time.perf_counter() - t0, 2)
    return cache


def train_split(split_kind: str, model_names: list[str], n_jobs: int) -> dict:
    df = data_mod.load_curated()
    split = splits.make_split(split_kind, df)
    splits.assert_no_leakage(split)

    summary = splits.split_summary(split)
    out_dir = cfg.ARTIFACTS / split_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record exactly which molecule went where, so a reviewer can audit it.
    for part, frame in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        assign = frame[[cfg.ID_COL, cfg.SMILES_COL, cfg.LABEL_COL]].copy()
        assign["partition"] = part
        assign.to_csv(out_dir / f"partition_{part}.csv", index=False)

    y_train = split.train[cfg.LABEL_COL].to_numpy()
    y_valid = split.valid[cfg.LABEL_COL].to_numpy()

    specs = [models.get(n) for n in model_names]
    cache = _featurize_blocks(split, {s.block for s in specs})

    cv = KFold(n_splits=cfg.N_CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    records: dict[str, dict] = {}

    for spec in specs:
        X_train = cache[spec.block]["train"]
        X_valid = cache[spec.block]["valid"]
        t0 = time.perf_counter()

        if spec.param_grid:
            search = GridSearchCV(
                spec.factory(),
                param_grid=spec.param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = {k: str(v) for k, v in search.best_params_.items()}
            cv_rmse = float(-search.best_score_)
            n_candidates = int(len(search.cv_results_["params"]))
        else:
            estimator = spec.factory()
            estimator.fit(X_train, y_train)
            best_params = {}
            # Still report a CV estimate on train for comparability.
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                spec.factory(),
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=n_jobs,
            )
            cv_rmse = float(-np.mean(scores))
            n_candidates = 1

        fit_seconds = round(time.perf_counter() - t0, 2)
        valid_metrics = metrics.compute_all(y_valid, estimator.predict(X_valid))

        dump(estimator, out_dir / f"{spec.name}.joblib")
        records[spec.name] = {
            "block": spec.block,
            "note": spec.note,
            "is_baseline": spec.is_baseline,
            "n_hyperparameter_candidates": n_candidates,
            "best_params": best_params,
            "cv_rmse_on_train": round(cv_rmse, 4),
            "valid": {k: round(v, 4) for k, v in valid_metrics.items()},
            "fit_seconds": fit_seconds,
        }
        print(
            f"  {spec.name:<18} cv_rmse={cv_rmse:.3f}  "
            f"valid_rmse={valid_metrics['rmse']:.3f}  "
            f"valid_kendall={valid_metrics['kendall']:.3f}  ({fit_seconds}s)"
        )

    ranked = sorted(records.items(), key=lambda kv: kv[1]["valid"]["rmse"])
    selected = ranked[0][0]

    payload = {
        "split": summary,
        "selection_rule": (
            "lowest validation RMSE; validation used ONLY for this choice. "
            "Hyperparameters came from 5-fold CV on train. Test set untouched."
        ),
        "selected_model": selected,
        "featurization_seconds": {b: cache[b]["seconds"] for b in cache},
        "models": records,
        "provenance": provenance.snapshot(),
    }
    provenance.write_json(cfg.RESULTS / f"validation_{split_kind}.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    p.add_argument("--models", nargs="*", default=list(models.DEFAULT_MODELS))
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args(argv)

    cfg.ensure_dirs()
    print(f"[train] split={args.split}  models={len(args.models)}")
    payload = train_split(args.split, args.models, args.jobs)
    print(f"\n[train] selected on validation: {payload['selected_model']}")
    print(f"[train] wrote results/validation_{args.split}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
