"""Final test-set evaluation. Run once, logged, on frozen models.

This module is the only place the test set is read, and it is built to make
"we did not tune on test" an auditable claim rather than a promise:

* It refuses to run without `--confirm-final-evaluation`.
* It loads models from disk and never calls `fit`. Hyperparameters cannot be
  changed from here.
* It evaluates the test partition frozen by `train.py`, read back from
  `artifacts/<split>/partition_test.csv`, so the set cannot drift.
* Every run appends a record to `results/test_access_log.jsonl` with the
  timestamp, git commit and data digest. If that log has more entries than
  there are reported result sets, someone was peeking, and the log says so.

Run:
    python -m logd.evaluate --split scaffold --confirm-final-evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import load  # noqa: E402

from . import config as cfg  # noqa: E402
from . import data as data_mod  # noqa: E402
from . import featurize, metrics, models, provenance, similarity  # noqa: E402


def _log_test_access(split_kind: str, model_names: list[str]) -> int:
    """Append an audit record; return the run index (1 = first ever)."""
    path = cfg.RESULTS / "test_access_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = 0
    if path.exists():
        previous = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    record = {
        "run_index": previous + 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_kind,
        "models": model_names,
        "git_commit": provenance.git_commit(),
        "curated_sha256": data_mod.sha256_file(cfg.CURATED_PARQUET),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record["run_index"]


def _plot(split_kind: str, model: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    lo = float(min(y_true.min(), y_pred.min())) - 0.3
    hi = float(max(y_true.max(), y_pred.max())) + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=14, alpha=0.5, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], lw=1.2, ls="--", color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("experimental logD$_{7.4}$")
    ax.set_ylabel("predicted logD$_{7.4}$")
    ax.set_title(f"{model} — {split_kind} split test set (n={len(y_true)})")
    ax.set_aspect("equal", adjustable="box")

    resid = y_pred - y_true
    ax = axes[1]
    ax.hist(resid, bins=35, edgecolor="white")
    ax.axvline(0.0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("residual (predicted − experimental), log units")
    ax.set_ylabel("count")
    ax.set_title(f"residuals: mean={resid.mean():+.3f}, sd={resid.std(ddof=1):.3f}")

    fig.savefig(cfg.FIGURES / f"test_{split_kind}_{model}.png", dpi=150)
    plt.close(fig)


def evaluate_split(split_kind: str, model_names: list[str]) -> dict:
    art_dir = cfg.ARTIFACTS / split_kind
    part_path = art_dir / "partition_test.csv"
    if not part_path.exists():
        raise FileNotFoundError(
            f"No frozen test partition at {part_path}. Run training first:\n"
            f"  python -m logd.train --split {split_kind}"
        )

    test = pd.read_csv(part_path)
    y_test = test[cfg.LABEL_COL].to_numpy()
    smiles = test[cfg.SMILES_COL].tolist()

    # Training molecules are read only to measure how far the test set sits
    # from them. No model is refitted here.
    train_smiles = pd.read_csv(art_dir / "partition_train.csv")[
        cfg.SMILES_COL
    ].tolist()

    available = [n for n in model_names if (art_dir / f"{n}.joblib").exists()]
    missing = sorted(set(model_names) - set(available))
    if not available:
        raise FileNotFoundError(f"no frozen models in {art_dir}")

    run_index = _log_test_access(split_kind, available)

    specs = [models.get(n) for n in available]
    blocks = {}
    for block in sorted({s.block for s in specs}):
        blocks[block] = featurize.featurize(smiles, block)

    rows: dict[str, dict] = {}
    points: dict[str, dict] = {}
    predictions = pd.DataFrame(
        {cfg.SMILES_COL: smiles, "experimental": y_test}
    )

    for spec in specs:
        estimator = load(art_dir / f"{spec.name}.joblib")
        t0 = time.perf_counter()
        y_pred = np.asarray(estimator.predict(blocks[spec.block]), dtype=float)
        predict_seconds = round(time.perf_counter() - t0, 3)

        ci = metrics.bootstrap_ci(y_test, y_pred, n_resamples=1000, seed=cfg.RANDOM_SEED)
        rows[spec.name] = ci
        points[spec.name] = {
            "block": spec.block,
            "note": spec.note,
            "is_baseline": spec.is_baseline,
            "predict_seconds": predict_seconds,
            "metrics": {m: round(ci[m]["point"], 4) for m in metrics.METRIC_NAMES},
            "ci95": {
                m: [round(ci[m]["lo"], 4), round(ci[m]["hi"], 4)]
                for m in metrics.METRIC_NAMES
            },
        }
        predictions[f"pred_{spec.name}"] = y_pred
        print(
            f"  {spec.name:<18} rmse={ci['rmse']['point']:.3f} "
            f"r2={ci['r2']['point']:.3f} "
            f"kendall={ci['kendall']['point']:.3f}"
        )

    best = min(points, key=lambda n: points[n]["metrics"]["rmse"])
    best_baseline = min(
        (n for n in points if points[n]["is_baseline"]),
        key=lambda n: points[n]["metrics"]["rmse"],
        default=None,
    )
    _plot(split_kind, best, y_test, predictions[f"pred_{best}"].to_numpy())

    # --- how far is the test set from the training set? ----------------------
    sims = similarity.nearest_train_similarity(smiles, train_smiles)
    predictions["nearest_train_tanimoto"] = sims
    audited = [n for n in (best, best_baseline) if n is not None]
    similarity_audit = {
        "description": (
            "ECFP4 Tanimoto from each test molecule to its nearest training "
            "molecule. A scaffold split guarantees distinct frameworks, not "
            "distant molecules, so the distance is measured rather than assumed."
        ),
        "nearest_train_tanimoto": similarity.similarity_summary(sims),
        "error_by_similarity_band": {
            name: similarity.error_by_similarity_band(
                y_test, predictions[f"pred_{name}"].to_numpy(), sims
            )
            for name in audited
        },
    }

    predictions.to_csv(cfg.RESULTS / f"test_predictions_{split_kind}.csv", index=False)
    table = metrics.format_table(rows)
    (cfg.RESULTS / f"test_table_{split_kind}.md").write_text(table + "\n", encoding="utf-8")

    verdict = None
    if best_baseline is not None:
        delta = (
            points[best_baseline]["metrics"]["rmse"] - points[best]["metrics"]["rmse"]
        )
        beats = best not in {n for n in points if points[n]["is_baseline"]}
        verdict = {
            "best_model": best,
            "best_baseline": best_baseline,
            "rmse_improvement_over_best_baseline": round(delta, 4),
            "learned_model_beats_all_baselines": bool(beats),
            "interpretation": (
                "A learned model is only worth shipping if this improvement is "
                "larger than the bootstrap intervals above."
            ),
        }

    payload = {
        "split": split_kind,
        "test_n": int(len(y_test)),
        "test_access_run_index": run_index,
        "models_missing_artifacts": missing,
        "results": points,
        "verdict": verdict,
        "similarity_audit": similarity_audit,
        "provenance": provenance.snapshot(),
    }
    provenance.write_json(cfg.RESULTS / f"test_{split_kind}.json", payload)
    print("\n" + table)
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    p.add_argument("--models", nargs="*", default=list(models.DEFAULT_MODELS))
    p.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="required; acknowledges this touches the held-out test set",
    )
    args = p.parse_args(argv)

    if not args.confirm_final_evaluation:
        print(
            "Refusing to run.\n\n"
            "This reads the held-out test set. Every run is recorded in\n"
            "results/test_access_log.jsonl. Re-run with "
            "--confirm-final-evaluation\nonly when model development is "
            "finished.",
            file=sys.stderr,
        )
        return 2

    cfg.ensure_dirs()
    print(f"[evaluate] FINAL test evaluation, split={args.split}")
    evaluate_split(args.split, args.models)
    print(f"[evaluate] wrote results/test_{args.split}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
