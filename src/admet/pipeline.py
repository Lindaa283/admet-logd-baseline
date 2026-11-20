"""End-to-end run: download -> QC -> split -> fit -> score -> figures.

Design choices that affect how the numbers should be read:

* **The headline evaluation is the deterministic Bemis-Murcko scaffold split.**
  Largest scaffold groups go to train, singletons to test.  It is reproducible
  without a seed and it is the hardest arrangement, so it is the one quoted.
* **Hyperparameters are chosen once**, on the validation part of that headline
  split, and then reused unchanged for every other split.  The test part is
  never consulted during selection.
* **Two extra families of splits are run for context only**: three seeded
  scaffold splits (how much the estimate moves with a different but still
  scaffold-disjoint partition) and three seeded random splits (how much a
  random split flatters the same models on the same data).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import download, features, leakage, metrics, models, plots, scaffold
from .chem import DESCRIPTOR_NAMES, mol_from_canonical
from .qc import load_and_qc

LABEL_NAME = "logD7.4"
FRACTIONS = (0.7, 0.1, 0.2)
N_SEEDS = 3
BOOTSTRAP_RESAMPLES = 1000


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(
    root: Path | None = None,
    *,
    n_seeds: int = N_SEEDS,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    make_figures: bool = True,
) -> dict:
    """Run everything and write results under ``root/results``."""
    started = time.time()
    root = Path(root) if root else project_root()
    raw_dir = root / "data" / "raw"
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    print("[1/7] data")
    csv_path, provenance = download.fetch(raw_dir)
    (raw_dir / "provenance.json").write_text(provenance.to_json(), encoding="utf-8")

    clean, qc_report = load_and_qc(
        csv_path,
        source_sha256=provenance.sha256,
        label_name=LABEL_NAME,
        plausible_range=download.LIPOPHILICITY["plausible_label_range"],
    )
    (tables / "qc_report.json").write_text(qc_report.to_json(), encoding="utf-8")
    print(
        f"      {qc_report.n_raw_rows} raw rows -> {qc_report.n_clean_rows} compounds "
        f"({qc_report.n_duplicate_compounds} measured more than once, "
        f"{qc_report.n_unparseable} unparseable)"
    )
    print(f"      unit check: {qc_report.unit_check}")

    # ----------------------------------------------------------- 2. featurise
    print("[2/7] featurising")
    mols = [mol_from_canonical(smiles) for smiles in clean["canonical_smiles"]]
    if any(mol is None for mol in mols):
        raise RuntimeError("a canonical SMILES failed to re-parse; QC is inconsistent")

    descriptor_block, descriptor_names = features.descriptor_matrix(
        mols, clean["n_fragments"].tolist()
    )
    morgan_block = features.morgan_counts(mols)
    bitvects = features.morgan_bitvects(mols)
    scaffolds = [scaffold.murcko_scaffold(mol) for mol in mols]
    groups = scaffold.scaffold_groups(scaffolds)
    y = clean["label"].to_numpy(dtype=float)
    identity_keys = clean["connectivity_key"].tolist()
    blocks = models.FeatureBlocks(descriptor_block, morgan_block)
    print(
        f"      {blocks.n_rows} compounds, {morgan_block.shape[1]} fingerprint bits, "
        f"{descriptor_block.shape[1]} descriptors, {len(groups)} scaffolds"
    )

    clean_out = clean.copy()
    clean_out["murcko_scaffold"] = scaffolds
    clean_out.to_csv(tables / "clean_dataset.csv", index=False)

    # --------------------------------------------------------------- 3. splits
    print("[3/7] splits + leakage audit")
    split_set: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    split_set["scaffold_deterministic"] = scaffold.scaffold_split(
        scaffolds, fractions=FRACTIONS, seed=None
    )
    for seed in range(n_seeds):
        split_set[f"scaffold_seed{seed}"] = scaffold.scaffold_split(
            scaffolds, fractions=FRACTIONS, seed=seed
        )
        split_set[f"random_seed{seed}"] = scaffold.random_split(
            blocks.n_rows, fractions=FRACTIONS, seed=seed
        )

    audits = []
    for name, (train, val, test) in split_set.items():
        audits.append(
            leakage.audit_split(
                name,
                train,
                val,
                test,
                scaffolds=scaffolds,
                identity_keys=identity_keys,
                fingerprints=bitvects,
                require_scaffold_disjoint=name.startswith("scaffold"),
            )
        )
    leakage.write_audits(audits, tables / "leakage_audit.json")
    leakage.assert_all_passed(audits)
    headline_audit = next(a for a in audits if a.split_name == "scaffold_deterministic")
    random_audit = next(a for a in audits if a.split_name == "random_seed0")
    print(
        f"      scaffold split: {headline_audit.n_shared_scaffolds_train_test} shared "
        f"scaffolds, median nearest-train similarity "
        f"{headline_audit.median_nearest_similarity:.3f}"
    )
    print(
        f"      random split:   {random_audit.n_shared_scaffolds_train_test} shared "
        f"scaffolds, median nearest-train similarity "
        f"{random_audit.median_nearest_similarity:.3f}"
    )

    # ------------------------------------------- 4. hyperparameters (headline)
    print("[4/7] selecting hyperparameters on the headline validation split")
    specs = models.model_specs()
    train, val, test = split_set["scaffold_deterministic"]
    chosen: dict[str, dict] = {}
    headline_fits: dict[str, models.FittedModel] = {}
    for spec in specs:
        fitted = models.fit_with_validation(spec, blocks, y, train, val)
        chosen[spec.name] = fitted.chosen_params
        headline_fits[spec.name] = fitted
        print(
            f"      {spec.name:<20} params={fitted.chosen_params} "
            f"val MAE={fitted.validation_mae:.3f}"
        )

    # --------------------------------------------------- 5. score every split
    print("[5/7] fitting and scoring every split")
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for split_name, (train, val, test) in split_set.items():
        refit_rows = np.concatenate([train, val])
        for spec in specs:
            if split_name == "scaffold_deterministic":
                fitted = headline_fits[spec.name]
            else:
                estimator = spec.build(chosen[spec.name])
                estimator.fit(blocks.get(spec.block, refit_rows), y[refit_rows])
                fitted = models.FittedModel(
                    spec=spec,
                    estimator=estimator,
                    chosen_params=chosen[spec.name],
                    validation_mae=float("nan"),
                )
            y_pred_test = fitted.predict(blocks, test)
            y_pred_train = fitted.predict(blocks, refit_rows)
            test_scores = metrics.score(y[test], y_pred_test)
            train_scores = metrics.score(y[refit_rows], y_pred_train)

            record = {
                "split": split_name,
                "split_family": _family(split_name),
                "model": spec.name,
                "description": spec.description,
                "params": json.dumps(chosen[spec.name], sort_keys=True),
                "n_train": int(refit_rows.size),
                "n_test": int(test.size),
                "train_mae": train_scores.mae,
                "train_rmse": train_scores.rmse,
                "train_r2": train_scores.r2,
                "test_mae": test_scores.mae,
                "test_rmse": test_scores.rmse,
                "test_r2": test_scores.r2,
                "test_spearman": test_scores.spearman,
                "test_bias": test_scores.bias,
                "test_max_abs_error": test_scores.max_abs_error,
            }
            if split_name == "scaffold_deterministic":
                low, high = metrics.bootstrap_ci(
                    y[test], y_pred_test, metric="mae",
                    n_resamples=bootstrap_resamples, seed=7,
                )
                rmse_low, rmse_high = metrics.bootstrap_ci(
                    y[test], y_pred_test, metric="rmse",
                    n_resamples=bootstrap_resamples, seed=7,
                )
                record.update(
                    test_mae_ci_low=low, test_mae_ci_high=high,
                    test_rmse_ci_low=rmse_low, test_rmse_ci_high=rmse_high,
                    validation_mae=headline_fits[spec.name].validation_mae,
                )
                predictions[spec.name] = y_pred_test
            rows.append(record)
        print(f"      {split_name} done")

    results = pd.DataFrame(rows)
    results.to_csv(tables / "metrics_all_splits.csv", index=False)

    headline = (
        results[results["split"] == "scaffold_deterministic"]
        .sort_values("test_mae")
        .reset_index(drop=True)
    )
    headline.to_csv(tables / "metrics_headline_scaffold.csv", index=False)

    summary = (
        results[results["split_family"].isin(["scaffold_seeded", "random_seeded"])]
        .groupby(["split_family", "model"])
        .agg(
            test_mae_mean=("test_mae", "mean"),
            test_mae_sd=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_sd=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            n_splits=("test_mae", "size"),
        )
        .reset_index()
    )
    summary.to_csv(tables / "metrics_split_family_summary.csv", index=False)

    # ---------------------------------------------------- 6. predictions file
    train, val, test = split_set["scaffold_deterministic"]
    test_fps = [bitvects[row] for row in test]
    train_fps = [bitvects[row] for row in np.concatenate([train, val])]
    nearest = features.nearest_train_similarity(test_fps, train_fps)
    best_name = headline.iloc[0]["model"]
    prediction_table = pd.DataFrame(
        {
            "compound_id": clean["compound_id"].to_numpy()[test],
            "canonical_smiles": clean["canonical_smiles"].to_numpy()[test],
            "murcko_scaffold": np.array(scaffolds)[test],
            "observed": y[test],
            "nearest_train_tanimoto": nearest,
            "n_measurements": clean["n_measurements"].to_numpy()[test],
        }
    )
    for name, values in predictions.items():
        prediction_table[f"pred_{name}"] = values
    prediction_table.to_csv(
        tables / "predictions_headline_scaffold.csv", index=False
    )

    # ------------------------------------------------------------ 7. figures
    written_figures = []
    if make_figures:
        print("[6/7] figures")
        group_sizes = np.array([len(v) for v in groups.values()])
        written_figures.append(
            plots.data_overview(y, group_sizes, figures / "01_data_overview.png",
                                label_name=LABEL_NAME)
        )
        spreads = clean.loc[clean["n_measurements"] > 1, "label_spread"].to_numpy()
        written_figures.append(
            plots.duplicate_spread(spreads, figures / "02_duplicate_spread.png",
                                   label_name=LABEL_NAME)
        )

        order = [spec.name for spec in specs]
        head_indexed = headline.set_index("model").loc[order]
        random_means = (
            summary[summary["split_family"] == "random_seeded"]
            .set_index("model")["test_mae_mean"]
            .reindex(order)
        )
        written_figures.append(
            plots.model_comparison(
                order,
                head_indexed["test_mae"].tolist(),
                head_indexed["test_mae_ci_low"].tolist(),
                head_indexed["test_mae_ci_high"].tolist(),
                random_means.tolist(),
                figures / "03_model_comparison.png",
                noise_floor=qc_report.implied_experimental_mae_floor or None,
            )
        )
        written_figures.append(
            plots.observed_vs_predicted(
                [
                    ("clogp_calibrated (baseline)", y[test],
                     predictions["clogp_calibrated"]),
                    (f"{best_name} (best)", y[test], predictions[best_name]),
                ],
                figures / "04_observed_vs_predicted.png",
                label_name=LABEL_NAME,
            )
        )
        written_figures.append(
            plots.residual_diagnostics(
                y[test], predictions[best_name], figures / "05_residuals.png",
                label_name=LABEL_NAME,
            )
        )
        written_figures.append(
            plots.applicability_domain(
                nearest, np.abs(predictions[best_name] - y[test]),
                figures / "06_applicability_domain.png",
            )
        )
        ridge = headline_fits["descriptor_ridge"].estimator.named_steps["ridge"]
        written_figures.append(
            plots.descriptor_coefficients(
                list(descriptor_names), np.asarray(ridge.coef_).ravel(),
                figures / "07_descriptor_coefficients.png",
            )
        )

    # --------------------------------------------------------- run metadata
    print("[7/7] metadata")
    import matplotlib
    import rdkit
    import scipy
    import sklearn

    metadata = {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "rdkit": rdkit.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "label": LABEL_NAME,
        "fractions": list(FRACTIONS),
        "n_seeds": n_seeds,
        "bootstrap_resamples": bootstrap_resamples,
        "random_state": models.RANDOM_STATE,
        "morgan": {"radius": features.MORGAN_RADIUS, "bits": features.MORGAN_BITS},
        "data_provenance": asdict(provenance),
        "chosen_hyperparameters": chosen,
        "best_model_headline": best_name,
        "figures": [str(path.relative_to(root)) for path in written_figures],
    }
    (root / "results" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nheadline (deterministic scaffold split), test MAE in {LABEL_NAME} units:")
    for _, row in headline.iterrows():
        print(
            f"  {row['model']:<20} MAE {row['test_mae']:.3f} "
            f"[{row['test_mae_ci_low']:.3f}, {row['test_mae_ci_high']:.3f}]  "
            f"RMSE {row['test_rmse']:.3f}  R2 {row['test_r2']:.3f}  "
            f"rho {row['test_spearman']:.3f}"
        )
    print(f"\ndone in {metadata['wall_seconds']}s")
    return {
        "qc": qc_report,
        "headline": headline,
        "summary": summary,
        "metadata": metadata,
        "audits": audits,
    }


def _family(split_name: str) -> str:
    if split_name == "scaffold_deterministic":
        return "scaffold_deterministic"
    if split_name.startswith("scaffold_seed"):
        return "scaffold_seeded"
    return "random_seeded"


if __name__ == "__main__":  # pragma: no cover - manual use
    run()
