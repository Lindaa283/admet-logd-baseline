"""The model ladder, from a constant to a fingerprint ensemble.

The point of the ladder is that each rung has to earn its place.  A result is
only meaningful relative to what a cheaper method already achieves, so the
report always shows:

* ``constant`` - predict the training mean.  Any model that does not clearly
  beat this has learned nothing.
* ``clogp_calibrated`` - a one-parameter rescaling of Wildman-Crippen cLogP,
  which is a published, zero-training-data estimate of lipophilicity.  For
  logD this is a genuinely strong baseline and the one that matters: a machine
  learning model on this endpoint is only interesting if it beats cLogP.
* ``descriptor_ridge`` - 24 interpretable descriptors, linear.
* ``morgan_ridge`` - substructure counts, linear.
* ``morgan_rf`` / ``morgan_desc_hgb`` - the non-linear models.

Hyperparameters are chosen on the validation split only.  The test split is
scored once, at the end, for every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import N_THREADS
from .chem import DESCRIPTOR_NAMES

RANDOM_STATE = 20261120


class FeatureBlocks:
    """The featurised dataset, sliced by row index on demand.

    Built once for the whole table.  Models receive only the rows of their
    split, so no model ever sees a test row during fitting.
    """

    def __init__(self, descriptors: np.ndarray, morgan: np.ndarray):
        if descriptors.shape[0] != morgan.shape[0]:
            raise ValueError("descriptor and fingerprint row counts differ")
        self.descriptors = descriptors
        self.morgan = morgan
        self.clogp_index = DESCRIPTOR_NAMES.index("MolLogP")

    @property
    def n_rows(self) -> int:
        return self.descriptors.shape[0]

    def get(self, block: str, rows: np.ndarray) -> np.ndarray:
        if block == "descriptors":
            return self.descriptors[rows]
        if block == "morgan":
            return self.morgan[rows]
        if block == "both":
            return np.hstack([self.morgan[rows], self.descriptors[rows]])
        if block == "clogp":
            return self.descriptors[rows][:, [self.clogp_index]]
        if block == "none":
            return np.zeros((len(rows), 1), dtype=np.float64)
        raise ValueError(f"unknown feature block {block!r}")


class ConstantRegressor:
    """Predicts the mean of the training labels, whatever the input."""

    def __init__(self) -> None:
        self.value_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantRegressor":
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value_, dtype=np.float64)


@dataclass
class ModelSpec:
    """One rung of the ladder: how to build it and which features it sees."""

    name: str
    block: str
    description: str
    build: Callable[[dict[str, Any]], Any]
    grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    # Set for the models whose coefficients we want to report.
    report_coefficients: bool = False


def _ridge(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=True)),
            ("ridge", Ridge(alpha=params.get("alpha", 1.0), random_state=None)),
        ]
    )


def _ridge_sparse(params: dict[str, Any]) -> Pipeline:
    # Fingerprint columns are counts on a common scale; centring 2048 mostly
    # zero columns destroys sparsity for no benefit, so only scale variance.
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=params.get("alpha", 1.0))),
        ]
    )


def _random_forest(params: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        random_state=RANDOM_STATE,
        n_jobs=int(N_THREADS),
    )


def _hgb(params: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 600),
        learning_rate=params.get("learning_rate", 0.06),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("l2_regularization", 1.0),
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def model_specs() -> list[ModelSpec]:
    """The ladder, in the order it is reported."""
    return [
        ModelSpec(
            name="constant",
            block="none",
            description="predict the training-set mean logD",
            build=lambda params: ConstantRegressor(),
        ),
        ModelSpec(
            name="clogp_calibrated",
            block="clogp",
            description=(
                "Wildman-Crippen cLogP rescaled by a one-feature linear fit "
                "(no learned chemistry beyond slope and intercept)"
            ),
            build=lambda params: _ridge({"alpha": 1e-6}),
            report_coefficients=True,
        ),
        ModelSpec(
            name="descriptor_ridge",
            block="descriptors",
            description="24 interpretable physicochemical descriptors, ridge regression",
            build=_ridge,
            grid=[{"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0)],
            report_coefficients=True,
        ),
        ModelSpec(
            name="morgan_ridge",
            block="morgan",
            description="Morgan count fingerprint (r=2, 2048 bits), ridge regression",
            build=_ridge_sparse,
            grid=[{"alpha": alpha} for alpha in (1.0, 10.0, 100.0)],
        ),
        ModelSpec(
            name="morgan_rf",
            block="morgan",
            description="Morgan count fingerprint, random forest (500 trees)",
            build=_random_forest,
            grid=[
                {"max_features": "sqrt", "min_samples_leaf": 1},
                {"max_features": 0.1, "min_samples_leaf": 1},
            ],
        ),
        ModelSpec(
            name="morgan_desc_hgb",
            block="both",
            description=(
                "Morgan counts + descriptors, histogram gradient boosting"
            ),
            build=_hgb,
            grid=[
                {"learning_rate": 0.06, "max_iter": 600, "max_leaf_nodes": 31},
                {"learning_rate": 0.03, "max_iter": 900, "max_leaf_nodes": 63},
            ],
        ),
    ]


@dataclass
class FittedModel:
    """A model after hyperparameter selection and a final fit on train."""

    spec: ModelSpec
    estimator: Any
    chosen_params: dict[str, Any]
    validation_mae: float

    def predict(self, blocks: FeatureBlocks, rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(blocks.get(self.spec.block, rows)), dtype=np.float64
        ).ravel()


def fit_with_validation(
    spec: ModelSpec,
    blocks: FeatureBlocks,
    y: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
) -> FittedModel:
    """Pick hyperparameters on the validation split, then refit on train+val.

    Refitting on train+val after selection uses the data efficiently while
    keeping the test split untouched.  The validation MAE that drove the
    choice is carried along and reported, so the selection is visible.
    """
    X_train = blocks.get(spec.block, train_rows)
    X_val = blocks.get(spec.block, val_rows)
    y_train, y_val = y[train_rows], y[val_rows]

    best_params: dict[str, Any] = {}
    best_mae = np.inf
    for params in spec.grid:
        candidate = spec.build(params)
        candidate.fit(X_train, y_train)
        mae = float(np.mean(np.abs(candidate.predict(X_val) - y_val)))
        if mae < best_mae:
            best_mae, best_params = mae, params

    refit_rows = np.concatenate([train_rows, val_rows])
    final = spec.build(best_params)
    final.fit(blocks.get(spec.block, refit_rows), y[refit_rows])
    return FittedModel(
        spec=spec,
        estimator=final,
        chosen_params=dict(best_params),
        validation_mae=best_mae,
    )
