"""Model registry: baselines first, then learned models.

The ordering is deliberate. A molecular ML result only means something
relative to what you get for free:

1. `mean_baseline`      -- predicts the training mean. Floor for RMSE/R2.
2. `clogp_identity`     -- predicts Crippen clogP directly. ZERO fitted
                           parameters. A knowledge-based calculator shipped
                           with RDKit since the 1990s.
3. `clogp_linear`        -- clogP rescaled by a 2-parameter fit on train.
4. `ridge_ecfp4`         -- linear model on 2048-bit ECFP4.
5. `svr_desc`            -- RBF kernel SVR on RDKit 2D descriptors.
6. `rf_desc`             -- random forest on RDKit 2D descriptors.
7. `hgb_desc`            -- histogram gradient boosting on descriptors.
8. `hgb_desc_ecfp4`      -- boosting on descriptors + fingerprint.

If a learned model cannot beat `clogp_linear` on a scaffold split, the honest
conclusion is that the model is not useful, and this repo says so rather than
quietly dropping the baseline.

Every fitted preprocessing step sits inside the Pipeline, so cross-validation
fits it on training folds only. There is no global scaler fitted on all data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as cfg


class ClogPIdentity(RegressorMixin, BaseEstimator):
    """Predict the first feature column unchanged. No fitted parameters."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def _dense_numeric(scale: bool) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


@dataclass(frozen=True)
class ModelSpec:
    name: str
    block: str
    factory: Callable[[], Pipeline]
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    note: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.name in {"mean_baseline", "clogp_identity", "clogp_linear"}


def _registry() -> dict[str, ModelSpec]:
    seed = cfg.RANDOM_SEED
    specs = [
        ModelSpec(
            name="mean_baseline",
            block="clogp",
            factory=lambda: Pipeline([("model", DummyRegressor(strategy="mean"))]),
            note="predicts the training-set mean; zero chemistry",
        ),
        ModelSpec(
            name="clogp_identity",
            block="clogp",
            factory=lambda: Pipeline([("model", ClogPIdentity())]),
            note="RDKit Crippen clogP used as-is; zero fitted parameters",
        ),
        ModelSpec(
            name="clogp_linear",
            block="clogp",
            factory=lambda: Pipeline([("model", LinearRegression())]),
            note="Crippen clogP rescaled by a 2-parameter fit on train",
        ),
        ModelSpec(
            name="ridge_ecfp4",
            block="ecfp4",
            factory=lambda: Pipeline(
                [("drop_constant", VarianceThreshold(0.0)), ("model", Ridge(random_state=None))]
            ),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            note="linear model on 2048-bit ECFP4",
        ),
        ModelSpec(
            name="svr_desc",
            block="desc",
            factory=lambda: Pipeline(
                _dense_numeric(scale=True) + [("model", SVR(kernel="rbf"))]
            ),
            param_grid={
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01],
                "model__epsilon": [0.1],
            },
            note="RBF SVR on standardised RDKit 2D descriptors",
        ),
        ModelSpec(
            name="rf_desc",
            block="desc",
            factory=lambda: Pipeline(
                _dense_numeric(scale=False)
                + [
                    (
                        "model",
                        # n_jobs=1 on purpose: GridSearchCV already
                        # parallelises across candidates and folds. Nesting
                        # forest threads inside CV workers oversubscribes the
                        # machine and has crashed the container engine here.
                        RandomForestRegressor(
                            n_estimators=500, random_state=seed, n_jobs=1
                        ),
                    )
                ]
            ),
            param_grid={
                "model__max_features": [0.3, "sqrt"],
                "model__min_samples_leaf": [1, 2],
            },
            note="500-tree random forest on RDKit 2D descriptors",
        ),
        ModelSpec(
            name="hgb_desc",
            block="desc",
            factory=lambda: Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=600,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            random_state=seed,
                        ),
                    )
                ]
            ),
            param_grid={
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [31, 63],
                "model__l2_regularization": [0.0, 1.0],
            },
            note="histogram gradient boosting; handles NaN descriptors natively",
        ),
        ModelSpec(
            name="hgb_desc_ecfp4",
            block="desc_ecfp4",
            factory=lambda: Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=600,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            random_state=seed,
                        ),
                    )
                ]
            ),
            param_grid={
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [31, 63],
            },
            note="boosting on descriptors concatenated with ECFP4",
        ),
    ]
    return {s.name: s for s in specs}


REGISTRY: dict[str, ModelSpec] = _registry()
DEFAULT_MODELS: tuple[str, ...] = tuple(REGISTRY)


def get(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
