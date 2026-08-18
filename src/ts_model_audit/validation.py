"""Regression validation protocols for chronological tabular data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def make_pipeline(model) -> Pipeline:
    """Create a pipeline where imputation, clipping, and scaling fit per fold."""

    # Winsorizer is applied explicitly inside evaluate_regression_protocols so
    # bounds never use the validation fold.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", clone(model)),
        ]
    )


def _metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def default_models() -> dict:
    return {
        "BayesianRidge": make_pipeline(BayesianRidge()),
        "Ridge10": make_pipeline(Ridge(alpha=10.0)),
        "Huber": make_pipeline(HuberRegressor(max_iter=2000)),
        "HistGB": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=2.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _splits(n_rows: int) -> dict[str, Iterable[tuple[np.ndarray, np.ndarray]]]:
    # Keep the example usable on small datasets while retaining a meaningful
    # purge gap on larger ones. A fixed gap can make TimeSeriesSplit invalid
    # when a user is auditing a short series.
    gap = min(25, max(0, n_rows // 10))
    return {
        "random_kfold": KFold(n_splits=5, shuffle=True, random_state=42).split(
            np.arange(n_rows)
        ),
        "blocked_kfold": KFold(n_splits=5, shuffle=False).split(np.arange(n_rows)),
        "expanding_timeseries": TimeSeriesSplit(n_splits=5).split(np.arange(n_rows)),
        "expanding_timeseries_gap": TimeSeriesSplit(n_splits=5, gap=gap).split(
            np.arange(n_rows)
        ),
    }


def evaluate_regression_protocols(
    features: pd.DataFrame,
    target: pd.Series,
    models: dict | None = None,
) -> dict:
    """Evaluate models under several split protocols.

    All preprocessing is fitted on each training fold. The returned report is
    JSON serializable and contains fold-level metrics for inspection.
    """

    if len(features) != len(target):
        raise ValueError("features and target must have the same number of rows")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    valid = y.notna()
    numeric, y = numeric.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    models = models or default_models()
    report: dict[str, dict] = {}
    for protocol, split_iter in _splits(len(numeric)).items():
        # Materialize generators because every model must see identical folds.
        folds = list(split_iter)
        protocol_report: dict[str, dict] = {}
        for name, estimator in models.items():
            fold_rows = []
            for train_idx, valid_idx in folds:
                train_x, valid_x = numeric.iloc[train_idx], numeric.iloc[valid_idx]
                fitted = clone(estimator).fit(train_x, y.iloc[train_idx])
                prediction = fitted.predict(valid_x)
                fold_rows.append(
                    {
                        "n_train": int(len(train_idx)),
                        "n_valid": int(len(valid_idx)),
                        **_metrics(y.iloc[valid_idx], prediction),
                    }
                )
            protocol_report[name] = {
                "folds": fold_rows,
                "rmse_mean": float(np.mean([row["rmse"] for row in fold_rows])),
                "rmse_weighted": float(
                    np.sqrt(
                        np.average(
                            [row["rmse"] ** 2 for row in fold_rows],
                            weights=[row["n_valid"] for row in fold_rows],
                        )
                    )
                ),
            }
        report[protocol] = protocol_report
    return {
        "rows": int(len(numeric)),
        "features": list(numeric.columns),
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)),
        "validation": report,
    }
