"""Regression validation protocols for chronological tabular data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model) -> Pipeline:
    """Create a pipeline where imputation and scaling are fitted per fold."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", clone(model)),
        ]
    )


def _metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float | None]:
    r2 = float(r2_score(y_true, prediction)) if len(y_true) >= 2 else None
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": r2,
    }


def default_models() -> dict:
    return {
        "BayesianRidge": make_pipeline(BayesianRidge()),
        "Ridge10": make_pipeline(Ridge(alpha=10.0)),
        "Huber": make_pipeline(HuberRegressor(max_iter=2000)),
        "HistGB": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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


def _splits(
    n_rows: int, n_splits: int, gap: int
) -> dict[str, Iterable[tuple[np.ndarray, np.ndarray]]]:
    # Keep the example usable on small datasets while retaining a meaningful
    # purge gap on larger ones. A fixed gap can make TimeSeriesSplit invalid
    # when a user is auditing a short series.
    return {
        "random_kfold": KFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            np.arange(n_rows)
        ),
        "blocked_kfold": KFold(n_splits=n_splits, shuffle=False).split(np.arange(n_rows)),
        "expanding_timeseries": TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_rows)),
        "expanding_timeseries_gap": TimeSeriesSplit(n_splits=n_splits, gap=gap).split(
            np.arange(n_rows)
        ),
    }


def evaluate_regression_protocols(
    features: pd.DataFrame,
    target: pd.Series,
    models: dict | None = None,
    n_splits: int = 5,
    gap: int | None = None,
) -> dict:
    """Evaluate models under several split protocols.

    All preprocessing is fitted on each training fold. The returned report is
    JSON serializable and contains fold-level metrics for inspection.
    """

    if len(features) != len(target):
        raise ValueError("features and target must have the same number of rows")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if gap is not None and gap < 0:
        raise ValueError("gap must be non-negative")

    y = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    valid_positions = valid.to_numpy()
    numeric = features.iloc[valid_positions].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = y.iloc[valid_positions].reset_index(drop=True)
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]
    if numeric.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found")
    if len(numeric) <= n_splits:
        raise ValueError(
            f"at least {n_splits + 1} rows with a numeric target are required "
            f"for {n_splits} splits"
        )

    effective_gap = min(25, max(0, len(numeric) // 10)) if gap is None else gap
    models = default_models() if models is None else models
    if not models:
        raise ValueError("models must contain at least one estimator")
    report: dict[str, dict] = {}
    for protocol, split_iter in _splits(len(numeric), n_splits, effective_gap).items():
        # Materialize generators because every model must see identical folds.
        try:
            folds = list(split_iter)
        except ValueError as exc:
            raise ValueError(
                f"invalid split configuration: rows={len(numeric)}, "
                f"n_splits={n_splits}, gap={effective_gap}"
            ) from exc
        protocol_report: dict[str, dict] = {}
        for name, estimator in models.items():
            fold_rows = []
            for train_idx, valid_idx in folds:
                train_x, valid_x = numeric.iloc[train_idx], numeric.iloc[valid_idx]
                fitted = clone(estimator).fit(train_x, y.iloc[train_idx])
                prediction = fitted.predict(valid_x)
                fold_rows.append(
                    {
                        "n_train": len(train_idx),
                        "n_valid": len(valid_idx),
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
        "rows": len(numeric),
        "features": list(numeric.columns),
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)),
        "configuration": {"n_splits": n_splits, "gap": effective_gap},
        "validation": report,
    }
