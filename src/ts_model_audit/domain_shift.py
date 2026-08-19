"""Train/test domain-shift diagnostics for numeric tabular data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def _numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Select numeric columns and make missing values explicit for diagnostics."""

    selected = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return selected.replace([np.inf, -np.inf], np.nan)


def adversarial_domain_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str] | None = None,
    random_state: int = 42,
) -> float:
    """Estimate how easily a classifier can distinguish train from test rows."""

    if columns is None:
        columns = [
            name
            for name in train.columns
            if name in test.columns
            and pd.api.types.is_numeric_dtype(train[name])
            and pd.api.types.is_numeric_dtype(test[name])
        ]
    columns = list(columns)
    if not columns:
        raise ValueError("No common numeric feature columns were found")
    smallest_domain = min(len(train), len(test))
    if smallest_domain < 2:
        raise ValueError("train and test must each contain at least 2 rows")

    combined = pd.concat(
        [_numeric_frame(train, columns), _numeric_frame(test, columns)],
        ignore_index=True,
    )
    labels = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    n_splits = min(5, smallest_domain)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifier = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=random_state,
        ),
    )
    probabilities = cross_val_predict(
        classifier, combined, labels, cv=folds, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, probabilities))


def standardized_mean_difference(
    train: pd.DataFrame, test: pd.DataFrame, columns: Iterable[str] | None = None
) -> pd.Series:
    """Return absolute standardized mean differences, sorted descending."""

    if columns is None:
        columns = [
            name
            for name in train.columns
            if name in test.columns
            and pd.api.types.is_numeric_dtype(train[name])
            and pd.api.types.is_numeric_dtype(test[name])
        ]
    columns = list(columns)
    train_num = _numeric_frame(train, columns)
    test_num = _numeric_frame(test, columns)
    pooled = np.sqrt((train_num.var(ddof=1) + test_num.var(ddof=1)) / 2.0)
    smd = ((test_num.mean() - train_num.mean()) / pooled.replace(0, np.nan)).abs()
    return smd.sort_values(ascending=False)


def clipping_rates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor: float = 3.0,
    columns: Iterable[str] | None = None,
) -> dict:
    """Measure rows/features outside train-only Tukey-style bounds."""

    if columns is None:
        columns = [
            name
            for name in train.columns
            if name in test.columns
            and pd.api.types.is_numeric_dtype(train[name])
            and pd.api.types.is_numeric_dtype(test[name])
        ]
    columns = list(columns)
    train_num = _numeric_frame(train, columns)
    test_num = _numeric_frame(test, columns)
    q1, q3 = train_num.quantile(0.25), train_num.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    flags = (test_num.lt(lower) | test_num.gt(upper)).fillna(False)
    return {
        "factor": factor,
        "test_rows_with_any_feature_clipped_pct": float(100 * flags.any(axis=1).mean()),
        "feature_clip_rate_pct": {
            str(name): float(100 * flags[name].mean())
            for name in flags.columns
        },
    }


def audit_domain_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str | None = None,
    random_state: int = 42,
) -> dict:
    """Build a JSON-serializable domain-shift report."""

    if target is not None and target not in train.columns:
        raise ValueError(f"target column not found in training data: {target}")
    train_features = train.drop(columns=[target], errors="ignore") if target else train
    columns = [
        name
        for name in train_features.columns
        if name in test.columns
        and pd.api.types.is_numeric_dtype(train_features[name])
        and pd.api.types.is_numeric_dtype(test[name])
    ]
    if not columns:
        raise ValueError("No common numeric feature columns were found")
    smd = standardized_mean_difference(train_features, test, columns)
    available_smd = smd.dropna()
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(columns),
        "features": columns,
        "adversarial_oof_auc": adversarial_domain_auc(
            train_features, test, columns, random_state
        ),
        "domain_cv_splits": min(5, len(train), len(test)),
        "top_standardized_mean_differences": {
            str(name): float(value) for name, value in available_smd.head(20).items()
        },
        "smd_unavailable_features": [str(name) for name in smd.index[smd.isna()]],
        "count_smd_gt_0_5": int((smd > 0.5).sum()),
        "count_smd_gt_1_0": int((smd > 1.0).sum()),
        "clipping": clipping_rates(train_features, test, columns=columns),
    }
    return report
