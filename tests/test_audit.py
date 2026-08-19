from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_model_audit import audit_domain_shift, evaluate_regression_protocols

ROOT = Path(__file__).resolve().parents[1]


def test_domain_shift_report_is_serializable():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=120), "y": rng.normal(size=120), "target": 0})
    test = pd.DataFrame({"x": rng.normal(1.5, size=60), "y": rng.normal(size=60)})
    report = audit_domain_shift(train, test, target="target")
    assert report["train_rows"] == 120
    assert report["test_rows"] == 60
    assert report["feature_count"] == 2
    assert report["adversarial_oof_auc"] > 0.5
    json.dumps(report, allow_nan=False)


def test_domain_shift_adapts_cross_validation_to_small_samples():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
    report = audit_domain_shift(train, test)
    assert report["domain_cv_splits"] == 3
    json.dumps(report, allow_nan=False)


def test_domain_shift_reports_constant_smd_as_unavailable():
    train = pd.DataFrame({"constant": [1.0] * 8})
    test = pd.DataFrame({"constant": [1.0] * 8})
    report = audit_domain_shift(train, test)
    assert report["smd_unavailable_features"] == ["constant"]
    assert report["top_standardized_mean_differences"] == {}
    json.dumps(report, allow_nan=False)


def test_domain_shift_rejects_single_row_domain():
    with pytest.raises(ValueError, match="at least 2 rows"):
        audit_domain_shift(pd.DataFrame({"x": [1.0]}), pd.DataFrame({"x": [2.0, 3.0]}))


def test_domain_shift_rejects_missing_target_column():
    frame = pd.DataFrame({"x": np.arange(10, dtype=float)})
    with pytest.raises(ValueError, match="target column not found"):
        audit_domain_shift(frame, frame.copy(), target="target")


def test_validation_uses_all_protocols():
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"x": np.arange(120), "noise": rng.normal(size=120)})
    y = 0.2 * x["x"] + rng.normal(scale=0.5, size=120)
    report = evaluate_regression_protocols(x, y)
    assert report["rows"] == 120
    assert set(report["validation"]) == {
        "random_kfold",
        "blocked_kfold",
        "expanding_timeseries",
        "expanding_timeseries_gap",
    }
    assert report["validation"]["expanding_timeseries"]["Ridge10"]["rmse_weighted"] >= 0
    assert report["configuration"] == {"n_splits": 5, "gap": 12}
    json.dumps(report, allow_nan=False)


def test_validation_accepts_custom_split_configuration():
    x = pd.DataFrame({"x": np.arange(30), "empty": np.nan})
    y = pd.Series(np.arange(30, dtype=float))
    report = evaluate_regression_protocols(x, y, n_splits=3, gap=2)
    assert report["configuration"] == {"n_splits": 3, "gap": 2}
    assert report["features"] == ["x"]


def test_validation_uses_positions_when_indexes_differ():
    x = pd.DataFrame({"x": np.arange(12)}, index=np.arange(100, 112))
    y = pd.Series(np.arange(12, dtype=float), index=np.arange(200, 212))
    report = evaluate_regression_protocols(x, y)
    assert report["rows"] == 12
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1}, "n_splits must be at least 2"),
        ({"gap": -1}, "gap must be non-negative"),
        ({"n_splits": 5}, "at least 6 rows"),
    ],
)
def test_validation_rejects_invalid_split_configuration(kwargs, message):
    x = pd.DataFrame({"x": np.arange(5)})
    y = pd.Series(np.arange(5, dtype=float))
    with pytest.raises(ValueError, match=message):
        evaluate_regression_protocols(x, y, **kwargs)


def test_validation_rejects_no_usable_numeric_features():
    x = pd.DataFrame({"label": ["unknown"] * 10})
    y = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="No usable numeric feature"):
        evaluate_regression_protocols(x, y)


@pytest.mark.parametrize("module", ["ts_model_audit", "ts_model_audit.cli"])
def test_python_module_cli_entry_points(module):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "domain-shift" in completed.stdout
    assert "regression-validation" in completed.stdout
