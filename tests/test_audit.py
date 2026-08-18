from __future__ import annotations

import numpy as np
import pandas as pd

from ts_model_audit import audit_domain_shift, evaluate_regression_protocols


def test_domain_shift_report_is_serializable():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=120), "y": rng.normal(size=120), "target": 0})
    test = pd.DataFrame({"x": rng.normal(1.5, size=60), "y": rng.normal(size=60)})
    report = audit_domain_shift(train, test, target="target")
    assert report["train_rows"] == 120
    assert report["test_rows"] == 60
    assert report["feature_count"] == 2
    assert report["adversarial_oof_auc"] > 0.5


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
