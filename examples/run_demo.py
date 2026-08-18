"""Run the audit toolkit on synthetic data without downloading any dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_model_audit import audit_domain_shift, evaluate_regression_protocols  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    n_train, n_test = 240, 100
    time = np.arange(n_train + n_test)
    feature_1 = np.sin(time / 12.0) + rng.normal(0, 0.08, len(time))
    feature_2 = np.cos(time / 25.0) + rng.normal(0, 0.08, len(time))
    target = 0.8 * feature_1 - 0.35 * feature_2 + rng.normal(0, 0.1, len(time))
    frame = pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "target": target})
    train = frame.iloc[:n_train].copy()
    test = frame.iloc[n_train:].copy()
    test["feature_2"] += 0.65  # intentional synthetic drift

    shift = audit_domain_shift(train, test, target="target")
    validation = evaluate_regression_protocols(
        train.drop(columns=["target"]), train["target"]
    )
    print("Domain-shift AUC:", round(shift["adversarial_oof_auc"], 4))
    print(
        "Expanding-window BayesianRidge RMSE:",
        round(validation["validation"]["expanding_timeseries"]["BayesianRidge"]["rmse_weighted"], 4),
    )
    print("Full report preview:")
    print(json.dumps({"shift": shift, "validation": validation}, indent=2)[:3000])


if __name__ == "__main__":
    main()
