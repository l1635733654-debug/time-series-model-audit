# Time-Series Model Audit

Reproducible diagnostics for time-series machine-learning projects.

This project helps answer two questions before model tuning:

1. Is the validation protocol aligned with the way the model will be used?
2. Are the training and evaluation datasets different enough that a random split is misleading?

The toolkit provides:

- random, blocked, expanding-window, and gap-aware time-series validation;
- model comparisons against simple baselines;
- adversarial train/test domain-shift AUC;
- standardized-mean-difference and out-of-range clipping diagnostics;
- JSON reports that can be checked into CI or attached to a model review.

The repository contains no private, competition, customer, or financial dataset. The demo uses synthetic data only.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

## Quick start

Run the self-contained demo:

```bash
python examples/run_demo.py
```

Run a domain-shift audit on two tabular CSV files. Both files must contain the same numeric feature columns; the training file may also contain the target column.

```bash
python -m ts_model_audit.cli domain-shift \
  --train data/train.csv \
  --test data/test.csv \
  --output reports/domain_shift.json \
  --target target
```

Run validation on a CSV whose row order is chronological:

```bash
python -m ts_model_audit.cli regression-validation \
  --data data/series.csv \
  --target target \
  --output reports/validation.json
```

The CLI does not assume that a random split is a valid estimate of deployment performance. It reports several protocols side by side so the decision is visible.

## Interpreting the reports

- A high adversarial domain AUC means the train/test boundary is easy to predict from features. Investigate drift, collection changes, leakage, or a mismatch between the evaluation window and deployment.
- A large gap between random and expanding-window performance means the model may be benefiting from future-like examples in a random split.
- The toolkit reports diagnostics; it does not decide that a model is deployable.
- For financial data, this project is an evaluation tool, not investment advice or a trading strategy.

## Development

```bash
pytest
ruff check .
```

The project is intentionally small and dependency-light. Pull requests should include a test or a reproducible example for new metrics and should avoid committing source datasets.

## License

MIT. See [LICENSE](LICENSE).
