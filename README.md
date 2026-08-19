# Time-Series Model Audit

[![CI](https://github.com/l1635733654-debug/time-series-model-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/l1635733654-debug/time-series-model-audit/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/l1635733654-debug/time-series-model-audit?display_name=tag)](https://github.com/l1635733654-debug/time-series-model-audit/releases)

Current release: `v0.1.1`

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

See [use cases and limitations](docs/use-cases.md) for guidance on when these diagnostics are appropriate.

## Install

Python 3.10+ is supported.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Quick start

Run the self-contained demo:

```bash
python examples/run_demo.py
```

Run a domain-shift audit on two tabular CSV files. Both files must contain the same numeric feature columns; the training file may also contain the target column.

```bash
ts-model-audit domain-shift \
  --train data/train.csv \
  --test data/test.csv \
  --output reports/domain_shift.json \
  --target target
```

Run validation on a CSV whose row order is chronological:

```bash
ts-model-audit regression-validation \
  --data data/series.csv \
  --target target \
  --n-splits 5 \
  --gap 12 \
  --output reports/validation.json
```

You can also invoke the installed CLI as `python -m ts_model_audit`.

The CLI does not assume that a random split is a valid estimate of deployment performance. It reports several protocols side by side so the decision is visible.

The complete runnable example is documented in [examples/README.md](examples/README.md).

## Interpreting the reports

- A high adversarial domain AUC means the train/test boundary is easy to predict from features. Investigate drift, collection changes, leakage, or a mismatch between the evaluation window and deployment.
- A large gap between random and expanding-window performance means the model may be benefiting from future-like examples in a random split.
- `blocked_kfold` uses contiguous validation blocks but may train on rows from both sides of a block. Treat it as a diagnostic comparison, not as a deployment-safe chronological estimate.
- The toolkit reports diagnostics; it does not decide that a model is deployable.
- For financial data, this project is an evaluation tool, not investment advice or a trading strategy.

## Development

```bash
pytest
ruff check .
```

The project is intentionally small and dependency-light. Pull requests should include a test or a reproducible example for new metrics and should avoid committing source datasets.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
