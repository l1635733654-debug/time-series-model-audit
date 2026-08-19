# Changelog

All notable changes to this project are documented here.

## [0.1.1] - 2026-08-19

### Fixed

- Made `python -m ts_model_audit` and `python -m ts_model_audit.cli` execute the CLI.
- Guaranteed that CLI reports use strict JSON without `NaN` values.
- Added clear validation errors for undersized datasets and invalid split settings.
- Preserved all-missing features safely inside fold-fitted imputers.
- Corrected validation pipeline documentation that incorrectly mentioned clipping.

### Added

- Configurable `--n-splits` and `--gap` options for regression validation.
- CLI, strict-JSON, missing-value, and input-boundary tests.
- CI coverage for Python 3.10 through 3.13 and package build verification.
- Package metadata, issue forms, a pull-request template, and Dependabot configuration.

## [0.1.0] - 2026-08-19

Initial public release.

### Added

- Adversarial domain-shift AUC for train/test diagnostics.
- Standardized mean difference and train-only clipping-rate reports.
- Random, blocked, expanding-window, and gap-aware validation protocols.
- Baseline model comparisons with fold-level JSON metrics.
- Command-line interface for CSV inputs and JSON reports.
- Synthetic-data demonstration that requires no private dataset.
- Automated pytest and Ruff checks through GitHub Actions.
