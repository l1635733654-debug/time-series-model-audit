# Use cases and limitations

## What this toolkit is for

Time-Series Model Audit is designed for chronological tabular machine-learning workflows where the order of observations matters. It helps a reviewer identify risks before comparing models or tuning hyperparameters.

Typical uses include:

- comparing random cross-validation with chronological validation;
- checking whether a train/test boundary is easy to predict from features;
- finding features whose distributions changed between two time windows;
- producing a JSON report for a model review or CI artifact;
- documenting why a validation protocol was selected.

## Interpreting the main diagnostics

### Adversarial domain AUC

The toolkit trains a classifier to distinguish training rows from test rows. An AUC near 0.5 means the two samples are difficult to distinguish using the selected numeric features. A larger AUC is a signal to investigate time-window drift, collection changes, missing-value behavior, or leakage.

This is a diagnostic signal, not a universal pass/fail threshold.

### Standardized mean difference

This reports the absolute difference in feature means divided by a pooled standard deviation. Large values identify features that deserve inspection. The result should be considered alongside plots, domain knowledge, and sample size.

### Chronological validation

Expanding-window and gap-aware protocols preserve the direction of time. A large performance gap between random and chronological splits can indicate that a random split is too optimistic for the intended deployment setting.

## What it does not do

- It does not prove that a model is leak-free or deployable.
- It does not choose a business-specific drift threshold.
- It does not replace domain review, monitoring, or backtesting.
- It does not provide investment advice or a trading strategy.
- It currently focuses on numeric tabular features and regression validation.

Use the reports to guide investigation, and record the final modeling decision together with its assumptions.
