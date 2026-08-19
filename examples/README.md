# Examples

## Synthetic demo

The demo creates a small chronological dataset in memory, introduces intentional synthetic drift, and prints both domain-shift and validation results:

```bash
python examples/run_demo.py
```

It does not download data or read files outside the repository.

## CSV workflows

After installing the package, the CLI can produce JSON reports from your own local CSV files:

```bash
ts-model-audit domain-shift \
  --train data/train.csv \
  --test data/test.csv \
  --target target \
  --output reports/domain_shift.json

ts-model-audit regression-validation \
  --data data/series.csv \
  --target target \
  --n-splits 5 \
  --gap 12 \
  --output reports/validation.json
```

The input rows for regression validation must already be ordered chronologically. Keep private or restricted datasets outside this repository.

`--gap` is the number of rows excluded between each expanding training window and its validation window. Choose it from the prediction horizon and feature construction process. If it is omitted, the CLI uses a conservative size-based default capped at 25 rows.
