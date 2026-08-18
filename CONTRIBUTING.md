# Contributing

Thanks for helping improve Time-Series Model Audit.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
pytest
ruff check .
python examples/run_demo.py
```

## Pull requests

- Explain the problem and the intended behavior.
- Include a focused test for new or changed diagnostics.
- Include a reproducible example when changing report output.
- Keep changes compatible with Python 3.10 or newer.
- Do not commit private, customer, competition, or source datasets.
- Do not include credentials, API keys, tokens, or personal filesystem paths.

For larger changes, open an issue first so the design can be discussed before implementation.
