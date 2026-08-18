"""Command-line interface for the audit toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .domain_shift import audit_domain_shift
from .validation import evaluate_regression_protocols


def _write(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ts-model-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shift = subparsers.add_parser("domain-shift", help="audit train/test distribution shift")
    shift.add_argument("--train", type=Path, required=True)
    shift.add_argument("--test", type=Path, required=True)
    shift.add_argument("--output", type=Path, required=True)
    shift.add_argument("--target", default=None)

    validation = subparsers.add_parser(
        "regression-validation", help="compare chronological validation protocols"
    )
    validation.add_argument("--data", type=Path, required=True)
    validation.add_argument("--target", required=True)
    validation.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "domain-shift":
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)
        _write(audit_domain_shift(train, test, target=args.target), args.output)
    elif args.command == "regression-validation":
        data = pd.read_csv(args.data)
        if args.target not in data.columns:
            raise SystemExit(f"target column not found: {args.target}")
        report = evaluate_regression_protocols(data.drop(columns=[args.target]), data[args.target])
        _write(report, args.output)
