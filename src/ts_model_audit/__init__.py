"""Small, reproducible diagnostics for time-series model validation."""

from .domain_shift import audit_domain_shift
from .validation import evaluate_regression_protocols

__all__ = ["audit_domain_shift", "evaluate_regression_protocols"]
