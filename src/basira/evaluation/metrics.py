"""
src/basira/evaluation/metrics.py
===============================
Alias to basira.eval.metrics for backward compatibility.
"""

from basira.eval.metrics import (
    compute_auc,
    expected_calibration_error,
    quadratic_weighted_kappa,
    sensitivity_at_specificity,
)

__all__ = [
    "compute_auc",
    "sensitivity_at_specificity",
    "expected_calibration_error",
    "quadratic_weighted_kappa",
]
