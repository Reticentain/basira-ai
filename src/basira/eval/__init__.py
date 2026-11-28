"""
basira.eval
===========
Evaluation metrics, calibration, and bootstrap confidence intervals for Basira AI.
"""

from basira.eval.bootstrap import bootstrap_ci, compute_point_metrics
from basira.eval.calibration import PlattScaler, compute_reliability_curve
from basira.eval.inference import (
    build_and_load_model,
    find_split_file,
    load_eval_checkpoint,
    run_inference,
)
from basira.eval.metrics import (
    compute_auc,
    compute_ppv_npv,
    expected_calibration_error,
    fraction_referable_per_grade,
    quadratic_weighted_kappa,
    sensitivity_at_specificity,
    wilson_score_interval,
)

__all__ = [
    "compute_auc",
    "sensitivity_at_specificity",
    "expected_calibration_error",
    "quadratic_weighted_kappa",
    "compute_ppv_npv",
    "fraction_referable_per_grade",
    "wilson_score_interval",
    "PlattScaler",
    "compute_reliability_curve",
    "bootstrap_ci",
    "compute_point_metrics",
    "find_split_file",
    "load_eval_checkpoint",
    "build_and_load_model",
    "run_inference",
]
