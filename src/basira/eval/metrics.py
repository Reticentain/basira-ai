"""
src/basira/eval/metrics.py
==========================
Evaluation metrics for Diabetic Retinopathy classification:
  - Area Under the ROC Curve (AUC)
  - Sensitivity at a fixed specificity (e.g. 90% for clinical screening)
  - Expected Calibration Error (ECE)
  - Quadratic Weighted Kappa (QWK)

All functions handle edge cases (e.g. all-one-class or empty inputs) gracefully
without crashing or raising uncaught exceptions.
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve


def compute_auc(y_true: np.ndarray | list, y_prob: np.ndarray | list) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0 or 1).
    y_prob : array-like of shape (N,)
        Predicted probabilities of the positive class in [0, 1].

    Returns
    -------
    float
        ROC-AUC score, or float('nan') if fewer than 2 classes are present.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob).ravel()

    if len(y_t) == 0:
        return float("nan")

    unique_classes = np.unique(y_t)
    if len(unique_classes) < 2:
        warnings.warn(
            f"compute_auc: only {len(unique_classes)} class present in y_true. "
            "Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    try:
        return float(roc_auc_score(y_t, y_p))
    except Exception as exc:
        warnings.warn(f"compute_auc failed with error: {exc}. Returning NaN.", UserWarning, stacklevel=2)
        return float("nan")


def sensitivity_at_specificity(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    target_specificity: float = 0.90,
) -> tuple[float, float]:
    """
    Compute Sensitivity (True Positive Rate) at a fixed target Specificity (1 - FPR).

    The decision threshold is chosen as the threshold achieving specificity >= target_specificity
    with the maximum sensitivity.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0 or 1).
    y_prob : array-like of shape (N,)
        Predicted probabilities of positive class in [0, 1].
    target_specificity : float, default 0.90
        Desired minimum specificity (e.g. 0.90 for clinical screening).

    Returns
    -------
    sensitivity : float
        True positive rate at the selected operating point (or NaN if undefined).
    threshold : float
        Operating probability threshold selected on the provided predictions.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob).ravel()

    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        warnings.warn(
            "sensitivity_at_specificity: ground truth has fewer than 2 classes. Returning (NaN, NaN).",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    specificity = 1.0 - fpr

    # Filter points meeting or exceeding the target specificity
    valid_idx = np.where(specificity >= target_specificity)[0]

    if len(valid_idx) == 0:
        # Fallback: take point with highest specificity available
        best_idx = int(np.argmax(specificity))
    else:
        # Among points with specificity >= target, pick the one with highest sensitivity
        best_sub_idx = np.argmax(tpr[valid_idx])
        best_idx = int(valid_idx[best_sub_idx])

    sens = float(tpr[best_idx])
    thresh = float(thresholds[best_idx])

    # Bound threshold to [0.0, 1.0] if sklearn added 1 + max(y_prob) sentinel
    thresh = min(1.0, max(0.0, thresh))

    return sens, thresh


def expected_calibration_error(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width probability bins.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0 or 1).
    y_prob : array-like of shape (N,)
        Predicted confidence probabilities in [0, 1].
    n_bins : int, default 10
        Number of equal-width bins spanning [0, 1].

    Returns
    -------
    float
        ECE in range [0, 1], or 0.0 if empty.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob).ravel()

    if len(y_t) == 0:
        return 0.0

    # Ensure probabilities are clipped to [0, 1]
    y_p = np.clip(y_p, 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_t)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_t[in_bin])
            bin_conf = np.mean(y_p[in_bin])
            weight = bin_count / total_samples
            ece += weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def quadratic_weighted_kappa(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    num_classes: int = 5,
) -> float:
    """
    Compute Cohen's Quadratic Weighted Kappa (QWK) across ordinal grades (0 to num_classes-1).

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground truth integer grades (e.g. 0 to 4).
    y_pred : array-like of shape (N,)
        Predicted integer grades (e.g. 0 to 4).
    num_classes : int, default 5
        Total number of ordinal classes.

    Returns
    -------
    float
        QWK score in [-1.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    if len(y_t) == 0:
        return float("nan")

    # If predictions and targets are identical, kappa is 1.0
    if np.array_equal(y_t, y_p):
        return 1.0

    # If either array has only one class and they don't match, or undefined
    labels = list(range(num_classes))
    try:
        score = cohen_kappa_score(y_t, y_p, labels=labels, weights="quadratic")
        if np.isnan(score):
            return 0.0
        return float(score)
    except Exception as exc:
        warnings.warn(f"quadratic_weighted_kappa failed with {exc}. Returning 0.0.", UserWarning, stacklevel=2)
        return 0.0
