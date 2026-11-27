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


def compute_ppv_npv(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float,
) -> tuple[float, float]:
    """
    Compute Positive Predictive Value (Precision) and Negative Predictive Value at fixed threshold.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0 or 1).
    y_prob : array-like of shape (N,)
        Predicted continuous probabilities in [0, 1].
    threshold : float
        Decision threshold.

    Returns
    -------
    ppv : float
        Positive Predictive Value (TP / (TP + FP)), or NaN if no positive predictions.
    npv : float
        Negative Predictive Value (TN / (TN + FN)), or NaN if no negative predictions.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    preds = (y_p >= threshold).astype(int)

    tp = float(np.sum((y_t == 1) & (preds == 1)))
    fp = float(np.sum((y_t == 0) & (preds == 1)))
    tn = float(np.sum((y_t == 0) & (preds == 0)))
    fn = float(np.sum((y_t == 1) & (preds == 0)))

    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    return ppv, npv


def fraction_referable_per_grade(
    y_grade: np.ndarray | list,
    y_pred_referable: np.ndarray | list,
    num_classes: int = 5,
) -> dict[int, float]:
    """
    Compute the empirical fraction of images flagged as referable per true DR grade (0 to 4).

    Parameters
    ----------
    y_grade : array-like of shape (N,)
        True integer grades in 0..num_classes-1.
    y_pred_referable : array-like of shape (N,)
        Binary predicted referable flags (1 if predicted referable, else 0).
    num_classes : int, default 5
        Total number of grades (default 5 for DR grades 0-4).

    Returns
    -------
    dict[int, float]
        Mapping from grade (0 to 4) to fraction flagged referable (or NaN if no samples).
    """
    grades = np.asarray(y_grade, dtype=int).ravel()
    preds = np.asarray(y_pred_referable, dtype=int).ravel()

    result: dict[int, float] = {}
    for g in range(num_classes):
        mask = (grades == g)
        count = int(np.sum(mask))
        if count > 0:
            result[g] = float(np.mean(preds[mask]))
        else:
            result[g] = float("nan")
    return result


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Handles edge cases (total == 0, successes == 0, successes == total)
    gracefully, returning values strictly bounded in [0.0, 1.0].

    Parameters
    ----------
    successes : int
        Number of positive events (successes).
    total : int
        Total number of trials.
    confidence : float, default 0.95
        Confidence level (e.g. 0.95 for 95% CI).

    Returns
    -------
    ci_lower : float
        Lower bound in [0.0, 1.0], or NaN if total <= 0.
    ci_upper : float
        Upper bound in [0.0, 1.0], or NaN if total <= 0.
    """
    if total <= 0:
        return float("nan"), float("nan")

    k = max(0, min(int(successes), int(total)))
    n = int(total)

    alpha = 1.0 - float(confidence)
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        if abs(confidence - 0.95) < 1e-4:
            z = 1.959963984540054
        elif abs(confidence - 0.99) < 1e-4:
            z = 2.5758293035489004
        elif abs(confidence - 0.90) < 1e-4:
            z = 1.6448536269514722
        else:
            z = 1.95996

    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))

    ci_lower = max(0.0, float(center - margin))
    ci_upper = min(1.0, float(center + margin))
    return ci_lower, ci_upper
