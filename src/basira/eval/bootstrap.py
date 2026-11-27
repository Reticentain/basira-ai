"""
src/basira/eval/bootstrap.py
============================
Bootstrap confidence interval estimation for classification metrics.

Features:
  - Patient-level cluster bootstrap (resamples patients with replacement, keeping both
    eyes together) for datasets with patient identifiers (e.g. EyePACS).
  - Standard image-level bootstrap for screening sets without patient tracking
    (APTOS, Messidor-2, IDRiD).
  - Calculates 95% percentile confidence intervals for AUC, Sensitivity, Specificity,
    Positive Predictive Value (PPV), and Negative Predictive Value (NPV).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from basira.eval.metrics import compute_auc


def compute_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute point estimates for AUC, Sensitivity, Specificity, PPV, and NPV.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Continuous probabilities in [0, 1].
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    dict[str, float]
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()

    auc = compute_auc(y_t, y_p)

    preds = (y_p >= threshold).astype(np.int64)

    tp = float(np.sum((y_t == 1) & (preds == 1)))
    fp = float(np.sum((y_t == 0) & (preds == 1)))
    tn = float(np.sum((y_t == 0) & (preds == 0)))
    fn = float(np.sum((y_t == 1) & (preds == 0)))

    sens = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else float("nan")

    return {
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    patient_ids: Optional[list[Any] | np.ndarray] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """
    Compute 95% bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Continuous probabilities in [0, 1].
    threshold : float
        Fixed operating threshold.
    patient_ids : list or np.ndarray, optional
        Patient IDs. If provided and containing valid non-None strings, cluster
        bootstrap by patient is performed. Otherwise, image-level bootstrap is used.
    n_bootstrap : int, default 1000
        Number of bootstrap replications.
    seed : int, default 0
        Random seed for reproducible resampling.
    ci_level : float, default 0.95
        Coverage for confidence intervals.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping metric_name -> {'val': float, 'ci_lower': float, 'ci_upper': float}
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    n_samples = len(y_t)

    # Point estimates on full dataset
    point_est = compute_point_metrics(y_t, y_p, threshold)

    rng = np.random.default_rng(seed)

    # Determine whether patient cluster bootstrap is applicable
    use_patient_clustering = False
    if patient_ids is not None:
        p_list = [str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None for x in patient_ids]
        if any(p is not None for p in p_list):
            use_patient_clustering = True

    bootstrap_results: dict[str, list[float]] = {k: [] for k in point_est.keys()}

    if use_patient_clustering:
        # Build mapping patient_id -> row indices
        patient_indices: dict[str, list[int]] = {}
        for idx, pid in enumerate(p_list):
            key = pid if pid is not None else f"orphan_{idx}"
            if key not in patient_indices:
                patient_indices[key] = []
            patient_indices[key].append(idx)

        unique_patients = np.array(list(patient_indices.keys()))
        n_patients = len(unique_patients)

        for _ in range(n_bootstrap):
            sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
            sampled_idx_chunks = [patient_indices[p] for p in sampled_pids]
            sample_idx = np.concatenate(sampled_idx_chunks)

            sample_metrics = compute_point_metrics(y_t[sample_idx], y_p[sample_idx], threshold)
            for k, v in sample_metrics.items():
                if not np.isnan(v):
                    bootstrap_results[k].append(v)

    else:
        # Standard image-level bootstrap
        indices = np.arange(n_samples)
        for _ in range(n_bootstrap):
            sample_idx = rng.choice(indices, size=n_samples, replace=True)
            sample_metrics = compute_point_metrics(y_t[sample_idx], y_p[sample_idx], threshold)
            for k, v in sample_metrics.items():
                if not np.isnan(v):
                    bootstrap_results[k].append(v)

    # Calculate percentile confidence intervals
    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    final_report: dict[str, dict[str, float]] = {}
    for metric_name, point_val in point_est.items():
        boot_vals = bootstrap_results[metric_name]
        if len(boot_vals) > 0:
            ci_low = float(np.percentile(boot_vals, lower_pct))
            ci_high = float(np.percentile(boot_vals, upper_pct))
        else:
            ci_low = float("nan")
            ci_high = float("nan")

        final_report[metric_name] = {
            "val": float(point_val),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        }

    return final_report
