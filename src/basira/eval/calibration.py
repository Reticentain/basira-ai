"""
src/basira/eval/calibration.py
==============================
Probability calibration via Platt scaling (logistic regression on logits)
and reliability diagram computation.

Crucial project rule:
  Platt scaling must be fit EXCLUSIVELY on the validation set.
  Test sets are evaluated with the frozen scaler and never used to fit calibration.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Optional

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """
    Parametric probability calibration using Platt scaling:
        P(y = 1 | z) = 1 / (1 + exp(-(slope * z + intercept)))
    where z is the raw model logit before sigmoid.

    Attributes
    ----------
    slope : float
        Learned scale factor on logits (must be > 0 for a ranking-aligned model).
    intercept : float
        Learned bias term.
    is_fitted : bool
        Whether the scaler has been fit on validation logits.
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self.slope: float = slope
        self.intercept: float = intercept
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray | list[float], y_true: np.ndarray | list[int]) -> PlattScaler:
        """
        Fit Platt scaling on validation logits and binary labels.

        Parameters
        ----------
        logits : array-like of shape (N,)
            Raw model logits from validation set.
        y_true : array-like of shape (N,)
            Binary ground truth labels (0 or 1).
        """
        z = np.asarray(logits, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=np.int64).ravel()

        if len(z) != len(y):
            raise ValueError(f"Length mismatch: {len(z)} logits vs {len(y)} labels.")
        if len(np.unique(y)) < 2:
            raise ValueError("Platt scaling requires at least 2 classes in the validation set.")

        # Fit 1D unregularized logistic regression (C=1e5) on validation logits
        clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        clf.fit(z.reshape(-1, 1), y)

        self.slope = float(clf.coef_[0][0])
        self.intercept = float(clf.intercept_[0])
        self.is_fitted = True
        return self

    def predict_proba(self, logits: np.ndarray | list[float]) -> np.ndarray:
        """
        Apply learned Platt transformation to logits to produce calibrated probabilities.

        Parameters
        ----------
        logits : array-like of shape (N,)
            Raw model logits.

        Returns
        -------
        np.ndarray of shape (N,)
            Calibrated probabilities in [0.0, 1.0].
        """
        z = np.asarray(logits, dtype=np.float64)
        scaled = self.slope * z + self.intercept
        return expit(scaled)

    def to_dict(self) -> dict[str, Any]:
        """Export parameters as a serializable dictionary."""
        return {
            "method": "platt_scaling",
            "slope": self.slope,
            "intercept": self.intercept,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlattScaler:
        """Reconstruct scaler from dictionary."""
        scaler = cls(slope=float(data["slope"]), intercept=float(data["intercept"]))
        scaler.is_fitted = bool(data.get("is_fitted", True))
        return scaler

    def save(self, filepath: pathlib.Path | str) -> None:
        """Save scaler parameters atomically to JSON."""
        fp = pathlib.Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = fp.parent / f".tmp_{os.getpid()}_{fp.name}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp_path, fp)

    @classmethod
    def load(cls, filepath: pathlib.Path | str) -> PlattScaler:
        """Load scaler parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute reliability curve points (bin confidences, bin accuracies, bin counts) and ECE.

    Returns
    -------
    bin_confs : np.ndarray
        Mean predicted probability in each non-empty bin.
    bin_accs : np.ndarray
        Empirical true positive rate in each non-empty bin.
    bin_counts : np.ndarray
        Number of samples in each non-empty bin.
    ece : float
        Expected Calibration Error across all bins.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    total = len(y_t)

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_t[mask]))
            conf = float(np.mean(y_p[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
            ece += (count / total) * abs(acc - conf)

    return (
        np.array(bin_confs),
        np.array(bin_accs),
        np.array(bin_counts),
        float(ece),
    )
