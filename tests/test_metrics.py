"""
tests/test_metrics.py
=====================
Unit tests for evaluation metrics (AUC, sensitivity at specificity, ECE, QWK).
Includes tests for:
  - Perfect classifiers
  - Random / imperfect classifiers
  - All-one-class edge cases that must not crash
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from basira.eval.metrics import (
    compute_auc,
    expected_calibration_error,
    quadratic_weighted_kappa,
    sensitivity_at_specificity,
)


class TestMetrics:
    def test_auc_perfect_classifier(self):
        """Perfect ranking yields AUC of 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
        auc = compute_auc(y_true, y_prob)
        assert auc == 1.0

    def test_auc_random_classifier(self):
        """Random probabilities yield AUC reasonably close to 0.5."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=500)
        y_prob = rng.uniform(0.0, 1.0, size=500)
        auc = compute_auc(y_true, y_prob)
        assert 0.40 <= auc <= 0.60

    def test_auc_all_one_class_does_not_crash(self):
        """All-zeros or all-ones ground truth must return NaN without crashing."""
        y_zeros = np.zeros(20, dtype=int)
        y_ones = np.ones(20, dtype=int)
        probs = np.linspace(0.1, 0.9, 20)

        auc_zeros = compute_auc(y_zeros, probs)
        assert math.isnan(auc_zeros)

        auc_ones = compute_auc(y_ones, probs)
        assert math.isnan(auc_ones)

    def test_sensitivity_at_specificity_perfect(self):
        """A well-separated classifier reaches sensitivity 1.0 at 90% specificity."""
        y_true = np.array([0] * 50 + [1] * 50)
        # Perfect separation
        y_prob = np.array([0.1] * 50 + [0.9] * 50)

        sens, thresh = sensitivity_at_specificity(y_true, y_prob, target_specificity=0.90)
        assert sens == 1.0
        assert 0.1 <= thresh <= 0.9

    def test_sensitivity_at_specificity_random(self):
        """Random classifier produces valid floats in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_prob = rng.uniform(0.0, 1.0, size=200)

        sens, thresh = sensitivity_at_specificity(y_true, y_prob, target_specificity=0.90)
        assert not math.isnan(sens)
        assert 0.0 <= sens <= 1.0
        assert 0.0 <= thresh <= 1.0

    def test_sensitivity_at_specificity_all_one_class_does_not_crash(self):
        """Ground truth with a single class must not crash."""
        y_zeros = np.zeros(30, dtype=int)
        y_prob = np.linspace(0.1, 0.8, 30)

        sens, thresh = sensitivity_at_specificity(y_zeros, y_prob, target_specificity=0.90)
        assert math.isnan(sens)
        assert math.isnan(thresh)

    def test_ece_perfect_and_uncalibrated(self):
        """ECE is 0.0 for perfectly calibrated binary output, and high for overconfident errors."""
        # Perfectly calibrated: probs match true labels exactly
        y_true_perf = np.array([0, 0, 1, 1])
        y_prob_perf = np.array([0.0, 0.0, 1.0, 1.0])
        assert expected_calibration_error(y_true_perf, y_prob_perf) == 0.0

        # Severely uncalibrated: all true are 0, but model predicts 1.0
        y_true_bad = np.zeros(50)
        y_prob_bad = np.ones(50)
        ece_bad = expected_calibration_error(y_true_bad, y_prob_bad)
        assert pytest.approx(ece_bad, abs=1e-5) == 1.0

    def test_ece_all_one_class_does_not_crash(self):
        """ECE runs without error when all labels are 0 or 1."""
        y_ones = np.ones(40, dtype=int)
        y_prob = np.full(40, 0.75)
        # Expected ECE = |1.0 - 0.75| = 0.25
        ece = expected_calibration_error(y_ones, y_prob)
        assert pytest.approx(ece, abs=1e-4) == 0.25

    def test_qwk_perfect_and_imperfect(self):
        """QWK is 1.0 for perfect prediction and lower for errors."""
        y_true = np.array([0, 1, 2, 3, 4, 1, 2, 3])
        # Perfect
        assert quadratic_weighted_kappa(y_true, y_true, num_classes=5) == 1.0

        # Close errors (off by 1)
        y_pred_close = np.array([0, 1, 2, 2, 4, 1, 3, 3])
        qwk_close = quadratic_weighted_kappa(y_true, y_pred_close, num_classes=5)
        assert 0.70 <= qwk_close < 1.0

    def test_qwk_all_one_class_does_not_crash(self):
        """Constant arrays must not crash."""
        # Perfect agreement on class 0
        y_const = np.zeros(20, dtype=int)
        assert quadratic_weighted_kappa(y_const, y_const, num_classes=5) == 1.0

        # Complete disagreement
        y_other = np.ones(20, dtype=int)
        qwk_disagree = quadratic_weighted_kappa(y_const, y_other, num_classes=5)
        assert qwk_disagree <= 0.0
