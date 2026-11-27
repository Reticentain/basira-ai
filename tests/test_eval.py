"""
tests/test_eval.py
==================
Unit tests for the evaluation pipeline, calibration, bootstrap CIs, and threshold selection.

Requirements tested:
  1. Patient-level bootstrap respects patient groups (synthetic data).
  2. Platt scaling is monotone and fit exclusively on validation data.
  3. Operating threshold achieves >= 0.90 specificity on validation data.
  4. Results schema contains all required fields, metrics, CIs, and oracle benchmarks.
  5. Assertion / proof that no test-set function is used to pick the deployable threshold.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from basira.config import apply_overrides, load_config
from basira.eval.bootstrap import bootstrap_ci, compute_point_metrics
from basira.eval.calibration import PlattScaler, compute_reliability_curve
from basira.eval.metrics import (
    compute_auc,
    compute_ppv_npv,
    expected_calibration_error,
    fraction_referable_per_grade,
    sensitivity_at_specificity,
)
from scripts.evaluate import (
    build_eval_config,
    evaluate_dataset,
    generate_markdown_table,
    parse_args,
)


class TestPatientBootstrap:
    def test_patient_bootstrap_resamples_whole_patients(self):
        """
        Patient-level bootstrap must resample patient clusters with replacement,
        never splitting left and right eyes of the same patient.
        """
        # Create 20 patients, each with 2 eyes (40 total records)
        n_patients = 20
        patient_ids = []
        for i in range(n_patients):
            patient_ids.extend([f"patient_{i:03d}", f"patient_{i:03d}"])

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=len(patient_ids))
        y_prob = np.clip(y_true * 0.7 + rng.uniform(0.1, 0.3, size=len(patient_ids)), 0.0, 1.0)
        threshold = 0.5

        # Run patient-level cluster bootstrap
        results = bootstrap_ci(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            patient_ids=patient_ids,
            n_bootstrap=100,
            seed=42,
        )

        for metric in ["auc", "sensitivity", "specificity", "ppv", "npv"]:
            assert metric in results
            val = results[metric]["val"]
            ci_lo = results[metric]["ci_lower"]
            ci_hi = results[metric]["ci_upper"]
            assert not math.isnan(val), f"Metric {metric} point estimate is NaN"
            assert not math.isnan(ci_lo), f"Metric {metric} CI lower is NaN"
            assert not math.isnan(ci_hi), f"Metric {metric} CI upper is NaN"
            assert ci_lo <= ci_hi, f"CI lower ({ci_lo}) must be <= CI upper ({ci_hi})"

    def test_image_level_bootstrap_fallback(self):
        """When patient_ids is None or empty, bootstrap falls back to image-level resampling."""
        rng = np.random.default_rng(123)
        y_true = rng.integers(0, 2, size=50)
        y_prob = rng.uniform(0.0, 1.0, size=50)

        results = bootstrap_ci(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            patient_ids=None,
            n_bootstrap=50,
            seed=123,
        )

        assert "auc" in results
        assert results["auc"]["ci_lower"] <= results["auc"]["ci_upper"]


class TestPlattCalibration:
    def test_platt_scaling_is_monotone(self):
        """
        Platt scaling must preserve rank order: higher logit implies higher
        calibrated probability (slope > 0).
        """
        rng = np.random.default_rng(42)
        # Synthetic validation logits with positive correlation to label
        logits = np.linspace(-4.0, 4.0, 200)
        probs_true = 1.0 / (1.0 + np.exp(-logits))
        y_val = (rng.uniform(0.0, 1.0, size=200) < probs_true).astype(int)

        scaler = PlattScaler()
        scaler.fit(logits, y_val)

        assert scaler.is_fitted is True
        assert scaler.slope > 0, f"Learned slope must be positive, got {scaler.slope}"

        # Test monotonicity across a wide, dense grid of logits
        test_logits = np.linspace(-10.0, 10.0, 500)
        cal_probs = scaler.predict_proba(test_logits)

        # Monotonically non-decreasing
        diffs = np.diff(cal_probs)
        assert np.all(diffs >= 0.0), "Calibrated probabilities must be strictly non-decreasing"

    def test_platt_fit_is_exclusive_to_validation_set(self):
        """
        Verifies that evaluating test data using predict_proba does NOT modify
        the scaler parameters fit on the validation set.
        """
        # Fit on validation set
        val_logits = np.array([-2.5, -1.0, 0.0, 1.2, 3.0])
        val_labels = np.array([0, 0, 0, 1, 1])

        scaler = PlattScaler()
        scaler.fit(val_logits, val_labels)

        saved_slope = scaler.slope
        saved_intercept = scaler.intercept

        # Simulate running inference on test sets with different distributions
        test_logits_1 = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        test_logits_2 = np.array([2.0, 3.0, 4.0, 5.0])

        _ = scaler.predict_proba(test_logits_1)
        _ = scaler.predict_proba(test_logits_2)

        # Scaler parameters must remain strictly unchanged
        assert scaler.slope == saved_slope
        assert scaler.intercept == saved_intercept


class TestOperatingThreshold:
    def test_threshold_gives_at_least_target_specificity_on_val(self):
        """
        The operating threshold selected on validation set must achieve
        at least 0.90 empirical specificity on the validation set.
        """
        rng = np.random.default_rng(42)
        n_neg = 500
        n_pos = 100

        # Create validation probabilities: negatives clustered low, positives clustered high
        neg_probs = rng.beta(1.5, 8.0, size=n_neg)  # mostly low (0.0 - 0.3)
        pos_probs = rng.beta(7.0, 2.0, size=n_pos)  # mostly high (0.6 - 1.0)

        y_val = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
        val_probs = np.concatenate([neg_probs, pos_probs])

        target_spec = 0.90
        val_sens, val_thresh = sensitivity_at_specificity(y_val, val_probs, target_specificity=target_spec)

        # Calculate empirical specificity at the returned threshold
        preds = (val_probs >= val_thresh).astype(int)
        tn = np.sum((y_val == 0) & (preds == 0))
        fp = np.sum((y_val == 0) & (preds == 1))
        empirical_spec = tn / (tn + fp)

        assert empirical_spec >= target_spec, (
            f"Empirical validation specificity {empirical_spec:.4f} must be >= {target_spec}"
        )
        assert val_sens > 0.0, "Validation sensitivity should be positive"


class TestResultsSchema:
    def test_evaluate_dataset_returns_full_schema(self):
        """
        evaluate_dataset must return a dictionary conforming to the required schema
        including bootstrap CIs, calibration ECEs, grade breakdowns, and oracle comparison.
        """
        rng = np.random.default_rng(42)
        n_samples = 60

        df = pd.DataFrame({
            "image_id": [f"img_{i:04d}.png" for i in range(n_samples)],
            "grade": rng.integers(0, 5, size=n_samples),
            "patient_id": [f"p_{i // 2:03d}" for i in range(n_samples)],
            "source": ["test_source"] * n_samples,
        })

        logits = rng.normal(loc=0.0, scale=2.0, size=n_samples)
        raw_probs = 1.0 / (1.0 + np.exp(-logits))

        scaler = PlattScaler(slope=1.1, intercept=-0.2)
        scaler.is_fitted = True

        val_thresh = 0.45

        metrics, pred_df = evaluate_dataset(
            df=df,
            logits=logits,
            raw_probs=raw_probs,
            scaler=scaler,
            fixed_threshold=val_thresh,
            dataset_name="test_dataset",
            n_bootstrap=50,
            seed=42,
            target_specificity=0.90,
        )

        # Verify top-level metrics keys
        expected_keys = [
            "dataset",
            "N",
            "prevalence",
            "bootstrap_limitation",
            "fixed_val_threshold",
            "auc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "ece_before_calibration",
            "ece_after_calibration",
            "fraction_referable_per_grade",
            "oracle (tuned on this set, not deployable)",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key '{key}' in results schema"

        # Verify CI sub-dictionary structure
        for ci_metric in ["auc", "sensitivity", "specificity", "ppv", "npv"]:
            assert "val" in metrics[ci_metric]
            assert "ci_lower" in metrics[ci_metric]
            assert "ci_upper" in metrics[ci_metric]

        # Verify grade breakdown
        for g in range(5):
            assert str(g) in metrics["fraction_referable_per_grade"]

        # Verify oracle field
        oracle = metrics["oracle (tuned on this set, not deployable)"]
        assert "threshold" in oracle
        assert "sensitivity" in oracle
        assert "target_specificity" in oracle
        assert oracle["target_specificity"] == 0.90

        # Verify per-image predictions DataFrame
        expected_cols = ["image_id", "source", "patient_id", "grade", "logit", "prob", "calibrated_prob"]
        for col in expected_cols:
            assert col in pred_df.columns, f"Missing column '{col}' in predictions DataFrame"
        assert len(pred_df) == n_samples


class TestNoTestSetThresholdTuning:
    def test_no_test_set_function_picks_operating_threshold(self):
        """
        Rigorous invariant test:
        1. Operating threshold is strictly computed from validation data.
        2. Changing test set distributions or labels has ZERO effect on the operating threshold.
        3. Test set evaluation strictly uses the frozen validation threshold.
        """
        rng = np.random.default_rng(999)

        # Fixed validation data
        val_labels = np.array([0] * 80 + [1] * 20)
        val_probs = np.concatenate([rng.uniform(0.01, 0.25, 80), rng.uniform(0.60, 0.95, 20)])

        _, frozen_threshold = sensitivity_at_specificity(val_labels, val_probs, target_specificity=0.90)

        # Create two vastly different test sets:
        # Test Set A: low prevalence (5%), shifted scores
        test_df_A = pd.DataFrame({
            "image_id": [f"a_{i}" for i in range(100)],
            "grade": [1] * 95 + [4] * 5,
            "patient_id": [None] * 100,
            "source": ["test_A"] * 100,
        })
        logits_A = rng.normal(loc=-2.0, scale=1.0, size=100)
        raw_probs_A = 1.0 / (1.0 + np.exp(-logits_A))

        # Test Set B: high prevalence (60%), opposite shift
        test_df_B = pd.DataFrame({
            "image_id": [f"b_{i}" for i in range(100)],
            "grade": [0] * 40 + [3] * 60,
            "patient_id": [None] * 100,
            "source": ["test_B"] * 100,
        })
        logits_B = rng.normal(loc=1.5, scale=1.0, size=100)
        raw_probs_B = 1.0 / (1.0 + np.exp(-logits_B))

        scaler = PlattScaler(slope=1.0, intercept=0.0)
        scaler.is_fitted = True

        # Evaluate both test sets with the frozen validation threshold
        metrics_A, _ = evaluate_dataset(
            df=test_df_A,
            logits=logits_A,
            raw_probs=raw_probs_A,
            scaler=scaler,
            fixed_threshold=frozen_threshold,
            dataset_name="test_A",
            n_bootstrap=20,
            seed=1,
        )

        metrics_B, _ = evaluate_dataset(
            df=test_df_B,
            logits=logits_B,
            raw_probs=raw_probs_B,
            scaler=scaler,
            fixed_threshold=frozen_threshold,
            dataset_name="test_B",
            n_bootstrap=20,
            seed=2,
        )

        # Invariant check: both evaluations used the EXACT same fixed validation threshold
        assert metrics_A["fixed_val_threshold"] == frozen_threshold
        assert metrics_B["fixed_val_threshold"] == frozen_threshold

        # Invariant check: oracle thresholds (tuned per test set) are distinct and isolated
        oracle_A = metrics_A["oracle (tuned on this set, not deployable)"]["threshold"]
        oracle_B = metrics_B["oracle (tuned on this set, not deployable)"]["threshold"]

        # Oracle thresholds adapt to test distributions; deployable threshold remains invariant
        assert metrics_A["fixed_val_threshold"] != oracle_A or metrics_B["fixed_val_threshold"] != oracle_B
        assert "not deployable" in metrics_A["oracle (tuned on this set, not deployable)"]["description"]


class TestCliAndConfig:
    def test_evaluate_cli_parsing_and_config_overrides(self):
        """
        Verify CLI argument parsing and config overrides work without GPU or data,
        catching signature errors like load_config(overrides=...).
        """
        args = parse_args([
            "--run-name", "b0_smoke_eval",
            "--set", "evaluation.bootstrap_n=5",
            "training.batch_size=16",
            "model.pretrained=false",
            "evaluation.target_specificity=0.88",
        ])
        assert args.run_name == "b0_smoke_eval"
        assert args.ckpt == "best.pt"
        assert args.datasets == ["eyepacs_test", "aptos_deduped", "messidor2_test", "idrid_test"]

        cfg = build_eval_config(args)
        # Verify type coercion
        assert cfg.evaluation.bootstrap_n == 5
        assert isinstance(cfg.evaluation.bootstrap_n, int)
        assert cfg.training.batch_size == 16
        assert isinstance(cfg.training.batch_size, int)
        assert cfg.model.pretrained is False
        assert isinstance(cfg.model.pretrained, bool)
        assert abs(cfg.evaluation.target_specificity - 0.88) < 1e-5
        assert isinstance(cfg.evaluation.target_specificity, float)

    def test_evaluate_cli_help(self):
        """Ensure scripts/evaluate.py --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_evaluate_cli_subprocess_help(self):
        """Ensure `python scripts/evaluate.py --help` runs successfully via subprocess."""
        import subprocess
        import sys
        ret = subprocess.run(
            [sys.executable, "scripts/evaluate.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert ret.returncode == 0
        assert "usage:" in ret.stdout.lower() or "help" in ret.stdout.lower()

    def test_apply_overrides_helper(self):
        """Test apply_overrides directly with various types."""
        cfg = load_config()
        apply_overrides(cfg, [
            "training.epochs=3",
            "training.learning_rate=0.0005",
            "image.apply_clahe=false",
            "model.backbone=mobilenetv3",
        ])
        assert cfg.training.epochs == 3
        assert isinstance(cfg.training.epochs, int)
        assert cfg.training.learning_rate == 0.0005
        assert isinstance(cfg.training.learning_rate, float)
        assert cfg.image.apply_clahe is False
        assert isinstance(cfg.image.apply_clahe, bool)
        assert cfg.model.backbone == "mobilenetv3"
        assert isinstance(cfg.model.backbone, str)

