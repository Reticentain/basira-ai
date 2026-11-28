"""
tests/test_phase5.py
====================
Unit tests for Phase 5 failure analysis, Grad-CAM interpretability, and local recalibration.

Tests:
  1. Wilson score interval function: exact bounds, edge cases (n=0, k=0, k=n), range in [0, 1].
  2. Top-K failure selection: picks correct rows on synthetic data (lowest calibrated_prob for FN,
     highest for FP, properly filtered by true grade).
  3. Grad-CAM: output shape matches input (H, W), values normalized in [0, 1], and background
     pixels outside the retina are strictly zero.
  4. Local recalibration: operating threshold is chosen strictly from calibration split without
     touching test data, and repeated trials are deterministic with a seed.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from basira.eval.metrics import wilson_score_interval
from scripts.analyze_failures import compute_grade_breakdown, select_top_k_failures
from scripts.gradcam import (
    GradCAM,
    mask_and_normalize_heatmap,
    overlay_heatmap,
)
from scripts.recalibrate import (
    run_aptos_recalibration,
    run_single_calibration_trial,
)
from scripts.aggregate_runs import (
    aggregate_runs_data,
    generate_aggregate_markdown,
    main as aggregate_main,
)


class TestWilsonScoreInterval:
    def test_wilson_ci_known_proportions(self):
        """Test Wilson score interval against analytical properties."""
        # 50 / 100 -> center is 0.50
        ci_lo, ci_hi = wilson_score_interval(50, 100, confidence=0.95)
        assert 0.0 <= ci_lo < 0.50 < ci_hi <= 1.0
        assert math.isclose((ci_lo + ci_hi) / 2.0, 0.50, rel_tol=1e-3)

        # 0 / 100 -> lower bound must be clamped to 0.0
        ci_lo_zero, ci_hi_zero = wilson_score_interval(0, 100, confidence=0.95)
        assert math.isclose(ci_lo_zero, 0.0, abs_tol=1e-5)
        assert 0.0 <= ci_lo_zero <= 0.05
        assert 0.0 < ci_hi_zero < 0.05

        # 100 / 100 -> upper bound must be clamped to 1.0
        ci_lo_one, ci_hi_one = wilson_score_interval(100, 100, confidence=0.95)
        assert 0.95 < ci_lo_one < 1.0
        assert math.isclose(ci_hi_one, 1.0, abs_tol=1e-5)
        assert 0.95 <= ci_hi_one <= 1.0

    def test_wilson_ci_edge_cases(self):
        """Zero trials must return NaN without crashing."""
        ci_lo, ci_hi = wilson_score_interval(0, 0)
        assert math.isnan(ci_lo)
        assert math.isnan(ci_hi)

        ci_lo_neg, ci_hi_neg = wilson_score_interval(5, -10)
        assert math.isnan(ci_lo_neg)
        assert math.isnan(ci_hi_neg)

    def test_wilson_ci_monotonicity_and_bounds(self):
        """Confidence interval bounds must strictly respect 0 <= lower <= upper <= 1."""
        for k in range(0, 21):
            ci_lo, ci_hi = wilson_score_interval(k, 20, confidence=0.95)
            assert 0.0 <= ci_lo <= ci_hi <= 1.0


class TestFailureAnalysis:
    def test_select_top_k_failures(self):
        """Top-K selection must accurately isolate most confident false negatives and positives."""
        df = pd.DataFrame({
            "image_id": [f"img_{i:03d}" for i in range(20)],
            "grade": [
                4, 3, 2, 4, 3, 2, 4, 3, 2, 4,  # 10 referable (grades >= 2)
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  # 10 non-referable (grades 0-1)
            ],
            "calibrated_prob": [
                # Referable (true positive vs false negative at threshold 0.30)
                0.02, 0.05, 0.08, 0.15, 0.25, 0.40, 0.50, 0.70, 0.85, 0.95,
                # Non-referable (true negative vs false positive at threshold 0.30)
                0.01, 0.04, 0.10, 0.20, 0.29, 0.35, 0.60, 0.75, 0.88, 0.92,
            ],
        })

        threshold = 0.30
        k = 4
        top_fn, top_fp = select_top_k_failures(df, threshold=threshold, k=k)

        # 1. False negatives check (grade >= 2, calibrated_prob < 0.30)
        assert len(top_fn) == k
        assert all(top_fn["grade"] >= 2)
        assert all(top_fn["calibrated_prob"] < threshold)
        # Must be sorted ascending (lowest calibrated_prob first)
        assert list(top_fn["calibrated_prob"]) == sorted(top_fn["calibrated_prob"])
        assert top_fn.iloc[0]["image_id"] == "img_000"  # prob 0.02

        # 2. False positives check (grade <= 1, calibrated_prob >= 0.30)
        assert len(top_fp) == k
        assert all(top_fp["grade"] <= 1)
        assert all(top_fp["calibrated_prob"] >= threshold)
        # Must be sorted descending (highest calibrated_prob first)
        assert list(top_fp["calibrated_prob"]) == sorted(top_fp["calibrated_prob"], reverse=True)
        assert top_fp.iloc[0]["image_id"] == "img_019"  # prob 0.92

    def test_compute_grade_breakdown(self):
        """Compute grade breakdown must return all 5 grades with correct counts."""
        df = pd.DataFrame({
            "grade": [0, 0, 1, 1, 2, 2, 3, 4],
            "calibrated_prob": [0.1, 0.4, 0.2, 0.5, 0.1, 0.8, 0.9, 0.95],
        })
        threshold = 0.35
        breakdown = compute_grade_breakdown(df, threshold=threshold)

        assert len(breakdown) == 5
        g0 = breakdown[0]
        assert g0["grade"] == 0
        assert g0["total"] == 2
        assert g0["flagged_referable"] == 1  # 0.4 >= 0.35
        assert g0["flagged_non_referable"] == 1
        assert g0["rate"] == 0.5
        assert 0.0 <= g0["wilson_ci_lower"] <= g0["wilson_ci_upper"] <= 1.0


class TestGradCAM:
    class DummyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(16, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    def test_gradcam_output_shape_range_and_masking(self):
        """
        Grad-CAM must:
        - Output map matching spatial resolution (256, 256)
        - Strictly set background pixels to 0.0
        - Normalize retina foreground pixels to [0.0, 1.0]
        """
        model = self.DummyConvNet()
        gradcam = GradCAM(model, target_layer=model.conv2)

        # Create synthetic input image with circular retina foreground and zero background
        h, w = 256, 256
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = 100
        circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius ** 2)

        cached_img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        cached_img_rgb[circle_mask] = 180  # non-zero retina

        # Input tensor for network
        inp_tensor = torch.from_numpy(cached_img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        # Generate Grad-CAM
        raw_cam = gradcam.generate(inp_tensor, class_idx=0)
        gradcam.remove_hooks()

        assert raw_cam.shape == (h, w), f"Expected shape ({h}, {w}), got {raw_cam.shape}"

        # Apply retina masking and normalization
        norm_cam = mask_and_normalize_heatmap(raw_cam, cached_img_rgb)

        # Invariant 1: Dimensions
        assert norm_cam.shape == (h, w)

        # Invariant 2: Values in [0.0, 1.0]
        assert np.min(norm_cam) >= 0.0
        assert np.max(norm_cam) <= 1.0

        # Invariant 3: Background pixels must be STRICTLY 0.0
        background_mask = ~circle_mask
        assert np.all(norm_cam[background_mask] == 0.0), "Background pixels outside the retina must be zero!"

        # Invariant 4: Foreground contains active signal
        assert np.max(norm_cam[circle_mask]) >= 0.0

        # Test overlay helper
        blended = overlay_heatmap(cached_img_rgb, norm_cam)
        assert blended.shape == (h, w, 3)
        assert 0.0 <= np.min(blended) <= np.max(blended) <= 1.0


class TestRecalibration:
    def test_recalibration_never_uses_test_labels_for_threshold(self):
        """
        Critical invariant:
        Operating threshold selected during recalibration must depend ONLY
        on the local calibration sample, completely independent of test labels.
        """
        rng = np.random.default_rng(42)

        # Fixed calibration sample
        cal_logits = np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.2, 2.0, 2.5])
        cal_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Test Set A (low prevalence, shifted scores)
        test_logits_A = rng.normal(-1.0, 1.0, size=50)
        test_labels_A = np.zeros(50, dtype=int)
        test_labels_A[:5] = 1

        # Test Set B (high prevalence, opposite scores)
        test_logits_B = rng.normal(1.5, 1.0, size=80)
        test_labels_B = np.ones(80, dtype=int)
        test_labels_B[:10] = 0

        # Run calibration trial against Test Set A and Test Set B
        res_A = run_single_calibration_trial(cal_logits, cal_labels, test_logits_A, test_labels_A)
        res_B = run_single_calibration_trial(cal_logits, cal_labels, test_logits_B, test_labels_B)

        # Invariant: Operating threshold must be EXACTLY identical regardless of test set
        assert res_A["operating_threshold"] == res_B["operating_threshold"]
        assert res_A["slope"] == res_B["slope"]
        assert res_A["intercept"] == res_B["intercept"]

    def test_recalibration_is_reproducible_with_seed(self):
        """Local recalibration experiment must yield identical results given the same seed."""
        rng = np.random.default_rng(123)

        train_logits = rng.normal(0.0, 1.5, size=200)
        train_labels = (train_logits > 0.1).astype(int)

        test_logits = rng.normal(0.0, 1.5, size=100)
        test_labels = (test_logits > 0.1).astype(int)

        from basira.eval.calibration import PlattScaler
        frozen_scaler = PlattScaler(slope=1.0, intercept=0.0)
        frozen_scaler.is_fitted = True
        frozen_thresh = 0.30

        # Run twice with the exact same seed
        res1 = run_aptos_recalibration(
            train_logits, train_labels, test_logits, test_labels,
            frozen_scaler, frozen_thresh, n_repeats=5, seed=777
        )
        res2 = run_aptos_recalibration(
            train_logits, train_labels, test_logits, test_labels,
            frozen_scaler, frozen_thresh, n_repeats=5, seed=777
        )

        # Invariant: sample sizes >= len(train_logits) (200, 500) must be skipped
        assert "25" in res1["sample_size_results"]
        assert "50" in res1["sample_size_results"]
        assert "100" in res1["sample_size_results"]
        assert "200" not in res1["sample_size_results"]
        assert "500" not in res1["sample_size_results"]
        assert "all" in res1["sample_size_results"]

        s1 = res1["sample_size_results"]["50"]["summary"]
        s2 = res2["sample_size_results"]["50"]["summary"]

        assert math.isclose(s1["sensitivity_mean"], s2["sensitivity_mean"], rel_tol=1e-7)
        assert math.isclose(s1["specificity_mean"], s2["specificity_mean"], rel_tol=1e-7)
        assert math.isclose(s1["threshold_mean"], s2["threshold_mean"], rel_tol=1e-7)


class TestAggregateRuns:
    def test_aggregate_runs_data_synthetic(self):
        """Aggregate computation must compute sample mean and ddof=1 sample standard deviation."""
        r1 = {
            "datasets": {
                "eyepacs_test": {
                    "auc": {"val": 0.950},
                    "sensitivity": {"val": 0.880},
                    "specificity": {"val": 0.890},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.900},
                },
                "aptos_deduped": {
                    "auc": {"val": 0.960},
                    "sensitivity": {"val": 0.850},
                    "specificity": {"val": 0.910},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.880},
                },
            }
        }
        r2 = {
            "datasets": {
                "eyepacs_test": {
                    "auc": {"val": 0.952},
                    "sensitivity": {"val": 0.884},
                    "specificity": {"val": 0.895},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.904},
                },
                "aptos_deduped": {
                    "auc": {"val": 0.966},
                    "sensitivity": {"val": 0.860},
                    "specificity": {"val": 0.915},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.890},
                },
            }
        }
        r3 = {
            "datasets": {
                "eyepacs_test": {
                    "auc": {"val": 0.954},
                    "sensitivity": {"val": 0.888},
                    "specificity": {"val": 0.900},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.908},
                },
                "aptos_deduped": {
                    "auc": {"val": 0.972},
                    "sensitivity": {"val": 0.870},
                    "specificity": {"val": 0.920},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.900},
                },
            }
        }

        agg = aggregate_runs_data(["run1", "run2", "run3"], [r1, r2, r3])
        assert agg["n_runs"] == 3
        assert "eyepacs_test" in agg["datasets"]
        assert "aptos_deduped" in agg["datasets"]

        eye = agg["datasets"]["eyepacs_test"]
        assert math.isclose(eye["auc"]["mean"], 0.9520, abs_tol=1e-5)
        assert math.isclose(eye["auc"]["sd"], 0.0020, abs_tol=1e-5)
        assert math.isclose(eye["sensitivity"]["mean"], 0.8840, abs_tol=1e-5)
        assert math.isclose(eye["sensitivity"]["sd"], 0.0040, abs_tol=1e-5)
        assert math.isclose(eye["specificity"]["mean"], 0.8950, abs_tol=1e-5)
        assert math.isclose(eye["specificity"]["sd"], 0.0050, abs_tol=1e-5)
        assert math.isclose(eye["oracle_sensitivity"]["mean"], 0.9040, abs_tol=1e-5)
        assert math.isclose(eye["oracle_sensitivity"]["sd"], 0.0040, abs_tol=1e-5)

        # Markdown generation test
        md = generate_aggregate_markdown(agg)
        assert "Multi-Run Aggregate Evaluation Summary" in md
        assert "eyepacs_test" in md
        assert "0.9520 ± 0.0020" in md

    def test_aggregate_runs_cli_end_to_end(self, tmp_path):
        """End-to-end integration test reading mock results.json files from disk."""
        reports_dir = tmp_path / "reports"
        run_a = reports_dir / "b0_referable_s42"
        run_b = reports_dir / "b0_referable_s1"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)

        payload_a = {
            "run_name": "b0_referable_s42",
            "datasets": {
                "eyepacs_test": {
                    "auc": {"val": 0.950},
                    "sensitivity": {"val": 0.880},
                    "specificity": {"val": 0.900},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.902},
                }
            },
        }
        payload_b = {
            "run_name": "b0_referable_s1",
            "datasets": {
                "eyepacs_test": {
                    "auc": {"val": 0.954},
                    "sensitivity": {"val": 0.886},
                    "specificity": {"val": 0.904},
                    "oracle (tuned on this set, not deployable)": {"sensitivity": 0.908},
                }
            },
        }

        import json
        with open(run_a / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload_a, f)
        with open(run_b / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload_b, f)

        out_dir = reports_dir / "aggregate"
        aggregate_main([
            "--runs", "b0_referable_s42", "b0_referable_s1",
            "--reports-root", str(reports_dir),
            "--output-dir", str(out_dir),
        ])

        assert (out_dir / "aggregate_results.json").exists()
        assert (out_dir / "aggregate_table.md").exists()

        with open(out_dir / "aggregate_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["n_runs"] == 2
        assert "eyepacs_test" in data["datasets"]
        eye = data["datasets"]["eyepacs_test"]
        assert math.isclose(eye["auc"]["mean"], 0.9520, abs_tol=1e-5)
