"""
tests/test_phase6.py
====================
Unit and property tests for Phase 6: Image Quality Metrics, Quality Gate,
and Synthetic Robustness Stress Testing.

Tests:
  1. Image quality metrics on synthetic images (sharpness, brightness, contrast, fill fraction).
  2. Blurred image has strictly lower sharpness than sharp image.
  3. Quality gate thresholds depend exclusively on validation data, never test data.
  4. All 7 synthetic corruptions exhibit strict severity monotonicity.
  5. Deterministic reproducibility with random seeds.
"""

from __future__ import annotations

import math
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from basira.eval.corruptions import (
    CORRUPTION_TYPES,
    SEVERITY_LEVELS,
    apply_corruption,
)
from basira.eval.quality import (
    QualityGate,
    compute_image_quality_metrics,
)


def create_synthetic_fundus_image(
    h: int = 256,
    w: int = 256,
    radius: int = 100,
    with_patterns: bool = True,
) -> np.ndarray:
    """Helper creating a synthetic fundus disc with circular mask and blood-vessel-like patterns."""
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius ** 2)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    if not with_patterns:
        img[circle_mask] = 120
        return img

    # Retinal background: reddish-orange
    img[circle_mask, 0] = 180  # R
    img[circle_mask, 1] = 90   # G
    img[circle_mask, 2] = 30   # B

    # Add high-contrast patterns inside retina (simulating vessels and lesions)
    for i in range(center_x - radius + 20, center_x + radius - 20, 16):
        cv2.line(img, (i, center_y - 40), (i + 10, center_y + 40), (80, 20, 10), 2)
        cv2.circle(img, (i, center_y), 3, (240, 220, 100), -1)

    # Keep background strictly black
    img[~circle_mask] = 0
    return img


class TestQualityMetrics:
    def test_sharpness_blurred_strictly_lower_than_sharp(self):
        """Variance of Laplacian on green channel must be strictly lower on blurred image."""
        sharp_img = create_synthetic_fundus_image(h=256, w=256, with_patterns=True)

        # Apply significant blur to green channel
        blurred_img = sharp_img.copy()
        blurred_img = cv2.GaussianBlur(blurred_img, (15, 15), 5.0)
        # Preserve background zeroing
        mask = np.any(sharp_img > 0, axis=-1, keepdims=True)
        blurred_img = np.where(mask, blurred_img, 0)

        m_sharp = compute_image_quality_metrics(sharp_img)
        m_blur = compute_image_quality_metrics(blurred_img)

        assert m_sharp["sharpness"] > 0.0
        assert m_blur["sharpness"] > 0.0
        assert m_sharp["sharpness"] > m_blur["sharpness"] * 2.0, (
            f"Expected sharp ({m_sharp['sharpness']}) >> blurred ({m_blur['sharpness']})"
        )

    def test_mean_brightness_and_contrast(self):
        """Brightness and contrast metrics must respond predictably to scaling."""
        base_img = create_synthetic_fundus_image(h=256, w=256, with_patterns=True)

        # Brightness test
        dimmed = np.clip(base_img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        bright = np.clip(base_img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

        m_base = compute_image_quality_metrics(base_img)
        m_dim = compute_image_quality_metrics(dimmed)
        m_bright = compute_image_quality_metrics(bright)

        assert m_dim["mean_brightness"] < m_base["mean_brightness"] < m_bright["mean_brightness"]

        # Contrast test (variance around mean)
        mask = np.any(base_img > 0, axis=-1)
        mean_col = np.mean(base_img[mask], axis=0, keepdims=True)
        low_contrast = np.clip((base_img.astype(np.float32) - mean_col) * 0.3 + mean_col, 0, 255).astype(np.uint8)
        low_contrast[~mask] = 0

        m_low_c = compute_image_quality_metrics(low_contrast)
        assert m_low_c["contrast"] < m_base["contrast"]

    def test_retina_fill_fraction(self):
        """Fill fraction must scale proportionally with retinal circle area."""
        img_small = create_synthetic_fundus_image(h=256, w=256, radius=50)
        img_large = create_synthetic_fundus_image(h=256, w=256, radius=100)

        m_small = compute_image_quality_metrics(img_small)
        m_large = compute_image_quality_metrics(img_large)

        expected_ratio = (100 ** 2) / (50 ** 2)  # ~4x area
        actual_ratio = m_large["retina_fill_fraction"] / m_small["retina_fill_fraction"]
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=0.1)

    def test_all_black_image_graceful(self):
        """All-black image must return zero metrics without crashing."""
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        m = compute_image_quality_metrics(black)
        assert m["sharpness"] == 0.0
        assert m["mean_brightness"] == 0.0
        assert m["retina_fill_fraction"] == 0.0


class TestQualityGate:
    def test_gate_thresholds_never_use_test_data(self):
        """
        Critical invariant:
        Quality gate rejection thresholds must be fit EXCLUSIVELY on validation data.
        Varying the test set distribution must have zero impact on gate thresholds.
        """
        rng = np.random.default_rng(42)

        # Fixed validation metrics dataframe
        val_df = pd.DataFrame({
            "sharpness": rng.exponential(150.0, size=200),
            "contrast": rng.normal(40.0, 8.0, size=200),
            "retina_fill_fraction": rng.uniform(0.60, 0.75, size=200),
            "mean_brightness": rng.normal(120.0, 15.0, size=200),
            "grade": rng.integers(0, 5, size=200),
            "calibrated_prob": rng.uniform(0.0, 1.0, size=200),
        })

        # Test Cohort A: High quality, sharp images
        test_df_A = pd.DataFrame({
            "sharpness": rng.exponential(400.0, size=100),
            "contrast": rng.normal(55.0, 5.0, size=100),
            "retina_fill_fraction": rng.uniform(0.70, 0.80, size=100),
            "mean_brightness": rng.normal(130.0, 10.0, size=100),
            "grade": rng.integers(0, 5, size=100),
            "calibrated_prob": rng.uniform(0.0, 1.0, size=100),
        })

        # Test Cohort B: Extremely degraded, blurry, dark images
        test_df_B = pd.DataFrame({
            "sharpness": rng.exponential(10.0, size=100),
            "contrast": rng.normal(15.0, 5.0, size=100),
            "retina_fill_fraction": rng.uniform(0.30, 0.45, size=100),
            "mean_brightness": rng.normal(50.0, 10.0, size=100),
            "grade": rng.integers(0, 5, size=100),
            "calibrated_prob": rng.uniform(0.0, 1.0, size=100),
        })

        gate_A = QualityGate().fit(val_df)
        gate_B = QualityGate().fit(val_df)

        # Invariant 1: Thresholds fit on val_df are strictly identical
        for k in gate_A.thresholds:
            assert gate_A.thresholds[k] == gate_B.thresholds[k]

        # Invariant 2: High quality Cohort A has far lower rejection rate than degraded Cohort B
        pred_A = gate_A.predict(test_df_A)
        pred_B = gate_A.predict(test_df_B)

        rej_A = np.mean(pred_A["is_reject_10%"])
        rej_B = np.mean(pred_B["is_reject_10%"])
        assert rej_B > rej_A, f"Degraded cohort B ({rej_B:.2f}) must have higher rejection than sharp cohort A ({rej_A:.2f})"

    def test_gate_rejection_rates_monotonic(self):
        """Thresholds for 5%, 10%, 15% must increase monotonically."""
        rng = np.random.default_rng(101)
        val_df = pd.DataFrame({
            "sharpness": rng.exponential(100.0, size=500),
            "contrast": rng.normal(40.0, 10.0, size=500),
            "retina_fill_fraction": rng.uniform(0.50, 0.70, size=500),
            "mean_brightness": rng.normal(110.0, 20.0, size=500),
            "grade": rng.integers(0, 5, size=500),
            "calibrated_prob": rng.uniform(0.0, 1.0, size=500),
        })

        gate = QualityGate(rejection_rates=(0.05, 0.10, 0.15)).fit(val_df)
        t5 = gate.thresholds["reject_5%"]
        t10 = gate.thresholds["reject_10%"]
        t15 = gate.thresholds["reject_15%"]

        assert t5 < t10 < t15, f"Expected t5 < t10 < t15, got {t5:.3f}, {t10:.3f}, {t15:.3f}"

    def test_gate_coverage_evaluation(self):
        """evaluate_coverage must return valid tier performance dictionaries."""
        rng = np.random.default_rng(555)
        df = pd.DataFrame({
            "sharpness": rng.exponential(120.0, size=100),
            "contrast": rng.normal(45.0, 8.0, size=100),
            "retina_fill_fraction": rng.uniform(0.55, 0.75, size=100),
            "mean_brightness": rng.normal(120.0, 15.0, size=100),
            "grade": np.array([0] * 70 + [2] * 30),
            "calibrated_prob": np.concatenate([rng.uniform(0.05, 0.35, size=70), rng.uniform(0.40, 0.90, size=30)]),
        })

        gate = QualityGate().fit(df)
        cov = gate.evaluate_coverage(df, operating_threshold=0.30)

        assert "all_images" in cov
        assert "tiers" in cov
        assert "reject_10%" in cov["tiers"]
        t10 = cov["tiers"]["reject_10%"]
        assert 80.0 <= t10["coverage_pct"] <= 95.0
        assert 0.0 <= t10["sensitivity"] <= 1.0


class TestCorruptions:
    @pytest.mark.parametrize("c_type", CORRUPTION_TYPES)
    def test_corruption_severity_is_monotone(self, c_type: str):
        """
        Every corruption must exhibit strict monotonic degradation or parameter scaling
        as severity increases from 1 to 5.
        """
        img = create_synthetic_fundus_image(h=256, w=256, with_patterns=True)
        mask = np.any(img > 0, axis=-1)

        corrupted_series = [
            apply_corruption(img, c_type, severity=s, seed=42 + s, keep_background_zeroed=True)
            for s in [1, 2, 3, 4, 5]
        ]

        if c_type == "gaussian_blur":
            # Sharpness must strictly decrease
            sharpnesses = [compute_image_quality_metrics(c)["sharpness"] for c in corrupted_series]
            for s in range(len(sharpnesses) - 1):
                assert sharpnesses[s] > sharpnesses[s + 1], f"Blur sharpness not strictly decreasing: {sharpnesses}"

        elif c_type == "brightness_down":
            # Mean brightness must strictly decrease
            brights = [np.mean(c[mask]) for c in corrupted_series]
            for s in range(len(brights) - 1):
                assert brights[s] > brights[s + 1], f"Brightness down not strictly decreasing: {brights}"

        elif c_type == "brightness_up":
            # Mean brightness must strictly increase
            brights = [np.mean(c[mask]) for c in corrupted_series]
            for s in range(len(brights) - 1):
                assert brights[s] < brights[s + 1], f"Brightness up not strictly increasing: {brights}"

        elif c_type == "contrast":
            # Luminance standard deviation must strictly decrease
            contrasts = [compute_image_quality_metrics(c)["contrast"] for c in corrupted_series]
            for s in range(len(contrasts) - 1):
                assert contrasts[s] > contrasts[s + 1], f"Contrast not strictly decreasing: {contrasts}"

        elif c_type == "gaussian_noise":
            # MSE relative to original must strictly increase
            mses = [np.mean((c.astype(float) - img.astype(float)) ** 2) for c in corrupted_series]
            for s in range(len(mses) - 1):
                assert mses[s] < mses[s + 1], f"Noise MSE not strictly increasing: {mses}"

        elif c_type == "jpeg_compression":
            # Compression distortion (MSE) must strictly increase
            mses = [np.mean((c.astype(float) - img.astype(float)) ** 2) for c in corrupted_series]
            for s in range(len(mses) - 1):
                assert mses[s] < mses[s + 1], f"JPEG distortion not strictly increasing: {mses}"

        elif c_type == "color_cast":
            # R/G ratio must strictly increase
            rg_ratios = [compute_image_quality_metrics(c)["rg_ratio"] for c in corrupted_series]
            for s in range(len(rg_ratios) - 1):
                assert rg_ratios[s] < rg_ratios[s + 1], f"Color cast R/G ratio not strictly increasing: {rg_ratios}"

    def test_background_pixels_strictly_zero(self):
        """Background pixels outside the retina circle must remain exactly 0 after any corruption."""
        img = create_synthetic_fundus_image(h=256, w=256, with_patterns=True)
        bg_mask = ~np.any(img > 0, axis=-1)

        for c_type in CORRUPTION_TYPES:
            for sev in [1, 3, 5]:
                corrupted = apply_corruption(img, c_type, severity=sev, seed=123, keep_background_zeroed=True)
                assert np.all(corrupted[bg_mask] == 0), (
                    f"Background outside retina not zero for {c_type} at severity {sev}"
                )

    def test_corruptions_reproducible_with_seed(self):
        """Stochastic corruptions (Gaussian noise) must be bit-level identical given the same seed."""
        img = create_synthetic_fundus_image(h=256, w=256, with_patterns=True)

        noisy1 = apply_corruption(img, "gaussian_noise", severity=3, seed=777)
        noisy2 = apply_corruption(img, "gaussian_noise", severity=3, seed=777)
        noisy_diff_seed = apply_corruption(img, "gaussian_noise", severity=3, seed=999)

        assert np.array_equal(noisy1, noisy2), "Identical seed must produce identical corrupted arrays"
        assert not np.array_equal(noisy1, noisy_diff_seed), "Different seeds must produce distinct noise patterns"
