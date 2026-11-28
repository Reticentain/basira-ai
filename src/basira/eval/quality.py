"""
src/basira/eval/quality.py
==========================
Image quality metrics and Quality Gate for fundus image validation.

Computes retina-specific image quality metrics:
  - Sharpness: Variance of Laplacian on the green channel (eroded interior retina).
  - Mean Brightness: Average intensity across retina pixels.
  - Contrast: Standard deviation of luminance across retina pixels.
  - Retina Fill Fraction: Fraction of total image pixels covered by the retina disc.
  - Mean R, G, B & Color Ratios: Quantifying camera color cast and spectral shifts.

Quality Gate:
  Rule-based scoring and rejection thresholds fit EXCLUSIVELY on EyePACS VAL.
  Evaluates coverage vs. diagnostic performance trade-offs on test sets.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

from basira.eval.metrics import compute_auc


def compute_image_quality_metrics(
    img_rgb: np.ndarray,
    fill_holes: bool = True,
) -> dict[str, float]:
    """
    Compute comprehensive quality metrics on a single fundus image (H, W, 3).
    Metrics are computed strictly over retina pixels.

    Parameters
    ----------
    img_rgb : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    fill_holes : bool, default True
        Whether to fill interior holes in the retina mask before analysis.

    Returns
    -------
    dict[str, float]
        Dictionary of computed quality metrics.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {img_rgb.shape}")

    h, w = img_rgb.shape[:2]
    total_pixels = h * w

    # 1. Retina mask
    retina_mask = np.any(img_rgb > 0, axis=-1)
    if fill_holes and np.any(retina_mask):
        try:
            retina_mask = binary_fill_holes(retina_mask)
        except Exception:
            pass

    retina_pixel_count = int(np.sum(retina_mask))
    fill_fraction = float(retina_pixel_count / total_pixels) if total_pixels > 0 else 0.0

    if retina_pixel_count == 0:
        return {
            "sharpness": 0.0,
            "mean_brightness": 0.0,
            "contrast": 0.0,
            "retina_fill_fraction": 0.0,
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "rg_ratio": 1.0,
            "rb_ratio": 1.0,
            "gb_ratio": 1.0,
        }

    # 2. Extract retina pixel RGB values
    retina_pixels_rgb = img_rgb[retina_mask].astype(np.float64)

    # 3. Sharpness: Variance of Laplacian on the green channel
    # To prevent the artificial outer circular boundary from inflating sharpness,
    # erode the retina mask slightly to analyze the interior retinal tissue.
    g_channel = img_rgb[:, :, 1].astype(np.float64)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(retina_mask.astype(np.uint8), erode_k) > 0

    eval_mask = inner_mask if np.sum(inner_mask) > 50 else retina_mask
    lap = cv2.Laplacian(g_channel, cv2.CV_64F, ksize=3)
    sharpness = float(np.var(lap[eval_mask]))

    # 4. Mean Brightness (average across R, G, B on retina)
    mean_brightness = float(np.mean(retina_pixels_rgb))

    # 5. Contrast: Standard deviation of luminance on retina
    luminance = (
        0.299 * retina_pixels_rgb[:, 0]
        + 0.587 * retina_pixels_rgb[:, 1]
        + 0.114 * retina_pixels_rgb[:, 2]
    )
    contrast = float(np.std(luminance))

    # 6. Mean R, G, B & Color Cast Ratios
    mean_r = float(np.mean(retina_pixels_rgb[:, 0]))
    mean_g = float(np.mean(retina_pixels_rgb[:, 1]))
    mean_b = float(np.mean(retina_pixels_rgb[:, 2]))

    rg_ratio = float(mean_r / (mean_g + 1e-6))
    rb_ratio = float(mean_r / (mean_b + 1e-6))
    gb_ratio = float(mean_g / (mean_b + 1e-6))

    return {
        "sharpness": sharpness,
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "retina_fill_fraction": fill_fraction,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "rg_ratio": rg_ratio,
        "rb_ratio": rb_ratio,
        "gb_ratio": gb_ratio,
    }


class QualityGate:
    """
    Rule-based Quality Gate for screening out ungradable/degraded fundus images.

    CRITICAL RULE:
      All gate thresholds and normalization statistics are fit strictly on
      EyePACS VAL. Test sets are strictly evaluated using frozen thresholds.

    Score Formulation:
      Composite Quality Score Q combines:
        - log-sharpness (higher is better)
        - contrast (higher is better)
        - retina fill fraction (higher is better)
        - brightness deviation penalty (|B - median_B|)
    """

    def __init__(self, rejection_rates: tuple[float, ...] = (0.05, 0.10, 0.15)) -> None:
        self.rejection_rates = rejection_rates
        self.thresholds: dict[str, float] = {}  # e.g. "reject_5%": score_thresh
        self.stats: dict[str, float] = {}
        self.is_fitted: bool = False

    def compute_composite_score(self, df_metrics: pd.DataFrame) -> np.ndarray:
        """Calculate composite quality score for a dataframe of quality metrics."""
        if not self.is_fitted:
            raise RuntimeError("QualityGate must be fit on validation data before scoring.")

        # Log-sharpness
        log_sharp = np.log1p(df_metrics["sharpness"].values)
        z_sharp = (log_sharp - self.stats["sharp_mean"]) / (self.stats["sharp_std"] + 1e-6)

        # Contrast
        contrast = df_metrics["contrast"].values
        z_contrast = (contrast - self.stats["contrast_mean"]) / (self.stats["contrast_std"] + 1e-6)

        # Fill fraction
        fill = df_metrics["retina_fill_fraction"].values
        z_fill = (fill - self.stats["fill_mean"]) / (self.stats["fill_std"] + 1e-6)

        # Brightness deviation penalty (symmetric penalty for under/over-exposure)
        bright = df_metrics["mean_brightness"].values
        bright_dev = np.abs(bright - self.stats["bright_median"])
        z_bright = - (bright_dev / (self.stats["bright_std"] + 1e-6))

        # Composite score
        q_score = z_sharp + 1.2 * z_contrast + 0.8 * z_fill + 1.0 * z_bright
        return q_score

    def fit(self, val_df: pd.DataFrame) -> QualityGate:
        """
        Fit normalization statistics and rejection thresholds on EyePACS VAL only.

        Parameters
        ----------
        val_df : pd.DataFrame
            DataFrame containing quality metrics from EyePACS VAL.
        """
        log_sharp = np.log1p(val_df["sharpness"].values)
        contrast = val_df["contrast"].values
        fill = val_df["retina_fill_fraction"].values
        bright = val_df["mean_brightness"].values

        self.stats = {
            "sharp_mean": float(np.mean(log_sharp)),
            "sharp_std": float(np.std(log_sharp)),
            "contrast_mean": float(np.mean(contrast)),
            "contrast_std": float(np.std(contrast)),
            "fill_mean": float(np.mean(fill)),
            "fill_std": float(np.std(fill)),
            "bright_median": float(np.median(bright)),
            "bright_std": float(np.std(bright)),
        }
        self.is_fitted = True

        # Compute composite quality score on validation set
        val_q = self.compute_composite_score(val_df)

        # Determine rejection thresholds for specified rates (e.g. bottom 5%, 10%, 15%)
        self.thresholds = {}
        for rate in self.rejection_rates:
            pct = rate * 100.0
            thresh = float(np.percentile(val_q, pct))
            rate_key = f"reject_{int(round(pct))}%"
            self.thresholds[rate_key] = thresh

        return self

    def predict(self, df_metrics: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict rejection decisions for each target rate.

        Returns
        -------
        dict[str, np.ndarray of bool]
            Boolean array per rate (True = Rejected, False = Accepted).
        """
        q_scores = self.compute_composite_score(df_metrics)
        decisions: dict[str, np.ndarray] = {"quality_score": q_scores}

        for rate_key, thresh in self.thresholds.items():
            # Rejected if quality score is below threshold
            decisions[f"is_{rate_key}"] = (q_scores < thresh)

        return decisions

    def evaluate_coverage(
        self,
        df_metrics_with_preds: pd.DataFrame,
        operating_threshold: float = 0.2785,
    ) -> dict[str, Any]:
        """
        Evaluate performance across accepted images vs. all images for each rejection threshold.

        Parameters
        ----------
        df_metrics_with_preds : pd.DataFrame
            DataFrame with 'grade', 'calibrated_prob', and quality metrics.
        operating_threshold : float
            Decision threshold for binary referable DR.

        Returns
        -------
        dict[str, Any]
            Performance metrics table per coverage tier.
        """
        y_true = (df_metrics_with_preds["grade"].values >= 2).astype(int)
        probs = df_metrics_with_preds["calibrated_prob"].values
        n_total = len(y_true)

        decisions = self.predict(df_metrics_with_preds)
        q_scores = decisions["quality_score"]

        # 1. Baseline: All images (100% coverage)
        auc_all = compute_auc(y_true, probs)
        preds_all = (probs >= operating_threshold).astype(int)
        tp_all = int(np.sum((y_true == 1) & (preds_all == 1)))
        fp_all = int(np.sum((y_true == 0) & (preds_all == 1)))
        tn_all = int(np.sum((y_true == 0) & (preds_all == 0)))
        fn_all = int(np.sum((y_true == 1) & (preds_all == 0)))

        sens_all = float(tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
        spec_all = float(tn_all / (tn_all + fp_all)) if (tn_all + fp_all) > 0 else 0.0

        results: dict[str, Any] = {
            "all_images": {
                "coverage_pct": 100.0,
                "rejection_rate_pct": 0.0,
                "n_evaluated": n_total,
                "n_rejected": 0,
                "auc": float(auc_all),
                "sensitivity": sens_all,
                "specificity": spec_all,
                "tp": tp_all,
                "fp": fp_all,
                "tn": tn_all,
                "fn": fn_all,
            },
            "tiers": {},
        }

        for rate_key, thresh in self.thresholds.items():
            is_rejected = (q_scores < thresh)
            is_accepted = ~is_rejected

            n_acc = int(np.sum(is_accepted))
            n_rej = int(np.sum(is_rejected))
            cov_pct = float((n_acc / n_total) * 100.0) if n_total > 0 else 0.0
            rej_pct = float((n_rej / n_total) * 100.0) if n_total > 0 else 0.0

            if n_acc >= 10 and len(np.unique(y_true[is_accepted])) >= 2:
                auc_acc = float(compute_auc(y_true[is_accepted], probs[is_accepted]))
            else:
                auc_acc = float("nan")

            preds_acc = (probs[is_accepted] >= operating_threshold).astype(int)
            y_acc = y_true[is_accepted]

            tp_acc = int(np.sum((y_acc == 1) & (preds_acc == 1)))
            fp_acc = int(np.sum((y_acc == 0) & (preds_acc == 1)))
            tn_acc = int(np.sum((y_acc == 0) & (preds_acc == 0)))
            fn_acc = int(np.sum((y_acc == 1) & (preds_acc == 0)))

            sens_acc = float(tp_acc / (tp_acc + fn_acc)) if (tp_acc + fn_acc) > 0 else 0.0
            spec_acc = float(tn_acc / (tn_acc + fp_acc)) if (tn_acc + fp_acc) > 0 else 0.0

            # Count errors in rejected subset
            preds_rej = (probs[is_rejected] >= operating_threshold).astype(int)
            y_rej = y_true[is_rejected]
            fn_filtered = int(np.sum((y_rej == 1) & (preds_rej == 0)))
            fp_filtered = int(np.sum((y_rej == 0) & (preds_rej == 1)))

            results["tiers"][rate_key] = {
                "threshold": float(thresh),
                "coverage_pct": cov_pct,
                "rejection_rate_pct": rej_pct,
                "n_evaluated": n_acc,
                "n_rejected": n_rej,
                "auc": auc_acc,
                "sensitivity": sens_acc,
                "specificity": spec_acc,
                "tp": tp_acc,
                "fp": fp_acc,
                "tn": tn_acc,
                "fn": fn_acc,
                "fn_filtered_out": fn_filtered,
                "fp_filtered_out": fp_filtered,
            }

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize QualityGate parameters for persistent storage."""
        return {
            "rejection_rates": list(self.rejection_rates),
            "thresholds": self.thresholds,
            "stats": self.stats,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityGate:
        """Instantiate QualityGate from serialized dictionary."""
        gate = cls(rejection_rates=tuple(data.get("rejection_rates", [0.05, 0.10, 0.15])))
        gate.thresholds = data.get("thresholds", {})
        gate.stats = data.get("stats", {})
        gate.is_fitted = data.get("is_fitted", False)
        return gate
