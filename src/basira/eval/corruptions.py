"""
src/basira/eval/corruptions.py
==============================
Synthetic image corruptions for fundus robustness stress testing.

Applies 7 clinical and acquisition image corruptions across 5 severity levels:
  1. Gaussian blur: Out-of-focus optics / motion blur.
  2. Brightness down: Under-exposure / cataract / small pupil.
  3. Brightness up: Over-exposure / excessive flash illumination.
  4. Contrast reduction: Corneal haze / media opacities.
  5. Gaussian noise: Sensor noise / low-light sensor amplification.
  6. JPEG compression: Telemedicine compression artifacts.
  7. Colour cast (R/G scaling): White balance shift / camera sensor variations.

Note:
  Corruptions are applied directly to preprocessed fundus images (after Stage-1).
  Retina masking is preserved so background remains strictly zeroed.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


CORRUPTION_TYPES = (
    "gaussian_blur",
    "brightness_down",
    "brightness_up",
    "contrast",
    "gaussian_noise",
    "jpeg_compression",
    "color_cast",
)

SEVERITY_LEVELS = (1, 2, 3, 4, 5)


def apply_gaussian_blur(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian blur with increasing sigma (1 to 5)."""
    # Monotonic sigmas: 1.0, 2.0, 3.0, 4.0, 5.0
    sigmas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    sigma = sigmas[severity]
    if sigma <= 0.0:
        return img_rgb.copy()
    k = int(2 * np.ceil(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(img_rgb, (k, k), sigma)
    return blurred


def apply_brightness_down(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Dim image brightness with decreasing scaling factors (1 to 5)."""
    # Monotonic scaling: 0.85, 0.70, 0.55, 0.40, 0.25
    factors = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]
    f = factors[severity]
    dimmed = np.clip(img_rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return dimmed


def apply_brightness_up(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Brighten image illumination with increasing scaling factors (1 to 5)."""
    # Monotonic scaling: 1.15, 1.30, 1.50, 1.75, 2.00
    factors = [1.0, 1.15, 1.30, 1.50, 1.75, 2.00]
    f = factors[severity]
    brightened = np.clip(img_rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return brightened


def apply_contrast_reduction(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Reduce dynamic contrast towards retinal mean luminance (1 to 5)."""
    # Monotonic contrast multipliers: 0.85, 0.70, 0.55, 0.40, 0.25
    factors = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]
    f = factors[severity]
    mask = np.any(img_rgb > 0, axis=-1)
    if np.any(mask):
        mean_lum = np.mean(img_rgb[mask].astype(np.float32), axis=0, keepdims=True)
    else:
        mean_lum = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    diff = img_rgb.astype(np.float32) - mean_lum
    reduced = np.clip(mean_lum + diff * f, 0, 255).astype(np.uint8)
    return reduced


def apply_gaussian_noise(
    img_rgb: np.ndarray,
    severity: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add zero-mean Gaussian sensor noise with increasing standard deviation (1 to 5)."""
    # Monotonic noise std: 8.0, 16.0, 28.0, 44.0, 64.0
    stds = [0.0, 8.0, 16.0, 28.0, 44.0, 64.0]
    std = stds[severity]
    if std <= 0.0:
        return img_rgb.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=img_rgb.shape).astype(np.float32)
    noisy = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_jpeg_compression(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Apply lossy JPEG compression with decreasing quality factors (1 to 5)."""
    # Monotonic quality degradation: 75, 55, 35, 20, 8
    qualities = [100, 75, 55, 35, 20, 8]
    q = qualities[severity]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, enc = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return img_rgb.copy()
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        return img_rgb.copy()
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def apply_color_cast(img_rgb: np.ndarray, severity: int) -> np.ndarray:
    """Apply R/G spectral shift simulating camera color temperature imbalance (1 to 5)."""
    # Monotonic scaling: red increases, green & blue decrease
    r_scales = [1.0, 1.12, 1.25, 1.45, 1.70, 2.00]
    g_scales = [1.0, 0.94, 0.86, 0.76, 0.64, 0.50]
    b_scales = [1.0, 0.94, 0.86, 0.76, 0.64, 0.50]

    rf = r_scales[severity]
    gf = g_scales[severity]
    bf = b_scales[severity]

    shifted = img_rgb.astype(np.float32).copy()
    shifted[:, :, 0] *= rf
    shifted[:, :, 1] *= gf
    shifted[:, :, 2] *= bf

    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_corruption(
    img_rgb: np.ndarray,
    corruption_type: str,
    severity: int,
    seed: Optional[int] = None,
    keep_background_zeroed: bool = True,
) -> np.ndarray:
    """
    Apply specified synthetic corruption at a given severity level (1 to 5).

    Parameters
    ----------
    img_rgb : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    corruption_type : str
        One of CORRUPTION_TYPES.
    severity : int
        Severity level from 1 to 5 (0 returns original image).
    seed : int, optional
        RNG seed for stochastic corruptions (e.g. gaussian_noise).
    keep_background_zeroed : bool, default True
        Whether to re-apply circular retina mask to keep background pixels at 0.

    Returns
    -------
    corrupted : np.ndarray
        uint8 RGB array of shape (H, W, 3).
    """
    if severity < 0 or severity > 5:
        raise ValueError(f"Severity must be in [0, 5], got {severity}")

    if severity == 0:
        return img_rgb.copy()

    c_type = corruption_type.strip().lower()

    if c_type == "gaussian_blur":
        corrupted = apply_gaussian_blur(img_rgb, severity)
    elif c_type == "brightness_down":
        corrupted = apply_brightness_down(img_rgb, severity)
    elif c_type == "brightness_up":
        corrupted = apply_brightness_up(img_rgb, severity)
    elif c_type == "contrast":
        corrupted = apply_contrast_reduction(img_rgb, severity)
    elif c_type == "gaussian_noise":
        corrupted = apply_gaussian_noise(img_rgb, severity, seed=seed)
    elif c_type in ("jpeg_compression", "jpeg"):
        corrupted = apply_jpeg_compression(img_rgb, severity)
    elif c_type in ("color_cast", "colour_cast"):
        corrupted = apply_color_cast(img_rgb, severity)
    else:
        raise ValueError(
            f"Unknown corruption type: '{corruption_type}'. Supported: {CORRUPTION_TYPES}"
        )

    if keep_background_zeroed:
        retina_mask = np.any(img_rgb > 0, axis=-1, keepdims=True)
        corrupted = np.where(retina_mask, corrupted, 0)

    return corrupted
