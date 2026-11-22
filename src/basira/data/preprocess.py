"""
src/basira/data/preprocess.py
=============================
Standard preprocessing pipeline for fundus images across all datasets.

Two-stage architecture:
  Stage 1 (Deterministic, cacheable):
    - Crop black borders (with fallback if mask is nearly empty)
    - Pad to square (preserve aspect ratio without stretching)
    - Resize to target size (default 256x256)
    - Apply CLAHE in LAB color space strictly on the L (luminance) channel
    -> Output: uint8 RGB numpy array (H, W, 3)

  Stage 2 (Tensor conversion & normalization):
    - Convert to torch.float32 [0.0, 1.0]
    - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    -> Output: torch.Tensor (3, H, W)

Augmentations (training only) sit between Stage 1 and Stage 2.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def crop_black_borders(
    img: np.ndarray,
    tol: int = 10,
    min_area_ratio: float = 0.02,
) -> np.ndarray:
    """
    Detect the circular fundus region and crop away black surrounding borders.

    Parameters
    ----------
    img : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    tol : int
        Intensity threshold below which pixels are considered background.
    min_area_ratio : float
        If non-background pixels cover less than this fraction of total area,
        the mask is considered empty/invalid and the crop is safely skipped.

    Returns
    -------
    np.ndarray
        Cropped uint8 RGB image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {img.shape}")

    h, w = img.shape[:2]
    # Check max channel intensity to robustly handle colored/dim retinas
    mask = np.max(img, axis=2) > tol

    # Safe fallback if mask is almost empty (e.g. nearly black photo or bad exposure)
    if np.count_nonzero(mask) < min_area_ratio * (h * w):
        return img

    y_indices, x_indices = np.where(mask)
    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())

    # Safeguard against degenerate boxes (< 10 px)
    if (ymax - ymin < 10) or (xmax - xmin < 10):
        return img

    return img[ymin : ymax + 1, xmin : xmax + 1]


def pad_to_square(img: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Pad the shorter dimension symmetrically with `pad_value` (black)
    to yield a square image without distorting the retinal circle.

    Parameters
    ----------
    img : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    pad_value : int
        Fill value for padding (default 0).

    Returns
    -------
    np.ndarray
        Square uint8 RGB image of shape (max(H, W), max(H, W), 3).
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        return np.pad(img, ((0, 0), (left, right), (0, 0)), mode="constant", constant_values=pad_value)
    else:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        return np.pad(img, ((top, bottom), (0, 0), (0, 0)), mode="constant", constant_values=pad_value)


def apply_clahe_lab(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Operating color space: LAB color space.
    CLAHE is applied STRICTLY to the L (luminance) channel to avoid chromaticity
    distortion, preserving the true hue of retinal haemorrhages and exudates.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def crop_and_enhance(
    img: np.ndarray | Image.Image,
    target_size: int = 256,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
    tol: int = 10,
) -> np.ndarray:
    """
    Stage 1: Deterministic fundus preprocessing pipeline.
    Steps:
      1. Crop black border
      2. Pad to square
      3. Resize to target_size x target_size
      4. Apply CLAHE on L channel in LAB color space (optional)

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (target_size, target_size, 3).
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = img.copy()

    # 1. Border crop
    cropped = crop_black_borders(img_np, tol=tol)

    # 2. Square pad (do not stretch)
    squared = pad_to_square(cropped)

    # 3. Resize
    interp = cv2.INTER_AREA if (squared.shape[0] >= target_size) else cv2.INTER_LINEAR
    resized = cv2.resize(squared, (target_size, target_size), interpolation=interp)

    # 4. CLAHE enhancement
    if apply_clahe:
        enhanced = apply_clahe_lab(resized, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)
    else:
        enhanced = resized

    return enhanced


def to_tensor_and_normalize(
    img: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Stage 2: Convert uint8 RGB array to normalized PyTorch tensor.

    Parameters
    ----------
    img : np.ndarray
        uint8 RGB image of shape (H, W, 3) in [0, 255].
    mean : tuple
        ImageNet per-channel mean.
    std : tuple
        ImageNet per-channel standard deviation.

    Returns
    -------
    torch.Tensor
        float32 Tensor of shape (3, H, W) normalized.
    """
    # uint8 [0, 255] -> float32 [0.0, 1.0]
    tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    return (tensor - mean_t) / std_t


def preprocess_image(
    img: np.ndarray | Image.Image,
    target_size: int = 256,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """End-to-end preprocessing helper (Stage 1 + Stage 2)."""
    stage1 = crop_and_enhance(
        img,
        target_size=target_size,
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
    )
    return to_tensor_and_normalize(stage1, mean=mean, std=std)
