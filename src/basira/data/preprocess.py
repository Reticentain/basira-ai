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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the circular fundus region and crop away black surrounding borders.

    Strategy (robust against JPEG background noise):
      1. Convert to grayscale and apply a Gaussian blur to suppress isolated noise.
      2. Threshold to obtain a binary foreground mask.
      3. Apply morphological opening to remove small isolated foreground blobs.
      4. Find connected components; keep only the LARGEST one (the retina disc).
      5. If the largest component covers < min_area_ratio of total pixels, skip crop.
      6. Use the tight bounding box of that component to crop.

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
    cropped : np.ndarray
        Cropped uint8 RGB image.
    retina_mask : np.ndarray
        uint8 binary mask (255 = retina, 0 = background) of the SAME spatial
        extent as `cropped`, suitable for downstream background zeroing.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {img.shape}")

    h, w = img.shape[:2]

    # --- Step 1: Grayscale + Gaussian blur to suppress JPEG noise ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_k = max(5, (min(h, w) // 100) * 2 + 1)  # odd kernel, ~1% of smallest dim
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # --- Step 2: Threshold ---
    _, raw_mask = cv2.threshold(blurred, tol, 255, cv2.THRESH_BINARY)

    # --- Step 3: Morphological opening to drop isolated noise pixels ---
    open_k = max(15, (min(h, w) // 50) * 2 + 1)  # ~2% of smallest dim
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    # --- Step 4: Largest connected component ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    # Label 0 is background; find the largest foreground label
    if num_labels < 2:
        # No foreground component found after opening → skip crop
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        return img, full_mask

    # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
    fg_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(fg_areas)) + 1
    largest_area = int(fg_areas[np.argmax(fg_areas)])

    # --- Step 5: Check minimum area ---
    if largest_area < min_area_ratio * (h * w):
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        return img, full_mask

    # --- Step 6: Tight bounding box of the largest component ---
    component_mask = (labels == largest_label).astype(np.uint8) * 255

    left   = int(stats[largest_label, cv2.CC_STAT_LEFT])
    top    = int(stats[largest_label, cv2.CC_STAT_TOP])
    cw     = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    ch     = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    right  = left + cw
    bottom = top + ch

    # Safeguard against degenerate boxes (< 10 px)
    if ch < 10 or cw < 10:
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        return img, full_mask

    cropped = img[top:bottom, left:right]
    retina_mask = component_mask[top:bottom, left:right]
    return cropped, retina_mask


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
      1. Crop black border (robust: blur → threshold → morph-open → largest CC bbox)
      2. Pad to square (black padding, no stretch)
      3. Resize to target_size x target_size
      4. Apply CLAHE on L channel in LAB color space (optional)
      5. Zero all pixels outside the retina mask (pure black background)

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (target_size, target_size, 3).
        Pixels outside the retina circle are exactly 0.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = img.copy()

    # 1. Border crop → returns (cropped_img, retina_mask_at_crop_resolution)
    cropped, retina_mask_crop = crop_black_borders(img_np, tol=tol)

    # 2. Square pad (do not stretch) — apply same padding to both image and mask
    squared = pad_to_square(cropped)
    # pad_to_square pads with 0, so just pad the mask with 0 (background) too
    mask_3d = retina_mask_crop[:, :, np.newaxis]  # (H, W, 1)
    squared_mask_3d = pad_to_square(mask_3d)
    squared_mask = squared_mask_3d[:, :, 0]  # back to (H, W)

    # 3. Resize image and mask to target_size
    interp = cv2.INTER_AREA if (squared.shape[0] >= target_size) else cv2.INTER_LINEAR
    resized = cv2.resize(squared, (target_size, target_size), interpolation=interp)
    resized_mask = cv2.resize(
        squared_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )

    # 4. CLAHE enhancement on L channel in LAB space
    if apply_clahe:
        enhanced = apply_clahe_lab(resized, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)
    else:
        enhanced = resized

    # 5. Zero pixels outside the retina mask so CLAHE noise in background is suppressed
    retina_bool = resized_mask > 0  # (H, W) bool
    enhanced[~retina_bool] = 0

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
