"""
tests/test_preprocess.py
========================
Unit tests for the fundus preprocessing pipeline in basira.data.preprocess:
  - Non-square image -> square output (padded, not stretched).
  - Almost-black / empty mask image -> does not crash (safely skips crop).
  - Dark retina on black background -> not cropped away.
  - Stage 1 outputs uint8 RGB of shape (target_size, target_size, 3).
  - Stage 2 outputs normalized float32 torch.Tensor of shape (3, target_size, target_size).
"""

import numpy as np
import pytest
import torch

from basira.data.preprocess import (
    crop_and_enhance,
    crop_black_borders,
    pad_to_square,
    preprocess_image,
    to_tensor_and_normalize,
)


class TestPreprocessPipeline:
    def test_pad_to_square_wide_image(self):
        """Pad a wide non-square image to a square."""
        img_wide = np.full((100, 200, 3), fill_value=128, dtype=np.uint8)
        squared = pad_to_square(img_wide)
        assert squared.shape == (200, 200, 3)
        # Verify original content is centered vertically
        assert np.array_equal(squared[50:150, :, :], img_wide)

    def test_pad_to_square_tall_image(self):
        """Pad a tall non-square image to a square."""
        img_tall = np.full((300, 150, 3), fill_value=128, dtype=np.uint8)
        squared = pad_to_square(img_tall)
        assert squared.shape == (300, 300, 3)
        # Verify original content is centered horizontally
        assert np.array_equal(squared[:, 75:225, :], img_tall)

    def test_non_square_synthetic_gives_square_output(self):
        """End-to-end: non-square input yields exact (target_size, target_size, 3)."""
        target_size = 256
        img_rect = np.zeros((400, 700, 3), dtype=np.uint8)
        # Draw a synthetic circular fundus in the center
        yy, xx = np.ogrid[:400, :700]
        circle_mask = ((yy - 200) ** 2 + (xx - 350) ** 2) <= (150**2)
        img_rect[circle_mask] = [180, 80, 40]

        stage1 = crop_and_enhance(img_rect, target_size=target_size, apply_clahe=True)

        assert isinstance(stage1, np.ndarray)
        assert stage1.dtype == np.uint8
        assert stage1.shape == (target_size, target_size, 3)

    def test_almost_black_image_does_not_crash(self):
        """Almost empty mask (e.g. all 0s or very few dim pixels) must not crash."""
        almost_black = np.zeros((200, 200, 3), dtype=np.uint8)
        # Place 3 dim pixels (well below min_area_ratio)
        almost_black[10, 10] = [15, 12, 11]

        # Must not raise or return 0-dimension slice
        # crop_black_borders now returns (cropped, mask) tuple
        cropped, _mask = crop_black_borders(almost_black, tol=10, min_area_ratio=0.02)
        assert cropped.shape == (200, 200, 3)

        out = crop_and_enhance(almost_black, target_size=256, apply_clahe=False)
        assert out.shape == (256, 256, 3)

    def test_dark_retina_not_cropped_away(self):
        """A dark retina (low intensity but visible fundus) must be preserved."""
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        # Low intensity retina circle: values around 25-30 (above tol=10)
        yy, xx = np.ogrid[:300, :300]
        circle = ((yy - 150) ** 2 + (xx - 150) ** 2) <= (100**2)
        canvas[circle] = [30, 22, 18]

        # crop_black_borders returns (cropped, mask) tuple
        cropped, _mask = crop_black_borders(canvas, tol=10)
        # Bounding box of a circle with radius 100 should be ~201x201
        assert 190 <= cropped.shape[0] <= 210
        assert 190 <= cropped.shape[1] <= 210
        # Retina contents should still be present
        assert np.mean(cropped) > 5

    def test_sparse_noise_at_edges_crops_tight(self):
        """
        Sparse bright noise pixels at the far edges must NOT widen the crop bbox.
        The crop bounding box must still be driven by the central circular retina,
        not by isolated noise blobs near the borders.
        """
        rng = np.random.default_rng(0)
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)

        # Draw a solid circular retina (radius=100) centred at (150, 150)
        yy, xx = np.ogrid[:300, :300]
        circle = ((yy - 150) ** 2 + (xx - 150) ** 2) <= (100 ** 2)
        canvas[circle] = [160, 70, 40]

        # Place 8 sparse bright noisy pixels at the extreme left and right edges
        for _ in range(4):
            row = int(rng.integers(10, 290))
            canvas[row, 0] = [200, 200, 200]   # leftmost column
            canvas[row, 299] = [200, 200, 200]  # rightmost column

        stage1 = crop_and_enhance(canvas, target_size=256, apply_clahe=False)
        assert stage1.shape == (256, 256, 3), f"Expected (256,256,3), got {stage1.shape}"

        # The output must be square (always true for crop_and_enhance)
        assert stage1.shape[0] == stage1.shape[1] == 256

        # The corner pixels (far outside any retina) must be black (zeroed background)
        for corner in [(0, 0), (0, 255), (255, 0), (255, 255)]:
            r, c = corner
            assert np.all(stage1[r, c] == 0), (
                f"Corner {corner} is not black: {stage1[r, c]}. "
                "Noise pixels at the edge may have expanded the crop."
            )

    def test_background_is_pure_black_after_stage1(self):
        """
        After Stage 1, all pixels outside the retina disc must be exactly [0, 0, 0].
        This ensures CLAHE amplification of background noise is suppressed.
        """
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:300, :300]
        circle = ((yy - 150) ** 2 + (xx - 150) ** 2) <= (100 ** 2)
        canvas[circle] = [140, 60, 30]

        stage1 = crop_and_enhance(canvas, target_size=256, apply_clahe=True)
        assert stage1.shape == (256, 256, 3)

        # Corners are outside the inscribed circle → must be pure black
        for corner in [(0, 0), (0, 255), (255, 0), (255, 255)]:
            r, c = corner
            assert np.all(stage1[r, c] == 0), (
                f"Background pixel at {corner} is not zero: {stage1[r, c]}"
            )

    def test_tensor_conversion_and_normalization(self):
        """Verify Stage 2 converts uint8 array to properly normalized torch Tensor."""
        img_np = np.full((256, 256, 3), fill_value=128, dtype=np.uint8)
        tensor = to_tensor_and_normalize(img_np)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert tensor.dtype == torch.float32

        # 128 / 255 ~= 0.50196
        # Expected normalized channel 0: (0.50196 - 0.485) / 0.229 ~= 0.074
        assert -3.0 < tensor.mean().item() < 3.0
