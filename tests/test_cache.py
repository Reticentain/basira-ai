"""
tests/test_cache.py
===================
Correctness and integrity tests for the Stage-1 preprocessed image cache:
  - Cache tag computation and path resolution
  - Manifest validation and tamper detection
  - Shape (256, 256, 3), dtype (uint8), and pure black corner verification
  - Exact pixel identity (np.array_equal) between cached PNG and on-the-fly Stage-1
  - Graceful pytest.skip when running without local/Colab cache files
"""

from __future__ import annotations

import json
import pathlib
import numpy as np
import pytest
from PIL import Image

from basira.config import load_config
from basira.data.cache import (
    cache_dir,
    cache_path,
    compute_cache_tag,
    get_manifest_path,
    validate_cache_manifest,
)
from basira.data.loaders import rebuild_path
from basira.data.preprocess import PREPROCESS_VERSION, crop_and_enhance


class TestCacheTagAndManifest:
    def test_cache_tag_deterministic(self):
        """Cache tag must be deterministic and include PREPROCESS_VERSION and image parameters."""
        cfg = load_config()
        tag1 = compute_cache_tag(cfg)
        tag2 = compute_cache_tag(cfg)
        assert tag1 == tag2
        assert f"v{PREPROCESS_VERSION}" in tag1
        assert f"s{cfg.image.size}" in tag1
        assert "crop" in tag1
        assert "clahe" in tag1

    def test_cache_path_format(self):
        """cache_path must resolve to <cache_root>/<tag>/<source>/<stem>.png."""
        cfg = load_config()
        p = cache_path("eyepacs_train", "1234_left.jpg", cfg)
        assert p.suffix == ".png"
        assert p.stem == "1234_left"
        assert p.parent.name == "eyepacs_train"
        assert p.parent.parent.name == compute_cache_tag(cfg)

    def test_manifest_validation_rejects_altered_config(self, tmp_path):
        """validate_cache_manifest must raise ValueError when settings do not match."""
        cfg = load_config()
        tag = compute_cache_tag(cfg)

        manifest_data = {
            "cache_tag": tag,
            "preprocess_version": PREPROCESS_VERSION,
            "git_commit": "testcommit",
            "settings": {
                "image_size": int(cfg.image.size) + 64,  # altered size!
                "apply_border_crop": bool(cfg.image.apply_border_crop),
                "apply_clahe": bool(cfg.image.apply_clahe),
                "clahe_clip_limit": float(cfg.image.clahe_clip_limit),
                "clahe_tile_grid": list(cfg.image.clahe_tile_grid),
            },
            "counts": {"aptos": 100},
        }

        # Override cache_root in a local copy of cfg
        cfg_copy = cfg.copy()
        cfg_copy.output.cache_root = str(tmp_path)

        fake_tag_dir = tmp_path / tag
        fake_tag_dir.mkdir(parents=True)
        with open(fake_tag_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # Must detect altered image_size
        with pytest.raises(ValueError, match="image_size mismatch"):
            validate_cache_manifest(cfg_copy)


class TestCachedImagesIntegrity:
    @pytest.fixture(autouse=True)
    def require_cache(self):
        """Skip all tests in this class if the cache directory or PNG files are absent."""
        cfg = load_config()
        c_dir = cache_dir(cfg)
        if not c_dir.exists():
            pytest.skip(f"Cache tag directory does not exist: {c_dir}")

        png_files = list(c_dir.glob("*/*.png"))
        if not png_files:
            pytest.skip(f"No cached PNG images found inside {c_dir}")

    def test_cache_manifest_valid(self):
        """Active cache folder must have a valid manifest matching current settings."""
        cfg = load_config()
        manifest = validate_cache_manifest(cfg)
        assert isinstance(manifest, dict)
        assert manifest["preprocess_version"] == PREPROCESS_VERSION

    def test_cached_images_shape_dtype_and_corners(self):
        """Sample 10 random cached images per source; assert 256x256x3 uint8 and zeroed corners."""
        cfg = load_config()
        c_dir = cache_dir(cfg)
        rng = np.random.default_rng(42)

        target_size = int(cfg.image.size)
        corners = [(0, 0), (0, target_size - 1), (target_size - 1, 0), (target_size - 1, target_size - 1)]

        for source_dir in c_dir.iterdir():
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue

            png_list = sorted(list(source_dir.glob("*.png")))
            if not png_list:
                continue

            sample_size = min(10, len(png_list))
            sampled_indices = rng.choice(len(png_list), size=sample_size, replace=False)

            for idx in sampled_indices:
                png_path = png_list[idx]
                with Image.open(png_path) as im:
                    arr = np.array(im)

                # 1. Shape and dtype check
                assert arr.shape == (target_size, target_size, 3), (
                    f"Unexpected shape {arr.shape} in {png_path}"
                )
                assert arr.dtype == np.uint8, f"Unexpected dtype {arr.dtype} in {png_path}"

                # 2. Pure black background check (corners must be [0, 0, 0])
                for r, c in corners:
                    assert np.all(arr[r, c] == 0), (
                        f"Non-zero corner ({r}, {c}) = {arr[r, c]} in {png_path}"
                    )

    def test_cached_vs_on_the_fly_identity(self):
        """
        Verify that cached PNG is bit-for-bit identical (np.array_equal)
        to on-the-fly execution of crop_and_enhance on the raw image.
        """
        cfg = load_config()
        c_dir = cache_dir(cfg)
        rng = np.random.default_rng(42)

        tested_comparisons = 0
        for source_dir in c_dir.iterdir():
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue

            source = source_dir.name
            png_list = sorted(list(source_dir.glob("*.png")))
            if not png_list:
                continue

            sample_size = min(10, len(png_list))
            sampled_indices = rng.choice(len(png_list), size=sample_size, replace=False)

            for idx in sampled_indices:
                png_path = png_list[idx]
                stem = png_path.stem

                # Reconstruct image_id (need extension: .jpg for eyepacs/messidor/idrid, .png for aptos)
                ext = ".png" if source == "aptos" else ".jpg"
                image_id = f"{stem}{ext}"

                try:
                    raw_path = rebuild_path(source, image_id, cfg)
                    raw_p = pathlib.Path(raw_path)
                except Exception:
                    continue

                if not raw_p.exists():
                    # Raw file missing on this system; skip raw comparison
                    continue

                # Load raw and run on-the-fly Stage-1
                with Image.open(raw_p) as pil_raw:
                    raw_rgb = pil_raw.convert("RGB")

                on_the_fly = crop_and_enhance(
                    raw_rgb,
                    target_size=int(cfg.image.size),
                    apply_clahe=bool(cfg.image.apply_clahe),
                    clahe_clip_limit=float(cfg.image.clahe_clip_limit),
                    clahe_tile_grid=tuple(cfg.image.clahe_tile_grid),
                    tol=10,
                )

                # Load cached PNG
                with Image.open(png_path) as pil_cached:
                    cached_arr = np.array(pil_cached)

                assert np.array_equal(cached_arr, on_the_fly), (
                    f"Pixel mismatch between cached PNG and on-the-fly Stage-1 for {png_path}!"
                )
                tested_comparisons += 1

        if tested_comparisons == 0:
            pytest.skip("Raw images not present on disk; skipped on-the-fly pixel comparison.")
