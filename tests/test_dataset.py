"""
tests/test_dataset.py
=====================
Unit tests for CachedFundusDataset:
  - Refuses construction on mismatched or missing manifest.json
  - Fails on missing cache files and lists up to 10 missing files
  - Accurate label mapping (referable float vs grade long)
  - Tensor shapes, dtypes, and train vs val augmentation behaviors
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from basira.config import load_config
from basira.data.cache import cache_dir, compute_cache_tag, write_manifest
from basira.data.dataset import CachedFundusDataset
from basira.data.preprocess import PREPROCESS_VERSION


@pytest.fixture
def fake_cache_env(tmp_path):
    """Fixture to set up a clean, isolated fake cache environment with valid manifest."""
    cfg = load_config()
    cfg_copy = cfg.copy()
    cfg_copy.output.cache_root = str(tmp_path)

    # Write a valid manifest for the active configuration
    write_manifest(counts={"eyepacs_train": 10}, cfg=cfg_copy, merge=False)
    return cfg_copy, tmp_path


class TestCachedFundusDataset:
    def test_refuses_missing_manifest(self, tmp_path):
        """Dataset must raise FileNotFoundError if manifest.json is absent."""
        cfg = load_config()
        cfg_copy = cfg.copy()
        cfg_copy.output.cache_root = str(tmp_path)

        df = pd.DataFrame([{"image_id": "001.jpg", "grade": 0, "source": "eyepacs_train"}])
        with pytest.raises(FileNotFoundError, match="Cache manifest not found"):
            CachedFundusDataset(df, cfg=cfg_copy, check_files=False)

    def test_refuses_mismatched_manifest(self, tmp_path):
        """Dataset must raise ValueError if manifest settings do not match configuration."""
        cfg = load_config()
        cfg_copy = cfg.copy()
        cfg_copy.output.cache_root = str(tmp_path)

        tag = compute_cache_tag(cfg_copy)
        tag_dir = tmp_path / tag
        tag_dir.mkdir(parents=True)

        # Corrupt settings: altered image_size
        manifest_data = {
            "cache_tag": tag,
            "preprocess_version": PREPROCESS_VERSION,
            "git_commit": "abc1234",
            "settings": {
                "image_size": int(cfg_copy.image.size) + 64,
                "apply_border_crop": bool(cfg_copy.image.apply_border_crop),
                "apply_clahe": bool(cfg_copy.image.apply_clahe),
                "clahe_clip_limit": float(cfg_copy.image.clahe_clip_limit),
                "clahe_tile_grid": list(cfg_copy.image.clahe_tile_grid),
            },
            "counts": {"eyepacs_train": 5},
        }
        with open(tag_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        df = pd.DataFrame([{"image_id": "001.jpg", "grade": 0, "source": "eyepacs_train"}])
        with pytest.raises(ValueError, match="image_size mismatch"):
            CachedFundusDataset(df, cfg=cfg_copy, check_files=False)

    def test_lists_missing_files_up_to_10(self, fake_cache_env):
        """Dataset must report missing files on construction and list first 10."""
        cfg, _ = fake_cache_env

        # 15 images in dataframe, none created on disk
        rows = [
            {"image_id": f"img_{i:03d}.jpg", "grade": i % 5, "source": "eyepacs_train"}
            for i in range(15)
        ]
        df = pd.DataFrame(rows)

        with pytest.raises(FileNotFoundError) as exc_info:
            CachedFundusDataset(df, cfg=cfg, check_files=True)

        err_msg = str(exc_info.value)
        assert "15 cached images are missing" in err_msg
        assert "First 10 missing:" in err_msg
        assert "img_000.png" in err_msg
        assert "img_009.png" in err_msg
        # img_010 should be beyond the first 10 list
        assert "img_010.png" not in err_msg

    def test_label_mapping_and_shapes(self, fake_cache_env):
        """Verify referable (float 0/1) vs grade (int64 0-4) label mapping and tensor shapes."""
        cfg, _ = fake_cache_env
        tag_dir = cache_dir(cfg) / "eyepacs_train"
        tag_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 synthetic 256x256 test PNG images
        rows = []
        for g in range(5):
            iid = f"sample_{g}.jpg"
            img_arr = np.full((256, 256, 3), fill_value=50 + g * 30, dtype=np.uint8)
            Image.fromarray(img_arr).save(tag_dir / f"sample_{g}.png")
            rows.append({"image_id": iid, "grade": g, "source": "eyepacs_train"})
        df = pd.DataFrame(rows)

        # 1. Referable task
        ds_ref = CachedFundusDataset(df, cfg=cfg, train=False, task="referable")
        assert len(ds_ref) == 5

        # Grades 0, 1 -> 0.0; Grades 2, 3, 4 -> 1.0
        expected_ref = [0.0, 0.0, 1.0, 1.0, 1.0]
        for i in range(5):
            tensor, label = ds_ref[i]
            assert tensor.shape == (3, 256, 256)
            assert tensor.dtype == torch.float32
            assert label.dtype == torch.float32
            assert label.item() == expected_ref[i]

        # 2. Grade task
        ds_grade = CachedFundusDataset(df, cfg=cfg, train=False, task="grade")
        for i in range(5):
            tensor, label = ds_grade[i]
            assert label.dtype == torch.long
            assert label.item() == i

    def test_train_vs_val_augmentation(self, fake_cache_env):
        """Train dataset exhibits stochastic transforms while Val dataset is deterministic."""
        cfg, _ = fake_cache_env
        tag_dir = cache_dir(cfg) / "eyepacs_train"
        tag_dir.mkdir(parents=True, exist_ok=True)

        # Asymmetric image pattern to easily detect rotation and flips
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        canvas[20:100, 20:80] = [200, 100, 50]
        Image.fromarray(canvas).save(tag_dir / "pattern.png")

        df = pd.DataFrame([{"image_id": "pattern.jpg", "grade": 1, "source": "eyepacs_train"}])

        ds_val = CachedFundusDataset(df, cfg=cfg, train=False, task="referable")
        t_val1, _ = ds_val[0]
        t_val2, _ = ds_val[0]
        # Val must be perfectly deterministic
        assert torch.equal(t_val1, t_val2)

        # Train should show variance across repeated samples
        ds_train = CachedFundusDataset(df, cfg=cfg, train=True, task="referable")
        draws = [ds_train[0][0] for _ in range(5)]
        has_variance = any(not torch.equal(draws[0], draws[k]) for k in range(1, 5))
        assert has_variance, "Train augmentations produced identical tensors across 5 draws."
