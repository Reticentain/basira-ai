"""
src/basira/data/cache.py
========================
Cache path construction, tagging, and manifest validation for preprocessed
Stage-1 fundus images.

Every cached image is stored at:
  /content/cache/<tag>/<source>/<image_id stem>.png

The <tag> uniquely encodes all Stage-1 preprocessing parameters and the
PREPROCESS_VERSION constant, preventing accidental reuse of outdated or
differently preprocessed caches.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any, Optional

from omegaconf import DictConfig

from basira.config import load_config
from basira.data.preprocess import PREPROCESS_VERSION


def compute_cache_tag(cfg: Optional[DictConfig] = None) -> str:
    """
    Generate a deterministic tag from current Stage-1 preprocessing settings.

    Components:
      - PREPROCESS_VERSION (bumped whenever Stage-1 code changes)
      - cfg.image.size
      - cfg.image.apply_border_crop
      - cfg.image.apply_clahe
      - cfg.image.clahe_clip_limit
      - cfg.image.clahe_tile_grid

    Example tag:
      v2_s256_crop1_clahe1_clip2.0_grid8x8
    """
    if cfg is None:
        cfg = load_config()

    grid = cfg.image.clahe_tile_grid
    grid_str = f"{grid[0]}x{grid[1]}"

    tag = (
        f"v{PREPROCESS_VERSION}"
        f"_s{cfg.image.size}"
        f"_crop{int(cfg.image.apply_border_crop)}"
        f"_clahe{int(cfg.image.apply_clahe)}"
        f"_clip{float(cfg.image.clahe_clip_limit):.1f}"
        f"_grid{grid_str}"
    )
    return tag


def cache_dir(cfg: Optional[DictConfig] = None) -> pathlib.Path:
    """
    Return the root directory for the current cache tag:
      <cfg.output.cache_root>/<tag>
    """
    if cfg is None:
        cfg = load_config()
    tag = compute_cache_tag(cfg)
    return pathlib.Path(cfg.output.cache_root) / tag


def cache_path(source: str, image_id: str, cfg: Optional[DictConfig] = None) -> pathlib.Path:
    """
    Return the expected filesystem path for a single cached image.

    Parameters
    ----------
    source : str
        One of 'aptos', 'eyepacs_train', 'eyepacs_test', 'messidor2', 'idrid'.
    image_id : str
        Filename with extension (e.g. '10_left.jpg', '0a4e1a29fc2d.png').
    cfg : DictConfig, optional
        Active configuration.

    Returns
    -------
    pathlib.Path
        Path ending in .png inside the tagged cache directory.
    """
    if cfg is None:
        cfg = load_config()

    stem = pathlib.Path(image_id).stem
    return cache_dir(cfg) / source / f"{stem}.png"


def get_manifest_path(cfg: Optional[DictConfig] = None) -> pathlib.Path:
    """Return path to manifest.json for the active cache tag."""
    return cache_dir(cfg) / "manifest.json"


def get_current_git_commit() -> str:
    """Retrieve HEAD git commit hash if available, otherwise return 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def write_manifest(
    counts: dict[str, int],
    cfg: Optional[DictConfig] = None,
    git_commit: Optional[str] = None,
) -> pathlib.Path:
    """
    Write manifest.json atomically to the tag folder.

    Parameters
    ----------
    counts : dict[str, int]
        Number of images cached per source.
    cfg : DictConfig, optional
        Active configuration.
    git_commit : str, optional
        Commit hash; auto-detected if None.
    """
    if cfg is None:
        cfg = load_config()

    if git_commit is None:
        git_commit = get_current_git_commit()

    tag = compute_cache_tag(cfg)
    c_dir = cache_dir(cfg)
    c_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "cache_tag": tag,
        "preprocess_version": PREPROCESS_VERSION,
        "git_commit": git_commit,
        "settings": {
            "image_size": int(cfg.image.size),
            "apply_border_crop": bool(cfg.image.apply_border_crop),
            "apply_clahe": bool(cfg.image.apply_clahe),
            "clahe_clip_limit": float(cfg.image.clahe_clip_limit),
            "clahe_tile_grid": list(cfg.image.clahe_tile_grid),
        },
        "counts": counts,
    }

    manifest_file = get_manifest_path(cfg)
    tmp_file = c_dir / f".tmp_{os.getpid()}_manifest.json"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    os.replace(tmp_file, manifest_file)
    return manifest_file


def read_manifest(cfg: Optional[DictConfig] = None) -> dict[str, Any]:
    """Read manifest.json for the active cache tag."""
    manifest_file = get_manifest_path(cfg)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Cache manifest not found at: {manifest_file}. "
            "Please run scripts/build_cache.py to generate the preprocessing cache."
        )
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_cache_manifest(cfg: Optional[DictConfig] = None) -> dict[str, Any]:
    """
    Verify that the existing cache manifest matches the current configuration.

    Raises
    ------
    FileNotFoundError
        If manifest.json does not exist.
    ValueError
        If settings, version, or tag do not match current config.
    """
    if cfg is None:
        cfg = load_config()

    manifest = read_manifest(cfg)
    expected_tag = compute_cache_tag(cfg)

    if manifest.get("cache_tag") != expected_tag:
        raise ValueError(
            f"Cache tag mismatch! Manifest has '{manifest.get('cache_tag')}', "
            f"current config generates '{expected_tag}'."
        )

    if manifest.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError(
            f"PREPROCESS_VERSION mismatch! Manifest has {manifest.get('preprocess_version')}, "
            f"current code is {PREPROCESS_VERSION}."
        )

    st = manifest.get("settings", {})
    if st.get("image_size") != int(cfg.image.size):
        raise ValueError(f"image_size mismatch: manifest={st.get('image_size')}, cfg={cfg.image.size}")
    if st.get("apply_border_crop") != bool(cfg.image.apply_border_crop):
        raise ValueError(f"apply_border_crop mismatch: manifest={st.get('apply_border_crop')}, cfg={cfg.image.apply_border_crop}")
    if st.get("apply_clahe") != bool(cfg.image.apply_clahe):
        raise ValueError(f"apply_clahe mismatch: manifest={st.get('apply_clahe')}, cfg={cfg.image.apply_clahe}")
    if float(st.get("clahe_clip_limit", 0)) != float(cfg.image.clahe_clip_limit):
        raise ValueError(f"clahe_clip_limit mismatch: manifest={st.get('clahe_clip_limit')}, cfg={cfg.image.clahe_clip_limit}")
    if list(st.get("clahe_tile_grid", [])) != list(cfg.image.clahe_tile_grid):
        raise ValueError(f"clahe_tile_grid mismatch: manifest={st.get('clahe_tile_grid')}, cfg={list(cfg.image.clahe_tile_grid)}")

    return manifest
