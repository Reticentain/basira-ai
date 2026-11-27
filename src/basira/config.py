"""
src/basira/config.py
====================
Central config loader for Basira AI.

Every other module imports `load_config()` instead of reading YAML directly,
so the merge logic lives in exactly one place.

Priority (highest wins)
-----------------------
1. `override_path` argument, if supplied.
2. `configs/local.yaml`, if it exists (git-ignored; machine- or run-specific).
3. `BASIRA_DATA_ROOT` environment variable (overrides data.root only).
4. `configs/default.yaml` (committed; Colab defaults).

Usage
-----
    from basira.config import load_config
    cfg = load_config()

    # Dot-access:
    print(cfg.data.aptos.images)   # /content/data/aptos/train_images
    print(cfg.image.size)          # 256
    print(cfg.seeds.global_seed)   # 42
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root is three levels above this file:
#   src/basira/config.py  →  src/basira/  →  src/  →  repo root
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CFG = _REPO_ROOT / "configs" / "default.yaml"
_LOCAL_CFG   = _REPO_ROOT / "configs" / "local.yaml"   # git-ignored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(override_path: Optional[str | Path] = None) -> DictConfig:
    """
    Load, merge, and return the Basira AI configuration as an OmegaConf
    DictConfig.

    Parameters
    ----------
    override_path : str or Path, optional
        Path to an additional YAML file whose values override everything else.
        Useful for per-experiment configs (e.g. configs/mobilenet_run.yaml).

    Returns
    -------
    omegaconf.DictConfig
        Merged config with dot-access and OmegaConf interpolation resolved.

    Raises
    ------
    FileNotFoundError
        If `configs/default.yaml` is missing.
    """
    if not _DEFAULT_CFG.exists():
        raise FileNotFoundError(
            f"Default config not found: {_DEFAULT_CFG}\n"
            "Make sure you are running from the repo root, or that the "
            "configs/ directory was not moved."
        )

    cfg = OmegaConf.load(_DEFAULT_CFG)

    # Layer 2: machine-local overrides (git-ignored)
    if _LOCAL_CFG.exists():
        local = OmegaConf.load(_LOCAL_CFG)
        cfg = OmegaConf.merge(cfg, local)

    # Layer 3: environment variable (convenient for Colab one-liners)
    env_root = os.environ.get("BASIRA_DATA_ROOT")
    if env_root:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"data": {"root": env_root}})
        )

    # Layer 4: per-run override file
    if override_path is not None:
        extra = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(cfg, extra)

    return cfg


def apply_overrides(cfg: DictConfig, items: list[str]) -> DictConfig:
    """
    Apply CLI key=value override strings to an OmegaConf DictConfig in-place.

    Coerces:
      - 'true' / 'false' (case-insensitive) -> bool
      - integer strings -> int
      - floating-point strings -> float
      - all other values remain str

    Parameters
    ----------
    cfg : DictConfig
        Configuration object to modify.
    items : list of str
        List of strings formatted as 'key=value' or 'section.key=value'.

    Returns
    -------
    DictConfig
        The modified configuration object.
    """
    if not items:
        return cfg

    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
            k_clean = k.strip()
            v_clean = v.strip()
            # Try type coercion (int, float, bool)
            if v_clean.lower() == "true":
                v_coerced: Any = True
            elif v_clean.lower() == "false":
                v_coerced = False
            else:
                try:
                    v_coerced = int(v_clean)
                except ValueError:
                    try:
                        v_coerced = float(v_clean)
                    except ValueError:
                        v_coerced = v_clean
            OmegaConf.update(cfg, k_clean, v_coerced)

    return cfg


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the resolved configuration (useful at the top of a run)."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        console = Console()
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        console.print(Syntax(yaml_str, "yaml", theme="monokai"))
    except ImportError:
        # Fall back to plain print if rich is not installed yet
        print(OmegaConf.to_yaml(cfg, resolve=True))
