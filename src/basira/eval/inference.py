"""
src/basira/eval/inference.py
============================
Inference utilities, checkpoint loading, model construction, and split path discovery.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from scipy.special import expit
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def find_split_file(filename: str, cfg: DictConfig) -> pathlib.Path:
    """Locate a split CSV in Drive output or local reports/splits directory."""
    candidates = [
        pathlib.Path(cfg.output.splits) / filename,
        _REPO_ROOT / "reports" / "splits" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Split file '{filename}' not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nPlease ensure splits have been generated."
    )


def load_eval_checkpoint(
    filepath: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint containing model weights, config, and metadata.

    Loads with map_location="cpu" and weights_only=False (for full dicts and RNG states).
    """
    return torch.load(filepath, map_location=map_location, weights_only=False)


def build_and_load_model(
    ckpt_path: pathlib.Path | str,
    device: torch.device,
    backbone_override: Optional[str] = None,
    cfg: Optional[DictConfig] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Instantiate model backbone and load trained weights.

    Parameters
    ----------
    ckpt_path : Path or str
        Path to checkpoint (best.pt).
    device : torch.device
        Device to move model to.
    backbone_override : str, optional
        Explicit model architecture override.
    cfg : DictConfig, optional
        Loaded configuration.

    Returns
    -------
    model : nn.Module
    ckpt : dict[str, Any]
    """
    ckpt = load_eval_checkpoint(ckpt_path, map_location="cpu")

    # Determine backbone name
    backbone = None
    if backbone_override:
        backbone = backbone_override
    elif isinstance(ckpt, dict) and ckpt.get("config"):
        cfg_in_ckpt = ckpt["config"]
        if isinstance(cfg_in_ckpt, dict):
            backbone = cfg_in_ckpt.get("model", {}).get("backbone")
        elif hasattr(cfg_in_ckpt, "model"):
            backbone = getattr(cfg_in_ckpt.model, "backbone", None)
    if not backbone and cfg is not None and hasattr(cfg, "model") and hasattr(cfg.model, "backbone"):
        backbone = cfg.model.backbone
    if not backbone:
        backbone = "efficientnet_b0"

    timm_name = "mobilenetv3_large_100" if backbone in ("mobilenetv3", "mobilenetv3_large_100") else backbone

    # Task is binary referable DR: 1 logit output
    model = timm.create_model(timm_name, pretrained=False, num_classes=1)

    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ckpt if isinstance(ckpt, dict) else {}


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass across DataLoader and return raw logits and sigmoid probabilities.

    Returns
    -------
    logits : np.ndarray of shape (N,)
    probs : np.ndarray of shape (N,)
    """
    model.eval()
    all_logits: list[float] = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(images).view(-1)
            else:
                out = model(images).view(-1)
            all_logits.extend(out.cpu().tolist())

    logits = np.array(all_logits, dtype=np.float64)
    probs = expit(logits)
    return logits, probs
