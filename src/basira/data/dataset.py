"""
src/basira/data/dataset.py
==========================
PyTorch Dataset for loading pre-cached Stage-1 fundus images.

Key requirements:
  - Loads lossless 256x256 PNGs from /content/cache/<tag>/<source>/<stem>.png
  - On construction: validates manifest.json (ensuring preprocess_version and settings match)
  - On construction: verifies that all required cached files exist; lists first 10 if missing
  - Training augmentations: random horizontal/vertical flip, random rotation up to 20 deg,
    small brightness/contrast jitter, mild random scale (0.9-1.1 with zero fill)
  - Validation/Test: no augmentations (only ToTensor and ImageNet normalisation)
  - Labels: "referable" (1.0 if grade >= 2 else 0.0, float32) or "grade" (0-4, int64)
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from basira.config import load_config
from basira.data.cache import cache_path, validate_cache_manifest


def resolve_cache_source(source: str) -> str:
    """Map split sources to their corresponding cache subdirectories."""
    src = str(source).lower().strip()
    if src in ("eyepacs", "eyepacs_train", "eyepacs_val"):
        return "eyepacs_train"
    elif src == "eyepacs_test":
        return "eyepacs_test"
    elif src in ("aptos", "aptos_train", "aptos_val", "aptos_test", "aptos_deduped"):
        return "aptos"
    elif src in ("messidor", "messidor2", "messidor2_test"):
        return "messidor2"
    elif src in ("idrid", "idrid_test"):
        return "idrid"
    return src


class CachedFundusDataset(Dataset):
    """
    Dataset reading from the Stage-1 preprocessed PNG cache.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['image_id', 'grade', 'source', ...].
    cfg : DictConfig, optional
        Loaded OmegaConf configuration.
    train : bool, default True
        If True, applies training augmentations. If False, only normalizes.
    task : str, default "referable"
        'referable' -> binary float32 target (1.0 if grade >= 2 else 0.0)
        'grade'     -> multiclass int64 target (0 to 4)
    check_files : bool, default True
        If True, validates that all image cache files exist on construction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Optional[DictConfig] = None,
        train: bool = True,
        task: str = "referable",
        check_files: bool = True,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.train = train
        self.task = task.lower().strip()

        if self.task not in ("referable", "grade"):
            raise ValueError(f"Task must be 'referable' or 'grade', got '{self.task}'")

        # 1. Validate manifest.json against active configuration
        # Raises ValueError or FileNotFoundError if cache is missing or invalidated
        validate_cache_manifest(self.cfg)

        # 2. Extract image records and construct expected cache paths
        self.records: list[dict[str, Any]] = []
        missing_paths: list[pathlib.Path] = []

        for idx, row in df.iterrows():
            iid = str(row["image_id"]).strip()
            raw_source = str(row.get("source", "eyepacs_train"))
            c_source = resolve_cache_source(raw_source)
            cp = cache_path(c_source, iid, self.cfg)

            grade = int(row["grade"])
            if self.task == "referable":
                label = 1.0 if grade >= 2 else 0.0
            else:
                label = grade

            record = {
                "image_id": iid,
                "source": c_source,
                "cache_path": cp,
                "grade": grade,
                "label": label,
            }
            self.records.append(record)

            if check_files and not cp.exists():
                missing_paths.append(cp)

        # 3. Fail immediately if any required cached files are missing
        if check_files and missing_paths:
            first_10 = "\n".join(f"  - {p}" for p in missing_paths[:10])
            raise FileNotFoundError(
                f"CachedFundusDataset: {len(missing_paths)} cached images are missing! "
                f"First {min(10, len(missing_paths))} missing:\n{first_10}\n"
                "Please build the Stage-1 cache first with scripts/build_cache.py."
            )

        # 4. Construct image transformations
        mean = list(self.cfg.image.mean)
        std = list(self.cfg.image.std)

        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=20,
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec["cache_path"]

        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")

        tensor = self.transform(img_rgb)

        if self.task == "referable":
            label = torch.tensor(rec["label"], dtype=torch.float32)
        else:
            label = torch.tensor(rec["label"], dtype=torch.long)

        return tensor, label
