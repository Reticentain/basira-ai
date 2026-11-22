"""
src/basira/data/loaders.py
==========================
Unified dataset loaders for Diabetic Retinopathy fundus datasets:
  - APTOS 2019
  - EyePACS (resized mirror)
  - Messidor-2 (external test)
  - IDRiD (external test)

Every loader returns a standardized pandas DataFrame with schema:
  - path: str (absolute path to image on disk)
  - image_id: str (image filename including extension, e.g. '10_left.jpg')
  - grade: int (0 to 4)
  - referable: int (1 if grade >= 2 else 0)
  - patient_id: str | None (EyePACS: text before first '_'; others: None)
  - eye: str | None ('left', 'right', or None)
  - source: str ('aptos', 'eyepacs', 'messidor2', 'idrid')

All paths are read dynamically from the OmegaConf config (never hardcoded).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from basira.config import load_config
from basira.data.splits import eyepacs_patient_id

UNIFIED_COLUMNS = [
    "path",
    "image_id",
    "grade",
    "referable",
    "patient_id",
    "eye",
    "source",
]


def rebuild_path(source: str, image_id: str, cfg: Optional[DictConfig] = None) -> str:
    """
    Reconstruct the full absolute file path from source and image_id using config.

    Parameters
    ----------
    source : str
        One of 'aptos', 'eyepacs_train', 'eyepacs_test', 'messidor2', 'idrid'.
    image_id : str
        Filename with extension (e.g. '10_left.jpg', '0a4e1a29fc2d.png').
    cfg : DictConfig, optional
        Loaded config. If None, `load_config()` is called.

    Returns
    -------
    str
        Full path to image.
    """
    if cfg is None:
        cfg = load_config()

    if source == "aptos":
        base_dir = pathlib.Path(cfg.data.aptos.images)
    elif source in ("eyepacs", "eyepacs_train"):
        # Could be in train or test folder; check train first, fallback to test
        p_train = pathlib.Path(cfg.data.eyepacs.train_images) / image_id
        if p_train.exists():
            return str(p_train)
        return str(pathlib.Path(cfg.data.eyepacs.test_images) / image_id)
    elif source == "eyepacs_test":
        base_dir = pathlib.Path(cfg.data.eyepacs.test_images)
    elif source == "messidor2":
        base_dir = pathlib.Path(cfg.data.messidor2.images)
    elif source == "idrid":
        base_dir = pathlib.Path(cfg.data.idrid.images)
    else:
        raise ValueError(f"Unknown source '{source}' for rebuilding path.")

    return str(base_dir / image_id)


def load_aptos(cfg: Optional[DictConfig] = None) -> pd.DataFrame:
    """
    Load APTOS 2019 dataset metadata into unified schema.

    Labels CSV expected columns: id_code, diagnosis (0-4).
    Images are *.png files in cfg.data.aptos.images.
    """
    if cfg is None:
        cfg = load_config()

    csv_path = pathlib.Path(cfg.data.aptos.labels_csv)
    img_dir = pathlib.Path(cfg.data.aptos.images)

    df_raw = pd.read_csv(csv_path)

    # Standardize column names
    col_map = {c.lower(): c for c in df_raw.columns}
    id_col = col_map.get("id_code", df_raw.columns[0])
    diag_col = col_map.get("diagnosis", df_raw.columns[1])

    rows = []
    for _, row in df_raw.iterrows():
        id_code = str(row[id_col]).strip()
        image_id = f"{id_code}.png" if not id_code.endswith(".png") else id_code
        img_path = str(img_dir / image_id)
        grade = int(row[diag_col])

        rows.append({
            "path": img_path,
            "image_id": image_id,
            "grade": grade,
            "referable": 1 if grade >= 2 else 0,
            "patient_id": None,  # APTOS provides no patient IDs
            "eye": None,
            "source": "aptos",
        })

    return pd.DataFrame(rows)[UNIFIED_COLUMNS]


def load_eyepacs(cfg: Optional[DictConfig] = None, split: str = "train") -> pd.DataFrame:
    """
    Load EyePACS metadata into unified schema.

    Parameters
    ----------
    split : str
        'train' for official trainLabels15.csv / resized train 15,
        'test' for official testLabels15.csv / resized test 15.
    """
    if cfg is None:
        cfg = load_config()

    if split == "train":
        csv_path = pathlib.Path(cfg.data.eyepacs.train_labels)
        img_dir = pathlib.Path(cfg.data.eyepacs.train_images)
    elif split == "test":
        csv_path = pathlib.Path(cfg.data.eyepacs.test_labels)
        img_dir = pathlib.Path(cfg.data.eyepacs.test_images)
    else:
        raise ValueError(f"EyePACS split must be 'train' or 'test', got '{split}'")

    df_raw = pd.read_csv(csv_path)

    col_map = {c.lower(): c for c in df_raw.columns}
    img_col = col_map.get("image", df_raw.columns[0])
    level_col = col_map.get("level", df_raw.columns[1])

    rows = []
    for _, row in df_raw.iterrows():
        raw_name = str(row[img_col]).strip()
        # Ensure .jpg extension
        image_id = f"{raw_name}.jpg" if not raw_name.endswith(".jpg") else raw_name
        img_path = str(img_dir / image_id)
        grade = int(row[level_col])

        # EyePACS filename pattern: <patient_id>_<left|right>.jpg
        pid = eyepacs_patient_id(image_id, sep=cfg.data.eyepacs.get("patient_id_sep", "_"))
        eye = "left" if "_left" in image_id.lower() else ("right" if "_right" in image_id.lower() else None)

        rows.append({
            "path": img_path,
            "image_id": image_id,
            "grade": grade,
            "referable": 1 if grade >= 2 else 0,
            "patient_id": pid,
            "eye": eye,
            "source": f"eyepacs_{split}",
        })

    return pd.DataFrame(rows)[UNIFIED_COLUMNS]


def load_messidor2(cfg: Optional[DictConfig] = None) -> pd.DataFrame:
    """
    Load Messidor-2 dataset metadata into unified schema (external test only).

    Labels CSV expected columns: image, grade.
    """
    if cfg is None:
        cfg = load_config()

    csv_path = pathlib.Path(cfg.data.messidor2.labels_csv)
    img_dir = pathlib.Path(cfg.data.messidor2.images)

    df_raw = pd.read_csv(csv_path)

    col_map = {c.lower(): c for c in df_raw.columns}
    img_col = col_map.get("image", df_raw.columns[0])
    grade_col = col_map.get("grade", df_raw.columns[1])

    rows = []
    for _, row in df_raw.iterrows():
        image_name = str(row[img_col]).strip()
        # Keep extension if present, otherwise default to .jpg
        if "." not in image_name:
            image_id = f"{image_name}.jpg"
        else:
            image_id = image_name

        img_path = str(img_dir / image_id)
        grade = int(row[grade_col])

        rows.append({
            "path": img_path,
            "image_id": image_id,
            "grade": grade,
            "referable": 1 if grade >= 2 else 0,
            "patient_id": None,
            "eye": None,
            "source": "messidor2",
        })

    return pd.DataFrame(rows)[UNIFIED_COLUMNS]


def load_idrid(cfg: Optional[DictConfig] = None) -> pd.DataFrame:
    """
    Load IDRiD dataset metadata into unified schema (external test only, 516 images).

    Labels CSV expected columns: image, grade, orig_split.
    """
    if cfg is None:
        cfg = load_config()

    csv_path = pathlib.Path(cfg.data.idrid.labels_csv)
    img_dir = pathlib.Path(cfg.data.idrid.images)

    df_raw = pd.read_csv(csv_path)

    col_map = {c.lower(): c for c in df_raw.columns}
    img_col = col_map.get("image", df_raw.columns[0])
    grade_col = col_map.get("grade", df_raw.columns[1])

    rows = []
    for _, row in df_raw.iterrows():
        image_name = str(row[img_col]).strip()
        if "." not in image_name:
            image_id = f"{image_name}.jpg"
        else:
            image_id = image_name

        img_path = str(img_dir / image_id)
        grade = int(row[grade_col])

        rows.append({
            "path": img_path,
            "image_id": image_id,
            "grade": grade,
            "referable": 1 if grade >= 2 else 0,
            "patient_id": None,
            "eye": None,
            "source": "idrid",
        })

    return pd.DataFrame(rows)[UNIFIED_COLUMNS]


def load_dataset(name: str, cfg: Optional[DictConfig] = None, split: str = "train") -> pd.DataFrame:
    """Convenience wrapper to load any dataset by name."""
    name_clean = name.lower().replace("-", "").replace("_", "")
    if "aptos" in name_clean:
        return load_aptos(cfg)
    elif "eyepacs" in name_clean:
        return load_eyepacs(cfg, split=split)
    elif "messidor" in name_clean:
        return load_messidor2(cfg)
    elif "idrid" in name_clean:
        return load_idrid(cfg)
    else:
        raise ValueError(f"Unknown dataset name '{name}'")
