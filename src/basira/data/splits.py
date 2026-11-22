"""
basira.data.splits
==================
Train / val / test splitting utilities with a leakage guard.

RULE: one patient MUST appear in exactly one split.
      `assert_no_patient_leakage()` enforces this and is called by pytest
      (tests/test_patient_split.py) and by every split function below.

Design decisions
----------------
* EyePACS  - patient ID is the text before the first "_" in the filename
             ("1234_left.jpg" -> "1234"). Both eyes of one patient share an ID, so
             they always land in the same split. The official EyePACS test set is
             used as the test set, so the official train set is split into
             train/val only (patient_split with test_frac=0).
* APTOS    - no patient IDs exist. We split BY IMAGE, stratified by grade, after
             near-duplicate removal. We cannot guarantee that two images of the same
             patient stay together, and reports must say so.
* Messidor-2 / IDRiD - external test only; no splitting.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# ---------------------------------------------------------------------------
# Patient-ID extractors
# ---------------------------------------------------------------------------

def eyepacs_patient_id(filename: str, sep: str = "_") -> str:
    """
    Extract the patient ID from an EyePACS filename or image name.

    >>> eyepacs_patient_id("10_left.jpg")
    '10'
    >>> eyepacs_patient_id("10_left")
    '10'
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem.split(sep, 1)[0]


def aptos_patient_id(id_code: str) -> str:
    """
    APTOS has no patient IDs, so each image is treated as its own "patient".
    This does NOT guarantee that images of one real patient stay together.
    """
    return id_code


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

def assert_no_patient_leakage(splits: Dict[str, Iterable[str]]) -> None:
    """
    Raise AssertionError if any patient ID appears in more than one split.

    A patient appearing several times INSIDE one split (for example both eyes)
    is fine; only appearing in two different splits is a leak.

    Parameters
    ----------
    splits : dict mapping split name -> iterable of patient IDs (repeats allowed)
    """
    seen: Dict[str, str] = {}                     # patient_id -> first split name
    leaks: List[Tuple[str, str, str]] = []

    for split_name, patient_ids in splits.items():
        for pid in set(patient_ids):              # set(): repeats inside a split are OK
            if pid in seen:
                leaks.append((pid, seen[pid], split_name))
            else:
                seen[pid] = split_name

    if leaks:
        details = "\n".join(
            f"  patient '{pid}' in '{first}' AND '{second}'"
            for pid, first, second in leaks[:20]
        )
        raise AssertionError(
            f"Patient-level split leakage detected! "
            f"{len(leaks)} patient(s) appear in multiple splits:\n{details}"
        )


# ---------------------------------------------------------------------------
# Grouped (patient-level) split: EyePACS
# ---------------------------------------------------------------------------

def patient_split(
    df: pd.DataFrame,
    patient_col: str,
    val_frac: float,
    test_frac: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / val / test so every patient is in exactly one part.

    With test_frac=0.0 only a train/val split is made and the returned test
    DataFrame is empty (used for EyePACS, whose official test set is separate).
    Fractions refer to the share of PATIENTS (not rows).
    """
    if val_frac <= 0 or val_frac + test_frac >= 1.0:
        raise ValueError("Need val_frac > 0 and val_frac + test_frac < 1.0")
    if patient_col not in df.columns:
        raise KeyError(f"Column '{patient_col}' not found in DataFrame")

    df = df.reset_index(drop=True)

    if test_frac > 0:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(df, groups=df[patient_col].values))
        df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
    else:
        df_trainval = df
        df_test = df.iloc[0:0].copy()

    relative_val_frac = val_frac / (1.0 - test_frac)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed + 1)
    train_idx, val_idx = next(
        gss_val.split(df_trainval, groups=df_trainval[patient_col].values)
    )
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    # Fail fast instead of silently producing leaky splits.
    assert_no_patient_leakage(
        {
            "train": df_train[patient_col].tolist(),
            "val": df_val[patient_col].tolist(),
            "test": df_test[patient_col].tolist(),
        }
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Stratified image-level split: APTOS (no patient IDs)
# ---------------------------------------------------------------------------

def stratified_image_split(
    df: pd.DataFrame,
    label_col: str,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    id_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by IMAGE, stratified by *label_col* (the DR grade), into train/val/test.

    Use only for datasets without patient IDs (APTOS). Images should be
    de-duplicated before calling this. If *id_col* is given, the function checks
    that no image id appears in two splits.
    """
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1.0:
        raise ValueError("Need val_frac > 0, test_frac > 0 and val_frac + test_frac < 1.0")

    df = df.reset_index(drop=True)

    df_trainval, df_test = train_test_split(
        df, test_size=test_frac, stratify=df[label_col], random_state=seed
    )
    relative_val_frac = val_frac / (1.0 - test_frac)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=relative_val_frac,
        stratify=df_trainval[label_col],
        random_state=seed + 1,
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if id_col is not None:
        assert_no_patient_leakage(
            {
                "train": df_train[id_col].tolist(),
                "val": df_val[id_col].tolist(),
                "test": df_test[id_col].tolist(),
            }
        )
    return df_train, df_val, df_test