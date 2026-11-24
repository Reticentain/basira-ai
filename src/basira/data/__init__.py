"""
basira.data
===========
Data loading, preprocessing, deduplication, and patient-level splitting.
"""

from basira.data.cache import (
    cache_dir,
    cache_path,
    compute_cache_tag,
    validate_cache_manifest,
)
from basira.data.dedup import compute_dhash_256, deduplicate_aptos
from basira.data.loaders import (
    UNIFIED_COLUMNS,
    load_aptos,
    load_dataset,
    load_eyepacs,
    load_idrid,
    load_messidor2,
    read_split_csv,
    rebuild_path,
)
from basira.data.preprocess import (
    PREPROCESS_VERSION,
    crop_and_enhance,
    crop_black_borders,
    pad_to_square,
    preprocess_image,
    to_tensor_and_normalize,
)
from basira.data.splits import (
    aptos_patient_id,
    assert_no_patient_leakage,
    eyepacs_patient_id,
    patient_split,
    stratified_image_split,
)

__all__ = [
    "load_dataset",
    "load_aptos",
    "load_eyepacs",
    "load_messidor2",
    "load_idrid",
    "read_split_csv",
    "rebuild_path",
    "UNIFIED_COLUMNS",
    "PREPROCESS_VERSION",
    "crop_black_borders",
    "pad_to_square",
    "crop_and_enhance",
    "to_tensor_and_normalize",
    "preprocess_image",
    "compute_dhash_256",
    "deduplicate_aptos",
    "patient_split",
    "stratified_image_split",
    "assert_no_patient_leakage",
    "eyepacs_patient_id",
    "aptos_patient_id",
    "cache_dir",
    "cache_path",
    "compute_cache_tag",
    "validate_cache_manifest",
]
