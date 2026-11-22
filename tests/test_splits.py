"""
tests/test_splits.py
====================
Validation suite for dataset split CSV files:
  1. Safe execution: tests that read real split files automatically pytest.skip()
     if the CSV files have not been generated yet.
  2. EyePACS split integrity:
     - Total train + val equals exactly 35,126 rows.
     - Test equals exactly 53,576 rows.
     - Zero patient_id leakage across train and val splits.
     - Both eyes of each patient remain in the same split (never divided).
  3. APTOS split integrity:
     - Train + val + test equals the deduped total (no hardcoded post-dedup count).
     - Zero image_id overlap across train, val, and test.
  4. Messidor-2 & IDRiD integrity:
     - Messidor-2 has 1,744 images.
     - IDRiD has 516 images.
"""

import pathlib
import pytest
import pandas as pd

from basira.config import load_config
from basira.data.splits import assert_no_patient_leakage

# Search paths for split CSVs: local reports/splits or Drive splits
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL_SPLITS_DIR = _REPO_ROOT / "reports" / "splits"


def get_split_path(filename: str) -> pathlib.Path | None:
    """Find split CSV in local reports/splits or configured Drive path."""
    # Check local mirror first
    p_local = LOCAL_SPLITS_DIR / filename
    if p_local.exists():
        return p_local

    # Fallback to configured output directory
    try:
        cfg = load_config()
        p_drive = pathlib.Path(cfg.output.splits) / filename
        if p_drive.exists():
            return p_drive
    except Exception:
        pass

    return None


class TestRealDatasetSplits:
    """Tests executing against generated split CSVs (skipped if files not yet generated)."""

    @pytest.fixture(autouse=True)
    def check_splits_exist(self):
        """Skip test suite if split CSVs are not present."""
        ep_train = get_split_path("eyepacs_train.csv")
        ep_val = get_split_path("eyepacs_val.csv")
        if ep_train is None or ep_val is None:
            pytest.skip("Dataset split CSV files not generated yet. Run scripts/create_splits.py in Colab first.")

    def test_eyepacs_split_counts_and_sizes(self):
        """Hardcode only EyePACS: exactly 35,126 (train+val) and 53,576 (test)."""
        ep_train_path = get_split_path("eyepacs_train.csv")
        ep_val_path = get_split_path("eyepacs_val.csv")
        ep_test_path = get_split_path("eyepacs_test.csv")

        df_train = pd.read_csv(ep_train_path)
        df_val = pd.read_csv(ep_val_path)

        # 1. Train + Val row count must equal 35,126
        train_val_total = len(df_train) + len(df_val)
        assert train_val_total == 35126, (
            f"EyePACS train + val count mismatch: expected 35126, got {train_val_total}"
        )

        # 2. Test row count must equal 53,576 (if test split CSV is present)
        if ep_test_path is not None and ep_test_path.exists():
            df_test = pd.read_csv(ep_test_path)
            assert len(df_test) == 53576, (
                f"EyePACS test count mismatch: expected 53576, got {len(df_test)}"
            )

    def test_eyepacs_patient_leakage_and_both_eyes_together(self):
        """
        Verify:
          (a) Zero patient_id leakage across EyePACS train and val.
          (b) Both eyes of each patient stay in the same split.
        """
        df_train = pd.read_csv(get_split_path("eyepacs_train.csv"))
        df_val = pd.read_csv(get_split_path("eyepacs_val.csv"))

        train_patients = set(df_train["patient_id"].astype(str))
        val_patients = set(df_val["patient_id"].astype(str))

        # (a) No patient in both splits
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, (
            f"EyePACS patient leakage detected: {len(overlap)} patients found in both train and val: {list(overlap)[:5]}"
        )

        # Call the strict assert_no_patient_leakage guard
        assert_no_patient_leakage({
            "train": df_train["patient_id"].astype(str).tolist(),
            "val": df_val["patient_id"].astype(str).tolist(),
        })

        # (b) Both eyes check: within each split, patients with 2 images have both in the same split
        # Combine and ensure every patient's images all reside in either train or val exclusively
        for pid, group in pd.concat([df_train, df_val]).groupby("patient_id"):
            in_train = pid in train_patients
            in_val = pid in val_patients
            # Exactly one of in_train or in_val must be True
            assert in_train != in_val, f"Patient {pid} images split across train and val!"

    def test_aptos_split_sums_to_deduped_total(self):
        """Do not hardcode post-dedup count: test that train + val + test == deduped_total."""
        apt_train_path = get_split_path("aptos_train.csv")
        apt_val_path = get_split_path("aptos_val.csv")
        apt_test_path = get_split_path("aptos_test.csv")
        apt_dedup_path = get_split_path("aptos_deduped.csv")

        if any(p is None for p in [apt_train_path, apt_val_path, apt_test_path, apt_dedup_path]):
            pytest.skip("APTOS split CSV files missing.")

        df_train = pd.read_csv(apt_train_path)
        df_val = pd.read_csv(apt_val_path)
        df_test = pd.read_csv(apt_test_path)
        df_dedup = pd.read_csv(apt_dedup_path)

        total_split_rows = len(df_train) + len(df_val) + len(df_test)
        total_dedup_rows = len(df_dedup)

        assert total_split_rows == total_dedup_rows, (
            f"APTOS split row sum mismatch: train({len(df_train)}) + val({len(df_val)}) + "
            f"test({len(df_test)}) = {total_split_rows} != deduped total ({total_dedup_rows})"
        )

        # Zero image_id overlap across the three partitions
        train_ids = set(df_train["image_id"])
        val_ids = set(df_val["image_id"])
        test_ids = set(df_test["image_id"])

        assert len(train_ids.intersection(val_ids)) == 0, "Overlap between APTOS train and val"
        assert len(train_ids.intersection(test_ids)) == 0, "Overlap between APTOS train and test"
        assert len(val_ids.intersection(test_ids)) == 0, "Overlap between APTOS val and test"

    def test_messidor_and_idrid_external_counts(self):
        """Verify external test set sizes: Messidor-2 has 1,744 images, IDRiD has 516."""
        mess_path = get_split_path("messidor2_test.csv")
        idrid_path = get_split_path("idrid_test.csv")

        if mess_path is not None and mess_path.exists():
            df_mess = pd.read_csv(mess_path)
            assert len(df_mess) == 1744, f"Messidor-2 count expected 1744, got {len(df_mess)}"

        if idrid_path is not None and idrid_path.exists():
            df_idrid = pd.read_csv(idrid_path)
            assert len(df_idrid) == 516, f"IDRiD count expected 516, got {len(df_idrid)}"
